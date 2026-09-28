"""
Pluggable embedding-model registry for step 6 (topic modeling).

Design: a `BaseEmbedder` abstract class plus a `register_embedder("key")`
class decorator and a `get_embedder(key, **kwargs)` factory. `extract_
embeddings.py` never names a model class directly — it looks the key up
in the registry — so adding a new backbone (including a future API-based
one) means writing one class and decorating it. Nothing else changes.

To add a new embedding model:
  1. Subclass BaseEmbedder.
  2. Implement `prepare_input(title, abstract) -> str` (how the model's
     original training convention combines title/abstract into one string)
     and `encode_batch(texts) -> np.ndarray` (tokenize/call API, pool,
     return an (len(texts), dim) float32 array).
  3. Decorate the class with @register_embedder("your-key").
  4. Either add the class to this file, or put it in a new module and
     `import` that module from here so the decorator runs.
  5. Select it via `--model your-key` — no other code needs to change.
"""
from __future__ import annotations

import abc
import os
from typing import Callable

import numpy as np
import torch

_EMBEDDER_REGISTRY: dict[str, type["BaseEmbedder"]] = {}


def register_embedder(key: str) -> Callable[[type], type]:
    def _decorator(cls: type) -> type:
        if key in _EMBEDDER_REGISTRY:
            raise ValueError(
                f"embedder key '{key}' already registered to "
                f"{_EMBEDDER_REGISTRY[key].__name__}"
            )
        _EMBEDDER_REGISTRY[key] = cls
        return cls
    return _decorator


def get_embedder(key: str, **kwargs) -> "BaseEmbedder":
    if key not in _EMBEDDER_REGISTRY:
        raise KeyError(
            f"Unknown embedder '{key}'. Available: {available_embedders()}"
        )
    return _EMBEDDER_REGISTRY[key](**kwargs)


def available_embedders() -> list[str]:
    return sorted(_EMBEDDER_REGISTRY)


class BaseEmbedder(abc.ABC):
    """
    One embedder = one model + its input convention + its pooling rule.

    Batching strategy, fp16 autocast, length-bucketing, and
    device selection all live in extract_embeddings.py and are identical
    for every backend — subclasses only need to know how to turn a
    (title, abstract) pair into a string, and a batch of strings into
    vectors.
    """

    #: embedding dimensionality; subclasses must set this after loading
    dim: int

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        fp16: bool = True,
        max_length: int = 512,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.fp16 = bool(fp16) and str(device).startswith("cuda")
        self.max_length = max_length
        self.cache_dir = cache_dir
        if cache_dir:
            # libraries that ignore a cache_dir= kwarg (e.g. adapters'
            # load_adapter) still honor this; must run before they import
            os.environ["HF_HUB_CACHE"] = str(cache_dir)

    @abc.abstractmethod
    def prepare_input(self, title: str, abstract: str) -> str:
        """Combine title/abstract into the single string this model expects."""

    @abc.abstractmethod
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), self.dim) float32 array of embeddings."""


def _clean(text) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    return "" if text.lower() == "nan" else text


def _cls_pool_encode(model, tokenizer, texts, device, max_length, fp16):
    """Shared forward pass + CLS-token pooling used by SPECTER2 and SciNCL."""
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    ).to(device)
    autocast_device = "cuda" if fp16 else "cpu"
    with torch.no_grad(), torch.autocast(
        device_type=autocast_device, enabled=fp16, dtype=torch.float16
    ):
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :]
    return cls.float().cpu().numpy()


@register_embedder("specter2")
class Specter2Embedder(BaseEmbedder):
    """
    SPECTER2 (allenai/specter2_base + the "proximity" adapter from
    allenai/specter2). Trained on `title [SEP] abstract` with CLS-token
    pooling of the last hidden state.

    specter2_base ships with no task head of its own — it is an
    adapter-hub base model that needs an adapter attached to produce the
    similarity-tuned embeddings we actually want. Loading it via plain
    transformers.AutoModel would silently give the un-adapted base model,
    so this uses AutoAdapterModel from the `adapters` package instead.
    """

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        adapter_name: str = "allenai/specter2",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir
        )
        self.model = AutoAdapterModel.from_pretrained(
            model_name, cache_dir=self.cache_dir
        )
        self.adapter_name_loaded = self.model.load_adapter(
            adapter_name, source="hf", load_as="proximity", set_active=True,
        )
        # set_active=True was not enough on some `adapters` versions (the
        # forward pass warned "adapters available but none activated"),
        # which silently gives plain-base-model embeddings. Set it
        # explicitly and refuse to continue if it still isn't active.
        self.model.set_active_adapters(self.adapter_name_loaded)
        if not self.model.active_adapters:
            raise RuntimeError("SPECTER2 'proximity' adapter is not active; "
                               "refusing to embed with the plain base model")
        print(f"[info] active adapter(s): {self.model.active_adapters}")
        self.model.to(self.device).eval()
        self.dim = self.model.config.hidden_size

    def prepare_input(self, title: str, abstract: str) -> str:
        return f"{_clean(title)}{self.tokenizer.sep_token}{_clean(abstract)}"

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return _cls_pool_encode(
            self.model, self.tokenizer, texts, self.device,
            self.max_length, self.fp16,
        )


@register_embedder("scincl")
class SciNCLEmbedder(BaseEmbedder):
    """
    SciNCL (malteos/scincl). Same convention as SPECTER2: `title [SEP]
    abstract` input, CLS-token pooling of the last hidden state. Unlike
    SPECTER2 it is a plain fine-tuned transformers model with no adapter
    required.
    """

    def __init__(self, model_name: str = "malteos/scincl", **kwargs):
        super().__init__(model_name, **kwargs)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=self.cache_dir
        )
        self.model.to(self.device).eval()
        self.dim = self.model.config.hidden_size

    def prepare_input(self, title: str, abstract: str) -> str:
        return f"{_clean(title)}{self.tokenizer.sep_token}{_clean(abstract)}"

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return _cls_pool_encode(
            self.model, self.tokenizer, texts, self.device,
            self.max_length, self.fp16,
        )


@register_embedder("scibert")
class SciBERTEmbedder(BaseEmbedder):
    """
    SciBERT (allenai/scibert_scivocab_uncased). This is scientific-domain
    BERT, NOT a model fine-tuned for sentence/document similarity, so its
    CLS token has no special pooling meaning the way SPECTER2/SciNCL's
    does. Uses attention-mask-weighted mean pooling over token embeddings
    instead — the standard off-the-shelf choice for a model with no
    embedding-specific pretraining head. Expect this to be a meaningfully
    weaker baseline than the other two backbones; that's expected, not a
    bug.
    """

    def __init__(
        self, model_name: str = "allenai/scibert_scivocab_uncased", **kwargs
    ):
        super().__init__(model_name, **kwargs)
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name, cache_dir=self.cache_dir
        )
        self.model.to(self.device).eval()
        self.dim = self.model.config.hidden_size

    def prepare_input(self, title: str, abstract: str) -> str:
        title, abstract = _clean(title), _clean(abstract)
        return f"{title}. {abstract}" if abstract else title

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        autocast_device = "cuda" if self.fp16 else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=autocast_device, enabled=self.fp16, dtype=torch.float16
        ):
            out = self.model(**enc)
        token_embeddings = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled.float().cpu().numpy()


@register_embedder("gtr_t5")
class GTRT5Embedder(BaseEmbedder):
    """
    GTR-T5 (sentence-transformers/gtr-t5-large: a T5 encoder with mean
    pooling and a projection to 768 dimensions). A general-purpose, not
    scientific, model. No instruction prefix is used. Always runs in fp32
    (T5 models can overflow in fp16), which is cheap for a 0.3B model.
    """

    def __init__(self, model_name: str = "sentence-transformers/gtr-t5-large", **kwargs):
        super().__init__(model_name, **kwargs)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name, device=self.device, cache_folder=self.cache_dir
        )
        self.model.max_seq_length = self.max_length
        self.dim = self.model.get_sentence_embedding_dimension()

    def prepare_input(self, title: str, abstract: str) -> str:
        title, abstract = _clean(title), _clean(abstract)
        return f"{title}. {abstract}" if abstract else title

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=len(texts), convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)


@register_embedder("mxbai")
class MxbaiEmbedder(BaseEmbedder):
    """
    mxbai-embed-large-v1 (mixedbread-ai/mxbai-embed-large-v1: a BERT-large
    sized model, 1024 dimensions). A general-purpose, not scientific, model.
    Documents take no instruction prefix (only search queries do). Runs in
    fp32, which is cheap for a 0.3B model.
    """

    def __init__(self, model_name: str = "mixedbread-ai/mxbai-embed-large-v1", **kwargs):
        super().__init__(model_name, **kwargs)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name, device=self.device, cache_folder=self.cache_dir
        )
        self.model.max_seq_length = self.max_length
        self.dim = self.model.get_sentence_embedding_dimension()

    def prepare_input(self, title: str, abstract: str) -> str:
        title, abstract = _clean(title), _clean(abstract)
        return f"{title}. {abstract}" if abstract else title

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=len(texts), convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
