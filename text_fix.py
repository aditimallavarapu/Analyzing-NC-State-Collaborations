"""
text_fix.py

Repairs mojibake (double-encoded UTF-8) in the publication text columns.

The OpenAlex/faculty export is valid UTF-8 on disk, but the text was
mis-decoded upstream (UTF-8 bytes read as Latin-1/cp1252, then saved as UTF-8
again), so e.g. "’" shows up as "â\\x80\\x99", "±" as "Â±", "ü" as "Ã¼".
ftfy undoes this and leaves already-correct text alone. Characters that were
already destroyed upstream (U+FFFD "�") cannot be recovered; they are counted
and reported.
"""

from __future__ import annotations

import re

import pandas as pd

try:
    import ftfy
except ImportError as e:  # pragma: no cover
    raise SystemExit("ftfy is required for UTF-8 repair: pip install ftfy") from e

# Typical leftovers of double-encoding: Ã/Â/â followed by a C1 control or a
# cp1252 "smart" glyph (€ ™ œ etc.), e.g. "â\x80\x99", "â€™", "Ã¼", "Â±".
_MOJIBAKE_RE = re.compile(
    "[ÂÃâÎ×][\u0080-¿€‚ƒ„"
    "…†‡ˆ‰Š‹ŒŽ‘’"
    "“”•–—˜™š›œžŸ]"
)
_C1_RE = re.compile("[\u0080-\u009f]")


def fix_text(value):
    """Repair one string; non-strings (NaN etc.) are returned unchanged."""
    if not isinstance(value, str):
        return value
    # NFC so accents are composed consistently; no other normalisation
    # (keep quotes/dashes as real Unicode rather than ASCII-fying them).
    return ftfy.fix_text(
        value, normalization="NFC", uncurl_quotes=False, fix_character_width=False
    )


def count_suspect(series: pd.Series) -> dict:
    """Counts of remaining mojibake-looking cells, C1 controls and U+FFFD."""
    s = series.dropna().astype(str)
    return {
        "mojibake_cells": int(s.str.contains(_MOJIBAKE_RE).sum()),
        "c1_control_cells": int(s.str.contains(_C1_RE).sum()),
        "replacement_char_cells": int(s.str.contains("�", regex=False).sum()),
    }


def repair_columns(df: pd.DataFrame, columns, verbose: bool = True) -> pd.DataFrame:
    """Apply fix_text to the given text columns (those present) and report.

    Prints a warning if recognisable mojibake survives the repair.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        before = count_suspect(df[col])
        original = df[col]
        df[col] = original.map(fix_text)
        n_changed = int(((original != df[col]) & original.notna()).sum())
        after = count_suspect(df[col])
        if verbose:
            print(
                f"[utf8] {col}: repaired {n_changed} cells | "
                f"mojibake {before['mojibake_cells']}->{after['mojibake_cells']}, "
                f"C1 controls {before['c1_control_cells']}->{after['c1_control_cells']}, "
                f"unrecoverable U+FFFD cells {after['replacement_char_cells']}",
                flush=True,
            )
        if after["mojibake_cells"] or after["c1_control_cells"]:
            print(
                f"[utf8] WARNING: column '{col}' still has suspect characters "
                f"after repair ({after}); inspect the source file.",
                flush=True,
            )
    return df
