"""
01_clean_data.py  (v2)
"""
import pandas as pd, re
from pathlib import Path

PUB_IN="publications.csv"; FAC_IN="Faculty.csv"
PUB_OUT="publications_clean.csv"; FAC_OUT="faculty_clean.csv"
COUNTS_OUT="faculty_pub_counts.csv"; TOP10_OUT="top10_collaborators.csv"

VALID_COLLEGES = {
    "College of Agriculture and Life Sciences","College of Design","College of Education",
    "College of Engineering","College of Humanities and Social Sciences",
    "College of Natural Resources","College of Sciences","College of Veterinary Medicine",
    "Poole College of Management","Wilson College of Textiles",
}

print("STEP 1 — Faculty")
fac_raw = pd.read_csv(FAC_IN, low_memory=False)
fac = fac_raw[['uid','name','shib_role','shib_title','college','department']].copy()
for c in fac.columns: fac[c] = fac[c].apply(lambda x: str(x).strip() if pd.notna(x) else '')
fac = fac[(fac['uid']!='')&(fac['name']!='')].drop_duplicates('uid',keep='first').reset_index(drop=True)
fac['is_asst_prof'] = fac['shib_title'].apply(lambda t: 'asst' in str(t).lower() or 'assistant' in str(t).lower())
print(f"  {len(fac)} faculty, {fac['is_asst_prof'].sum()} asst profs")
fac.to_csv(FAC_OUT, index=False)
uid_to_info = fac.set_index('uid').to_dict('index')
valid_uids  = set(fac['uid'])

print("STEP 2 — Publications")
pub_raw = pd.read_csv(PUB_IN, low_memory=False)
keep=['title','authors','nc_state_people','DOI','PMID','year','url','topics','abstract','openalex_cited_by_count']
pub = pub_raw[keep].copy()
pub['year'] = pd.to_numeric(pub['year'], errors='coerce')
pub = pub[pub['year'].between(2006,2025)].copy()
pub['year'] = pub['year'].astype(int)
pub = pub[pub['nc_state_people'].notna()&(pub['nc_state_people'].str.strip()!='')].copy()
def normalise_nc(v):
    if pd.isna(v) or not str(v).strip(): return ''
    return '; '.join(p.strip() for p in str(v).split(';') if p.strip())
pub['nc_state_people'] = pub['nc_state_people'].apply(normalise_nc)
pub_doi = pub[pub['DOI'].notna()].drop_duplicates('DOI',keep='first')
pub_ndoi= pub[pub['DOI'].isna()].drop_duplicates(subset=['title','year'],keep='first')
pub = pd.concat([pub_doi,pub_ndoi],ignore_index=True)
pub['title'] = pub['title'].fillna('').str.strip()
pub = pub[pub['title']!=''].copy()
print(f"  {len(pub)} papers after dedup")

def parse_people(val):
    if not val: return []
    out=[]
    for part in str(val).split(';'):
        part=part.strip()
        m=re.match(r'^(.+?)\s*\(([^)]+)\)\s*$',part)
        if m: out.append((m.group(1).strip(),m.group(2).strip()))
        elif part: out.append((part,''))
    return out

def filter_to_faculty(val):
    ppl=parse_people(val)
    kept=[f'{n} ({u})' for n,u in ppl if u in valid_uids]
    return '; '.join(kept) if kept else ''

pub['nc_state_people_all'] = pub['nc_state_people']
pub['nc_state_people']     = pub['nc_state_people'].apply(filter_to_faculty)
pub = pub[pub['nc_state_people']!=''].copy().reset_index(drop=True)
pub = pub.sort_values('year').reset_index(drop=True)
print(f"  {len(pub)} papers with faculty")
pub.to_csv(PUB_OUT, index=False)

print("STEP 3 — Counts")
rows=[]
for _,row in pub.iterrows():
    for (name,uid) in parse_people(row['nc_state_people']):
        if uid in valid_uids: rows.append({'uid':uid,'name':name,'year':row['year']})
person_df=pd.DataFrame(rows)
counts=(person_df.groupby(['uid','name'])
        .agg(total_pubs=('year','count'),first_year=('year','min'),last_year=('year','max'))
        .reset_index().sort_values('total_pubs',ascending=False))
counts=counts.merge(fac[['uid','college','department','shib_title','is_asst_prof']],on='uid',how='left')
counts['college']=counts['college'].fillna('Unknown')
counts['department']=counts['department'].fillna('')
counts['is_asst_prof']=counts['is_asst_prof'].fillna(False)
counts.to_csv(COUNTS_OUT, index=False)
print(f"  {len(counts)} faculty with publications")

print("STEP 4 — Top 10 per college")
collab_counts={}
for _,row in pub.iterrows():
    ppl=[uid for (name,uid) in parse_people(row['nc_state_people']) if uid in valid_uids]
    if len(ppl)<2: continue
    for uid in ppl:
        if uid not in collab_counts: collab_counts[uid]=set()
        for other in ppl:
            if other!=uid: collab_counts[uid].add(other)
counts['n_collaborators']=counts['uid'].map(lambda u:len(collab_counts.get(u,set())))

top10_rows=[]
for college in sorted(VALID_COLLEGES):
    sub=counts[counts['college']==college].sort_values('total_pubs',ascending=False).head(10).copy()
    sub['rank']=range(1,len(sub)+1); sub['college_group']=college
    top10_rows.append(sub)
top10=pd.concat(top10_rows,ignore_index=True)
top10=top10[['college_group','rank','uid','name','shib_title','is_asst_prof','total_pubs','first_year','last_year','n_collaborators','department']]
top10.to_csv(TOP10_OUT, index=False)

print("\nTOP 10 PER COLLEGE:")
for college, grp in top10.groupby('college_group'):
    print(f"\n{college}")
    for _,r in grp.iterrows():
        flag="  ★ Asst Prof" if r['is_asst_prof'] else ""
        print(f"  {r['rank']:>2}. {r['name']:<35} {r['total_pubs']:>4} pubs{flag}")
print("\nDONE")
