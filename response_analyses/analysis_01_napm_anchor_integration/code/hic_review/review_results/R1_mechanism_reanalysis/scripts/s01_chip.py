import pandas as pd, numpy as np, re
ROOT='/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review'
OUT=f'{ROOT}/hic_review/review_results/R1_mechanism_reanalysis'
GENOME=4411532; BIN=5000; NBINS=883

# ---- parse annotation ----
genes=[]
pat=re.compile(r'gene_id "([^"]+)"')
with open(f'{ROOT}/hic_review/annotation.gtf') as fh:
    for line in fh:
        if line.startswith('#'): continue
        f=line.rstrip('\n').split('\t')
        if len(f)<9 or f[2]!='gene': continue
        m=pat.search(f[8])
        if not m: continue
        genes.append(dict(gene_id=m.group(1),start=int(f[3]),end=int(f[4]),strand=f[6]))
ann=pd.DataFrame(genes).drop_duplicates('gene_id').reset_index(drop=True)
ann['tss']=np.where(ann['strand']=='+',ann['start'],ann['end'])
print('annotated genes:',len(ann))

# ---- load peaks ----
x=pd.read_excel(f'{ROOT}/hic_review/review_data/chip_seq_data/41467_2015_BFncomms6829_MOESM46_ESM.xlsx',
                sheet_name='Sheet1',header=2)
# drop embedded Heading/Descriptor rows (Unnamed:10 non-null)
mask_heading=x['Unnamed: 10'].notna()
pk=x[(x['Regulator']=='Rv0047c') & (~mask_heading)].copy()
assert len(pk)==634, len(pk)
cur=pd.DataFrame(dict(
    chrom='Chromosome',
    start=pk['Peak Start (Fstart)'].astype(int),
    end=pk['Peak Stop (Rstop)'].astype(int),
    center=pk['Peak Center (Ccenter)'].astype(float),
    target_gene=pk['Gene'].astype(str),
    score=pk['Score'].astype(float),
    vpm=pk['VPM'].astype(int),
    p_value=pk['p-Value'].astype(float),
    source_sheet='41467_2015_BFncomms6829_MOESM46_ESM.xlsx:Sheet1(header=2,Regulator==Rv0047c)'))
cur=cur.sort_values('center').reset_index(drop=True)
cur['center_bin']=(cur['center']//BIN).astype(int)
assert cur['center_bin'].max()<NBINS
cur.to_csv(f'{OUT}/01_chip_curated/NapM_binding_curated.tsv',sep='\t',index=False)
print('peaks:',len(cur),'unique genes(col):',cur['target_gene'].nunique())

# ---- peak location classification vs annotation (coordinate-based, strand-aware) ----
locs=[]; nearest=[]; ndist=[]
cs=cur['center'].values
tss=ann['tss'].values.astype(float)
for c in cs:
    d=np.abs(tss-c); j=int(np.argmin(d))
    nearest.append(ann['gene_id'].iloc[j]); ndist.append(d[j])
cur['nearest_tss_gene']=nearest
cur['dist_to_nearest_tss_bp']=ndist
def classify(c):
    # gene body?
    inb=ann[(ann['start']<=c)&(ann['end']>=c)]
    if len(inb): return 'gene_body', ';'.join(inb['gene_id'])
    # promoter: within 300bp upstream of a TSS (strand-aware)
    up=ann.copy()
    dist=np.where(up['strand']=='+', up['start']-c, c-up['end'])
    cand=up[(dist>=0)&(dist<=300)]
    if len(cand): return 'promoter_up300',';'.join(cand['gene_id'])
    return 'intergenic',''
res=[classify(c) for c in cs]
cur['peak_location']=[r[0] for r in res]
cur['location_gene']=[r[1] for r in res]

# adjacent peak spacing (circular)
cen=cur['center'].values
gaps=np.diff(cen)
gaps=np.append(gaps, GENOME-cen[-1]+cen[0])
qc={}
qc['n_peaks']=len(cur)
qc['n_bound_bins(center rule)']=cur['center_bin'].nunique()
qc['n_unique_target_gene_col']=cur['target_gene'].nunique()
qc['score_min/med/max']=(cur['score'].min(),cur['score'].median(),cur['score'].max())
qc['vpm_min/med/max']=(cur['vpm'].min(),cur['vpm'].median(),cur['vpm'].max())
qc['vpm_median']=cur['vpm'].median()
qc['pvalue_med']=cur['p_value'].median()
qc['peak_width_med_bp']=(cur['end']-cur['start']).median()
qc['gap_bp_min/med/max']=(gaps.min(),np.median(gaps),gaps.max())
qc['location_counts']=cur['peak_location'].value_counts().to_dict()
cur['peak_location'].value_counts().to_csv(f'{OUT}/01_chip_curated/qc_peak_location_counts.csv')
pd.Series(gaps,name='gap_bp').to_frame().to_csv(f'{OUT}/01_chip_curated/qc_adjacent_peak_gaps.csv',index=False)
cur.groupby('center_bin').size().value_counts().sort_index().to_csv(f'{OUT}/01_chip_curated/qc_peaks_per_bin_counts.csv')

# ---- bin-level mapping ----
# RULE: a peak maps to the 5kb bin containing its CENTER (floor(center/5000), 0-based).
grp=cur.groupby('center_bin')
bins=pd.DataFrame({'bin':np.arange(NBINS)})
bins['chip_bin_npeaks']=bins['bin'].map(grp.size()).fillna(0).astype(int)
bins['chip_bin_score_max']=bins['bin'].map(grp['score'].max()).fillna(0.0)
bins['chip_bin_vpm_max']=bins['bin'].map(grp['vpm'].max()).fillna(0.0)
bins['chip_bound']=bins['chip_bin_npeaks']>=1
bins['chip_bound_score09']=bins['chip_bin_score_max']>=0.9
bins['chip_bound_vpm_ge_med']=bins['chip_bin_vpm_max']>=qc['vpm_median']
bins.to_csv(f'{OUT}/01_chip_curated/bin_level_binding.csv',index=False)
qc['n_bound_bins']=int(bins['chip_bound'].sum())
qc['n_bound_bins_score09']=int(bins['chip_bound_score09'].sum())
qc['n_bound_bins_vpm_ge_med']=int(bins['chip_bound_vpm_ge_med'].sum())

import json
with open(f'{OUT}/01_chip_curated/qc_summary.json','w') as fh:
    json.dump({k:(v if not isinstance(v,(np.floating,np.integer)) else float(v)) for k,v in qc.items()},fh,indent=2,default=str)
print(json.dumps(qc,indent=2,default=str))
