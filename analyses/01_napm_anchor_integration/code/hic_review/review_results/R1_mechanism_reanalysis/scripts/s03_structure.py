import pandas as pd, numpy as np
from scipy import stats
ROOT='/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review'
OUT=f'{ROOT}/hic_review/review_results/R1_mechanism_reanalysis'
NBINS=883; BIN=5000
rng=np.random.default_rng(42)

# ---------- local contact (circular distance <= 10 bins), per-rep O/E ----------
# O/E expected = per-matrix mean of each circular diagonal (excluding zeros/NaN).
# localcont[bin] = mean of O/E over partners with circular distance 1..10.
def load_matrix(path):
    a=np.loadtxt(path)
    assert a.shape[1]==3
    i=a[:,0].astype(int); j=a[:,1].astype(int); v=a[:,2]
    assert i.min()>=1 and i.max()<=NBINS and j.min()>=1 and j.max()<=NBINS  # 1-based
    m=np.zeros((NBINS,NBINS))
    m[i-1,j-1]=v
    return m

def local_contact(m):
    idx=np.arange(NBINS)
    i,j=np.meshgrid(idx,idx,indexing='ij')
    s=(j-i)%NBINS                     # circular shift offset
    dc=np.minimum(s,NBINS-s)          # circular distance
    exp=np.full(NBINS,np.nan)
    for off in range(1,NBINS//2+1):
        vals=np.concatenate([np.diag(m,off),np.diag(m,-off)]) if off*2!=NBINS else np.diag(m,off)
        vals=vals[(vals>0)&np.isfinite(vals)]
        exp[off]=vals.mean() if len(vals) else np.nan
    expmap=np.where(dc==0,np.nan,exp[dc])
    with np.errstate(invalid='ignore',divide='ignore'):
        oe=np.where((m>0)&np.isfinite(m)&np.isfinite(expmap), m/expmap, np.nan)
    oe[dc==0]=np.nan
    localmask=(dc>=1)&(dc<=10)
    oe_loc=np.where(localmask,oe,np.nan)
    lc=np.nanmean(oe_loc,axis=1)
    return lc

res={}
for cond,pref,n in [('WT','wt',4),('KO','napm_ko',4)]:
    for r in range(1,n+1):
        m=load_matrix(f'{ROOT}/hic_review/review_data/rep_icedmatrix/{pref}_{r}_5000_iced.matrix')
        res[f'localcont_{cond}_rep{r}']=local_contact(m)
        print('done',cond,r)

lc=pd.DataFrame(res)
lc.insert(0,'bin',np.arange(NBINS))
wt_cols=[f'localcont_WT_rep{r}' for r in range(1,5)]
ko_cols=[f'localcont_KO_rep{r}' for r in range(1,5)]
lc['localcont_WT_mean']=lc[wt_cols].mean(axis=1)
lc['localcont_KO_mean']=lc[ko_cols].mean(axis=1)
lc['d_localcont_KO_WT']=lc['localcont_KO_mean']-lc['localcont_WT_mean']
# replicate-level Welch CI on the delta
def welch_ci(a,b):
    # a,b: arrays (nbins, nrep)
    d=a.mean(axis=1)-b.mean(axis=1)
    se=np.sqrt(a.var(axis=1,ddof=1)/a.shape[1]+b.var(axis=1,ddof=1)/b.shape[1])
    return d-1.96*se, d+1.96*se
lo,hi=welch_ci(lc[ko_cols].values,lc[wt_cols].values)
lc['d_localcont_KO_WT_lo']=lo; lc['d_localcont_KO_WT_hi']=hi
lc.to_csv(f'{OUT}/03_structure_metrics/local_contact_perbin.csv',index=False)

# ---------- bin_level_v2 ----------
b=pd.read_csv(f'{ROOT}/hic_review/review_results/R1_mechanism/R1_mechanism_zhenghe/tables/bin_level_integrated.csv')
DEPRECATED=['chip_mean_signal','chip_bound','chip_bound_alt','bound_region_frac','matched_background','matched_to_bin']
missing=[c for c in DEPRECATED if c not in b.columns]
assert not missing, missing
b=b.drop(columns=DEPRECATED)
bind=pd.read_csv(f'{OUT}/01_chip_curated/bin_level_binding.csv')
b=b.merge(bind,on='bin',how='left',validate='1:1')
b=b.merge(lc[['bin','localcont_WT_mean','localcont_KO_mean','d_localcont_KO_WT','d_localcont_KO_WT_lo','d_localcont_KO_WT_hi']],on='bin',how='left',validate='1:1')
b.to_csv(f'{OUT}/03_structure_metrics/bin_level_v2.csv',index=False)
print('bin_level_v2:',b.shape)
print('bound bins:',b['chip_bound'].sum())
print('d_localcont describe:',b['d_localcont_KO_WT'].describe().to_dict())
print('NaN d_localcont:',b['d_localcont_KO_WT'].isna().sum())
