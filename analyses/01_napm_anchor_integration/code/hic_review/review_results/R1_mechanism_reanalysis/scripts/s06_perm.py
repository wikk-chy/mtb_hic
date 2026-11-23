import pandas as pd, numpy as np, json
from scipy import stats
ROOT='/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review'
OUT=f'{ROOT}/hic_review/review_results/R1_mechanism_reanalysis'
NBINS=883; BIN=5000; GENOME=4411532
NPERM=10000
rng=np.random.default_rng(42)
shifts=rng.integers(0,NBINS,NPERM)

b=pd.read_csv(f'{OUT}/03_structure_metrics/bin_level_v2.csv')
pairs=pd.read_csv(f'{OUT}/05_matched_controls/matched_pairs.csv')
loops=pd.read_csv(f'{OUT}/02_loop_curated/loops_41_curated.tsv',sep='\t')
pk=pd.read_csv(f'{OUT}/01_chip_curated/NapM_binding_curated.tsv',sep='\t')
g=pd.read_csv(f'{OUT}/04_rna_integration/gene_level_rna_v2.csv')
gpairs=pd.read_csv(f'{OUT}/05_matched_controls/matched_pairs_genes.csv')

bound=b['chip_bound'].values.astype(bool)
absd=np.abs(b['d_ins_KO_WT'].values)

def p_ge(obs,null): return (1+np.sum(null>=obs))/(1+len(null))
def p_le(obs,null): return (1+np.sum(null<=obs))/(1+len(null))
def p_2s(obs,null): return (1+np.sum(np.abs(null)>=abs(obs)))/(1+len(null))

summary=[]  # rows: test, observed, null_mean, effect, ci_lo, ci_hi, p, p_type, n, null_model, notes

# ============ T1: Fig1B metaprofile ============
peak_bins=np.sort(pk['center_bin'].unique())
offsets=np.arange(-20,21)
def metaprofile(pbins):
    prof=np.empty(len(offsets))
    for k,off in enumerate(offsets):
        prof[k]=np.nanmean(absd[(pbins+off)%NBINS])
    return prof
obs_prof=metaprofile(peak_bins)
null_prof=np.empty((NPERM,len(offsets)))
for i,s in enumerate(shifts):
    null_prof[i]=metaprofile((peak_bins+s)%NBINS)
np.save(f'{OUT}/06_permutation_tests/null_fig1b_metaprofile.npy',null_prof)
env_lo=np.percentile(null_prof,2.5,axis=0)
env_hi=np.percentile(null_prof,97.5,axis=0)
mp=pd.DataFrame(dict(offset_bins=offsets,offset_kb=offsets*5,observed=obs_prof,
                     null_mean=null_prof.mean(axis=0),null_lo95=env_lo,null_hi95=env_hi))
mp.to_csv(f'{OUT}/06_permutation_tests/fig1b_metaprofile.csv',index=False)

# window enrichment: bins within W of any peak-center bin
def window_bins(pbins,W):
    idx=set()
    for p in pbins:
        for o in range(-W,W+1):
            idx.add((p+o)%NBINS)
    return np.array(sorted(idx))
null_win={}
for W,lab in [(1,'5kb'),(2,'10kb'),(5,'25kb')]:
    wb=window_bins(peak_bins,W)
    obs=np.nanmean(absd[wb])
    ns=np.empty(NPERM)
    for i,s in enumerate(shifts):
        ns[i]=np.nanmean(absd[window_bins((peak_bins+s)%NBINS,W)])
    null_win[lab]=ns
    np.save(f'{OUT}/06_permutation_tests/null_fig1b_window_{lab}.npy',ns)
    summary.append(dict(test=f'Fig1B_window_pm{lab}',observed=obs,null_mean=ns.mean(),
        effect=obs-ns.mean(),ci_lo=np.nan,ci_hi=np.nan,p=p_ge(obs,ns),p_type='one-sided(obs>=null)',
        n=len(wb),null_model='circular-shift of peak-center bins (10000)',notes='mean |d_ins_KO_WT| in window'))

# ============ T2: Fig1C continuous ============
def spearman_rows(x,y,label):
    m=~(np.isnan(x)|np.isnan(y))
    r,p=stats.spearmanr(x[m],y[m])
    summary.append(dict(test=label,observed=r,null_mean=np.nan,effect=r,ci_lo=np.nan,ci_hi=np.nan,
        p=p,p_type='two-sided asymptotic',n=int(m.sum()),null_model='none (Spearman)',notes=''))
score=b['chip_bin_score_max'].values; vpm=b['chip_bin_vpm_max'].values
spearman_rows(score,absd,'Fig1C_spearman_score_vs_absdins_allbins')
spearman_rows(np.log10(vpm+1),absd,'Fig1C_spearman_logvpm_vs_absdins_allbins')
spearman_rows(score[bound],absd[bound],'Fig1C_spearman_score_vs_absdins_boundonly')
spearman_rows(np.log10(vpm[bound]+1),absd[bound],'Fig1C_spearman_logvpm_vs_absdins_boundonly')

# ============ paired-test machinery (shift null keeps pair offsets) ============
def bootstrap_ci(diffs,n=10000):
    bs=rng.choice(diffs,size=(n,len(diffs)),replace=True).mean(axis=1)
    return np.percentile(bs,[2.5,97.5])
def paired_test(value,label,value_name):
    B=pairs['bound_bin'].values; M=pairs['matched_bin'].values
    vb=value[B]; vm=value[M]
    ok=~(np.isnan(vb)|np.isnan(vm))
    B2,M2=B[ok],M[ok]
    diffs=value[B2]-value[M2]
    obs=diffs.mean()
    ci=bootstrap_ci(diffs)
    wp=stats.wilcoxon(diffs).pvalue if len(diffs)>10 else np.nan
    ns=np.empty(NPERM)
    for i,s in enumerate(shifts):
        ns[i]=np.nanmean(value[(B2+s)%NBINS]-value[(M2+s)%NBINS])
    np.save(f'{OUT}/06_permutation_tests/null_{label}.npy',ns)
    mw=stats.mannwhitneyu(value[B2],value[M2]).pvalue
    summary.append(dict(test=label,observed=obs,null_mean=ns.mean(),effect=obs,
        ci_lo=ci[0],ci_hi=ci[1],p=p_2s(obs,ns),p_type='two-sided circular-shift(10000)',
        n=int(ok.sum()),null_model='circular-shift of matched pair offsets',
        notes=f'{value_name}; paired Wilcoxon p={wp:.3g}; naive MW p={mw:.3g} (naive)'))

paired_test(absd,'Fig1Da_paired_absdins','|d_ins_KO_WT|')
paired_test(b['d_localcont_KO_WT'].values,'Fig1Db_paired_dlocalcont_signed','d_localcont_KO_WT (signed)')
paired_test(np.abs(b['d_localcont_KO_WT'].values),'Fig1Db_paired_abslocalcontmag','|d_localcont_KO_WT|')

# ============ T3c: loops ============
b1=loops['bin1'].values; b2=loops['bin2'].values
obs_both=int((bound[b1]&bound[b2]).sum())
obs_either=int((bound[b1]|bound[b2]).sum())
ns_both=np.empty(NPERM); ns_either=np.empty(NPERM)
for i,s in enumerate(shifts):
    b1s=(b1+s)%NBINS; b2s=(b2+s)%NBINS
    ns_both[i]=(bound[b1s]&bound[b2s]).sum()
    ns_either[i]=(bound[b1s]|bound[b2s]).sum()
np.save(f'{OUT}/06_permutation_tests/null_fig1dc_loops_both.npy',ns_both)
np.save(f'{OUT}/06_permutation_tests/null_fig1dc_loops_either.npy',ns_either)
summary.append(dict(test='Fig1Dc_loops_both_bound',observed=obs_both,null_mean=ns_both.mean(),
    effect=obs_both-ns_both.mean(),ci_lo=np.nan,ci_hi=np.nan,p=p_ge(obs_both,ns_both),
    p_type='one-sided(obs>=null)',n=41,null_model='circular-shift of whole loop set (10000)',
    notes=f'null 2.5/97.5 pct = {np.percentile(ns_both,[2.5,97.5])}'))
summary.append(dict(test='Fig1Dc_loops_either_bound',observed=obs_either,null_mean=ns_either.mean(),
    effect=obs_either-ns_either.mean(),ci_lo=np.nan,ci_hi=np.nan,p=p_ge(obs_either,ns_either),
    p_type='one-sided(obs>=null)',n=41,null_model='circular-shift of whole loop set (10000)',
    notes=f'null 2.5/97.5 pct = {np.percentile(ns_either,[2.5,97.5])}'))

# ============ T3d: structure_detected enrichment ============
det=b['structure_detected'].values.astype(bool)
a=int((bound&det).sum()); bb=int((bound&~det).sum()); c=int((~bound&det).sum()); d=int((~bound&~det).sum())
odds,p_fisher=stats.fisher_exact([[a,bb],[c,d]])
se=np.sqrt(1/a+1/bb+1/c+1/d); ci_or=np.exp(np.log(odds)+np.array([-1.96,1.96])*se)
obs_ov=a
ns=np.empty(NPERM)
bindices=np.where(bound)[0]
for i,s in enumerate(shifts):
    bm=np.zeros(NBINS,bool); bm[(bindices+s)%NBINS]=True
    ns[i]=(bm&det).sum()
np.save(f'{OUT}/06_permutation_tests/null_fig1dd_detected_overlap.npy',ns)
summary.append(dict(test='Fig1Dd_bound_x_structure_detected',observed=obs_ov,null_mean=ns.mean(),
    effect=odds,ci_lo=ci_or[0],ci_hi=ci_or[1],p=p_ge(obs_ov,ns),
    p_type='one-sided(obs>=null)',n=NBINS,null_model='circular-shift of bound mask (10000)',
    notes=f'Fisher OR={odds:.3f} ({ci_or[0]:.3f}-{ci_or[1]:.3f}), Fisher p={p_fisher:.3g}; table=[[{a},{bb}],[{c},{d}]]'))

# ============ T3e: CID boundary distance ============
bd=pd.read_csv(f'{OUT}/02_loop_curated/wt_cid_boundaries_curated.tsv',sep='\t')
bdbins=np.sort(bd['bin'].unique())
def mean_min_dist(bm):
    idx=np.where(bm)[0][:,None]
    d=np.abs(idx-bdbins[None,:]); d=np.minimum(d,NBINS-d)
    return d.min(axis=1).mean()
obs_dist=mean_min_dist(bound)
ns=np.empty(NPERM)
for i,s in enumerate(shifts):
    bm=np.zeros(NBINS,bool); bm[(bindices+s)%NBINS]=True
    ns[i]=mean_min_dist(bm)
np.save(f'{OUT}/06_permutation_tests/null_fig1de_cid_dist.npy',ns)
summary.append(dict(test='Fig1De_bound_dist_to_wtCIDboundary',observed=obs_dist,null_mean=ns.mean(),
    effect=obs_dist-ns.mean(),ci_lo=np.nan,ci_hi=np.nan,p=p_le(obs_dist,ns),
    p_type='one-sided(obs<=null)',n=int(bound.sum()),null_model='circular-shift of bound mask (10000)',
    notes=f'mean min distance in bins (x5kb); null 2.5/97.5={np.percentile(ns,[2.5,97.5])}'))

# ============ T5: Fig3 hypoxia interaction ============
paired_test(b['d_ins_interaction'].values,'Fig3_paired_d_ins_interaction','d_ins_interaction')
paired_test(np.abs(b['d_ins_interaction'].values),'Fig3_paired_abs_d_ins_interaction','|d_ins_interaction|')
paired_test(b['d_lrfrac_interaction'].values,'Fig3_paired_d_lrfrac_interaction','d_lrfrac_interaction')

sdf=pd.DataFrame(summary)
sdf.to_csv(f'{OUT}/06_permutation_tests/stats_fig1_fig3_summary.csv',index=False)
print(sdf[['test','observed','null_mean','effect','p','n']].to_string())
