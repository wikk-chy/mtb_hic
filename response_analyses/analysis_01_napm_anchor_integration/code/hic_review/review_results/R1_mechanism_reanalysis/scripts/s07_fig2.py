import pandas as pd, numpy as np, json
from scipy import stats
ROOT='/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review'
OUT=f'{ROOT}/hic_review/review_results/R1_mechanism_reanalysis'
rng=np.random.default_rng(42)

b=pd.read_csv(f'{OUT}/03_structure_metrics/bin_level_v2.csv')
g=pd.read_csv(f'{OUT}/04_rna_integration/gene_level_rna_v2.csv')
gpairs=pd.read_csv(f'{OUT}/05_matched_controls/matched_pairs_genes.csv')

# ---- Fig2a: 3x3 classification (bound x structure class x rna class), bin level ----
def cls3(s):
    return s.map(lambda v: 'detected' if str(v).startswith('detected')
                 else ('supports_small' if str(v).startswith('supports_small') else 'cannot_determine'))
b['struct3']=cls3(b['ins_change_class'])
b['rna3']=b['rna_bin_class'].map(lambda v: 'detected' if v=='detected_change'
             else ('supports_small' if v=='supports_small_change' else 'cannot_determine'))
ct=(b.groupby(['chip_bound','struct3','rna3']).size().rename('n_bins').reset_index())
ct.to_csv(f'{OUT}/06_permutation_tests/stats_fig2_classes.csv',index=False)

# ---- Fig2b: gene level bound vs matched ----
g['is_deg']=g['class_KOvsWT'].isin(['detected_increase','detected_decrease'])
g['has_rna']=g['class_KOvsWT'].notna()
g['gene_bound']=g['binding_class'].isin(['promoter_bound','gene_body_bound'])
gm=gpairs.merge(g[['gene_id','is_deg','has_rna','log2FC_KOvsWT','class_KOvsWT']],
                left_on='bound_gene',right_on='gene_id',how='left').drop(columns=['gene_id'])
gm=gm.merge(g[['gene_id','is_deg','has_rna','log2FC_KOvsWT','class_KOvsWT']],
            left_on='matched_gene',right_on='gene_id',how='left',suffixes=('_bound','_matched'))
rows=[]
def fisher_row(a_deg,a_n,b_deg,b_n,label,notes=''):
    tab=[[a_deg,a_n-a_deg],[b_deg,b_n-b_deg]]
    odds,p=stats.fisher_exact(tab)
    aa,bb_,cc,dd=tab[0][0]+0.5,tab[0][1]+0.5,tab[1][0]+0.5,tab[1][1]+0.5
    lor=np.log(aa*dd/(bb_*cc)); se=np.sqrt(1/aa+1/bb_+1/cc+1/dd)
    ci=np.exp(lor+np.array([-1.96,1.96])*se)
    rows.append(dict(test=label,observed=a_deg/a_n,effect=odds,ci_lo=ci[0],ci_hi=ci[1],p=p,
        n=a_n+b_n,null_model='Fisher exact',notes=f'{notes}; fractions {a_deg}/{a_n} vs {b_deg}/{b_n}'))

gb_ok=gm[gm['has_rna_bound']&gm['has_rna_matched']]
fisher_row(int(gb_ok['is_deg_bound'].sum()),len(gb_ok),
           int(gb_ok['is_deg_matched'].sum()),len(gb_ok),
           'Fig2b_DEGfrac_bound_vs_matched_genes','paired gene sets, genes with RNA class')
# all bound genes vs all unbound genes (naive reference)
gbd=g[g['gene_bound']&g['has_rna']]; gun=g[~g['gene_bound']&g['has_rna']]
fisher_row(int(gbd['is_deg'].sum()),len(gbd),int(gun['is_deg'].sum()),len(gun),
           'Fig2b_DEGfrac_bound_vs_unbound_genes_NAIVE','naive, unmatched')

# |log2FC| MW + bootstrap median CI
lb=gb_ok['log2FC_KOvsWT_bound'].abs().values; lm=gb_ok['log2FC_KOvsWT_matched'].abs().values
d=np.nanmedian(lb)-np.nanmedian(lm)
bs=np.empty(10000)
for i in range(10000):
    idx=rng.integers(0,len(lb),len(lb))
    bs[i]=np.nanmedian(lb[idx])-np.nanmedian(lm[idx])
ci=np.percentile(bs,[2.5,97.5])
mw=stats.mannwhitneyu(lb,lm).pvalue
wil=stats.wilcoxon(lb-lm).pvalue
rows.append(dict(test='Fig2b_abslog2FC_bound_vs_matched_genes',observed=d,effect=d,
    ci_lo=ci[0],ci_hi=ci[1],p=wil,n=len(lb),null_model='paired Wilcoxon + bootstrap CI',
    notes=f'median diff; naive MW p={mw:.3g} (naive)'))

# ---- Fig2c: joint enrichment (gene bound x bin structure change) ----
g['struct_changed']=g['bin_ins_change_class'].astype(str).str.startswith('detected')
grp_defs={'bound_structchange':(g['gene_bound']&g['struct_changed']&g['has_rna']),
          'bound_structstable':(g['gene_bound']&~g['struct_changed']&g['has_rna']),
          'unbound_structchange':(~g['gene_bound']&g['struct_changed']&g['has_rna']),
          'unbound_structstable':(~g['gene_bound']&~g['struct_changed']&g['has_rna'])}
deg={k:int(g.loc[m,'is_deg'].sum()) for k,m in grp_defs.items()}
nn={k:int(m.sum()) for k,m in grp_defs.items()}
for k in grp_defs:
    rows.append(dict(test=f'Fig2c_group_{k}',observed=deg[k]/nn[k],effect=np.nan,
        ci_lo=np.nan,ci_hi=np.nan,p=np.nan,n=nn[k],null_model='descriptive',notes=f'DEG {deg[k]}/{nn[k]}'))
fisher_row(deg['bound_structchange'],nn['bound_structchange'],deg['unbound_structchange'],nn['unbound_structchange'],
           'Fig2c_OR_boundSC_vs_unboundSC','structure-changed stratum')
fisher_row(deg['bound_structchange'],nn['bound_structchange'],deg['bound_structstable'],nn['bound_structstable'],
           'Fig2c_OR_boundSC_vs_boundSS','bound stratum')
fisher_row(deg['bound_structstable'],nn['bound_structstable'],deg['unbound_structstable'],nn['unbound_structstable'],
           'Fig2c_OR_boundSS_vs_unboundSS','structure-stable stratum')

# ---- Fig2d: Spearman RNA vs d_ins by stratum (bin level) ----
pairsM=pd.read_csv(f'{OUT}/05_matched_controls/matched_pairs.csv')
mb=set(pairsM['matched_bin'])
for lab,mask in [('all',np.ones(len(b),bool)),
                 ('bound',b['chip_bound'].values),
                 ('matched',b['bin'].isin(mb).values)]:
    x=b.loc[mask,'rna_mean_log2FC_KOvsWT'].values
    y=b.loc[mask,'d_ins_KO_WT'].values
    m=~(np.isnan(x)|np.isnan(y))
    r,p=stats.spearmanr(x[m],y[m])
    rows.append(dict(test=f'Fig2d_spearman_rna_vs_dins_{lab}',observed=r,effect=r,
        ci_lo=np.nan,ci_hi=np.nan,p=p,n=int(m.sum()),null_model='Spearman asymptotic',notes=''))

rdf=pd.DataFrame(rows)
rdf.to_csv(f'{OUT}/06_permutation_tests/stats_fig2_summary.csv',index=False)
print(rdf.to_string())
print(ct.to_string())
