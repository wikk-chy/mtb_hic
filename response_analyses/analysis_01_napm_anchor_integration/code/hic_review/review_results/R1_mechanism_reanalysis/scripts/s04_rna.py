import pandas as pd, numpy as np
ROOT='/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review'
OUT=f'{ROOT}/hic_review/review_results/R1_mechanism_reanalysis'
GENOME=4411532; BIN=5000; NBINS=883

pk=pd.read_csv(f'{OUT}/01_chip_curated/NapM_binding_curated.tsv',sep='\t')
g=pd.read_csv(f'{ROOT}/hic_review/review_results/R1_mechanism/R1_mechanism_zhenghe/tables/gene_level_integrated.csv')
# drop deprecated chip columns
g=g.drop(columns=[c for c in ['chip_mean_gene','chip_mean_upstream500','bin_chip_bound'] if c in g.columns])

cen=pk['center'].values
def circdist_bp(a,b):
    d=np.abs(a-b)
    return np.minimum(d,GENOME-d)

rows=[]
for _,r in g.iterrows():
    tss=r['tss']
    d=np.array([circdist_bp(tss,c) for c in cen])
    j=int(np.argmin(d))
    # promoter: peak center within 300bp upstream of TSS, strand-aware
    if r['strand']=='+':
        upstream=((tss-cen)>=0)&((tss-cen)<=300)
    else:
        upstream=((cen-tss)>=0)&((cen-tss)<=300)
    # circular-safe enough: genes not near ori boundary issues handled by circdist for nearest only
    body=(pk['start'].values<=r['end'])&(pk['end'].values>=r['start'])
    prom_idx=np.where(upstream)[0]; body_idx=np.where(body)[0]
    rows.append(dict(
        gene_id=r['gene_id'],
        nearest_peak_dist_bp=float(d[j]),
        nearest_peak_score=float(pk['score'].iloc[j]),
        nearest_peak_vpm=int(pk['vpm'].iloc[j]),
        n_peaks_promoter_up300=len(prom_idx),
        n_peaks_gene_body=len(body_idx),
        promoter_peak_score_max=float(pk['score'].values[prom_idx].max()) if len(prom_idx) else np.nan,
        promoter_peak_vpm_max=int(pk['vpm'].values[prom_idx].max()) if len(prom_idx) else np.nan,
        body_peak_score_max=float(pk['score'].values[body_idx].max()) if len(body_idx) else np.nan,
        body_peak_vpm_max=int(pk['vpm'].values[body_idx].max()) if len(body_idx) else np.nan))
ann=pd.DataFrame(rows)
ann['binding_class']=np.select(
    [ann['n_peaks_promoter_up300']>0, ann['n_peaks_gene_body']>0],
    ['promoter_bound','gene_body_bound'],default='unbound')
# gene-level peak score/vpm: max over promoter+body peaks
ann['gene_peak_score_max']=ann[['promoter_peak_score_max','body_peak_score_max']].max(axis=1)
ann['gene_peak_vpm_max']=ann[['promoter_peak_vpm_max','body_peak_vpm_max']].max(axis=1)

# WT expression from raw counts via EBG mapping
rna=pd.read_csv(f'{ROOT}/hic_review/review_data/rnaseq/AKR_RNA_seq_merged.csv')
rna['expr_wt']=np.log2(rna[['WT-rep1','WT-rep2']].mean(axis=1)+1)
emap=pd.read_csv(f'{ROOT}/hic_review/review_results/R1_mechanism/R1_mechanism_zhenghe/tables/ebg_to_rv_mapping.csv')
rv2ebg=dict(zip(emap['gene_id'],emap['merged_ebg_id']))
ebg2expr=dict(zip(rna['gene'],rna['expr_wt']))
def get_expr(gid):
    if gid in ebg2expr: return ebg2expr[gid]
    e=rv2ebg.get(gid)
    return ebg2expr.get(e,np.nan)
g['expr_wt_gene']=g['gene_id'].map(get_expr)
g['gene_length']=g['end']-g['start']+1

g2=g.merge(ann,on='gene_id',how='left',validate='1:1')
g2.to_csv(f'{OUT}/04_rna_integration/gene_level_rna_v2.csv',index=False)
print('genes:',len(g2),'expr mapped:',g2['expr_wt_gene'].notna().sum())
print(g2['binding_class'].value_counts().to_dict())

# bin-level rna v2: rna + binding columns from bin_level_v2
b=pd.read_csv(f'{OUT}/03_structure_metrics/bin_level_v2.csv')
cols=['bin','start','end','chip_bin_npeaks','chip_bin_score_max','chip_bin_vpm_max','chip_bound',
      'chip_bound_score09','chip_bound_vpm_ge_med',
      'n_genes','rna_mean_log2FC_KOvsWT','rna_mean_abs_log2FC_KOvsWT','n_rna_up','n_rna_down',
      'n_rna_small','n_rna_undet','expr_wt_mean','rna_bin_class']
b[cols].to_csv(f'{OUT}/04_rna_integration/bin_level_rna_v2.csv',index=False)
print('bin rna v2 written', b[cols].shape)
