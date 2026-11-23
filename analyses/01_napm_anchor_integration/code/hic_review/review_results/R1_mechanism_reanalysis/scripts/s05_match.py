import pandas as pd, numpy as np, json
ROOT='/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review'
OUT=f'{ROOT}/hic_review/review_results/R1_mechanism_reanalysis'
rng=np.random.default_rng(42)
CALIPER=1.0  # Euclidean distance in z-scored feature space (4 features => ~0.5 SD per feature)

def greedy_match(bound_df, pool_df, feats, caliper=CALIPER):
    mu=pd.concat([bound_df[feats],pool_df[feats]]).mean()
    sd=pd.concat([bound_df[feats],pool_df[feats]]).std(ddof=1)
    B=(bound_df[feats]-mu)/sd; P=(pool_df[feats]-mu)/sd
    Pv=P.values; Bv=B.values
    avail=np.ones(len(P),bool)
    pairs=[]
    order=np.argsort(bound_df.index)  # deterministic
    # greedy: process bound units in order of their best available distance ascending (iterative)
    best=np.full(len(B),np.inf); bestj=np.full(len(B),-1)
    rem=set(range(len(B)))
    Pidx=np.arange(len(P))
    while rem:
        progress=False
        cand=[]
        for bi in list(rem):
            if not avail.any():
                pairs.append((bi,-1,np.nan)); rem.discard(bi); continue
            d=np.sqrt(((Pv[avail]-Bv[bi])**2).sum(axis=1))
            jloc=int(np.argmin(d)); pj=Pidx[avail][jloc]
            cand.append((d[jloc],bi,pj))
        cand.sort()
        dmin,bi,pj=cand[0]
        pairs.append((bi,pj,dmin)); avail[pj]=False; rem.discard(bi); progress=True
        if not progress: break
    out=[]
    for bi,pj,d in pairs:
        out.append(dict(bound_index=int(bound_df.index[bi]),
                        matched_index=int(pool_df.index[pj]) if pj>=0 else -1,
                        match_distance=float(d) if pj>=0 else np.nan,
                        caliper_exceeded=bool(d>caliper) if pj>=0 else True))
    return pd.DataFrame(out)

# ---------------- bin-level ----------------
b=pd.read_csv(f'{OUT}/03_structure_metrics/bin_level_v2.csv')
feats=['weight_WT','ins_WT','expr_wt_mean','n_genes']
use=b.dropna(subset=feats).copy()
bound_df=use[use['chip_bound']].copy()
pool_df=use[~use['chip_bound']].copy()
m=greedy_match(bound_df,pool_df,feats)
m['bound_bin']=m['bound_index'].map(use['bin'])
m['matched_bin']=m['matched_index'].map(use['bin'])
m[['bound_bin','matched_bin','match_distance','caliper_exceeded']].to_csv(
    f'{OUT}/05_matched_controls/matched_pairs.csv',index=False)
qc=dict(level='bin',feats=feats,caliper=CALIPER,
        n_bound=int(len(bound_df)),n_matched=int((m['matched_index']>=0).sum()),
        n_caliper_exceeded=int(m['caliper_exceeded'].sum()),
        match_dist_med=float(m['match_distance'].median()),
        match_dist_max=float(m['match_distance'].max()))
# balance check
bal={}
mb=use.set_index(use['bin'])
for f in feats:
    bvals=mb.loc[m['bound_bin'],f]; mvals=mb.loc[m.loc[m['matched_index']>=0,'matched_bin'],f]
    bal[f]=dict(bound_mean=float(bvals.mean()),matched_mean=float(mvals.mean()),
                std_diff=float((bvals.mean()-mvals.mean())/use[f].std()))
qc['balance']=bal
# ---------------- gene-level ----------------
g=pd.read_csv(f'{OUT}/04_rna_integration/gene_level_rna_v2.csv')
g['gene_bound']=g['binding_class'].isin(['promoter_bound','gene_body_bound'])
gfeats=['expr_wt_gene','gene_length']
gu=g.dropna(subset=gfeats).copy()
gb=gu[gu['gene_bound']]; gp=gu[~gu['gene_bound']]
gm=greedy_match(gb,gp,gfeats)
gm['bound_gene']=gm['bound_index'].map(gu['gene_id'])
gm['matched_gene']=gm['matched_index'].map(gu['gene_id'])
gm[['bound_gene','matched_gene','match_distance','caliper_exceeded']].to_csv(
    f'{OUT}/05_matched_controls/matched_pairs_genes.csv',index=False)
qcg=dict(level='gene',feats=gfeats,caliper=CALIPER,
         n_bound=int(len(gb)),n_matched=int((gm['matched_index']>=0).sum()),
         n_caliper_exceeded=int(gm['caliper_exceeded'].sum()),
         match_dist_med=float(gm['match_distance'].median()))
bal={}
gu2=gu.set_index('gene_id')
for f in gfeats:
    bvals=gu2.loc[gm['bound_gene'],f]; mvals=gu2.loc[gm.loc[gm['matched_index']>=0,'matched_gene'],f]
    bal[f]=dict(bound_mean=float(bvals.mean()),matched_mean=float(mvals.mean()),
                std_diff=float((bvals.mean()-mvals.mean())/gu[f].std()))
qcg['balance']=bal
with open(f'{OUT}/05_matched_controls/qc_matching.json','w') as fh:
    json.dump({'bin':qc,'gene':qcg},fh,indent=2)
print(json.dumps({'bin':qc,'gene':qcg},indent=2))
