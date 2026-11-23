#!/usr/bin/env python3
"""Changed-loop anchor-proximal DEGs and three-way NapM/insulation/RNA integration."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA=ROOT/'hic_review/review_results/R1_mechanism_reanalysis'
BASE=RA/'09_combined/R1_review_delivery_20261027/mechanism_reorganization_20261028'
CACHE=ROOT/'hic_review/review_results/R2_comments/go_kegg_enrichment/annotation_cache'
OUT=BASE/'anchor_centered_integration_20261029'
for d in ['tables','source_data','logs']:(OUT/d).mkdir(parents=True,exist_ok=True)
GENOME=4415000
DET={'detected_increase','detected_decrease'}

loops=pd.read_csv(BASE/'loop_change_enrichment/per_loop_change_and_binding.csv')
changed=loops[loops.abs_d_log2OE.ge(.5)].copy()
assert changed.loop_id.tolist()==['L00','L12','L14','L15','L16','L17','L19','L23','L28','L29','L32']
g=pd.read_csv(RA/'04_rna_integration/gene_level_rna_v2.csv')
g['RNA_tested']=g.log2FC_KOvsWT.notna()&g.padj_KOvsWT.notna()
g['DEG']=g.class_KOvsWT.isin(DET)
g['up']=g.class_KOvsWT.eq('detected_increase');g['down']=g.class_KOvsWT.eq('detected_decrease')
g['NapM_occupied']=g.binding_class.isin(['promoter_bound','gene_body_bound'])
g['insulation_changed']=g.bin_ins_change_class.isin(DET)
assert g.RNA_tested.sum()==4060 and g.DEG.sum()==820

def anchor_table(frame):
    left=frame[['loop_id','center1','anchor1_bound','distance_kb','d_log2OE','WT_mean','KO_mean']].rename(
        columns={'center1':'center','anchor1_bound':'anchor_NapM_occupied'})
    right=frame[['loop_id','center2','anchor2_bound','distance_kb','d_log2OE','WT_mean','KO_mean']].rename(
        columns={'center2':'center','anchor2_bound':'anchor_NapM_occupied'})
    a=pd.concat([left.assign(side='left'),right.assign(side='right')],ignore_index=True)
    a['anchor_id']=a.loop_id+'_'+a.side
    return a.sort_values(['loop_id','center']).reset_index(drop=True)
anchors=anchor_table(changed);all_anchors=anchor_table(loops)
assert len(anchors)==22 and anchors.center.nunique()==21
anchors.to_csv(OUT/'tables/changed_loop_22_anchor_instances.csv',index=False)

def circdist(x,c):
    d=np.abs(np.asarray(x,float)-float(c));return np.minimum(d,GENOME-d)

def associate(frame,prefix,window=10000):
    membership=[[] for _ in range(len(g))]
    for a in frame.itertuples():
        idx=np.flatnonzero(circdist(g.tss,a.center)<=window)
        for i in idx:membership[i].append(a.anchor_id)
    g[f'in_{prefix}']=[bool(x) for x in membership]
    g[f'{prefix}_ids']=[';'.join(x) for x in membership]
    rows=[]
    for a in frame.itertuples():
        z=g[(circdist(g.tss,a.center)<=window)&g.RNA_tested]
        rows.append(dict(anchor_id=a.anchor_id,loop_id=a.loop_id,side=a.side,center=a.center,
            anchor_NapM_occupied=a.anchor_NapM_occupied,window_bp=window,n_RNA_tested=len(z),n_DEG=z.DEG.sum(),
            n_up=z.up.sum(),n_down=z.down.sum(),DEG_fraction=z.DEG.mean() if len(z) else np.nan,
            genes=';'.join(z.gene_id),DEGs=';'.join(z.loc[z.DEG,'gene_id'])))
    return pd.DataFrame(rows)

per_anchor=associate(anchors,'changed_anchor',10000)
all_anchor_summary=associate(all_anchors,'eligible_anchor',10000)
per_anchor.to_csv(OUT/'tables/changed_anchor_gene_overlap_summary.csv',index=False)
all_anchor_summary.to_csv(OUT/'source_data/all_39_loop_anchor_gene_overlap_summary.csv',index=False)
sensitivity=[]
for w in [5000,10000,25000]:
    near=np.zeros(len(g),bool);eligible=np.zeros(len(g),bool)
    for a in anchors.itertuples():near|=circdist(g.tss,a.center)<=w
    for a in all_anchors.itertuples():eligible|=circdist(g.tss,a.center)<=w
    sensitivity.append(dict(window_bp=w,changed_anchor_RNA_tested=int((near&g.RNA_tested).sum()),
        changed_anchor_DEGs=int((near&g.DEG).sum()),changed_anchor_up=int((near&g.up).sum()),
        changed_anchor_down=int((near&g.down).sum()),eligible_anchor_RNA_tested=int((eligible&g.RNA_tested).sum()),
        eligible_anchor_DEGs=int((eligible&g.DEG).sum())))
pd.DataFrame(sensitivity).to_csv(OUT/'tables/anchor_window_sensitivity.csv',index=False)

# Exact three-way membership among RNA-tested genes.
g['triple_intersection']=g.RNA_tested&g.DEG&g.NapM_occupied&g.insulation_changed
g['combination_code']=g.apply(lambda r:''.join(['N' if r.NapM_occupied else '-', 'I' if r.insulation_changed else '-', 'D' if r.DEG else '-']),axis=1)
cols=['gene_id','gene_name','start','end','strand','tss','bin_tss','RNA_tested','DEG','up','down',
      'log2FC_KOvsWT','padj_KOvsWT','binding_class','NapM_occupied','bin_d_ins_KO_WT','bin_ins_change_class',
      'insulation_changed','triple_intersection','combination_code','in_changed_anchor','changed_anchor_ids',
      'in_eligible_anchor','eligible_anchor_ids','expr_wt_gene']
g[cols].to_csv(OUT/'source_data/anchor_and_three_way_gene_membership.csv',index=False)
combos=g[g.RNA_tested].groupby('combination_code').agg(n_genes=('gene_id','size'),genes=('gene_id',lambda x:';'.join(x))).reset_index()
combos.to_csv(OUT/'tables/three_way_exact_combinations.csv',index=False)
triple=g[g.triple_intersection].copy().sort_values('log2FC_KOvsWT',key=abs,ascending=False)
triple.to_csv(OUT/'tables/NapM_insulation_DEG_triple_genes.csv',index=False)
anchor_deg=g[g.in_changed_anchor&g.DEG].copy().sort_values('log2FC_KOvsWT',key=abs,ascending=False)
anchor_deg.to_csv(OUT/'tables/changed_anchor_DEGs.csv',index=False)

# Pre-declared representative ranking: anchor DEG + direct occupancy + detected insulation
# change + occupied anchor; then absolute RNA effect. dnaA is separately labelled as
# a literature-anchored control and never claimed to be selected by enrichment.
anchor_map=anchors.assign(anchor_NapM_occupied=anchors.anchor_NapM_occupied.astype(bool)).set_index('anchor_id').anchor_NapM_occupied.to_dict()
anchor_deg['at_NapM_occupied_anchor']=anchor_deg.changed_anchor_ids.map(lambda s:any(anchor_map.get(x,False) for x in s.split(';') if x))
anchor_deg['representative_score']=(anchor_deg.NapM_occupied.astype(int)+anchor_deg.insulation_changed.astype(int)+
    anchor_deg.at_NapM_occupied_anchor.astype(int))*100+anchor_deg.log2FC_KOvsWT.abs()
anchor_deg=anchor_deg.sort_values(['representative_score','gene_id'],ascending=[False,True])
anchor_deg.to_csv(OUT/'tables/representative_anchor_loci_ranked.csv',index=False)

# GO/KEGG annotations.
k2p=pd.read_csv(CACHE/'kegg_gene2pathway.tsv',sep='\t',header=None,names=['gene','term'])
k2p['gene']=k2p.gene.str.replace('mtu:','',regex=False);k2p['term']=k2p.term.str.replace('path:','',regex=False)
pn=pd.read_csv(CACHE/'kegg_pathway_names.tsv',sep='\t',header=None,names=['term','name'])
pn['name']=pn.name.str.replace(' - Mycobacterium tuberculosis H37Rv','',regex=False)
kegg_names=dict(zip(pn.term,pn.name))
rv_pat=re.compile(r'Rv\d{3,4}[A-Za-z]?(?![A-Za-z0-9])');rows=[]
with open(CACHE/'30.M_tuberculosis_ATCC_25618.goa') as h:
    for line in h:
        if line.startswith('!'):continue
        f=line.rstrip('\n').split('\t')
        if f[8]=='P':
            for gene in set(rv_pat.findall(f[10])):rows.append((gene,f[4]))
gaf=pd.DataFrame(rows,columns=['gene','term']).drop_duplicates()
go_names={};cid=cname=None
with open(CACHE/'go-basic.obo') as h:
    for line in h:
        line=line.rstrip('\n')
        if line=='[Term]':
            if cid and cname:go_names[cid]=cname
            cid=cname=None
        elif line.startswith('id: '):cid=line[4:]
        elif line.startswith('name: '):cname=line[6:]
if cid and cname:go_names[cid]=cname
dbs={'GO_BP':(gaf,go_names),'KEGG':(k2p[['gene','term']],kegg_names)}

def bh(p):
    p=np.asarray(p,float);order=np.argsort(p);r=p[order]
    q=np.minimum.accumulate((r*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty_like(q);out[order]=np.minimum(q,1);return out

def enrichment(query,universe,mapping,names,gene_set,db):
    U=set(g.loc[universe,'gene_id'])&set(mapping.gene);Q=set(g.loc[query,'gene_id'])&U
    M,N=len(U),len(Q);records=[]
    if not(M>N>0):return pd.DataFrame()
    for term,genes in mapping.groupby('term').gene.apply(set).items():
        T=genes&U;n=len(T)
        if n<3:continue
        overlap=Q&T;k=len(overlap)
        records.append(dict(gene_set=gene_set,database=db,term=term,name=names.get(term,term),overlap=k,
            term_size=n,query_size=N,background_size=M,fold_enrichment=(k/N)/(n/M),
            p_value=stats.hypergeom.sf(k-1,M,n,N),genes=','.join(sorted(overlap))))
    out=pd.DataFrame(records);out['q_BH_all_tested_terms']=bh(out.p_value);return out

specs=[]
for direction in ['all','up','down']:
    q=g.in_changed_anchor&g.DEG;u=g.in_eligible_anchor&g.DEG
    if direction!='all':q=q&g[direction];u=u&g[direction]
    specs.append(dict(gene_set=f'changed_anchor_{direction}_vs_eligible_anchor_DEGs',query=q,universe=u,role='primary_conditional'))
specs += [
 dict(gene_set='changed_anchor_DEGs_vs_eligible_anchor_RNA_tested',query=g.in_changed_anchor&g.DEG,
      universe=g.in_eligible_anchor&g.RNA_tested,role='sensitivity_conventional_anchor_background'),
 dict(gene_set='triple_vs_all_DEGs',query=g.triple_intersection,universe=g.DEG,role='primary_conditional_triple'),
 dict(gene_set='triple_vs_RNA_tested',query=g.triple_intersection,universe=g.RNA_tested,role='sensitivity_conventional_triple')]
results=[]
for db,(mapping,names) in dbs.items():
    for s in specs:
        z=enrichment(s['query'],s['universe'],mapping,names,s['gene_set'],db)
        if not z.empty:z['role']=s['role'];results.append(z)
res=pd.concat(results,ignore_index=True).sort_values(['gene_set','database','p_value','term'])
res.to_csv(OUT/'tables/anchor_and_three_way_GO_KEGG_enrichment.csv',index=False)
res[res.q_BH_all_tested_terms.lt(.05)].to_csv(OUT/'tables/anchor_and_three_way_FDR_significant.csv',index=False)
summary=[]
for s in specs:
    summary.append(dict(gene_set=s['gene_set'],role=s['role'],n_query=int(s['query'].sum()),n_universe=int(s['universe'].sum())))
pd.DataFrame(summary).to_csv(OUT/'tables/enrichment_gene_set_summary.csv',index=False)
metadata=dict(primary_window='TSS within +/-10 kb of anchor center, circular distance',sensitivity_windows=[5000,25000],
    changed_loops=changed.loop_id.tolist(),anchor_instances=22,unique_changed_anchor_centers=21,
    DEG='detected_increase/decrease: FDR<0.05 and |log2FC|>=1',NapM='promoter_bound or gene_body_bound; baseline occupancy, not differential binding',
    insulation='Existing bin_ins_change_class detected_increase/decrease',three_way_universe='4060 RNA-tested genes',
    enrichment='BH over all terms with >=3 background genes, including zero-overlap terms',
    representative_rule='Anchor DEG ranked by direct NapM occupancy, detected insulation change, occupied-anchor status, then absolute RNA effect; dnaA considered separately as literature control',
    limitation='Overlap and enrichment do not establish direct regulation, contact-to-RNA causality, chromosomal plasticity, or dormancy establishment.')
(OUT/'tables/analysis_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('WINDOWS\n',pd.DataFrame(sensitivity).to_string(index=False))
print('\nTHREE-WAY COMBINATIONS\n',combos.to_string(index=False))
print('\nTOP ANCHOR DEG LOCI\n',anchor_deg[['gene_id','gene_name','changed_anchor_ids','log2FC_KOvsWT','binding_class','bin_ins_change_class','at_NapM_occupied_anchor','representative_score']].head(20).to_string(index=False))
print('\nGENE SETS\n',pd.DataFrame(summary).to_string(index=False))
print('\nFDR SIGNIFICANT\n',res[res.q_BH_all_tested_terms.lt(.05)][['gene_set','database','name','overlap','query_size','fold_enrichment','q_BH_all_tested_terms','genes']].to_string(index=False))
