import pandas as pd, numpy as np, re, json
ROOT='/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review'
OUT=f'{ROOT}/hic_review/review_results/R1_mechanism_reanalysis'
GENOME=4411532; BIN=5000; NBINS=883

l=pd.read_csv(f'{ROOT}/hic_review/review_results/R4_comment3_loops/chromosight_out/wt.tsv',sep='\t')
assert len(l)==41
cur=pd.DataFrame(dict(
    loop_id=[f'L{i:02d}' for i in range(len(l))],
    chrom1=l['chrom1'],start1=l['start1'],end1=l['end1'],
    center1=(l['start1']+l['end1'])/2,
    chrom2=l['chrom2'],start2=l['start2'],end2=l['end2'],
    center2=(l['start2']+l['end2'])/2,
    score=l['score'],qvalue=l['qvalue'],bin1=l['bin1'],bin2=l['bin2']))
# distance: linear (matrix is linear contact map); also circular for reference
lin=(cur['center2']-cur['center1']).abs()
cur['distance_kb']=lin/1000.0
cur['distance_circular_kb']=np.minimum(lin,GENOME-lin)/1000.0
cur.to_csv(f'{OUT}/02_loop_curated/loops_41_curated.tsv',sep='\t',index=False)

bind=pd.read_csv(f'{OUT}/01_chip_curated/bin_level_binding.csv')
bound=bind['chip_bound'].values
cur['anchor1_bound']=bound[cur['bin1'].values]
cur['anchor2_bound']=bound[cur['bin2'].values]
cur['both_bound']=cur['anchor1_bound']&cur['anchor2_bound']
cur['either_bound']=cur['anchor1_bound']|cur['anchor2_bound']
cur.to_csv(f'{OUT}/02_loop_curated/loops_41_curated.tsv',sep='\t',index=False)

# annotation genes overlapping anchor bins
genes=[]
pat=re.compile(r'gene_id "([^"]+)"')
with open(f'{ROOT}/hic_review/annotation.gtf') as fh:
    for line in fh:
        if line.startswith('#'): continue
        f=line.rstrip('\n').split('\t')
        if len(f)<9 or f[2]!='gene': continue
        m=pat.search(f[8])
        if m: genes.append((m.group(1),int(f[3]),int(f[4])))
ann=pd.DataFrame(genes,columns=['gene_id','start','end']).drop_duplicates('gene_id')
def genes_in_bin(b):
    s=b*BIN; e=min((b+1)*BIN,GENOME)
    gg=ann[(ann['start']<e)&(ann['end']>s)]
    return ';'.join(gg['gene_id'])
cur['anchor1_genes']=[genes_in_bin(b) for b in cur['bin1']]
cur['anchor2_genes']=[genes_in_bin(b) for b in cur['bin2']]
cur.to_csv(f'{OUT}/02_loop_curated/loops_41_curated.tsv',sep='\t',index=False)

# CID boundaries (wt)
bd=pd.read_csv(f'{ROOT}/hic_review/review_data/cid_boundary/wt-boundary.bed',sep='\t',header=None,
               names=['chrom','start','end'])
bd['bin']=(bd['start']//BIN).astype(int)
bd.to_csv(f'{OUT}/02_loop_curated/wt_cid_boundaries_curated.tsv',sep='\t',index=False)
bdbins=np.sort(bd['bin'].unique())
def min_circ_dist_bins(b):
    d=np.abs(bdbins-b); d=np.minimum(d,NBINS-d)
    return int(d.min())
cur['anchor1_dist_to_wtCID_bins']=[min_circ_dist_bins(b) for b in cur['bin1']]
cur['anchor2_dist_to_wtCID_bins']=[min_circ_dist_bins(b) for b in cur['bin2']]
cur.to_csv(f'{OUT}/02_loop_curated/loops_41_curated.tsv',sep='\t',index=False)

qc=dict(
  n_loops=len(cur),
  distance_kb_min=float(cur['distance_kb'].min()),
  distance_kb_med=float(cur['distance_kb'].median()),
  distance_kb_max=float(cur['distance_kb'].max()),
  anchor1_bound=int(cur['anchor1_bound'].sum()),
  anchor2_bound=int(cur['anchor2_bound'].sum()),
  either_bound=int(cur['either_bound'].sum()),
  both_bound=int(cur['both_bound'].sum()),
  n_anchor_on_wtCID_boundary=int(((cur['anchor1_dist_to_wtCID_bins']==0)|(cur['anchor2_dist_to_wtCID_bins']==0)).sum()),
  n_wtCID_boundaries=len(bd),
  score_med=float(cur['score'].median()),
  qvalue_max=float(cur['qvalue'].max()))
with open(f'{OUT}/02_loop_curated/qc_summary.json','w') as fh:
    json.dump(qc,fh,indent=2)
print(json.dumps(qc,indent=2))
