import os, glob
import pandas as pd
ROOT='/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review'
OUT=f'{ROOT}/hic_review/review_results/R1_mechanism_reanalysis'
rows=[]
def add(dataset,sample,condition,replicate,path,resolution,notes):
    p=path.replace(ROOT+'/','')
    exists=os.path.exists(path)
    size=os.path.getsize(path) if exists and os.path.isfile(path) else ''
    rows.append(dict(dataset=dataset,sample=sample,condition=condition,replicate=replicate,
                     file=p,resolution=resolution,exists=exists,size_bytes=size,notes=notes))

add('chip_seq','Minch2015_ChIPseq_summary','WT_log_phase','NA',
    f'{ROOT}/hic_review/review_data/chip_seq_data/41467_2015_BFncomms6829_MOESM46_ESM.xlsx','peak-level',
    'AUTHORITATIVE new NapM binding definition; Sheet1 header=2; 634 peaks for Regulator==Rv0047c; first 10 rows are embedded Heading/Descriptor rows (not Rv0047c)')
add('annotation','H37Rv_ASM19595v2','NA','NA',f'{ROOT}/hic_review/annotation.gtf','gene','ENA GTF; gene_id Rv*')
add('genome','H37Rv','NA','NA',f'{ROOT}/hic_review/genome.fa','bp','circular 4,411,532 bp')
add('genome_index','H37Rv','NA','NA',f'{ROOT}/hic_review/genome.fa.fai','bp','')
add('loops','chromosight_wt','WT','pooled',
    f'{ROOT}/hic_review/review_results/R4_comment3_loops/chromosight_out/wt.tsv','5kb',
    '41 high-confidence loops; bin1/bin2 0-based 5kb; ONLY loop set used')
add('bin_integrated_prev','R1_mechanism_zhenghe','multi','pooled',
    f'{ROOT}/hic_review/review_results/R1_mechanism/R1_mechanism_zhenghe/tables/bin_level_integrated.csv','5kb',
    '883 bins; reuse all except chip_mean_signal/chip_bound/chip_bound_alt/bound_region_frac/matched_background/matched_to_bin (deprecated, recomputed from Excel)')
add('gene_integrated_prev','R1_mechanism_zhenghe','multi','pooled',
    f'{ROOT}/hic_review/review_results/R1_mechanism/R1_mechanism_zhenghe/tables/gene_level_integrated.csv','gene',
    '4109 genes; chip_mean_gene/chip_mean_upstream500/bin_chip_bound deprecated')
add('ebg_rv_map','R1_mechanism_zhenghe','NA','NA',
    f'{ROOT}/hic_review/review_results/R1_mechanism/R1_mechanism_zhenghe/tables/ebg_to_rv_mapping.csv','gene',
    'Rv gene_id -> merged EBG id for RNA-seq count table join')
add('rnaseq_counts','AKR_RNA_seq_merged','WT/Hypoxia/Latent/KO/KO-hypoxia','2 each',
    f'{ROOT}/hic_review/review_data/rnaseq/AKR_RNA_seq_merged.csv','gene','raw counts, EBG gene ids')
for cond,pref,n in [('WT','wt',4),('KO','napm_ko',4),('hypoxia','hypoxia',4),('latent','latent',4),('KO-hypoxia','ko-hypoxia',3)]:
    for i in range(1,n+1):
        add('iced_matrix',f'{pref}_{i}',cond,i,
            f'{ROOT}/hic_review/review_data/rep_icedmatrix/{pref}_{i}_5000_iced.matrix','5kb',
            '883x883 ICE-normalized contact matrix (text)')
for f in sorted(glob.glob(f'{ROOT}/hic_review/review_data/cool_file/*')):
    add('cool_file',os.path.basename(f),'multi','pooled',f,'5kb','cooler file (not needed unless recomputation)')
for f in sorted(glob.glob(f'{ROOT}/hic_review/review_data/cid_boundary/*.bed')):
    add('cid_boundary',os.path.basename(f),os.path.basename(f).replace('-boundary.bed',''),'pooled',f,'5kb','CID boundary bed')
add('chip_seq_legacy','Rv0047c.bedgraph','WT','NA',
    f'{ROOT}/hic_review/review_data/chip_seq_data/Rv0047c.bedgraph','bp','LEGACY heuristic source; NOT used for binding definition')

df=pd.DataFrame(rows)
df.to_csv(f'{OUT}/data_manifest.tsv',sep='\t',index=False)
print(df.to_string())
print('all exist:', df['exists'].all())
