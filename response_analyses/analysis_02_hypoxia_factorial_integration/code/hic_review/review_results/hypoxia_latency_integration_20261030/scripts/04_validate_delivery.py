#!/usr/bin/env python3
"""Independent consistency checks for the integrated-analysis delivery."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path('/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review')
RA = ROOT / 'hic_review/review_results/R1_mechanism_reanalysis'
OUT = ROOT / 'hic_review/review_results/hypoxia_latency_integration_20261030'
TAB, SRC, FIG = OUT / 'tables', OUT / 'source_data', OUT / 'figures'
checks = []

def check(name, condition, detail=''):
    checks.append(dict(check=name, passed=bool(condition), detail=str(detail)))
    if not condition:
        raise AssertionError(f'{name}: {detail}')

required = [
    'Hypoxia_gene_level.csv', 'Hypoxia_bin_level.csv', 'Latency_gene_level.csv', 'Latency_bin_level.csv',
    'HiC_insulation_per_replicate.csv.gz', 'Representative_locus_contact_maps_per_replicate.csv.gz',
    'Representative_locus_mean_contact_maps.csv', 'Representative_locus_insulation_tracks.csv',
    'Representative_locus_NapM_peaks.csv', 'Representative_locus_RNA_tracks.csv',
    'Representative_locus_gene_models.csv']
for f in required:
    check(f'exists {f}', (SRC / f).is_file() and (SRC / f).stat().st_size > 0)
for f in ['Hypoxia_multiomics_figure.pdf', 'Hypoxia_multiomics_figure.png',
          'Latency_multiomics_figure.pdf', 'Latency_multiomics_figure.png']:
    check(f'exists {f}', (FIG / f).is_file() and (FIG / f).stat().st_size > 10000)
for f in ['English_manuscript_and_reviewer_text.md', '中文多组学整合分析报告.md']:
    check(f'exists {f}', (OUT / f).is_file() and (OUT / f).stat().st_size > 1000)

hb = pd.read_csv(SRC / 'Hypoxia_bin_level.csv'); hg = pd.read_csv(SRC / 'Hypoxia_gene_level.csv')
lb = pd.read_csv(SRC / 'Latency_bin_level.csv'); lg = pd.read_csv(SRC / 'Latency_gene_level.csv')
orig = pd.read_csv(RA / '03_structure_metrics/bin_level_v2.csv')
check('883 bins', len(hb) == len(lb) == 883)
check('4109 genes', len(hg) == len(lg) == 4109)
check('unique bins', hb['bin'].is_unique and lb['bin'].is_unique)
check('unique genes', hg.gene_id.is_unique and lg.gene_id.is_unique)
check('TSS coordinate mapping', np.array_equal(hg.bin_tss.to_numpy(), ((hg.tss.to_numpy()-1)//5000)))

# Independently recalculate the WT-latent robust contrast from the curated replicate columns.
wt = orig[[f'ins_WT_rep{i}' for i in range(1,5)]].to_numpy(float).T
lat = orig[[f'ins_latent_rep{i}' for i in range(1,5)]].to_numpy(float).T
wt = np.where(np.isfinite(wt),wt,np.nan); lat = np.where(np.isfinite(lat),lat,np.nan)
ds = (lat[:,None,:] - wt[None,:,:]).reshape(-1,883)
pt = np.nanmedian(ds, axis=0); lo = np.nanpercentile(ds,5,axis=0); hi = np.nanpercentile(ds,95,axis=0)
check('latent effect exact', np.allclose(pt, lb.d_ins_latent_WT, equal_nan=True), np.nanmax(np.abs(pt-lb.d_ins_latent_WT)))
check('latent interval exact', np.allclose(lo,lb.d_ins_latent_WT_lo,equal_nan=True) and np.allclose(hi,lb.d_ins_latent_WT_hi,equal_nan=True))
expected_class = np.select([lo>.25,hi<-.25,(lo>=-.25)&(hi<=.25)],['detected_increase','detected_decrease','supports_small_change'],default='cannot_determine')
check('latent class exact', np.array_equal(expected_class,lb.latent_ins_class))

# No degenerate matched nulls after the corrected mask-shift implementation.
ms = pd.read_csv(TAB / 'Hypoxia_NapM_matched_tests.csv')
check('matched tests nonzero null SD', (ms.null_sd>0).all(), ms.null_sd.min())
check('matched tests p range', ms.circular_shift_p.between(0,1).all())
check('primary matched tests null', (ms.iloc[[0,4,8]].circular_shift_p>.6).all())
nulls = pd.read_csv(SRC / 'Hypoxia_NapM_circular_shift_nulls.csv.gz')
check('10000 nulls per primary comparison', (nulls.groupby('comparison').size()==10000).all())
check('empirical null SD nonzero', (nulls.groupby('comparison').null_statistic.std()>0).all())

# Frozen selection must agree with the completed genome-wide evidence tables.
sel = pd.read_csv(TAB / 'Representative_locus_selection.csv')
check('two hypoxia loci', list(sel[sel.model.eq('Hypoxia')].center)==[119,862])
check('three latency loci', list(sel[sel.model.eq('Latency')].center)==[0,267,133])
regions = pd.read_csv(TAB / 'Latency_candidate_regions.csv').sort_values('evidence_score',ascending=False)
check('latency top3 exact', regions.iloc[:3].region_id.tolist()==['LATR01','LATR39','LATR18'])
check('no KO-latent design', json.load(open(TAB/'analysis_metadata.json'))['HiC_replicates']['KO_latent']==0)

# Plotted mean O/E must equal the mean of per-replicate source values.
reps = pd.read_csv(SRC / 'Representative_locus_contact_maps_per_replicate.csv.gz')
mean_maps = pd.read_csv(SRC / 'Representative_locus_mean_contact_maps.csv')
agg = reps.groupby(['model','locus','condition','bin1','bin2'],as_index=False).OE.mean()
chk = mean_maps.merge(agg,on=['model','locus','condition','bin1','bin2'],validate='1:1')
err = np.nanmax(np.abs(chk.mean_OE-chk.OE))
check('mean maps equal replicate means', np.allclose(chk.mean_OE,chk.OE,equal_nan=True),err)

# Every browser feature must lie in the same locus coordinate range.
ins = pd.read_csv(SRC / 'Representative_locus_insulation_tracks.csv')
rna = pd.read_csv(SRC / 'Representative_locus_RNA_tracks.csv')
genes = pd.read_csv(SRC / 'Representative_locus_gene_models.csv')
for loc, z in ins.groupby('locus'):
    lo_x,hi_x=z.position_kb.min()-2.5,z.position_kb.max()+2.5
    check(f'{loc} RNA coordinates',rna[rna.locus.eq(loc)].tss_kb.between(lo_x,hi_x).all())
    zg=genes[genes.locus.eq(loc)]
    check(f'{loc} gene TSS coordinates',zg.tss_kb.between(lo_x,hi_x).all())

# Verify headline counts directly from source tables and the independent limma output.
det={'detected_increase','detected_decrease'}
check('WT hypoxia DEG count',hg.WT_hypoxia_RNA_class.isin(det).sum()==1328)
check('KO hypoxia DEG count',hg.KO_hypoxia_RNA_class.isin(det).sum()==669)
check('RNA interaction DEG count',hg.hypoxia_interaction_RNA_class.isin(det).sum()==730)
check('RNA interaction FDR count',hg.padj_int.lt(.05).sum()==2453)
limma_long = pd.read_csv(TAB / 'Hypoxia_limma_voom_gene_results_long.csv')
check('limma long table complete',len(limma_long)==4*3814 and limma_long.contrast.nunique()==4)
limma_ko = limma_long[limma_long.contrast.eq('KO_hypoxia')]
check('limma KO DEG count exact',(limma_ko.DEG & limma_ko.adj_P_Val.lt(.05) & limma_ko.log2FC.abs().ge(1)).sum()==669)
check('latency dual DEG count',lg.latency_dual_DEG.sum()==437)
check('latency dual structure count',lb.latency_dual_structure.sum()==47)
check('interaction structure classes',hb.interaction_ins_class.eq('detected_increase').sum()==79 and hb.interaction_ins_class.eq('detected_decrease').sum()==236)

# Report files must retain the design caveat.
en=(OUT/'English_manuscript_and_reviewer_text.md').read_text()
cn=(OUT/'中文多组学整合分析报告.md').read_text()
check('English no KO-latent caveat','No KO-latent comparison is performed' in en)
check('Chinese no KO-latent caveat','不构造 KO-latent' in cn)
check('same-direction caveat','same-direction patterns indicate partial molecular similarity' in en.lower())
check('limma methods in reports','limma-voom' in en and 'limma-voom' in cn)
check('stale RNA claims removed','No individual gene passed FDR' not in en and 'KO: 16' not in en and '| KO 缺氧响应 | 14 | 2 | 16 | 17 |' not in cn)

result = dict(n_checks=len(checks),all_passed=all(x['passed'] for x in checks),
              mean_map_max_abs_error=float(err),figures={},checks=checks)
for f in FIG.glob('*'):
    result['figures'][f.name]=f.stat().st_size
with open(TAB/'validation_report.json','w') as fh:
    json.dump(result,fh,indent=2)
pd.DataFrame(checks).to_csv(TAB/'validation_checks.csv',index=False)

# File-level traceability manifest (excluding the manifest itself).
rows=[]
for path in sorted(OUT.rglob('*')):
    if not path.is_file() or path.name in ['source_manifest.csv','.DS_Store']:
        continue
    h=hashlib.sha256()
    with open(path,'rb') as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b''):
            h.update(chunk)
    rows.append(dict(file=str(path.relative_to(OUT)),size_bytes=path.stat().st_size,sha256=h.hexdigest()))
pd.DataFrame(rows).to_csv(SRC/'source_manifest.csv',index=False)
print(json.dumps({'n_checks':len(checks),'all_passed':True,'mean_map_max_abs_error':err,
                  'manifest_files':len(rows)},indent=2))
