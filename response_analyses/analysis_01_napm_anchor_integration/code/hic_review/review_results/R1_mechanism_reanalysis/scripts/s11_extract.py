import pandas as pd, numpy as np
OUT='/Users/caihuaiyuan/Desktop/mtb_hic_analysis/mtb_hic_review/hic_review/review_results/R1_mechanism_reanalysis'
s=pd.read_csv(f'{OUT}/06_permutation_tests/stats_fig1_fig3_summary.csv')
# two-sided & depletion p for window tests from saved nulls
for lab,row_lab in [('5kb','Fig1B_window_pm5kb'),('10kb','Fig1B_window_pm10kb')]:
    ns=np.load(f'{OUT}/06_permutation_tests/null_fig1b_window_{lab}.npy')
    obs=s.loc[s['test']==row_lab,'observed'].iloc[0]
    p_le=(1+np.sum(ns<=obs))/(1+len(ns)); p_ge=(1+np.sum(ns>=obs))/(1+len(ns))
    p2=min(1,2*min(p_le,p_ge))
    print(lab,'obs',round(obs,4),'null_mean',round(ns.mean(),4),'p_ge',round(p_ge,4),'p_le',round(p_le,4),'p_2s',round(p2,4),
          'null2.5-97.5',np.round(np.percentile(ns,[2.5,97.5]),4))
print()
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',200)
print(s[['test','observed','null_mean','ci_lo','ci_hi','p','n','notes']].to_string())
