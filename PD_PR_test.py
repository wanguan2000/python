#import matplotlib.pyplot as plt
import pandas as pd
import scipy as scipy
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy import create_engine, text
import scipy as scipy
from scipy.stats import wasserstein_distance

myfile = '/share_storage/home/nacoo/DI/20230425/GCKB-WES-202210-patients_en.csv'
mydata = pd.read_csv(myfile, sep='\t', low_memory=False)
mydata_RNA = mydata[mydata['projectTypes'].str.contains('RNA')]
mydata_RNA_mutations = mydata_RNA[['code', 'mutations','tmb']].copy()
mydata_RNA_mutations = mydata_RNA_mutations.query('tmb > 100')
print(mydata_RNA_mutations)
code_RNA_tmb100 = (list(mydata_RNA_mutations.code))

data_snp='/share_storage/home/nacoo/DI/20230425/code_RNA_tmb100_snp2.xls'
data_indel='/share_storage/home/nacoo/DI/20230425/code_RNA_tmb100_indel2.xls'

mydata_snp = pd.read_csv(data_snp, sep='\t', low_memory=False,index_col=0)
mydata_snp = mydata_snp[["patient_id", "normal_id", "tumor_id", "project_id","symbol"]]
mydata_indel = pd.read_csv(data_indel, sep='\t', low_memory=False,index_col=0)
mydata_indel = mydata_indel[["patient_id", "normal_id", "tumor_id", "project_id","symbol"]]
result = pd.concat([mydata_snp, mydata_indel], axis=0)
result2=result[["patient_id", "symbol"]]
result2 = result2.drop_duplicates()
counts = result2.symbol.value_counts()
counts40 = counts[counts > 10]
counts40dict = counts40.to_dict()
counts40dict['RPA1']
##RNA
myRNA='/share_storage/home/nacoo/DI/20230425/all_RNA.csv'
mydata_RNA = pd.read_csv(myRNA, sep='\t', low_memory=False)
mydata_RNA2=mydata_RNA[["patient_id", "gene","fpkm"]]
mydata_RNA2 = mydata_RNA2.drop_duplicates()
mydata_RNA3 = mydata_RNA2[mydata_RNA2.patient_id.isin(result2.patient_id)]



def standardized_wasserstein_distance(a, b):
  """a and b are numpy arrays."""
  numerator = wasserstein_distance(a, b)
  denominator = np.std(np.concatenate([a, b]))
  return numerator / denominator if denominator != .0 else .0


def getstats(gene,testgene,PD,PRCR):
    ttest = scipy.stats.ttest_ind(PD, PRCR, equal_var=False)
    ks_2samp = scipy.stats.ks_2samp(PD, PRCR)
    foldchange = np.mean(PD)/(np.mean(PRCR)+0.001)
    try:
        ttest_pvalue = ttest.pvalue
    except:
        ttest_pvalue =1
    try:
        ks_2samp_pvalue = ks_2samp.pvalue
    except:
        ks_2samp_pvalue =1
    
    redf = pd.DataFrame(
            {'gene': gene,
             'testgene':testgene,
             'ttest':ttest_pvalue,
             'ks_2samp': ks_2samp_pvalue,
             'PD_q10': np.quantile(PD,0.10),
             'PD_q25': np.quantile(PD,0.25),
             'PD_q50': np.quantile(PD,0.5),
             'PD_q75': np.quantile(PD,0.75),
             'PD_q90': np.quantile(PD,0.90),
             'PRCR_q10': np.quantile(PRCR,0.10),
             'PRCR_q25': np.quantile(PRCR,0.25),
             'PRCR_q50': np.quantile(PRCR,0.5),
             'PRCR_q75': np.quantile(PRCR,0.75),
             'PRCR_q90': np.quantile(PRCR,0.90),
             'foldchange_median': foldchange,
             'foldchange_mean': np.mean(PD)/(np.mean(PRCR)+0.001),
             'foldchange_q25': np.quantile(PD,0.25)/(np.quantile(PRCR,0.25)+0.001),
             'foldchange_q75': np.quantile(PD,0.75)/(np.quantile(PRCR,0.75)+0.001),
             'foldchange_q90': np.quantile(PD,0.90)/(np.quantile(PRCR,0.90)+0.001),
             'stand_wassdis':standardized_wasserstein_distance(PD, PRCR),
             'PD_count': len(PD),
             'PRCR_count': len(PRCR),
             },index=[0])
    return redf

RNA_gene = 'CXCL9'
CXCL9= mydata_RNA3.query("gene == '%s'" % RNA_gene)
re_distrutiontest =[]
for gene in counts40dict.keys():
    pids = result2.query("symbol == '%s'" % gene).patient_id
    group_mt=CXCL9[CXCL9.patient_id.isin(pids)]
    group_wt=CXCL9[~CXCL9.patient_id.isin(pids)]
    jie1 = getstats(gene,RNA_gene,group_mt.fpkm,group_wt.fpkm)
    re_distrutiontest.append(jie1)

resultre = pd.concat(re_distrutiontest, ignore_index=True)
resultre.to_csv('/share_storage/home/nacoo/DI/20230425/mutations10_%s_ok1.xls' % RNA_gene, sep='\t',index=False)


