from statistics import mean, median
import scipy as scipy
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import heapq
import pandas as pd
from scipy.stats import chi2
import collections
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import rpy2.robjects as robjects
from matplotlib.backends.backend_pdf import PdfPages

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 10)

mydata1= pd.read_csv('HCC_immuno_response_RNA_list20230208.txt',sep='\t')
#print(mydata1)

myenst=pd.read_csv('HCC_genetranscript20230209.csv.zip',sep=',',index_col=False)

myenst = myenst[['patient_id','sample_id','project_id','gene','enst','fpkm']]
myenst = myenst[myenst.patient_id.isin(list(mydata1.Code))]

myenst_re = mydata1.merge(myenst,left_on='Sample_id',right_on='sample_id',how='left')
myenst_re.drop_duplicates(subset=['Sample_id','enst'], keep='first',inplace=True)
myenst_re['fpkm'] =myenst_re['fpkm'].astype(int)


def myrwaddR(a,b):
    rwaddR = robjects.r('''
        library("waddR")
        myf <- function(a,b){
        spec.output<-c("pval","d.wass^2","perc.loc","perc.size","perc.shape")
        wasserstein.test(a,b,method="SP",permnum=10000)[spec.output]-> re
          return(re)
        }
        ''')
    myre = rwaddR(robjects.FloatVector(a),robjects.FloatVector(b))
    return myre


re_distrutiontest =[]
for  enst, enst_FPKM in myenst_re.groupby('enst'):
    gene=list(enst_FPKM['gene'])[0]

    PD = enst_FPKM[enst_FPKM['Response'] == 'PD'].fpkm.values
    PRCR = enst_FPKM[enst_FPKM['Response'] == 'PR/CR'].fpkm.values

    ttest = scipy.stats.ttest_ind(PD, PRCR, equal_var=False)
    ks_2samp = scipy.stats.ks_2samp(PD, PRCR)
    foldchange = np.median(PD)/(np.median(PRCR)+0.001)

    try:
        ttest_pvalue = ttest.pvalue
    except:
        ttest_pvalue =1
    try:
        ks_2samp_pvalue = ks_2samp.pvalue
    except:
        ks_2samp_pvalue =1
    

    waddR_re = myrwaddR(PD,PRCR)
    waddR_pd = pd.DataFrame(np.array([np.array(waddR_re)]),columns=['pval', 'dwass2', 'perc_loc','perc_size','perc_shape'])


    redf = pd.DataFrame(
            {'gene': gene,
            'enst': enst,
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
             'waddR_pval': waddR_pd.pval,
             'waddR_dwass2': waddR_pd.dwass2,
             'waddR_perc_loc': waddR_pd.perc_loc,
             'waddR_perc_size': waddR_pd.perc_size,
             'waddR_perc_shape': waddR_pd.perc_shape,
             },index=[0])
    re_distrutiontest.append(redf)

result = pd.concat(re_distrutiontest, ignore_index=True)
result.to_csv('HCC_PD_PR_enst_int_ok2.xls', sep='\t',index=False)

