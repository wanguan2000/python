from re import I
import pandas as pd
import scipy as scipy
import numpy as np
from scipy import stats
from scipy.stats import fisher_exact
from Causal_HR_column import RNA_groupsurvival
from joblib import Parallel, delayed
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False,nb_workers=8)

#################
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 5)

mydata_FPKM = pd.read_csv('/Users/nacoo/Downloads/TTNT/TTNT2/HCC_RNA.xls', low_memory=False, encoding='utf-8', sep='\t',index_col=0)
geneset = (set(mydata_FPKM.gene))
mydata_FPKM = mydata_FPKM[['patient_id',  'gene', 'fpkm', 'tpm', 'count']]
mydata_FPKM.drop_duplicates(subset=['patient_id','gene'], keep='first',inplace=True)

mydata_FPKM2 = pd.pivot_table(mydata_FPKM, index='patient_id', columns='gene', values=['fpkm'])
mydata_FPKM2.columns = mydata_FPKM2.columns.droplevel()
#######AZIN1editor
AZIN1edited = '/Users/nacoo/Downloads/TTNT/AZIN1edited/azin1_re.xls'
mydata_AZIN1edited = pd.read_csv(AZIN1edited, low_memory=False,encoding='utf-8', sep='\t') 
mydata_AZIN1edited=mydata_AZIN1edited[['ratio','patient_id']]
mydata_AZIN1edited.drop_duplicates(subset=['patient_id'], keep='first',inplace=True)
mydata_FPKM3 = mydata_FPKM2.merge(mydata_AZIN1edited,left_on='patient_id',right_on='patient_id',how='left')
mydata_FPKM4 = (mydata_FPKM3[['AZIN1','ratio','patient_id']])
mydata_FPKM4['AZIN1editor'] = mydata_FPKM4['AZIN1']*mydata_FPKM4['ratio']
mydata_FPKM4['AZIN1ratio'] = mydata_FPKM4['ratio']*100
#mydata_FPKM4.drop(columns=['ratio'],inplace = True)

mydata_FPKM4 = mydata_FPKM4[['AZIN1editor','AZIN1ratio','patient_id']]




mydata_FPKM5 = mydata_FPKM2.merge(mydata_FPKM4,left_on='patient_id',right_on='patient_id',how='left')
print(mydata_FPKM5)
##########AZIN1editor

factor_gene  = 'AZIN1ratio'
CD8A_high = mydata_FPKM5.query("(%s  > 17) and (ADAR > 70)" % factor_gene)

mydata_survivalMonth = pd.read_csv('/Users/nacoo/Downloads/TTNT/TTNT2/HCC_RNA_survival_withoutIC.csv', low_memory=False,encoding='utf-8', sep='\t')
id2code = pd.read_csv('/Users/nacoo/Downloads/TTNT/TTNT2/external_id_202202181530.csv', low_memory=False,encoding='utf-8', )  
mydata_survivalMonth2 = mydata_survivalMonth.merge(id2code,left_on='code',right_on='internal_id',how='inner')

mydata_survivalMonth2['mylevel'] = mydata_survivalMonth2['patient_id'].apply(lambda x: 1 if x in list(CD8A_high.patient_id) else 0)
mydata_survivalMonth2['myscore'] = mydata_survivalMonth2['patient_id'].apply(lambda x: 1 if x in list(CD8A_high.patient_id) else 0)

mydata_factor = mydata_survivalMonth2[["patient_id",'death_observed', 'survivalMonth',"myscore","mylevel"]]
mydata_factor.sort_values(by=['survivalMonth'],inplace=True)


mydata_FPKM6 = mydata_factor.merge(mydata_FPKM5, left_on='patient_id', right_on='patient_id',how='inner')
mydata_FPKM6.dropna(axis=0,inplace=True)
mydata_FPKM6.sort_values(by=['survivalMonth'],inplace=True)

mydata_FPKM6['myscore'] = mydata_FPKM6[factor_gene]
print(mydata_FPKM6)

start = time.time()
import warnings
warnings.filterwarnings("ignore")
myfun=RNA_groupsurvival(mydata_FPKM6)
c2 = mydata_FPKM6.columns.to_series()[5:].parallel_apply(myfun.distrution_pipelinemap, args=('death_observed','survivalMonth','myscore','mylevel'))
result = pd.concat(list(c2), ignore_index=True)
print(result)

result.to_csv('HCC_RNA_withoutIC_AZIN1ratioADAR_high1770_OS_parallel2.xls', sep='\t')

end = time.time()
print(end - start)
