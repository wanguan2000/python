#import matplotlib.pyplot as plt
import pandas as pd
import scipy as scipy
import numpy as np
from scipy import stats
from scipy.stats import chi2
from scipy.stats import fisher_exact
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import heapq
from sklearn.cluster import DBSCAN
import pingouin as pg
from numba import njit
import collections
import warnings
import rpy2.robjects as robjects
from scipy.stats import wasserstein_distance

def myrwaddR(a,b):
    rwaddR = robjects.r('''
        library("waddR")
        myf <- function(a,b){
        spec.output<-c("pval","d.wass^2","perc.loc","perc.size","perc.shape")
        wasserstein.test(a,b,method="SP",permnum=10000)[spec.output]-> re
          return(re)
        }
        ''')
    waddR_re = rwaddR(robjects.FloatVector(a),robjects.FloatVector(b))
    waddR_pd = pd.DataFrame(np.array([np.array(waddR_re)]),columns=['pval', 'dwass2', 'perc_loc','perc_size','perc_shape'])
    return waddR_pd
@njit
def logrank_test(d_myfactor, d_survivalMonth, d_death_observed, d_mylength):
    nall_timetotal = d_mylength
    n2i_timetotal = 0
    d2i_deathtotal = 0  # observed_death
    T2i_sum = 0.0  # expected
    patient_count_death = 0
    V2i_sum = 0.0

    # [HR, chisq, observed, expected]
    for n in range(0, d_mylength, 1):
        if (d_death_observed[n] == 1):
            patient_count_death += 1
        if (d_myfactor[n] == 1):
            n2i_timetotal += 1
            if (d_death_observed[n] == 1):
                d2i_deathtotal += 1
    n = 0
    i = 0
    while i < (d_mylength):
        dall_death_censored = 0
        dall_death = 0
        d2i_death = 0
        d2i_death_censored = 0
        # T2i = 0.0
        # V2i = 0.0
        if ((i == (d_mylength - 1)) or (d_survivalMonth[i] != d_survivalMonth[i + 1])):
            while (n <= i):
                dall_death_censored += 1
                if d_death_observed[n] == 1:
                    dall_death += 1
                if (d_death_observed[n] == 1 and d_myfactor[n] == 1):
                    d2i_death += 1
                if (d_myfactor[n] == 1):
                    d2i_death_censored += 1
                n += 1

            T2i = dall_death / nall_timetotal * n2i_timetotal
            T2i_sum = T2i_sum + T2i
            V2i = (n2i_timetotal / nall_timetotal) * (1 - (n2i_timetotal / nall_timetotal)) * (
                    (nall_timetotal - dall_death) / (nall_timetotal - 0.99999)) * dall_death
            V2i_sum = V2i_sum + V2i

            nall_timetotal = nall_timetotal - dall_death_censored
            n2i_timetotal = n2i_timetotal - d2i_death_censored
        i += 1

    # chisq = (observed_death - expected) * (observed_death - expected) / (myvar + 0.00001)
    chisq = (d2i_deathtotal - T2i_sum) * (d2i_deathtotal - T2i_sum) / (V2i_sum + 0.00001)
    # HR = (observed_death * (patient_count_death - expected)) / (expected * (patient_count_death - observed_death) + 0.00001)
    if V2i_sum == 0:
        HR = 1
    else:
        HR = (d2i_deathtotal * (patient_count_death - T2i_sum)) / (
                T2i_sum * (patient_count_death - d2i_deathtotal) + 0.00001)
    # [HR, chisq, observed, expected]
    d_hrresult = [HR, chisq, d2i_deathtotal, T2i_sum]
    # pvalue = chisq
    # pvalue = chi2.sf(chisq, 1)
    # pvalue chi2.sf(d_hrresult[1], 1)
    re = np.array([1.0, 1.0, 0.0, 0.0])

    re[0] = HR
    re[1] = chisq
    re[2] = np.count_nonzero(d_myfactor == 0)
    re[3] = (d_myfactor == 1).sum()
    return re


class RNA_groupsurvival:
    def __init__(self,mydata_FPKM3, nstep=1000, bw_adjust=0.5, eps_score=10, my_min_samples=7, pvaluecut=0.2, HRcut=0.7):
        self.mydata_FPKM3 = mydata_FPKM3
        self.nstep = nstep
        self.bw_adjust = bw_adjust
        self.eps_score = eps_score
        self.my_min_samples = my_min_samples
        self.pvaluecut = pvaluecut
        self.HRcut = HRcut
    
    def standardized_wasserstein_distance(self, a,b):
        numerator = wasserstein_distance(a, b)
        denominator = np.std(np.concatenate([a, b]))
        stand_wassd = 0
        if denominator != 0:
            stand_wassd = numerator / denominator
        return stand_wassd

    def getDBSCAN(self, X, eps_score, my_min_samples):
        def do(X, eps_score, my_min_samples):
            topcutoff = 0
            X_sort = np.sort(X, axis=0)
            my_std = np.std(X_sort)
            clustering = DBSCAN(eps=my_std / eps_score, min_samples=my_min_samples).fit(X_sort.reshape(-1, 1))
            clustering_diff = (np.diff(clustering.labels_))
            clustering_diff2 = np.where(clustering_diff != 0, 1, clustering_diff)
            clustering_diff_points = np.where(clustering_diff2 == 1)[0]
            if clustering_diff_points.size > 0:
                topcutoff = X_sort[clustering_diff_points][-1]
                return topcutoff
            else:
                return topcutoff

        topcutoff = do(X, eps_score, my_min_samples)
        if topcutoff == 0:
            topcutoff = do(X, eps_score * 2, my_min_samples)
        return topcutoff

    def get_peakdiff(self, negative_density, positive_density, negative_numpy, x_lab):
        diff_negative_positive = negative_density - positive_density

        diff_negative_peaks, diff_negative_peakheights = find_peaks(diff_negative_positive, height=0)
        if len(diff_negative_peaks) == 0:
            return 0, 0, 0
        negative_diff_heapq = diff_negative_peaks[
            heapq.nlargest(2, range(len(diff_negative_peakheights['peak_heights'])),
                           diff_negative_peakheights['peak_heights'].take)]
        negative_difftop1 = x_lab[negative_diff_heapq[0]]

        negative_difftop1_std = negative_difftop1 + 1.5 * np.std(negative_numpy[negative_numpy < negative_difftop1])
        if len(negative_diff_heapq) > 1:
            negative_difftop2 = x_lab[negative_diff_heapq[1]]
        else:
            negative_difftop2 = x_lab[0]
        return negative_difftop1, negative_difftop1_std, negative_difftop2

    def get_peaktop(self, negative_numpy, bw_adjust, x_lab):
        negative_kde = stats.gaussian_kde(dataset=negative_numpy, bw_method='scott')
        negative_kde.set_bandwidth(negative_kde.factor * bw_adjust)
        negative_density = negative_kde(x_lab)
        negative_peaks, negative_peakheights = find_peaks(negative_density, height=0)
        if len(negative_peaks) == 0:
            return 0, 0, 0, negative_density
        negative_heapq = heapq.nlargest(2, range(len(negative_peakheights['peak_heights'])),
                                        negative_peakheights['peak_heights'].take)
        negative_maxtop1_index = negative_peaks[negative_heapq[0]]
        negative_maxtop1 = x_lab[negative_maxtop1_index]
        ###std as normal distribution
        negative_maxstdtop1 = negative_maxtop1 + 1.5 * np.std(negative_numpy[negative_numpy < negative_maxtop1])
        if len(negative_heapq) > 1:
            negative_maxtop2 = x_lab[negative_peaks[negative_heapq[1]]]
        else:
            negative_maxtop2 = x_lab[0]
        return negative_maxtop1, negative_maxstdtop1, negative_maxtop2, negative_density

    def get_crosspoint(self, negative_density, positive_density, x_lab):
        a = 0 - (np.abs(negative_density - positive_density))
        b = np.abs(min(a)) + a
        cross_peaks, cross_peakheights = find_peaks(b, height=0)
        # cross peaks rank by positive_density
        cross_peaks_positive_density = positive_density[cross_peaks] * 1000
        cross_heapq = cross_peaks[
            heapq.nlargest(2, range(len(cross_peaks_positive_density)), cross_peaks_positive_density.take)]
                
        if len(cross_heapq) == 0:
            cross_top1 = x_lab[0]
            cross_top2 = x_lab[0]
        elif len(cross_heapq) == 1:
             cross_top1 = x_lab[cross_heapq[0]]
             cross_top2 = x_lab[cross_heapq[0]]
        else:
             cross_top1 = x_lab[cross_heapq[0]]
             cross_top2 = x_lab[cross_heapq[1]]

        return cross_top1, cross_top2,

    def getpeak(self, negative_group, positive_group):
        nstep = self.nstep
        bw_adjust = self.bw_adjust
        negative_numpy = negative_group
        positive_numpy = positive_group

        allscore = (np.append(negative_numpy, positive_numpy))

        x_lab = np.linspace(min(allscore), max(allscore), nstep)
        #####max
        negative_top1, negative_top1std, negative_top2, negative_density = self.get_peaktop(negative_numpy, bw_adjust,
                                                                                            x_lab)
        positive_top1, positive_top1std, positive_top2, positive_density = self.get_peaktop(positive_numpy, bw_adjust,
                                                                                            x_lab)
        #####diff

        negative_difftop1, negative_difftop1_std, negative_difftop2 = self.get_peakdiff(negative_density,
                                                                                        positive_density,
                                                                                        negative_numpy, x_lab)
        positive_difftop1, positive_difftop1_std, positive_difftop2 = self.get_peakdiff(positive_density,
                                                                                        negative_density,
                                                                                        positive_numpy, x_lab)

        ###cross point peak in highest positive density
        cross_top1, cross_top2 = self.get_crosspoint(negative_density, positive_density, x_lab)

        ###DBSCAN
        negative_DBSCANtop = self.getDBSCAN(negative_numpy, self.eps_score, self.my_min_samples)
        positive_DBSCANtop = self.getDBSCAN(positive_numpy, self.eps_score, self.my_min_samples)

        a = 0 - (np.abs(negative_density - positive_density))
        b = np.abs(min(a)) + a

        # plt.plot(x_lab, positive_density - negative_density, color='#ffd600')
        # plt.plot(x_lab, b, color='#00c3ff')
        # plt.plot(x_lab, negative_density, color='#61b15a')
        # plt.plot(x_lab, positive_density, color='#c75643')
        # plt.hlines(0, 0, 90, color="black")

        return negative_top1, negative_top1std, negative_top2, positive_top1, positive_top1std, positive_top2, negative_difftop1, negative_difftop1_std, negative_difftop2, positive_difftop1, positive_difftop1_std, positive_difftop2, cross_top1, cross_top2, negative_DBSCANtop, positive_DBSCANtop


    def peak_HR(self, peak_x, negative_group, positvie_group, prefix):
        # peak_x = 30
        #negative_group.loc[:, 'myfpkm_level'] = pd.cut(negative_group['fpkm'], [-0.1, peak_x, 1000000], labels=[1, 0])
        #positvie_group['myfpkm_level'] = pd.cut(positvie_group['fpkm'], [-0.1, peak_x, 10000000], labels=[1, 0])
        # print(negative_group)

        negative_group.loc[:,'myfpkm_level'] = negative_group.fpkm.apply(lambda a: 1 if (a < peak_x) else 0)
        positvie_group.loc[:,'myfpkm_level'] = positvie_group.fpkm.apply(lambda a: 1 if (a < peak_x) else 0)

        neg_HR, neg_chisq, neg_0, neg_1 = logrank_test(negative_group['myfpkm_level'].to_numpy(),
                                                        negative_group['survivalMonth'].to_numpy(),
                                                        negative_group['death_observed'].to_numpy(),
                                                        negative_group['myfpkm_level'].to_numpy().size)
                        

        pos_HR, pos_chisq, pos_0, pos_1 = logrank_test(positvie_group['myfpkm_level'].to_numpy(),
                                                        positvie_group['survivalMonth'].to_numpy(),
                                                        positvie_group['death_observed'].to_numpy(),
                                                        positvie_group['myfpkm_level'].to_numpy().size)
        
        neg_pvalue = chi2.sf(neg_chisq, 1)
        pos_pvalue = chi2.sf(pos_chisq, 1)

        redf = collections.OrderedDict({prefix + '_cut': [peak_x],
                                        prefix + '_neg_HR': [neg_HR],
                                        prefix + '_neg_pvalue': [neg_pvalue],
                                        prefix + '_neg_1': [neg_1],
                                        prefix + '_neg_1_mscore': [
                                            (negative_group.myscore[negative_group.myfpkm_level == 1]).median()],
                                        prefix + '_neg_0': [neg_0],
                                        prefix + '_neg_0_mscore': [
                                            (negative_group.myscore[negative_group.myfpkm_level == 0]).median()],
                                        prefix + '_pos_HR': [pos_HR],
                                        prefix + '_pos_pvalue': [pos_pvalue],
                                        prefix + '_pos_1': [pos_1],
                                        prefix + '_pos_1_mscore': [
                                            (positvie_group.myscore[positvie_group.myfpkm_level == 1]).median()],
                                        prefix + '_pos_0': [pos_0],
                                        prefix + '_pos_0_mscore': [
                                            (positvie_group.myscore[positvie_group.myfpkm_level == 0]).median()]})
        return redf

    def HR_corr(self, gene, mydata):
        pvaluecut = self.pvaluecut
        HRcut = self.HRcut
        b = mydata[mydata.gene == gene]
        # print(b)
        mycut = (b[b.columns[b.columns.str.endswith('_cut')]]).to_numpy()[0]
        mycut_neg_HR = (b[b.columns[b.columns.str.endswith('_neg_HR')]]).to_numpy()[0]
        mycut_pos_HR = (b[b.columns[b.columns.str.endswith('_pos_HR')]]).to_numpy()[0]

        neg_1_mscore = (b[b.columns[b.columns.str.endswith('_neg_1_mscore')]]).to_numpy()[0]
        pos_1_mscore = (b[b.columns[b.columns.str.endswith('_pos_1_mscore')]]).to_numpy()[0]

        mycut_neg_HR[mycut_neg_HR < 0.05] = 0.05
        mycut_neg_HR[mycut_neg_HR > 20] = 20

        mycut_pos_HR[mycut_pos_HR < 0.05] = 0.05
        mycut_pos_HR[mycut_pos_HR > 20] = 20

        log_neg_HR = np.log2(1 / mycut_neg_HR)
        log_pos_HR = np.log2(1 / mycut_pos_HR)

        cutHR = pd.DataFrame(
            {'mycut_numpy': mycut, 'log_neg_HR': log_neg_HR, 'log_pos_HR': log_pos_HR,
             'neg_1_mscore': neg_1_mscore, 'pos_1_mscore': pos_1_mscore
             },
        )

        cutHR.dropna(axis='rows', how='any', inplace=True)

        cut_HR_pearsonr_neg = scipy.stats.pearsonr(cutHR.mycut_numpy, cutHR.log_neg_HR)
        cut_HR_spearmanr_neg = scipy.stats.spearmanr(cutHR.mycut_numpy, cutHR.log_neg_HR)

        socre_HR_pearsonr_neg = scipy.stats.pearsonr(cutHR.neg_1_mscore, cutHR.log_neg_HR)
        socre_HR_spearmanr_neg = scipy.stats.spearmanr(cutHR.neg_1_mscore, cutHR.log_neg_HR)

        cut_HR_pearsonr_pos = scipy.stats.pearsonr(cutHR.mycut_numpy, cutHR.log_pos_HR)
        cut_HR_spearmanr_pos = scipy.stats.spearmanr(cutHR.mycut_numpy, cutHR.log_pos_HR)

        socre_HR_pearsonr_pos = scipy.stats.pearsonr(cutHR.pos_1_mscore, cutHR.log_pos_HR)
        socre_HR_spearmanr_pos = scipy.stats.spearmanr(cutHR.pos_1_mscore, cutHR.log_pos_HR)

        # <0.2 and selectcutoff
        cutoff_neg_HR = np.array([np.nan])
        cutoff_pos_HR = np.array([np.nan])

        neg_pvalue = (b[b.columns[b.columns.str.endswith('_neg_pvalue')]]).to_numpy()[0]
        pos_pvalue = (b[b.columns[b.columns.str.endswith('_pos_pvalue')]]).to_numpy()[0]

        pos_1 = (b[b.columns[b.columns.str.endswith('_pos_1')]]).to_numpy()[0]
        pos_0 = (b[b.columns[b.columns.str.endswith('_pos_0')]]).to_numpy()[0]
        pos_mypercent = pos_0 / (pos_1 + pos_0)

        neg_1 = (b[b.columns[b.columns.str.endswith('_neg_1')]]).to_numpy()[0]
        neg_0 = (b[b.columns[b.columns.str.endswith('_neg_0')]]).to_numpy()[0]
        neg_mypercent = neg_0 / (neg_1 + neg_0)

        count_neg_pvalue = 0
        count_pos_pvalue = 0

        for i in range(len(mycut)):
            if (mycut_neg_HR[i] > (1 / HRcut)) and neg_pvalue[i] <= pvaluecut:
                count_neg_pvalue = count_neg_pvalue + 1
                cutoff_neg_HR = np.append(cutoff_neg_HR, mycut[i])
                cutoff_neg_HR = cutoff_neg_HR[~np.isnan(cutoff_neg_HR)]

            if mycut_pos_HR[i] < HRcut and pos_pvalue[i] <= pvaluecut:
                count_pos_pvalue = count_pos_pvalue + 1
                cutoff_pos_HR = np.append(cutoff_pos_HR, mycut[i])
                cutoff_pos_HR = cutoff_pos_HR[~np.isnan(cutoff_pos_HR)]

        if not np.isnan(min(cutoff_pos_HR)):
            pos_cut_percent = min(pos_mypercent[np.where(mycut == min(cutoff_pos_HR))])
            neg_cut_percent = min(neg_mypercent[np.where(mycut == min(cutoff_pos_HR))])
        else:
            pos_cut_percent = np.nan
            neg_cut_percent = np.nan
        
        try:
           cut_mscore_HR_neg = pg.linear_regression(cutHR[['mycut_numpy', 'neg_1_mscore']], cutHR['log_neg_HR'],remove_na=True)
           cut_mscore_HR_neg_r2 =  cut_mscore_HR_neg.adj_r2[0]
        except:
            cut_mscore_HR_neg_r2 =0
        try:
           cut_mscore_HR_pos = pg.linear_regression(cutHR[['mycut_numpy', 'pos_1_mscore']], cutHR['log_pos_HR'],remove_na=True)
           cut_mscore_HR_pos_r2 = cut_mscore_HR_pos.adj_r2[0],
        except:
            cut_mscore_HR_pos_r2 =0

        HRcorr_df = pd.DataFrame(
            {'gene': gene,
             'cut_HR_pearsonr_neg_r': cut_HR_pearsonr_neg[0],
             'cut_HR_pearsonr_neg_pvalue': cut_HR_pearsonr_neg[1],
             'cut_HR_spearmanr_neg_r': cut_HR_spearmanr_neg[0],
             'cut_HR_spearmanr_neg_pvalue': cut_HR_spearmanr_neg[1],

             'cut_HR_pearsonr_pos_r': cut_HR_pearsonr_pos[0],
             'cut_HR_pearsonr_pos_pvalue': cut_HR_pearsonr_pos[1],
             'cut_HR_spearmanr_pos_r': cut_HR_spearmanr_pos[0],
             'cut_HR_spearmanr_pos_pvalue': cut_HR_spearmanr_pos[1],

             'socre_HR_pearsonr_neg_r': socre_HR_pearsonr_neg[0],
             'socre_HR_pearsonr_neg_pvalue': socre_HR_pearsonr_neg[1],
             'socre_HR_spearmanr_neg_r': socre_HR_spearmanr_neg[0],
             'socre_HR_spearmanr_neg_pvalue': socre_HR_spearmanr_neg[1],

             'socre_HR_pearsonr_pos_r': socre_HR_pearsonr_pos[0],
             'socre_HR_pearsonr_pos_pvalue': socre_HR_pearsonr_pos[1],
             'socre_HR_spearmanr_pos_r': socre_HR_spearmanr_pos[0],
             'socre_HR_spearmanr_pos_pvalue': socre_HR_spearmanr_pos[1],

             'cut_mscore_HR_neg_r2': cut_mscore_HR_neg_r2,
             'cut_mscore_HR_pos_r2': cut_mscore_HR_pos_r2,

             'count_neg_pvalue': count_neg_pvalue,
             'count_pos_pvalue': count_pos_pvalue,

             'neg_HR_cutoff': min(cutoff_neg_HR),
             'pos_HR_cutoff': min(cutoff_pos_HR),
             'neg_cut_percent': neg_cut_percent,
             'pos_cut_percent': pos_cut_percent,

             'pos_neg_cut_ratio': pos_cut_percent / (neg_cut_percent + 0.0000001),
             },
            index=[0])
        allresult = mydata.merge(HRcorr_df, left_on='gene', right_on='gene')
        return allresult

    def distrution_pipelinemap(self, gene,death_observed,survivalMonth,myscore,mylevel):
        #mydata_factor must be sorted!
        
        #gene_FPKM = mydata_FPKM[mydata_FPKM.gene == gene]
        #c1 = mydata_factor.merge(gene_FPKM, left_on='patient_id', right_on='patient_id',how='inner')
        #c1= mydata_FPKM2[[gene,'death_observed','survivalMonth','myscore','mylevel']]
        c1= self.mydata_FPKM3[[gene,death_observed,survivalMonth,myscore,mylevel]]
        c1.loc[:,'fpkm'] = c1[gene]
        try:
            pearson = c1.myscore.corr(c1.fpkm, method='pearson')
            spearman = c1.myscore.corr(c1.fpkm, method='spearman')
        except:
            print(gene)
        # print(c1.describe(include=[np.number]))

        fpkm_numpy = c1.fpkm.to_numpy()

        allq10 = np.quantile(fpkm_numpy, 0.10)
        allq25 = np.quantile(fpkm_numpy, 0.25)
        allq50 = np.quantile(fpkm_numpy, 0.50)
        allq75 = np.quantile(fpkm_numpy, 0.75)
        allq90 = np.quantile(fpkm_numpy, 0.90)
        allmean = np.mean(fpkm_numpy)
        allmeanq90 = allq90 / 2
        all2FPKM = 2

        positive_group = (c1[c1.mylevel == True]).copy()
        negative_group = (c1[c1.mylevel == False]).copy()

        positive_value = positive_group.fpkm.to_numpy()
        negative_value = negative_group.fpkm.to_numpy()

        ttest = scipy.stats.ttest_ind(positive_value, negative_value, equal_var=False)

        ks_2samp_pvalue = 1
        try:
           ks_2samp = scipy.stats.ks_2samp(positive_value, negative_value)
           ks_2samp_pvalue = ks_2samp.pvalue
        except:
            ks_2samp_pvalue = 1

        waddR_pd_pval = 1
        waddR_pd_dwass2 = 0
        waddR_pd_perc_loc = 0
        waddR_pd_perc_size = 0
        waddR_pd_perc_shape = 0
        try:
            waddR_re = myrwaddR(positive_value, negative_value)
            waddR_pd_pval = waddR_re.pval
            waddR_pd_dwass2 = waddR_re.dwass2
            waddR_pd_perc_loc = waddR_re.perc_loc
            waddR_pd_perc_size = waddR_re.perc_size
            waddR_pd_perc_shape = waddR_re.perc_shape
        except:
            pass
        
        negative_numpy = negative_value
        if len(np.unique(negative_numpy)) == 1:
            negative_numpy = negative_numpy + (
                    np.random.randint(low=1, high=10, size=len(negative_numpy), dtype='l') / 100.0)

        positive_numpy =positive_value
        if len(np.unique(positive_numpy)) == 1:
            positive_numpy = positive_numpy + np.random.randint(low=1, high=10, size=len(positive_numpy),
                                                                dtype='l') / 100.0

        negative_top1, negative_top1std, negative_top2, positive_top1, positive_top1std, positive_top2, negative_difftop1, negative_difftop1_std, negative_difftop2, positive_difftop1, positive_difftop1_std, positive_difftop2, cross_top1, cross_top2, negative_DBSCANtop, positive_DBSCANtop = self.getpeak(
            negative_numpy, positive_numpy)

        # peak_x = 30
        negative_top1_HR = self.peak_HR(negative_top1, negative_group, positive_group, 'negative_top1')
        negative_top1std_HR = self.peak_HR(negative_top1std, negative_group, positive_group, 'negative_top1std')
        negative_top2_HR = self.peak_HR(negative_top2, negative_group, positive_group, 'negative_top2')

        positive_top1_HR = self.peak_HR(positive_top1, negative_group, positive_group, 'positive_top1')
        positive_top1std_HR = self.peak_HR(positive_top1std, negative_group, positive_group, 'positive_top1std')
        positive_top2_HR = self.peak_HR(positive_top2, negative_group, positive_group, 'positive_top2')

        negative_difftop1_HR = self.peak_HR(negative_difftop1, negative_group, positive_group, 'negative_difftop1')
        negative_difftop1_std_HR = self.peak_HR(negative_difftop1_std, negative_group, positive_group,
                                                'negative_difftop1_std')
        negative_difftop2_HR = self.peak_HR(negative_difftop2, negative_group, positive_group, 'negative_difftop2')

        positive_difftop1_HR = self.peak_HR(positive_difftop1, negative_group, positive_group, 'positive_difftop1')
        positive_difftop1_std_HR = self.peak_HR(positive_difftop1_std, negative_group, positive_group,
                                                'positive_difftop1_std')
        positive_difftop2_HR = self.peak_HR(positive_difftop2, negative_group, positive_group, 'positive_difftop2')

        cross_top1_HR = self.peak_HR(cross_top1, negative_group, positive_group, 'cross_top1')
        cross_top2_HR = self.peak_HR(cross_top2, negative_group, positive_group, 'cross_top2')

        negative_DBSCANtop_HR = self.peak_HR(negative_DBSCANtop, negative_group, positive_group, 'negative_DBSCANtop')
        positive_DBSCANtop_HR = self.peak_HR(positive_DBSCANtop, negative_group, positive_group, 'positive_DBSCANtop')

        allq10_df_HR = self.peak_HR(allq10, negative_group, positive_group, 'allq10')
        allq25_df_HR = self.peak_HR(allq25, negative_group, positive_group, 'allq25')
        allq50_df_HR = self.peak_HR(allq50, negative_group, positive_group, 'allq50')
        allq75_df_HR = self.peak_HR(allq75, negative_group, positive_group, 'allq75')
        allq90_df_HR = self.peak_HR(allq90, negative_group, positive_group, 'allq90')

        allmean_df_HR = self.peak_HR(allmean, negative_group, positive_group, 'allmean')
        allmeanq90_df_HR = self.peak_HR(allmeanq90, negative_group, positive_group, 'allmeanq90')
        all2FPKM_df_HR = self.peak_HR(all2FPKM, negative_group, positive_group, 'all2FPKM')


        stat_df = collections.OrderedDict(
            {'gene': [gene], 'pearson': [pearson], 'spearman': [spearman], 'allq10': [allq10], 'allq25': [allq25],
             'allq50': [allq50],
             'allq75': [allq75], 'allq90': [allq90], 'allmean': [allmean], 'allmeanq90': [allmeanq90],
             'mylevellow_std': [np.std(negative_value)],
             'mylevellow_q25': [np.quantile(negative_value, 0.25)],
             'mylevellow_median': [np.median(negative_value)],
             'mylevellow_q75': [np.quantile(negative_value, 0.75)],
             'mylevellow_q95': [np.quantile(negative_value, 0.95)],
             'mylevelhigh_std': [np.std(positive_value)],
             'mylevelhigh_q25': [np.quantile(positive_value, 0.25)],
             'mylevelhigh_median': [np.median(positive_value)],
             'mylevelhigh_q75': [np.quantile(positive_value, 0.75)],
             'mylevelhigh_q95': [np.quantile(positive_value, 0.95)],
             'foldchange_median':  np.median(positive_value)/(np.median(negative_value)+0.001),
             'foldchange_mean': np.mean(positive_value)/(np.mean(negative_value)+0.001),
             'foldchange_q25': np.quantile(positive_value,0.25)/(np.quantile(negative_value,0.25)+0.001),
             'foldchange_q75': np.quantile(positive_value,0.75)/(np.quantile(negative_value,0.75)+0.001),
             'foldchange_q90': np.quantile(positive_value,0.90)/(np.quantile(negative_value,0.90)+0.001),
             'ttest': [ttest.pvalue], 'ks_2samp': [ks_2samp_pvalue],
             'waddR_pval': waddR_pd_pval,
             'stand_wassd': self.standardized_wasserstein_distance(positive_value,negative_value),
             'waddR_dwass2': waddR_pd_dwass2,
             'waddR_perc_loc': waddR_pd_perc_loc,
             'waddR_perc_size': waddR_pd_perc_size,
             'waddR_perc_shape': waddR_pd_perc_shape
             })

        HR_result0 = stat_df | negative_top1_HR | negative_top1std_HR | negative_top2_HR | positive_top1_HR | positive_top1std_HR | positive_top2_HR | negative_difftop1_HR | negative_difftop1_std_HR \
                    | negative_difftop2_HR | positive_difftop1_HR | positive_difftop1_std_HR | positive_difftop2_HR | cross_top1_HR | cross_top2_HR | allq10_df_HR | allq25_df_HR | allq50_df_HR | allq75_df_HR | allq90_df_HR | allmean_df_HR | allmeanq90_df_HR | all2FPKM_df_HR
        HR_result = pd.DataFrame.from_dict(HR_result0)
        all_result = self.HR_corr(gene, HR_result)
        # positive_difftop1_cut-negative_difftop1_cut
        all_result.loc[:, 'CS_BFscore'] = (all_result['positive_difftop1_cut'] - all_result['negative_difftop1_cut'])
        # (positive_difftop1_cut-negative_difftop1_cut)/mylevellow_median
        all_result.loc[:, 'CS_foldchang'] = (all_result['positive_difftop1_cut'] - all_result[
            'negative_difftop1_cut']) / (all_result['mylevellow_median'] + 0.001)
        # (positive_difftop1_cut-negative_difftop1_cut)/mylevellow_std
        all_result.loc[:, 'CS_zscore'] = (all_result['positive_difftop1_cut'] - all_result['negative_difftop1_cut']) / (
                    all_result['mylevellow_std'] + 0.001)
        return all_result
