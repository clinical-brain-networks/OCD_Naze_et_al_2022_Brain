################################################################################
# Structural analysis
#
# Author: Sebastien Naze
# QIMR Berghofer
# 2021
################################################################################

import argparse
import bct
import itertools
import json
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
from nilearn.image import load_img
from nilearn.plotting import plot_matrix, plot_glass_brain
import numpy as np
import os
import pickle
import pandas as pd
import platform
import scipy
from scipy.io import loadmat
import statsmodels
from statsmodels.stats import multitest
import ttest_ind_FWE

# get computer name to set paths
if platform.node()=='qimr18844':
    working_dir = '/home/sebastin/working/'
elif 'hpcnode' in platform.node():
    working_dir = '/mnt/lustre/working/'
else:
    print('Computer unknown! Setting working dir as /working')
    working_dir = '/working/'


### Global variables ###
#----------------------#

# paths
proj_dir = working_dir+'lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
subjs = pd.read_table(os.path.join(code_dir, 'subject_list.txt'), names=['name'])['name']

#atlas_dir = '/home/sebastin/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/'
atlas_dir = os.path.join(proj_dir, 'utils')
atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)

atlas_suffix = {'schaefer100_tianS1':'MNI_lps_mni.nii.gz', \
                'schaefer200_tianS2':'MNI_lps_mni.nii.gz', \
                'schaefer400_tianS4':'MNI_lps_mni.nii.gz', \
                'schaefer400_harrison2009':'.nii.gz', \
                'ocdOFClPFC_ocdAccdPut':'.nii.gz'}

### PART I : Extract connectomes ###
#----------------------------------#

def get_connectomes_from_mat(subjs, atlases, metrics, opts={'rfe':'dhollander','verbose':True}):
    """ Get connectomes from controls and patients from .mat file in derivative folder
    inputs:
        subjs: list of subjects
        atlas: list of  atlases
        metrics: list of metrics used in tck2connectome
        opts: dict of extras
    outputs:
        connectivity matrices: dict. with keys 'controls' and 'patients'
    """
    cohorts = ['controls', 'patients']
    conns = dict( ( ((atl,met,coh),[]) for atl,met,coh in itertools.product(atlases,metrics,cohorts) ) )
    discarded = []
    for i,subj in enumerate(subjs):
        subj_dir = os.path.join(deriv_dir, 'qsirecon', subj)
        fname = subj+'_acq-88_dir-AP_space-T1w_desc-preproc_space-T1w_connectivity_'+opts['rfe']+'.mat'
        fpath = os.path.join(subj_dir,'dwi', fname)
        if os.path.exists(fpath):
            f = loadmat(fpath)
        else:
            discarded.append((i,subj))
            #subjs.drop(i)
            print('Subject {} preprocessing not found, subject has been removed for subjects list.'.format([subj]))
            continue;

        for atlas,metric in itertools.product(atlases, metrics):
            c = f['_'.join([atlas, metric, 'connectivity'])]
            if 'control' in subj:
                conns[(atlas,metric,'controls')].append(c)
            elif 'patient' in subj:
                conns[(atlas,metric,'patients')].append(c)
            else:
                discarded.append((i,subj))
                #subjs.drop(i)
                print('Subject {} neither control nor patient? Subjected discarded, check name spelling'.format([subj]))
                continue;

    for k,v in conns.items():
        conns[k] = np.transpose(np.array(v),(1,2,0))
        if opts['verbose']:
            print(' --> shape: '.join([str(k), str(conns[k].shape)]))

    subjs = subjs.drop([disc[0] for disc in discarded])
    return conns,subjs,discarded


# Reconstruct nosift because QSIPrep did only SIFT
def get_connectomes_from_csv(subjs, atlases, metrics, opts={'rfe':'dhollander','verbose':True}):
    """ Get connectomes from controls and patients from CSV file (in postprocessing folder)
    inputs:
        subjs: list of subjects
        atlas: list of  atlases
        metrics: list of metrics used in tck2connectome
        opts: dict of extras
    outputs:
        connectivity matrices: dict. with keys 'controls' and 'patients'
    """
    cohorts = ['controls', 'patients']
    conns = dict( ((at,met,coh),[]) for at,met,coh in itertools.product(atlases,metrics,cohorts))
    for subj,atlas,metric in itertools.product(subjs,atlases,metrics):
        csv_name = '_'.join([subj,atlas,metric,'connectome.csv'])
        fpath = os.path.join(proj_dir,'postprocessing',subj,csv_name)
        if os.path.exists(fpath):
            ns = pd.read_csv(fpath, sep=' ', header=None, index_col=False)
        else:
            print('Subject {} preprocessing not found.'.format([subj]))
            continue;
        if 'control' in subj:
            conns[(atlas,metric,'controls')].append(np.array(ns))
        elif 'patient' in subj:
            conns[(atlas,metric,'patients')].append(np.array(ns))

    for k,v in conns.items():
        conns[k] = np.transpose(np.array(v),(1,2,0))
        if opts['verbose']:
            print(' --> shape: '.join([str(k), str(conns[k].shape)]))
    return conns



### Part II: Statistical Analysis ###
#-----------------------------------#
def create_nifti_imgs(atlas_img, stats, pvals, node_ids, s_thresh=1.8, p_thresh=1.):
    ''' Return Nifti images of stats and pvals '''
    atlas_data = atlas_img.get_fdata()
    stats_data = np.zeros((atlas_data.shape))
    pvals_data = np.ones((atlas_data.shape))
    sig_stats_data = np.zeros((atlas_data.shape))

    # populate stats data
    sig_ids = np.where(np.abs(stats)>=s_thresh)[0]
    sig_node_ids = node_ids[sig_ids]
    for i,idx in enumerate(sig_node_ids):
        atlas_ids = np.where(atlas_data==idx)
        stats_data[atlas_ids] = stats[sig_ids[i]]

    # populate pvals data
    sig_ids = np.where(pvals<=p_thresh)[0]
    sig_node_ids = node_ids[sig_ids]
    for i,idx in enumerate(sig_node_ids):
        atlas_ids = np.where(atlas_data==idx)
        pvals_data[atlas_ids] = pvals[sig_ids[i]]

    # populate union data (only stats for which p<0.05)
    sig_ids = np.where( (1-pvals_data) >= 0.95 )
    if len(sig_ids)>0:
        sig_stats_data[sig_ids] = stats_data[sig_ids]

    # create and return new nifti img
    stats_img = nib.Nifti1Image(stats_data, atlas_img.affine, atlas_img.header)
    pvals_img = nib.Nifti1Image(1-pvals_data, atlas_img.affine, atlas_img.header)
    sig_stats_img = nib.Nifti1Image(sig_stats_data, atlas_img.affine, atlas_img.header)
    return stats_img, pvals_img, sig_stats_img


def get_fspt_node_ids(atlas):
    """ returns Fronto-Striato-Pallido-Thalamic node ids of input atlas """
    # read FrStrPalTh atlas config
    with open(os.path.join(proj_dir, 'utils', 'atlas_cfg_FrStrPalThal_'+atlas+'.json')) as jf:
        fspt_atlas_cfg = json.load(jf)
    fspt_node_ids = np.array(fspt_atlas_cfg[atlas]['node_ids']).astype(int)
    return fspt_node_ids

def get_fspt_Fr_node_ids(atlas, subctx=['Thal', 'Pal', 'Acc', 'Put', 'Caud']):
    """ Extract only cortical frontal regions indices from FrStrPalThal atlas """
    # filter-in frontal regions
    fspt_node_ids = get_fspt_node_ids(atlas)
    fspt_names = np.array(atlas_cfg[atlas]['node_names'])[fspt_node_ids-1]
    Fr_fspt_ids = []
    for i,name in enumerate(fspt_names):
        cnt=0
        for sctx in subctx:
            if sctx in name:
                cnt+=1
        if (cnt==0):
            Fr_fspt_ids.append(i)
    Fr_fspt_ids = np.array(Fr_fspt_ids).astype(int)
    Fr_node_ids = fspt_node_ids[Fr_fspt_ids]
    return Fr_node_ids, Fr_fspt_ids

def plot_pq_values(outp, atlas, metric, roi, p_alpha=0.05, q_alpha=0.2):
    """ Plot visual reprensations of p-values and FDR corrected p-values (q-values)"""
    d = outp[atlas,metric,roi]
    stats,pvals,qvals,names = d['stats'],d['pvals'],d['qvals'],d['nn_Fr_node_names']
    p_corrected = d['p_corrected']

    sorted_inds = np.argsort(pvals)

    plt.figure(figsize=[16,4])
    title='Statistics {} {} -- ROI: {}'.format(atlas,metric,roi)
    plt.suptitle(title, fontsize=16)

    ax1 = plt.subplot(1,3,1)
    plt.plot(np.arange(len(pvals))+1, pvals[sorted_inds], '.-', label='p')
    for p_meth,p in p_corrected.items():
        plt.plot(np.arange(len(p))+1, p[sorted_inds], '.-', label=p_meth)
    #plt.plot(np.arange(len(pvals)), np.linspace(0,1,len(pvals))*p_alpha)
    plt.legend()
    plt.axhline(y=p_alpha, linestyle='--', color='red')
    plt.ylim([0,1])
    plt.xlabel(r'$i$', fontsize=14)
    plt.ylabel(r'$p_{(i)}$', fontsize=14)

    ax2 = plt.subplot(1,3,2)
    plt.plot(pvals[sorted_inds], qvals[sorted_inds], '.-')
    #plt.plot(np.linspace(0,1,len(pvals)), np.linspace(0,1,len(pvals))*q_alpha)
    plt.axhline(y=q_alpha, color='red')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel(r'$p_{(i)}$', fontsize=14)
    plt.ylabel(r'$q_{(i)}$', fontsize=14)

    if np.any(qvals <= q_alpha):
        sorted_inds = np.argsort(qvals)
        sorted_qvals = qvals[sorted_inds]
        rev_sort_inds = np.argsort(sorted_inds)
        sig_sort_inds, = np.where(sorted_qvals <= q_alpha)
        sig_inds = rev_sort_inds[sig_sort_inds]
        ax3 = plt.subplot(1,3,3)
        plt.hist(d['row_con'][sorted_inds[0]], bins=10, alpha=0.5)
        plt.hist(d['row_pat'][sorted_inds[0]], bins=10, alpha=0.5)
        plt.legend(['controls', 'patients'])
        #print(names[sorted_inds])
        plt.title(names[sorted_inds[0]])

    plt.show(block=False)


def print_sig_names(roi, stats, pvals, qvals, fwe_pvals, names, alpha=0.05, verbose=True):
    """ Print to stdout the names, t-values, p-values and q-values (FDR corrected p-value) of statistically significant ROIs """
    names = np.asarray(names)
    # print ROIs with significant p-values
    ids = np.where(pvals<=alpha)[0]
    for i,idx in enumerate(ids):
        print('{:<5} <--> {:<25} \t \t pval = {:.3f} \t qvals = {:.3f} \t FWE pval = {:.3f} \t T(c-p) = {:.1f}'.format(
            roi, names[idx], pvals[idx], qvals[idx], fwe_pvals[idx], stats[idx]))
    #return names[ids], pvals[idx], stats[idx]

def threshold_connectivity(c, quantile=0.5, level='subject'):
    """ Threshold matrix c at quantile value at subject or population level.
        The outputs connectivity matrices are ajusted to threshold value (new zero = threshold)"""
    cc = c.copy().astype(float)
    if level=='subject':
        c_tildes = []
        for i in range(c.shape[-1]):
            c_ = np.abs(c[:,:,i].copy().astype(float))
            c_tilde = c_[c_!=0]
            c_tilde = np.quantile(c_tilde,q=quantile)
            c_ -= c_tilde
            cc[:,:,i] -= c_tilde
            neg_inds = np.where(c_<0)
            cc[neg_inds[0], neg_inds[1],i]=0
            c_tildes = np.append(c_tildes, c_tilde)
        c_tilde = c_tildes
    else:
        # threshold at population level
        c_tilde = cc[cc!=0]
        c_tilde = np.quantile(c_tilde,q=quantile)
        cc -= c_tilde
        cc[cc<0]=0
    return cc, c_tilde

# Plot histograms of weight distributions
def plot_conns_hists(conns, atlases, metrics, scale='log'):
    """ Plot whole brain vs fronto-striatal histograms of weight matrices of controls and patient """
    for atlas,metric in itertools.product(atlases,metrics):
        fspt_node_ids = get_fspt_node_ids(atlas)
        print('{}   {}'.format(atlas, metric))

        # CONTROLS
        #----------
        # whole brain
        cns = conns[(atlas,metric,'controls')].flatten()
        dns = cns.copy().astype(float)
        dns[dns==0]=np.nan
        cns_ = cns[cns!=0]
        cns_tilde = np.median(cns_)
        mad_cns = np.median(np.abs(cns_-cns_tilde))
        print('CONTROLS: mean={}; std={}; median={}; mad={}'.format(np.mean(cns_), np.std(cns_), cns_tilde, mad_cns))

        # FrStr
        ens = conns[(atlas,metric,'controls')][np.ix_(fspt_node_ids-1,fspt_node_ids-1)].flatten().astype(float)
        ens_tilde = np.median(ens[ens!=0])
        ens[ens==0]=np.nan

        # PATIENTS
        #---------
        # whole brain
        pns = conns[(atlas,metric,'patients')].flatten()
        qns = pns.copy().astype(float)
        qns[qns==0]=np.nan
        pns_ = pns[pns!=0]
        pns_tilde = np.median(pns_)
        mad_pns = np.median(np.abs(pns_-pns_tilde))
        print('PATIENTS: mean={}; std={}; median={}; mad={}'.format(np.mean(pns_), np.std(pns_), pns_tilde, mad_pns))

        # Fronto-BG-Thal
        rns = conns[(atlas,metric,'patients')][np.ix_(fspt_node_ids-1,fspt_node_ids-1)].flatten().astype(float)
        rns_tilde = np.median(rns[rns!=0])
        rns[rns==0]=np.nan

        ## PLOTTING ##

        plt.figure(figsize=(12,6))
        plt.suptitle('{}   {}'.format(atlas,metric))
        plt.grid(True)

        # controls
        ax1 = plt.subplot(1,2,1)
        if scale=='log':
            plt.hist(np.log10(dns), bins=np.arange(-1,6,0.2))
            plt.hist(np.log10(ens), bins=np.arange(-1,6,0.2))
            plt.legend(['Whole brain', 'Fronto-BG-Thal'])
            plt.axvline(x=np.log10(cns_tilde), color='black')
            plt.axvline(x=np.log10(ens_tilde), color='black')
            plt.xlabel(r'weights $(10^x)$', fontsize=12)
        else:
            plt.hist(dns, bins=20)
            plt.hist(ens, bins=20)
            plt.legend(['Whole brain', 'Fronto-BG-Thal'])
            plt.axvline(x=cns_tilde, color='black')
            plt.axvline(x=ens_tilde, color='black')
            plt.xlabel(r'weights', fontsize=12)
        plt.title('controls '+metric, fontsize=14)
        plt.ylabel(metric, fontsize=12)

        # patients
        ax2 = plt.subplot(1,2,2)
        if scale=='log':
            plt.hist(np.log10(qns), bins=np.arange(-1,6,0.2))
            plt.hist(np.log10(rns), bins=np.arange(-1,6,0.2))
            plt.axvline(x=np.log10(pns_tilde), color='black')
            plt.axvline(x=np.log10(rns_tilde), color='black')
            plt.xlabel(r'weights $(10^x)$', fontsize=12)
        else:
            plt.hist(qns, bins=20)
            plt.hist(rns, bins=20)
            plt.axvline(x=pns_tilde, color='black')
            plt.axvline(x=rns_tilde, color='black')
            plt.xlabel(r'weights', fontsize=12)
        plt.title('patients '+metric, fontsize=14)
        plt.ylabel(metric, fontsize=12)

        plt.show(block=False)

def plot_conns_matrices(conns, atlases, metrics, scale='log', vmin=None, vmax=None):
    """ Plot connectivity matrices of controls and patients """
    for atlas,metric in itertools.product(atlases,metrics):
        node_names = atlas_cfg[atlas]['node_names']
        if scale=='log':
            c = np.log10(conns[(atlas,metric,'controls')].mean(axis=-1))
            p = np.log10(conns[(atlas,metric,'patients')].mean(axis=-1))
        else:
            c = conns[(atlas,metric,'controls')].mean(axis=-1)
            p = conns[(atlas,metric,'patients')].mean(axis=-1)
        # plotting
        fig = plt.figure(figsize=(32,12))
        plt.suptitle('{}  {}'.format(atlas,metric))
        ax1 = plt.subplot(1,2,1)
        im1 = plot_matrix(c, labels=node_names, axes=ax1, reorder=False, colorbar=False, vmin=vmin, vmax=vmax)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('controls')
        ax2 = plt.subplot(1,2,2)
        im2 = plot_matrix(p, labels=node_names, axes=ax2, reorder=False, colorbar=True, vmin=vmin, vmax=vmax)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('patients')
        plt.show(block=False)


def run_stat_analysis(conns, atlases, metrics, subrois, suprois=['Thal', 'Pal', 'Put', 'Caud', 'Acc'], threshold=True, quantile=0.8, n_min_edges=0, verbose=True):
    """ Perform 'row-wise' t-test on ROIs, with FDR correction """
    outp = dict( ( ( (atlas,metric,roi), None ) for atlas,metric,roi in itertools.product(atlases,metrics,subrois) ) )
    mcm = ['bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg', 'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'] # multiple comparison method
    for atlas,metric in itertools.product(atlases, metrics):
        if verbose:
            print(atlas, metric)
        fspt_node_ids = get_fspt_node_ids(atlas)
        Fr_node_ids, Fr_fspt_ids = get_fspt_Fr_node_ids(atlas, subctx=suprois)
        fspt_node_names = np.array(atlas_cfg[atlas]['node_names'])[fspt_node_ids-1]
        Fr_node_names = np.array(atlas_cfg[atlas]['node_names'])[Fr_node_ids-1]

        # perform t-test
        if threshold:
            con, _ = threshold_connectivity(conns[(atlas,metric,'controls')][np.ix_(fspt_node_ids-1, fspt_node_ids-1)], \
                    quantile=quantile)
            pat, _ = threshold_connectivity(conns[(atlas,metric,'patients')][np.ix_(fspt_node_ids-1, fspt_node_ids-1)], \
                    quantile=quantile)
        else:
            con = conns[(atlas,metric,'controls')][np.ix_(fspt_node_ids-1, fspt_node_ids-1)]
            pat = conns[(atlas,metric,'patients')][np.ix_(fspt_node_ids-1, fspt_node_ids-1)]

        def perform_stat_on_roi(row_con, row_pat, nni):
            stats, pvals = scipy.stats.ttest_ind(row_con[nni,:], row_pat[nni,:], axis=1, permutations=1000)
            rej, qvals, _, _ = multitest.multipletests(pvals, method='fdr_bh')
            p_corrected = dict( ( (m,None) for m in mcm ) )
            fwe_stats, fwe_pvals = ttest_ind_FWE.ttest_ind_FWE(row_con[nni,:].T, row_pat[nni,:].T, n_perm=1000, verbose=False)
            p_corrected['fwe'] = fwe_pvals
            for m in mcm:
                r = multitest.multipletests(pvals, method=m)
                p_corrected[m] = r[1]
            if verbose:
                print_sig_names(roi, stats, pvals, qvals, fwe_pvals, names=Fr_node_names[nni])
            return stats, pvals, qvals, p_corrected

        def non_null_inds(row_con, row_pat):
            """ eliminate nonzero entries when on both con and pat (to avoid being counted in the FDR correction) """
            nn_con = (row_con!=0)
            nn_pat = (row_pat!=0)
            nni, = np.where( (nn_con.sum(axis=1)>n_min_edges) & (nn_pat.sum(axis=1)>n_min_edges) )
            return nni

        # plot stats with FDR correction
        all_roi_ids = np.array([])
        for roi in subrois:
            roi_ids = np.where([roi in name for name in fspt_node_names])[0]
            all_roi_ids = np.concatenate([all_roi_ids,  roi_ids]).astype(int)
            row_con = np.sum(con[np.ix_(roi_ids,Fr_fspt_ids)], axis=0)
            row_pat = np.sum(pat[np.ix_(roi_ids,Fr_fspt_ids)], axis=0)
            nni = non_null_inds(row_con, row_pat)
            if len(nni)!=0:
                stats, pvals, qvals, p_corrected = perform_stat_on_roi(row_con, row_pat, nni)
                outp[atlas,metric,roi] = {  'stats':stats, 'pvals':pvals, 'qvals':qvals, 'p_corrected':p_corrected, \
                                        'row_con':row_con[nni,:], 'row_pat':row_pat[nni,:], \
                                        'nn_Fr_node_names':Fr_node_names[nni], 'nn_Fr_node_ids':Fr_node_ids[nni] }
        # plot stats for all ROIs aggregated (i.e. all Striatum)
        row_con = np.sum(con[np.ix_(all_roi_ids,Fr_fspt_ids)], axis=0)
        row_pat = np.sum(pat[np.ix_(all_roi_ids,Fr_fspt_ids)], axis=0)
        nni = non_null_inds(row_con, row_pat)
        if len(nni)!=0:
            stats, pvals, qvals, p_corrected = perform_stat_on_roi(row_con, row_pat, nni)
            outp[atlas,metric,'all'] = dict( stats=stats, pvals=pvals, qvals=qvals, p_corrected=p_corrected, \
                                             row_con=row_con[nni,:], row_pat=row_pat[nni,:], \
                                             nn_Fr_node_ids=Fr_node_ids[nni], nn_Fr_node_names=Fr_node_names[nni] )
    return outp

def plot_stats_on_glass_brain(atlas, metric, roi, outp, p_corrected=None):
    """ Display color-coded T stats and p-values on glass brain """
    # get atlas and ids
    atlas_img = load_img(os.path.join(atlas_dir, atlas+atlas_suffix[atlas]))
    # create nifti imgs
    if (p_corrected==None):
        stats_img, pvals_img, sig_stats_img = create_nifti_imgs(atlas_img, stats=outp[atlas,metric,roi]['stats'], pvals=outp[atlas,metric,roi]['pvals'], node_ids=outp[atlas,metric,roi]['nn_Fr_node_ids'], p_thresh=1.)
    else:
        stats_img, pvals_img, sig_stats_img = create_nifti_imgs(atlas_img, stats=outp[atlas,metric,roi]['stats'], pvals=outp[atlas,metric,roi]['p_corrected'][p_corrected], node_ids=outp[atlas,metric,roi]['nn_Fr_node_ids'], p_thresh=1.)
    # plotting
    fig = plt.figure(figsize=[30,4])
    plt.suptitle('{} {}'.format(atlas,metric))
    ax_s = fig.add_subplot(1,3,1)
    ax_p = fig.add_subplot(1,3,2)
    ax_sp = fig.add_subplot(1,3,3)
    display1 = plot_glass_brain(stats_img, display_mode='lzry', colorbar=True, title=roi, plot_abs=False, vmin=-5, vmax=5, axes=ax_s)
    display2 = plot_glass_brain(None, display_mode='lzry', colorbar=True, axes=ax_p)
    display2.add_contours(pvals_img, filled='green', levels=[0.95])
    display3 = plot_glass_brain(sig_stats_img, display_mode='lzry', colorbar=True, title=roi, plot_abs=False, vmin=-5, vmax=5, axes=ax_sp)
    plt.show(block=False)


### MAIN ###
# TODO: add saving options for figures and results
# TODO: add flags to perform only part of the analysis
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--compute_conns', default=False, action='store_true', help='compute connectivity matrices (from .mat for sift and .csv from no_sift)')
    parser.add_argument('--compute_conns_from_csv', default=False, action='store_true', help='compute connectivity matrices (from .csv for both sift and no_sift)')
    parser.add_argument('--plot_conns', default=False, action='store_true', help='plot connectivity matrices')
    parser.add_argument('--plot_weight_distrib', default=False, action='store_true', help='plot connectivity weight distributions')
    parser.add_argument('--plot_pq_values', default=False, action='store_true', help='plot graphs of p-values and FDR corrected p-values, with most significant weight distrib')
    parser.add_argument('--plot_stats_on_glass_brain', default=False, action='store_true', help='plot t-values and signicant p-values (uncorrected) on glass brain')
    parser.add_argument('--save_suffix', default='_SC', type=str, action='store', help='text to append ot end of connectivity files for saving')
    args = parser.parse_args()

    #atlases = ['schaefer400_tianS4', 'schaefer200_tianS2', 'schaefer100_tianS1']
    atlases = ['ocdOFClPFC_ocdAccdPut'] #['schaefer400_harrison2009']
    metrics = ['count_sift'] #['count_sift', 'count_nosift']
    subrois = ['Acc', 'Put'] # 'Caud'
    #subrois = ['Left_NucleusAcc', 'Left_Caud', 'Left_Put', 'Right_NucleusAcc', 'Right_Caud', 'Right_Put']

    ### PART I: Aggregate connectomes from controls and patients ###
    #--------------------------------------------------------------#
    if args.compute_conns:
        # import connectivity matrices from all controls and patients
        conns, subjs, discarded = get_connectomes_from_mat(subjs, atlases, metrics)

        # udpdate nosift connectomes (as qsiprep only did w/ sift)
        conn_ns = get_connectomes_from_csv(subjs, atlases, metrics=['count_nosift'])
        cohorts = ['controls', 'patients']
        for atlas,cohort in itertools.product(atlases,cohorts):
            conns[(atlas,'count_nosift',cohort)] = conn_ns[(atlas,'count_nosift',cohort)]

        if args.save_outputs:
            # save connectivity matrices
            fname = os.path.join(proj_dir,'postprocessing','conns'+args.save_suffix+'.pkl')
            with open(fname, 'wb') as pf:
                pickle.dump(conns,pf)

    elif args.compute_conns_from_csv:
        conns = get_connectomes_from_csv(subjs, atlases, metrics)

        if args.save_outputs:
            # save connectivity matrices
            fname = os.path.join(proj_dir,'postprocessing','conns'+args.save_suffix+'.pkl')
            with open(fname, 'wb') as pf:
                pickle.dump(conns,pf)

    else:
        fname = os.path.join(proj_dir,'postprocessing','conns'+args.save_suffix+'.pkl')
        with open(fname, 'rb') as pf:
            conns = pickle.load(pf)

    # plot average connectivity matrices
    if args.plot_conns:
        plot_conns_matrices(conns, atlases, metrics)

    # Plot histograms of weight distributions
    if args.plot_weight_distrib:
        plot_conns_hists(conns, atlases, metrics)

    ### PART II: Statistical Analysis ###
    #-----------------------------------#
    outp = run_stat_analysis(conns, atlases, metrics, subrois, suprois=['Thal', 'Pal', 'Put', 'Caud', 'Acc'])

    if args.save_outputs:
        # save stats
        fname = os.path.join(proj_dir,'postprocessing','outp'+args.save_suffix+'.pkl')
        with open(fname, 'wb') as pf:
            pickle.dump(outp,pf)

    rois = np.concatenate([subrois, ['all']])
    # plot results on glass brain
    for atlas,metric,roi in itertools.product(atlases, metrics, rois):
        if outp[atlas,metric,roi] != None:
            if np.any(outp[atlas,metric,roi]['pvals']<=0.05):
                if args.plot_pq_values:
                    plot_pq_values(outp, atlas, metric, roi)
                if args.plot_stats_on_glass_brain:
                    plot_stats_on_glass_brain(atlas, metric, roi, outp, p_corrected='fwe')
