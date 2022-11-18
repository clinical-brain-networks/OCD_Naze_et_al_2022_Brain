# Script to perform FA analysis on masks derived from track density
# Author: Sebastien Naze
# QIMR Berghofer 2021-2022

import argparse
import bct
import datetime
import h5py
import itertools
import joblib
from joblib import Parallel, delayed
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import nibabel as nib
import nilearn
from nilearn.image import load_img, binarize_img, threshold_img, mean_img, new_img_like
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison, plot_roi, view_img
import numpy as np
import os
import pickle
import pandas as pd
import platform
import scipy
from scipy.io import loadmat
import seaborn as sbn
import sklearn
import statsmodels
from statsmodels.stats import multitest
import time
from time import time
import warnings
warnings.filterwarnings('once')

# get computer name to set paths
if platform.node()=='qimr18844':
    working_dir = '/home/sebastin/working/'
elif 'hpcnode' in platform.node():
    working_dir = '/mnt/lustre/working/'
else:
    print('Computer unknown! Setting working dir as /working')
    working_dir = '/working/'

# general paths
proj_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCDbaseline')
lukeH_proj_dir = os.path.join(working_dir, 'lab_lucac/lukeH/projects/OCDbaseline')
code_dir = os.path.join(proj_dir, 'docs/code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
lukeH_deriv_dir = os.path.join(lukeH_proj_dir, 'data/derivatives')
atlas_dir = os.path.join(proj_dir, 'utils')

atlas_cfg_path = os.path.join(atlas_dir, 'atlas_config.json')
with open(atlas_cfg_path) as jsf:
    atlas_cfg = json.load(jsf)
subjs = pd.read_table(os.path.join(code_dir, 'subject_list.txt'), names=['name'])['name']

# options
sc_metrics = ['sift'] # unecessary but kept for compatibility
rois = ["AccR_dPutR_vPutL_lvPFC_lPFC_dPFC_sphere6-12mm"] #['ACC_OFC_ROI', 'PUT_PFC_ROI']
dsi_metrics = ['gfa'] #, 'iso', 'fa0', 'fa1', 'fa2']
subrois = ['AccOFC', 'PutPFC', 'vPutdPFC']
cohorts = ['controls', 'patients']

# coords for plotting global masks
xyz = {'ACC_OFC_ROI':(20, 42, -4), 'PUT_PFC_ROI':None, "AccR_dPutR_vPutL_lvPFC_lPFC_dPFC_sphere6-12mm":None}


def get_tdi_img(subj, scm, roi, subroi):
    """ import track density images """
    fpath = os.path.join(proj_dir, 'postprocessing', subj)
    fname = '_'.join([roi, scm, subroi]) + '.nii.gz'
    fname = os.path.join(fpath, fname)
    if os.path.exists(fname):
        img = load_img(fname)
    else:
        img = None
    return img


# import FA & mean diffusivity (MD) images
def get_diffusion_metric(subj, metric='gfa'):
    """ import diffusion metrics from DSI Studio such as generalized fractional anisotropy (GFA) """
    fpath = os.path.join(proj_dir, 'data/derivatives/qsirecon', subj, 'dwi')
    fname = subj+'_acq-88_dir-AP_space-T1w_desc-preproc_space-T1w_desc-'+metric+'_gqiscalar.nii.gz'
    fname = os.path.join(fpath, fname)
    if os.path.exists(fname):
        img = load_img(fname)
    else:
        img = None
    return img

def compute_tdi_maps(thresholds, args):
    """ computes track density maps between pathways ROIs """
    dsi_m = dict( ( ((dsm,scm,roi,subroi,thr,coh),{'mean_dsi':[], 'masks':[]}) for dsm,scm,roi,subroi,thr,coh in itertools.product(
        dsi_metrics, sc_metrics, rois, subrois, thresholds, cohorts)) )
    for dsm,scm,roi,subroi,thr in itertools.product(dsi_metrics, sc_metrics, rois, subrois, thresholds):
        for i,subj in enumerate(subjs):
            # import track density images (TDI)
            img = get_tdi_img(subj, scm, roi, subroi)
            if img==None :
                #subjs.pop(i)
                #print("{} TDI img is not found, will apply group level TD mask instead".format(subj))
                mask = None
            elif np.sum(img.get_fdata())==0:
                 #print("{} TDI img is empty, will apply group level TD mask instead".format(subj))
                 mask = None
            else:
                # threshold TDI to get binary mask
                img_data = img.get_fdata().copy()
                q_thr = np.percentile(img_data[img_data!=0], thr)
                mask = binarize_img(img, threshold=q_thr)

                if args.plot_indiv_masks:
                    plot_roi(mask, title='_'.join([subj, roi, subroi, str(thr)+'%']),
                             output_file=os.path.join(proj_dir, 'postprocessing', subj, '_'.join([subj, roi, subroi, str(thr)])+'.png'))

                if 'control' in subj:
                    dsi_m[dsm,scm,roi,subroi,thr, 'controls']['masks'].append(mask)
                else:
                    dsi_m[dsm,scm,roi,subroi,thr, 'patients']['masks'].append(mask)
            dsi_m[dsm,scm,roi,subroi,thr,subj, 'mask'] = mask

    return dsi_m


def compute_diffusion_maps(dsi_m, args):
    """ computes the diffusion metric accross the track density mask """
    for dsm,scm,roi,subroi, thr in itertools.product(dsi_metrics, sc_metrics, rois, subrois, args.thresholds):
        for i,subj in enumerate(subjs):
            # import TD map
            mask = dsi_m[dsm,scm,roi,subroi,thr,subj, 'mask']
            # import diffusion maps (i.e. GFA)
            img = get_diffusion_metric(subj, metric=dsm)
            if img==None:
                subjs.pop(i)
                print("{} removed, DSI metric img not found".format(subj))
                continue

            # use TDI mask to extract mean FA & MD values for each subject
            data = threshold_img(img, threshold=0, mask_img=mask).get_fdata()
            mean_dsi = np.mean(data[data!=0])
            if 'control' in subj:
                dsi_m[dsm,scm,roi,subroi,thr, 'controls']['mean_dsi'].append(mean_dsi)
            else:
                dsi_m[dsm,scm,roi,subroi,thr, 'patients']['mean_dsi'].append(mean_dsi)
            dsi_m[dsm,scm,roi,subroi,thr,subj, 'mean_dsi'] = mean_dsi
    return dsi_m

def create_df_dsi(dsi_m, thr, scm='sift', dsm='gfa'):
    """ create dataframe from dict with DSI metrics """
    df_dsi = pd.DataFrame(columns=['mean_dsi', 'cohorts', 'roi', 'subroi'])
    for roi, subroi, coh in itertools.product(rois, subrois, cohorts):
        n = len(dsi_m[dsm,scm,roi,subroi,thr,coh]['mean_dsi'])
        df_ = pd.DataFrame.from_dict({'mean_dsi':dsi_m[dsm,scm,roi,subroi,thr,coh]['mean_dsi'],
                                      'cohorts':np.repeat(coh,n),
                                      'roi':np.repeat(roi,n),
                                      'subroi':np.repeat(subroi,n)})
        df_dsi = df_dsi.append(df_, ignore_index=True)
    return df_dsi

def create_summary(dsi_m, args):
    """ Summarize TDI outputs in dataframe """
    summary_df = pd.DataFrame()
    for dsm,scm,roi,subroi,thr in itertools.product(dsi_metrics, sc_metrics, rois, subrois, args.thresholds):
        df = pd.DataFrame()
        df['subj'] = subjs.copy()
        df['mean_dsi'] = np.concatenate([dsi_m[dsm,scm,roi,subroi,thr, 'controls']['mean_dsi'], dsi_m[dsm,scm,roi,subroi,thr, 'patients']['mean_dsi']])
        df['cohort'] = np.concatenate([np.repeat('controls', len(dsi_m[dsm,scm,roi,subroi,thr, 'controls']['mean_dsi'])),
                                       np.repeat('patients', len(dsi_m[dsm,scm,roi,subroi,thr, 'patients']['mean_dsi']))])
        df['thr'] = np.repeat(thr, len(subjs))
        df['roi'] = np.repeat(roi, len(subjs))
        df['subroi'] = np.repeat(subroi, len(subjs))
        df['dsi_metric'] = np.repeat(dsm, len(subjs))

        summary_df = summary_df.append(df, ignore_index=True)

    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'tdi_fa_summary_df.pkl'), 'wb') as f:
            pickle.dump(summary_df, f)


def threshold_normalize_img(img, thr=0, min_val=0, max_val=1):
    """  Threshold image and normalize the remaining so that it takes values between min_val and max_val """
    data = img.get_fdata().copy()
    thr_data = data[data>thr]
    rescaled = (thr_data - thr_data.min()) / (thr_data.max() - thr_data.min())
    rescaled_data = np.zeros(data.shape)
    rescaled_data[data>thr] = rescaled
    return new_img_like(img, rescaled_data)


def cohen_d(x,y):
    """ Calculates effect size as cohen's d """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def compute_stats(dsi_m, thresholds, args):
    """ performs student's t-test between groups' mean TDI in pathways """
    stats = dict()
    for dsm,scm,roi,subroi,thr in itertools.product(dsi_metrics, sc_metrics, rois, subrois, thresholds):
        cons = np.array(dsi_m[dsm,scm,roi,subroi,thr, 'controls']['mean_dsi'])
        pats = np.array(dsi_m[dsm,scm,roi,subroi,thr, 'patients']['mean_dsi'])
        t,p = scipy.stats.ttest_ind(cons, pats)
        ks,pks = scipy.stats.ks_2samp(cons, pats)
        print("{} {} {} {} -- thr={} -- t={:.3f} p={:.3f} -- ks={:.2f} p={:.3f}".format(dsm,scm,roi,subroi,int(thr),t,p,ks,pks))
        stats[dsm,scm,roi,subroi,thr] = {'t':t, 'p':p}
    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'tdi_fa_stats_dict.pkl'), 'wb') as f:
            pickle.dump(stats, f)



## PLOTTING FUNCTIONS ##
def get_group_masks(dsi_m, args):
    for dsm,scm,roi,subroi,thr in itertools.product(dsi_metrics, sc_metrics, rois, subrois, args.thresholds):
        unscaled_img = mean_img(np.concatenate([dsi_m[dsm,scm,roi,subroi,thr,'controls']['masks'], \
                                                dsi_m[dsm,scm,roi,subroi,thr,'patients']['masks']]))
        rescaled_img = threshold_normalize_img(unscaled_img)
        mask = binarize_img(rescaled_img, threshold=args.group_mask_threshold)

        for i,subj in enumerate(subjs):
            if dsi_m[dsm,scm,roi,subroi,thr,subj,'mask'] == None :
                dsi_m[dsm,scm,roi,subroi,thr,subj,'mask'] = mask

        if args.plot_group_masks:
            if args.save_figs:
                plot_roi(rescaled_img, cmap='Reds', threshold=0.00001, draw_cross=False,
                          cut_coords=xyz[roi], alpha=0.8, title='{} {} avg mask at {}%'.format(roi,subroi, str(thr)),
                          output_file=os.path.join(proj_dir, 'postprocessing', 'avg_mask_{}_{}_{}percent.svg'.format(roi, subroi, str(thr))) )
            if args.plot_figs:
                plot_roi(rescaled_img, cmap='Reds', threshold=0.00001, draw_cross=False,
                          cut_coords=xyz[roi], alpha=0.8)

    return dsi_m


def plot_tdi_distrib(df_dsi, thr, args, ylims=[0.05, 0.15]):
    plt.rcParams.update({'font.size':12, 'font.family':['Arial'], 'pdf.fonttype': 42})

    fig = plt.figure(figsize=[8,4])

    ax1 = plt.subplot(1,3,1)
    sbn.violinplot(data=df_dsi, y='mean_dsi', x='subroi', hue='cohorts', orient='v', ax=ax1, split=True, scale_hue=True,
                   inner='quartile', dodge=True, width=0.8, cut=2)
    #sbn.stripplot(data=df_dsi, y='mean_dsi', x='roi', hue='cohorts', orient='v', ax=ax1, split=True, dodge=False,
    #              size=2, edgecolor='black', linewidth=0.5, jitter=0.25)
    plt.ylim(ylims);
    ax1.legend([])

    ax2 = plt.subplot(1,3,2)
    sbn.stripplot(data=df_dsi, y='mean_dsi', x='subroi', hue='cohorts', orient='v', ax=ax2, dodge=True, linewidth=1, size=2)
    plt.ylim(ylims);

    ax3 = plt.subplot(1,3,3)
    bplot = sbn.boxplot(data=df_dsi, y='mean_dsi', x='subroi', hue='cohorts', orient='v', ax=ax3, fliersize=0)
    #bplot.set_facecolor('blue')
    #bplot['boxes'][1].set_facecolor('orange')
    #splot = sbn.swarmplot(data=df_dsi, y='mean_dsi', x='subroi', hue='cohorts', orient='v', ax=ax3, dodge=True, linewidth=1, size=6, alpha=0.6)
    plt.ylim(ylims);
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    lgd = ax3.legend(handles=bplot.patches, labels=['HC', 'OCD'], bbox_to_anchor=(1, 1))
    #ax3.set_xticklabels(labels=['NAcc', 'dPut'], minor=True)
    fig.tight_layout()

    if args.save_figs:
        plt.savefig(os.path.join(proj_dir, 'img', 'TD_FA_con_vs_pat_3pathways_'+str(thr)+'_14-06-2022.svg'))

    if args.plot_figs:
        plt.show(block=False)
    plt.close()

    # Pathway specific
    fig = plt.figure(figsize=[7,3])
    for i,pathway in enumerate(['AccOFC', 'PutPFC', 'vPutdPFC']):
        ax = plt.subplot(1,3,i+1)
        splot = sbn.swarmplot(data=df_dsi[df_dsi['subroi']==pathway], y='mean_dsi', x='subroi', hue='cohorts', orient='v', ax=ax, dodge=True, linewidth=0.5, size=3, alpha=0.8)
        #plt.ylim(ylims);
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_legend().set_visible(False)
        #lgd = ax.legend(handles=splot.patches, labels=['HC', 'OCD'], bbox_to_anchor=(1, 1))

    fig.tight_layout()

    if args.save_figs:
        plt.savefig(os.path.join(proj_dir, 'img', 'TD_FA_con_vs_pat_3_ind_pathways_'+str(thr)+'_14-06-2022.pdf'))

    if args.plot_figs:
        plt.show(block=False)

    plt.close()


def create_atlas_from_VOIs():
    """ create and export new atlas based on VOIs from seed-to-voxel analysis """
    voi_loc = {'AccR':[9,9,-8], 'dPutR':[28,1,3], 'vPutL':[-20,12,-3],
           'lvPFC':[23.5, 57.5, -6.5], 'lPFC':[53.5, 13.5, 19.5], 'dPFC':[-24.5, 55.5, 35.5]}

    voi_radius = 6 #mm
    dPFC_radius = 12
    mask = None
    i = 1
    for voi,loc in voi_loc.items():
        if voi=='dPFC':
            img = nltools.create_sphere(loc, radius=dPFC_radius)
        else:
            img = nltools.create_sphere(loc, radius=voi_radius)
        if mask == None:
            mask = img
        else:
            img = new_img_like(img, img.get_fdata()*i)
            mask = nilearn.image.math_img("img1 + img2", img1=mask, img2=img)
        i += 1

    # saving
    fname = os.path.join(proj_dir, 'utils', '_'.join(list(voi_loc.keys()))+'_sphere{}-{}mm.nii.gz'.format(str(int(voi_radius)), str(int(dPFC_radius))))
    nib.save(mask, fname)

    # exporting to .mif for use in MrTrix3
    cmd = "/home/davidS/mrtrix3/bin/mrconvert {} {}".format(fname, fname[:-7]+'.mif')
    os.system(cmd)



def create_df_streamline(seeds=['Acc', 'dPut', 'vPut'], vois = ['OFC', 'lPFC', 'dPFC'], args=None):
    """ get number of streamlines between ROIs from csv file for each subject and put in dataframe"""
    regions = np.hstack([seeds,vois])
    df_lines = []
    for subj in subjs:
        if 'control' in subj:
            coh = 'controls'
        else:
            coh = 'patients'

        fname = os.path.join(proj_dir, 'postprocessing', subj, subj+'_AccR_dPutR_vPutL_lvPFC_lPFC_dPFC_sphere6-12mm_count_nosift_connectome.csv')
        if os.path.exists(fname):
            df = pd.read_csv(fname, sep=' ', names=regions)
            df.index = regions
        else:
            print(subj+' discarded, no streamline count file.')
            continue

        for seed,voi in zip(seeds,vois):
            df_lines.append({'subj':subj, 'pathway':seed+voi, 'count':df[seed][voi], 'cohort':coh})

    df_streamlines = pd.DataFrame(df_lines)
    return df_streamlines

def plot_streamline_counts(df_streamlines, seeds=['Acc', 'dPut', 'vPut'], vois = ['OFC', 'lPFC', 'dPFC'], args=None):
    """ plot distribution of streamline count in each pathway """
    colors = ['lightgray', 'darkgray']
    plt.figure(figsize=[8,10])
    plt.rcParams.update({'font.size':14, 'axes.linewidth':1, 'pdf.fonttype':42})

    for i, (seed,voi) in enumerate(zip(seeds,vois)):
        ax = plt.subplot(2,3,i+1)
        sbn.violinplot(data=df_streamlines[df_streamlines['pathway']==seed+voi], y='count', x='pathway', hue='cohort', \
                    split=True, dodge=False, linewidth=0, palette=colors, cut=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=1)
        plt.ylabel('streamline count');

        if i==len(seeds)-1:
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        else:
            ax.get_legend().remove()

        ax = plt.subplot(2,3,3+i+1)
        sbn.boxplot(data=df_streamlines[df_streamlines['pathway']==seed+voi], y='count', x='pathway', hue='cohort', \
                    linewidth=1, palette=colors)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=1)
        plt.ylabel('streamline count');

        if i==len(seeds)-1:
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        else:
            ax.get_legend().remove()
    plt.tight_layout()

    if args.save_figs:
        fname = 'streamline_count_'+datetime.datetime.now().strftime("%Y%m%d")+'.pdf'
        plt.savefig(os.path.join(proj_dir, 'img', fname), transparent=True)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_indiv_masks', default=False, action='store_true', help='flag to plot individual masks')
    parser.add_argument('--plot_group_masks', default=False, action='store_true', help='flag to plot group masks')
    parser.add_argument('--keep_masks', default=False, action='store_true', help='flag to keep subjects in dict (take ~16Gb)')
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--compute_tdi', default=False, action='store_true', help='flag to compute track density maps, if not switched it tries to load them')
    parser.add_argument('--plot_tdi_distrib', default=False, action='store_true', help='Plot difference in track density distributions (violin, strip, box plots)')
    parser.add_argument('--group_mask_threshold', type=float, default=0.5, action='store', help="threshold used to create group mask in pathway when no streamlines available")
    parser.add_argument('--compute_stats', default=False, action='store_true', help="compute and print stats of structural differences")
    parser.add_argument('--plot_streamline_counts', default=False, action='store_true', help="get streamline counts from connectomes csv files and plot distributions")
    args = parser.parse_args()

    # parameters
    thresholds = np.arange(20,101,20) # in %
    args.thresholds = thresholds
    tdi_threshold_to_plot = thresholds[3] # plot 80%


    if args.compute_tdi:
        dsi_m = compute_tdi_maps(thresholds, args)
        dsi_m = get_group_masks(dsi_m, args)
        dsi_m = compute_diffusion_maps(dsi_m, args)
    else:
        with open(os.path.join(proj_dir, 'postprocessing', 'dsi_m.pkl'), 'rb') as f:
            dsi_m = pickle.load(f)

    if args.compute_stats:
        compute_stats(dsi_m, thresholds, args)

        # Effect size
        df_dsi = create_df_dsi(dsi_m, tdi_threshold_to_plot)
        for subroi in subrois:
          print(subroi)
          cons = df_dsi[(df_dsi['cohorts']=='controls') & (df_dsi['subroi']==subroi)].mean_dsi
          pats = df_dsi[(df_dsi['cohorts']=='patients') & (df_dsi['subroi']==subroi)].mean_dsi
          print('Controls: mean={:.5f} std={:.5f} n={}'.format(cons.mean(), cons.std(), str(len(cons))))
          print('Patients: mean={:.5f} std={:.5f} n={}'.format(pats.mean(), pats.std(), str(len(pats))))
          print('cohen\'s d at threshold {} = {:.2f}'.format(str(tdi_threshold_to_plot), cohen_d(cons, pats)))

        create_summary(dsi_m, args)

    #avg_mask = plot_group_masks(dsi_m, thresholds, args)

    if args.plot_tdi_distrib:
        plot_tdi_distrib(df_dsi, tdi_threshold_to_plot, args)

    if args.save_outputs:
        with open(os.path.join(proj_dir, 'postprocessing', 'dsi_m.pkl'), 'wb') as f:
            pickle.dump(dsi_m, f)

    if args.plot_streamline_counts:
        df_streamlines = create_df_streamline(args=args)
        plot_streamline_counts(df_streamlines, args=args)



