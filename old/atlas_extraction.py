# Extract timeseries of brain regions based on atlas and
# fMRI preprocessing method.
#
# Original author: Luke Hearne
# Modified by: Sebastien Naze
#
# 2021 - Clinical Brain Networks - QIMR Berghofer
###########################################################
import argparse
import h5py
from joblib import Parallel, delayed
import os
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
import socket
from time import time

# global variables
# paths
if 'hpc' in socket.gethostname():
    working_path = '/working/lab_lucac/'
else:
    working_path = '/home/sebastin/working/lab_lucac/'
lukeH_proj_dir = working_path+'lukeH/projects/OCDbaseline/'
sebN_proj_dir = working_path+'sebastiN/projects/OCDbaseline/'
lukeH_deriv_dir = lukeH_proj_dir+'data/derivatives/post-fmriprep-fix/'
sebN_deriv_dir = sebN_proj_dir+'data/derivatives/post-fmriprep-fix/'
parc_dir = sebN_proj_dir+'utils/'
#scratch_dir = proj_dir+'data/scratch/nilearn/'

# task
task_list = ['rest']

# preprocessed image space
img_space = 'MNI152NLin2009cAsym'

# denoising label (no smoothing needed)
denoise_label = {
    'rest': ['detrend_filtered_scrub', 'detrend_gsr_filtered_scrub']}

# parcellation nifti files used in timeseries extraction
ts_parc_dict = {#'schaefer100': parc_dir+'schaefer100MNI_lps_mni.nii.gz',
                #'schaefer200': parc_dir+'schaefer200MNI_lps_mni.nii.gz',
                #'schaefer400': parc_dir+'schaefer400MNI_lps_mni.nii.gz',
                #'harrison2009': parc_dir+'Harrison2009.nii.gz',
                'ocdOFClPFC': parc_dir+'ocdOFClPFC.nii.gz',
                'ocdAccdPut': parc_dir+'ocdAccdPut.nii.gz'
               }


smoothing_fwhm = {'schaefer100': 8,
                  'schaefer200': 8,
                  'schaefer400': 8,
                  'harrison2009': None,
                  'ocdOFClPFC': None,
                  'ocdAccdPut': None
                 }

# lists of parcellations to create fc matrices from
# these will match the above dict in name
#con_cortical_parcs = ['Schaefer2018_100_7', 'Schaefer2018_200_7',
#                      'Schaefer2018_300_7', 'Schaefer2018_400_7']
#con_subcortical_parcs = ['Tian_S1', 'Tian_S2', 'Tian_S3', 'Tian_S4']

#ctx_parc = ['schaefer100', 'schaefer200', 'schaefer400']
#subctx_parc = ['harrison2009']

ctx_parc = ['ocdOFClPFC']
subctx_parc = ['ocdAccdPut', 'harrison2009']


def extract_timeseries(subj):
    # Extracts atlas based timeseries using nilearn
    for task in task_list:
        for denoise in denoise_label[task]:
            for parc in ts_parc_dict.keys():
                print('\t', subj, '\t', task, ':', denoise, '\t', parc)

                # get subject / task / denoise specific BOLD nifti filename
                bold_file = (lukeH_deriv_dir+subj+'/func/'+subj+'_task-'+task
                             + '_space-'+img_space+'_desc-'+denoise+'.nii.gz')

                # use nilearn to extract timeseries
                masker = NiftiLabelsMasker(ts_parc_dict[parc], smoothing_fwhm=smoothing_fwhm[parc], \
                                t_r=0.81, low_pass=0.1, high_pass=0.01, \
                                memory='nilearn_cache', memory_level=1, verbose=0)
                time_series = masker.fit_transform(bold_file)

                # save timeseries out as h5py file
                out_file = (sebN_deriv_dir+subj+'/timeseries/'+subj+'_task-'
                            + task+'_atlas-'+parc+'_desc-'+denoise+'_fwhm'+str(smoothing_fwhm[parc])+'.h5')
                hf = h5py.File(out_file, 'w')
                hf.create_dataset(parc, data=time_series)
                hf.close()


def generate_fc(subj):
    # generate the tian-schaefer connectivity matrices
    for task in task_list:
        for denoise in denoise_label[task]:
            for parc in ctx_parc:

                # load the cortical timeseries
                in_file = (sebN_deriv_dir+subj+'/timeseries/'+subj+'_task-'
                           + task+'_atlas-'+parc+'_desc-'+denoise+'_fwhm'+str(smoothing_fwhm[parc])+'.h5')
                hf = h5py.File(in_file, 'r')
                ctx_time_series = hf[parc][:]
                hf.close()

                for sparc in subctx_parc:
                    # load the sub-cortical timeseries
                    in_file = (sebN_deriv_dir+subj+'/timeseries/'+subj+'_task-'
                               + task+'_atlas-'+sparc+'_desc-'+denoise+'_fwhm'+str(smoothing_fwhm[sparc])+'.h5')
                    hf = h5py.File(in_file, 'r')
                    subctx_time_series = hf[sparc][:]
                    hf.close()

                    # combine ctx and subctx timeseries
                    time_series = np.hstack((ctx_time_series, subctx_time_series))

                    # perform fc (here, correlation only)
                    fc = np.corrcoef(time_series.T)

                    # save out
                    parc_out = '_'.join([parc,sparc])
                    out_file = (sebN_deriv_dir+subj+'/fc/'+subj+'_task-'+task
                                + '_atlas-'+parc_out+'_desc-correlation-'+denoise+'_fwhm'+str(smoothing_fwhm[parc])+'.h5')
                    hf = h5py.File(out_file, 'w')
                    hf.create_dataset(name='fc', data=fc)
                    hf.close()

def process_subj(subj):
    for newdir in ['timeseries', 'fc']:
        os.makedirs(os.path.join(sebN_deriv_dir, subj, newdir), exist_ok=True)
    start = time()
    print(subj)
    extract_timeseries(subj)
    generate_fc(subj)
    finish = time()
    print(subj, ' time elapsed:', finish - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', action='store', default=None, help='subject to process')
    args = parser.parse_args()

    if (args.subj == None):
        # loop through everyone and run:
        subj_list = list(np.loadtxt('./subject_list.txt', dtype='str'))
        Parallel(n_jobs=12)(delayed(process_subj)(subj) for subj in subj_list)
    else:
        process_subj(args.subj)
