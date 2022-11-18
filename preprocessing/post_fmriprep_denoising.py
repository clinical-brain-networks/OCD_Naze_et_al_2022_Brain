"""
Wrapper code that uses Paula's 'fmripop' code to perform basic denoising.
For this project denoising is very minimal because FIX has already been run.

see https://github.com/brain-modelling-group/fmripop/blob/master/post_fmriprep.py

"""
# %%
import sys
import time
import os
import json
import numpy as np
from nilearn.image import new_img_like

code_dir = '/mnt/lustre/working/lab_lucac/sebastiN/projects/OCDbaseline/docs/code/'
sys.path.insert(0, code_dir)
sys.path.insert(0, os.path.join(code_dir, 'preprocessing'))

# import my own code
from functions.data_helpers import get_computer, make_dirs

# add Paula's fmripop to path and import functions
computer, proj_dir = get_computer()
conf_dir = proj_dir+'data/derivatives/fmriprep/'
bold_dir = proj_dir+'data/derivatives/fmriprep-fix/'
out_dir = '/mnt/lustre/working/lab_lucac/sebastiN/projects/OCDbaseline/data/derivatives/post-fmriprep-fix/'

if computer == 'lucky2' or computer == 'lucky3':
    fmripop_path = '/home/lukeh/hpcworking/lukeH/fmripop-master/'
else:  # assume on hpc
    fmripop_path = '/mnt/lustre/working/lab_lucac/sebastiN/fmripop/'
sys.path.insert(0, fmripop_path)
from post_fmriprep import parser, fmripop_check_args, fmripop_remove_confounds, fmripop_scrub_data, fmripop_smooth_data

# define subj
subj = sys.argv[1]
print(subj)

# files and pipelines
img_space = 'MNI152NLin2009cAsym'

# get files for this subject
task_nii = (bold_dir+subj+'/func/'+subj+'_task-fearRev_space-'
                    + img_space+'_desc-preproc_bold.nii.gz')
rest_nii = (bold_dir+subj+'/func/'+subj+'_task-rest_space-'
                    + img_space+'_desc-preproc_bold.nii.gz')

task_msk = (bold_dir+subj+'/func/'+subj+'_task-fearRev_space-'
                    + img_space+'_desc-brain_mask.nii.gz')
rest_msk = (bold_dir+subj+'/func/'+subj+'_task-rest_space-'
                    + img_space+'_desc-brain_mask.nii.gz')

task_tsv = conf_dir+subj+'/func/'+subj + \
    '_task-fearRev_desc-confounds_timeseries.tsv'
rest_tsv = conf_dir+subj+'/func/'+subj + \
    '_task-rest_desc-confounds_timeseries.tsv'

# list the models I would like to run:
# pipelines = {'detrend': {'niipath': task_nii,
#                          'maskpath': task_msk,
#                          'tsvpath': task_tsv,
#                          'add_orig_mean_img': True,
#                          'confound_list': [None],
#                          'detrend': True,
#                          'fmw_disp_th': None,
#                          'fwhm': 0,
#                          'high_pass': None,
#                          'low_pass': None,
#                          'remove_volumes': False,
#                          'scrubbing': False,
#                          'num_confounds': 0,
#                          'tr': 0.81,
#                          'task': 'fearRev'
#                          },
#              'detrend_smooth-6mm': {'niipath': task_nii,
#                                     'maskpath': task_msk,
#                                     'tsvpath': task_tsv,
#                                     'add_orig_mean_img': True,
#                                     'confound_list': [None],
#                                     'detrend': True,
#                                     'fmw_disp_th': None,
#                                     'fwhm': 6,
#                                     'high_pass': None,
#                                     'low_pass': None,
#                                     'scrubbing': False,
#                                     'remove_volumes': False,
#                                     'num_confounds': 0,
#                                     'tr': 0.81,
#                                     'task': 'fearRev'
#                                     },
#              'detrend_filtered_scrub': {'niipath': rest_nii,
#                                         'maskpath': rest_msk,
#                                         'tsvpath': rest_tsv,
#                                         'add_orig_mean_img': True,
#                                         'confound_list': [None],
#                                         'detrend': True,
#                                         'fmw_disp_th': 0.5,
#                                         'fwhm': 0,
#                                         'high_pass': 0.01,
#                                         'low_pass': 0.10,
#                                         'scrubbing': True,
#                                         'remove_volumes': True,
#                                         'tr': 0.81,
#                                         'num_confounds': 0,
#                                         'task': 'rest'
#                                         },
#              'detrend_filtered_scrub_smooth-6mm': {'niipath': rest_nii,
#                                                    'maskpath': rest_msk,
#                                                    'tsvpath': rest_tsv,
#                                                    'add_orig_mean_img': True,
#                                                    'confound_list': [None],
#                                                    'detrend': True,
#                                                    'fmw_disp_th': 0.5,
#                                                    'fwhm': 6,
#                                                    'high_pass': 0.01,
#                                                    'low_pass': 0.10,
#                                                    'scrubbing': True,
#                                                    'remove_volumes': True,
#                                                    'tr': 0.81,
#                                                    'num_confounds': 0,
#                                                    'task': 'rest'
#                                                    },
#              'detrend_filtered_scrub_csf_wm': {'niipath': rest_nii,
#                                                'maskpath': rest_msk,
#                                                'tsvpath': rest_tsv,
#                                                'add_orig_mean_img': True,
#                                                'confound_list': ['csf', 'white_matter'],
#                                                'detrend': True,
#                                                'fmw_disp_th': 0.5,
#                                                'fwhm': 0,
#                                                'high_pass': 0.01,
#                                                'low_pass': 0.10,
#                                                'scrubbing': True,
#                                                'remove_volumes': True,
#                                                'tr': 0.81,
#                                                'num_confounds': 2,
#                                                'task': 'rest'
#                                                },
#              'detrend_filtered_scrub_csf_wm_smooth-6mm': {'niipath': rest_nii,
#                                                           'maskpath': rest_msk,
#                                                           'tsvpath': rest_tsv,
#                                                           'add_orig_mean_img': True,
#                                                           'confound_list': ['csf', 'white_matter'],
#                                                           'detrend': True,
#                                                           'fmw_disp_th': 0.5,
#                                                           'fwhm': 6,
#                                                           'high_pass': 0.01,
#                                                           'low_pass': 0.10,
#                                                           'scrubbing': True,
#                                                           'remove_volumes': True,
#                                                           'tr': 0.81,
#                                                           'num_confounds': 2,
#                                                           'task': 'rest'
#                                                           },
#              }
"""
pipelines = {'detrend_gsr': {'niipath': task_nii,
                             'maskpath': task_msk,
                             'tsvpath': task_tsv,
                             'add_orig_mean_img': True,
                             'confound_list': ['global_signal'],
                             'detrend': True,
                             'fmw_disp_th': None,
                             'fwhm': 0,
                             'high_pass': None,
                             'low_pass': None,
                             'remove_volumes': False,
                             'scrubbing': False,
                             'num_confounds': 1,
                             'tr': 0.81,
                             'task': 'fearRev'
                             },
             'detrend_gsr_filtered_scrub': {'niipath': rest_nii,
                                            'maskpath': rest_msk,
                                            'tsvpath': rest_tsv,
                                            'add_orig_mean_img': True,
                                            'confound_list': ['global_signal'],
                                            'detrend': True,
                                            'fmw_disp_th': 0.5,
                                            'fwhm': 0,
                                            'high_pass': 0.01,
                                            'low_pass': 0.10,
                                            'scrubbing': True,
                                            'remove_volumes': True,
                                            'tr': 0.81,
                                            'num_confounds': 1,
                                            'task': 'rest'
                                            }
             }

pipelines = {'detrend_filtered_gsr_smooth-6mm': {'niipath': rest_nii,
                                                          'maskpath': rest_msk,
                                                          'tsvpath': rest_tsv,
                                                          'add_orig_mean_img': True,
                                                          'confound_list': ['global_signal'],
                                                          'detrend': True,
                                                          'fmw_disp_th': False,
                                                          'fwhm': 6,
                                                          'high_pass': 0.01,
                                                          'low_pass': 0.10,
                                                          'scrubbing': False,
                                                          'remove_volumes': False,
                                                          'tr': 0.81,
                                                          'num_confounds': 1,
                                                          'task': 'rest'}}

pipelines = {'detrend_filtered_smooth-6mm': { 'niipath': rest_nii,
                                              'maskpath': rest_msk,
                                              'tsvpath': rest_tsv,
                                              'add_orig_mean_img': True,
                                              'confound_list': [],
                                              'detrend': True,
                                              'fmw_disp_th': None,
                                              'fwhm': 6,
                                              'high_pass': 0.01,
                                              'low_pass': 0.10,
                                              'scrubbing': False,
                                              'remove_volumes': False,
                                              'tr': 0.81,
                                              'num_confounds': 0,
                                              'task': 'rest'},
             'detrend_smooth-6mm': {  'niipath': rest_nii,
                                      'maskpath': rest_msk,
                                      'tsvpath': rest_tsv,
                                      'add_orig_mean_img': True,
                                      'confound_list': [],
                                      'detrend': True,
                                      'fmw_disp_th': None,
                                      'fwhm': 6,
                                      'high_pass': None,
                                      'low_pass': None,
                                      'scrubbing': False,
                                      'remove_volumes': False,
                                      'tr': 0.81,
                                      'num_confounds': 0,
                                      'task': 'rest'},
            'filtered_smooth-6mm': {  'niipath': rest_nii,
                                    'maskpath': rest_msk,
                                    'tsvpath': rest_tsv,
                                    'add_orig_mean_img': True,
                                    'confound_list': [],
                                    'detrend': False,
                                    'fmw_disp_th': None,
                                    'fwhm': 6,
                                    'high_pass': 0.01,
                                    'low_pass': 0.10,
                                    'scrubbing': False,
                                    'remove_volumes': False,
                                    'tr': 0.81,
                                    'num_confounds': 0,
                                    'task': 'rest'},
            'smooth-6mm': {   'niipath': rest_nii,
                              'maskpath': rest_msk,
                              'tsvpath': rest_tsv,
                              'add_orig_mean_img': True,
                              'confound_list': [],
                              'detrend': False,
                              'fmw_disp_th': None,
                              'fwhm': 6,
                              'high_pass': None,
                              'low_pass': None,
                              'scrubbing': False,
                              'remove_volumes': False,
                              'tr': 0.81,
                              'num_confounds': 0,
                              'task': 'rest'}}
"""

pipelines = {'detrend_gsr_filtered_scrubFD05': { 'niipath': rest_nii,
                                              'maskpath': rest_msk,
                                              'tsvpath': rest_tsv,
                                              'add_orig_mean_img': True,
                                              'confound_list': ['global_signal'],
                                              'detrend': True,
                                              'fmw_disp_th': 0.5,
                                              'fwhm': 0,
                                              'high_pass': 0.01,
                                              'low_pass': 0.1,
                                              'scrubbing': True,
                                              'remove_volumes': True,
                                              'tr': 0.81,
                                              'num_confounds': 1,
                                              'task': 'rest'}}

"""             'detrend_gsr_smooth-6mm': { 'niipath': rest_nii,
                                              'maskpath': rest_msk,
                                              'tsvpath': rest_tsv,
                                              'add_orig_mean_img': True,
                                              'confound_list': ['global_signal'],
                                              'detrend': True,
                                              'fmw_disp_th': None,
                                              'fwhm': 6,
                                              'high_pass': None,
                                              'low_pass': None,
                                              'scrubbing': False,
                                              'remove_volumes': False,
                                              'tr': 0.81,
                                              'num_confounds': 1,
                                              'task': 'rest'}}"""


for pl_label in pipelines:
    print('Running: '+pl_label)
    # use my own wrapper code (similar to __main__ in fmripop)
    start_time = time.time()
    pl = pipelines[pl_label]

    # set up args obj
    args = parser.parse_args('')

    # Modify the arguments based on dict
    args.niipath = pl['niipath']
    args.maskpath = pl['maskpath']
    args.tsvpath = pl['tsvpath']
    args.add_orig_mean_img = pl['add_orig_mean_img']
    args.confound_list = pl['confound_list']
    args.detrend = pl['detrend']
    args.fmw_disp_th = pl['fmw_disp_th']
    args.fwhm = pl['fwhm']
    args.high_pass = pl['high_pass']
    args.low_pass = pl['low_pass']
    args.num_confounds = pl['num_confounds']
    args.remove_volumes = pl['remove_volumes']
    args.scrubbing = pl['scrubbing']
    args.tr = pl['tr']

    # Set derived Parameters according to user specified parameters
    args = fmripop_check_args(args)

    # Convert to dict() for saving later
    params_dict = vars(args)
    params_dict['fwhm'] = args.fwhm.tolist()

    # Performs main task -- removing confounds
    out_img = fmripop_remove_confounds(args)

    # Perform additional actions on data
    if args.scrubbing:
        out_img, params_dict = fmripop_scrub_data(out_img, args, params_dict)

    if np.array(args.fwhm).sum() > 0.0:  # If fwhm is not zero, performs smoothing
        out_img = fmripop_smooth_data(out_img, args.fwhm)

    # Save output image and parameters used in this script
    out_path = (out_dir+subj+'/func/')
    out_file = (out_path+subj+'_task-'+pl['task']+'_space-'
                + img_space+'_desc-'+pl_label+'.nii.gz')
    make_dirs([out_path])

    # make sure the out img has the correct header
    out_img = new_img_like(
        pl['niipath'], out_img.get_fdata(), copy_header=True)

    # Save the clean data in a separate file
    out_img.to_filename(out_file)

    # Save the input arguments in a json file with a timestamp
    timestamp = time.strftime("%Y-%m-%d-%H%M%S")
    out_file = 'fmripop_'+pl_label+'_parameters.json'
    with open(os.path.sep.join((out_path, out_file)), 'w') as file:
        file.write(json.dumps(params_dict, indent=4, sort_keys=True))

    print("--- %s seconds ---" % (time.time() - start_time))

print('Finished all pipelines')

# %%
