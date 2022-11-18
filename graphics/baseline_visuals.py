# Script to visualize baseline fMRI analysis outputs
# Author: Sebastien Naze
# QIMR Berghofer 2022

import argparse
from argparse import Namespace
import bct
from datetime import datetime
import h5py
import importlib
from itkwidgets import view
import itertools
import joblib
from joblib import Parallel, delayed
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import nilearn
from nilearn.image import load_img
from nilearn.plotting import plot_matrix, plot_glass_brain, plot_stat_map, plot_img_comparison
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import pyvista as pv
from pyvista import examples
import scipy
from scipy.io import loadmat
import seaborn as sbn
import sklearn
from sklearn.decomposition import PCA
import sys
import time
from time import time, sleep
import warnings

# paths
proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'code')
deriv_dir = os.path.join(proj_dir, 'data/derivatives')
atlas_dir = os.path.join(proj_dir, 'utils')
fs_dir = '/usr/local/freesurfer/'

sys.path.insert(0, os.path.join(code_dir, 'functional'))
from functional import seed_to_voxel_analysis

# uncomment in case of using freesurfer surfaces
#coords, faces, info, stamp = nib.freesurfer.io.read_geometry(os.path.join(fs_dir, 'subjects', 'fsaverage4', 'surf', 'lh.white'), read_metadata=True, read_stamp=True)

imgs_info = {
                'base': {'path': os.path.join(proj_dir, 'utils', 'empty.nii.gz'),
                                'name': 'base',
                                'cmap': 'binary',
                                'clim': [0, 1.],
                                'opacity': 1.,
                                'nan_opacity': 1. },
                'ventromed_probmap': {  'path': os.path.join(proj_dir, 'utils', 'ProbabilityMaps_CorticoStriatalConnectivity_RightStriatum', 'ProbMap_R_N3_ventromed.nii'),
                                        'name':'ventromed_probmap',
                                        'cmap':'Reds',
                                        'clim': [50,70],
                                        'opacity': 0.5,
                                        'nan_opacity':0.},
                'putamen_probmap': {  'path': os.path.join(proj_dir, 'utils', 'ProbabilityMaps_CorticoStriatalConnectivity_RightStriatum', 'ProbMap_R_N3_putamen.nii'),
                                        'name':'putamen_probmap',
                                        'cmap':'Reds',
                                        'clim': [50,100],
                                        'opacity': 1,
                                        'nan_opacity':0.},
                'tian_acc': {  'path': os.path.join(proj_dir, 'utils', 'hcp_masks', 'Acc_pathway_mask_group_by_hemi_Ftest_grp111_100hcpThr100SD_GM_23092022.nii.gz'), 
                                        'name':'tian_acc',
                                        'cmap':'Oranges',
                                        'clim': [0, 0.6],
                                        'opacity': 1.,
                                        'nan_opacity':0.},
                'tian_putamen': {  'path': os.path.join(proj_dir, 'utils', 'hcp_masks', 'dPut_pathway_mask_group_by_hemi_Ftest_grp111_100hcpThr250SD_GM_23092022.nii.gz'),,
                                        'name':'tian_putamen',
                                        'cmap':'Greens',
                                        'clim': [0, 0.6],
                                        'opacity': 1.,
                                        'nan_opacity':0.},
                'dCaud_hcp_mask': {  'path': os.path.join(proj_dir, 'utils', 'hcp_masks', 'dCaud_pathway_mask_group_by_hemi_Ftest_grp111_100hcpThr275SD_GM_23092022.nii.gz'),
                                        'name':'dCaud_hcp_mask',
                                        'cmap':'Blues',
                                        'clim': [0.4, 0.6],
                                        'opacity': 1.,
                                        'nan_opacity':0.},
                'vPut_hcp_mask': {  'path': os.path.join(proj_dir, 'utils', 'hcp_masks', 'vPut_pathway_mask_group_by_hemi_Ftest_grp111_100hcpThr225SD_GM_23092022.nii.gz'),
                                        'name':'vPut_hcp_mask',
                                        'cmap':'Purples',
                                        'clim': [0.4, 0.6],
                                        'opacity': 0.8,
                                        'nan_opacity':0.},
                'acc_seed': {   'path': os.path.join(proj_dir, 'utils', 'Acc.nii.gz'),
                                'name': 'acc_seed',
                                'cmap': 'Reds',
                                'clim': [0,0.5],
                                'opacity': 1. ,
                                'nan_opacity':0},
                'acc_pathway': {'path': os.path.join(proj_dir, 'utils', 'frontal_Acc_mapping.nii.gz'),
                                'name': 'acc_pathway',
                                'cmap': 'Blues',
                                'clim': [0, 1.5],
                                'opacity': 0.5,
                                'nan_opacity': 0. },
                'dPut_pathway': {'path': os.path.join(proj_dir, 'utils', 'frontal_dPut_mapping.nii.gz'),
                                'name': 'dPut_pathway',
                                'cmap': 'Blues',
                                'clim': [0, 1.5],
                                'opacity': 0.5,
                                'nan_opacity': 0. },
                'Acc_con': {'path': os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/detrend_gsr_filtered_scrubFD05/brainFWHM8mm/Acc/randomise', 'Acc_outputs_n5000_TFCE_OCD_minus_HC_13062022_tstat1.nii.gz'),
                                'name': 'Acc_con',
                                'cmap': 'Reds',
                                'clim': [1., 2.],
                                'opacity': 1,
                                'nan_opacity': 0. },
                'dPut_con': {'path': os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/detrend_gsr_filtered_scrubFD05/brainFWHM8mm/dPut/randomise', 'dPut_outputs_n5000_TFCE_HC_minus_OCD_13062022_tstat1.nii.gz'),
                                'name': 'dPut_con',
                                'cmap': 'Blues',
                                'clim': [1., 2.],
                                'opacity': 1,
                                'nan_opacity': 0. },
                'Acc_hcp_con': {'path': os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/detrend_gsr_filtered_scrubFD05/brainFWHM8mm/Acc/randomise', 'Acc_outputs_n5000_c35_group_by_hemi_Ftest_grp111_100hcpThr100SD_GM_22092022_tstat1.nii.gz'),
                                'name': 'Acc_hcp_con',
                                'cmap': 'RdBu',
                                'clim': [-3, 3],
                                'opacity': 1,
                                'nan_opacity': 0. },
                'dPut_hcp_con': {'path': os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/detrend_gsr_filtered_scrubFD05/brainFWHM8mm/dPut/randomise', 'dPut_outputs_n5000_c30_group_by_hemi_Ftest_grp111_100hcpThr275SD_GM_22092022_tstat1.nii.gz'),
                                'name': 'dPut_hcp_con',
                                'cmap': 'RdBu',
                                'clim': [-3, 3],
                                'opacity': 1,
                                'nan_opacity': 0. },
                'dCaud_hcp_con': {'path': os.path.join(proj_dir, 'postprocessing/SPM/outputs/Harrison2009Rep/smoothed_but_sphere_seed_based/detrend_gsr_filtered_scrubFD05/brainFWHM8mm/dCaud/randomise', 'dCaud_outputs_n5000_c30_group_by_hemi_Ftest_grp111_100hcpThr275SD_GM_23092022_tstat1.nii.gz'),
                                'name': 'dCaud_hcp_con',
                                'cmap': 'RdBu',
                                'clim': [-3, 3],
                                'opacity': 1,
                                'nan_opacity': 0. }
}

# Additionally, all the modules other than ipygany and pythreejs require a framebuffer, which can be setup on a headless environment with pyvista.start_xvfb().
pv.start_xvfb()


def get_brainnet_surf(surf_name):
    """ Import brain net viewer surface into pyvista polyData type """
    brainnet_path = '/home/sebastin/Downloads/BrainNetViewer/BrainNet-Viewer/Data/SurfTemplate/'
    fname = os.path.join(brainnet_path, surf_name+'.nv')
    with open(fname, 'r') as f:
        n_vertices = int(f.readline())

    icbm_txt = pd.read_csv(fname, sep=' ', header=None, index_col=False, skiprows=[0,n_vertices+1])
    coords = np.array(icbm_txt.iloc[:n_vertices])
    faces = np.array(icbm_txt.iloc[n_vertices:], dtype=int) - 1

    nfaces, fdim = faces.shape
    c = np.ones((nfaces,1))*fdim
    icbm_surf = pv.PolyData(coords, np.hstack([c,faces]).astype(int))
    return icbm_surf, coords, faces


def volume_to_surface(vol_img, coords, faces, radius=5.):
    """ project volume niftii image to cortical surface mesh """
    left_surf = nilearn.surface.vol_to_surf(img=vol_img, surf_mesh=[coords.left, faces.left], radius=radius, interpolation='linear')
    right_surf = nilearn.surface.vol_to_surf(img=vol_img, surf_mesh=[coords.right, faces.right], radius=radius, interpolation='linear')
    both_surf = nilearn.surface.vol_to_surf(img=vol_img, surf_mesh=[coords.both, faces.both], radius=radius, interpolation='linear')
    return Namespace(**{'left':left_surf, 'right':right_surf, 'both':both_surf})


def get_icbm_surf(args):
    """ imports ICBM152 surfaces into Namespace """
    if args.smoothed_surface:
        icbm_left, coords_left, faces_left = get_brainnet_surf('BrainMesh_ICBM152Left_smoothed')
        icbm_right, coords_right, faces_right = get_brainnet_surf('BrainMesh_ICBM152Right_smoothed')
        icbm_both, coords_both, faces_both = get_brainnet_surf('BrainMesh_ICBM152_smoothed')
    else:
        icbm_left, coords_left, faces_left = get_brainnet_surf('BrainMesh_ICBM152Left')
        icbm_right, coords_right, faces_right = get_brainnet_surf('BrainMesh_ICBM152Right')
        icbm_both, coords_both, faces_both = get_brainnet_surf('BrainMesh_ICBM152')
    coords = Namespace(**{'left':coords_left, 'right':coords_right, 'both':coords_both})
    faces = Namespace(**{'left':faces_left, 'right':faces_right, 'both':faces_both})
    surfs = Namespace(**{'left':icbm_left, 'right':icbm_right, 'both':icbm_both})
    return surfs, coords, faces


def project_surface(template, img, name):
    """ fill template surface with img surface data for rendering """
    img.left[img.left==0] = np.NaN
    img.right[img.right==0] = np.NaN
    img.both[img.both==0] = np.NaN
    template.left.point_data[name] = img.left
    template.right.point_data[name] = img.right
    template.both.point_data[name] = img.both


def plot_surface(surfs, names=imgs_info.keys(), args=None):
    """  """
    cam_pos = {'front':[-3, 2, -1], 'medial':[1,1,-0.3]}

    # Plot
    pl = pv.Plotter(window_size=[800, 600], shape=(1,4), border=False)
    pl.set_plot_theme = 'document'

    # between-group row
    pl.subplot(0,0)
    for img_name in names:
        img_info = imgs_info[img_name]
        pl.add_mesh(surfs.left.copy(), scalars=img_name, cmap=img_info['cmap'], smooth_shading=True, opacity=img_info['opacity'], clim=img_info['clim'],
                    nan_color='white', nan_opacity=img_info['nan_opacity'], interpolate_before_map=False, show_scalar_bar=False)
    pl.camera_position = cam_pos['front']
    pl.camera.zoom(1.5)
    pl.background_color = 'white'

    pl.subplot(0,1)
    for img_name in names:
        img_info = imgs_info[img_name]
        pl.add_mesh(surfs.left.copy(), scalars=img_name, cmap=img_info['cmap'], smooth_shading=True, opacity=img_info['opacity'], clim=img_info['clim'],
                    nan_color='white', nan_opacity=img_info['nan_opacity'], interpolate_before_map=False, show_scalar_bar=False)
    pl.camera_position = cam_pos['medial']
    pl.camera.zoom(1.5)
    pl.background_color = 'white'

    pl.subplot(0,2)
    for img_name in names:
        img_info = imgs_info[img_name]
        pl.add_mesh(surfs.right.copy(), scalars=img_name, cmap=img_info['cmap'], smooth_shading=True, opacity=img_info['opacity'], clim=img_info['clim'],
                    nan_color='white', nan_opacity=img_info['nan_opacity'], interpolate_before_map=False, show_scalar_bar=False)
    pl.camera_position = cam_pos['front']
    pl.camera.zoom(1.5)
    pl.background_color = 'white'

    pl.subplot(0,3)
    for img_name in names:
        img_info = imgs_info[img_name]
        pl.add_mesh(surfs.right.copy(), scalars=img_name, cmap=img_info['cmap'], smooth_shading=True, opacity=img_info['opacity'], clim=img_info['clim'],
                    nan_color='white', nan_opacity=img_info['nan_opacity'], interpolate_before_map=False, show_scalar_bar=False)
    pl.camera_position = cam_pos['medial']
    pl.camera.zoom(1.5)
    pl.background_color = 'white'

    if args.save_figs:
        fname = '_'.join(names)+'_'+datetime.now().strftime('%d%m%Y')+'.pdf'
        plt.savefig(os.path.join(proj_dir, 'img','plot_'+fname))




    pl.show(jupyter_backend='panel')
    pl.deep_clean()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_figs', default=False, action='store_true', help='plot figures')
    parser.add_argument('--save_figs', default=False, action='store_true', help='save figures')
    parser.add_argument('--save_outputs', default=False, action='store_true', help='save outputs')
    parser.add_argument('--surface_template', type=str, default='icbm', action='store', help='defines which surface template to use. default: icbm')
    parser.add_argument('--plot_surface', default=False, action='store_true', help='plot surface mesh with stim  locations and mask')
    parser.add_argument('--smoothed_surface', default=False, action='store_true', help='use smooth cortical mesh')
    args = parser.parse_args()

    names=['base', 'Acc_con', 'dPut_con'] # <-- must be decalred in imgs_info on top of the file

    if args.plot_surface:
        # get template ICBM surfaces (left, right and both hemispheres meshes)
        surfs, coords, faces = get_icbm_surf(args)

        # create projections on surfaces
        for img_name in names:
            img_info = imgs_info[img_name]
            # load image of interest
            img = load_img(img_info['path'])
            # project image to surface
            img_surfs = volume_to_surface(img, coords, faces)
            # prepare surface for rendering/plotting
            project_surface(surfs, img_surfs, name=img_name)

        plot_surface(surfs, names, args=args)
