#
# Atlaser
#
# Utility to access and modify existing atlas
#
# Author: Sebastien Naze
# -----------------------------------------------------------------------

import itertools
import json
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
from nilearn.image import load_img
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
import numpy as np
import os
import pandas as pd
import scipy
import socket
import sys

from IPython.core.debugger import set_trace

if 'hpc' in socket.gethostname():
    working_dir = '/working/'
else:
    working_dir = '/home/sebastin/working/'

atlas_suffix = {'schaefer100_tianS1':'MNI_lps_mni.nii.gz', \
                'schaefer200_tianS2':'MNI_lps_mni.nii.gz', \
                'schaefer400_tianS4':'MNI_lps_mni.nii.gz', \
                'schaefer400_harrison2009':'.nii.gz',
                'ocdOFClPFC_ocdAccdPut':'.nii.gz'}
class Atlaser:
    def __init__(self, atlas='schaefer100_tianS1'):
        """ Constructor: set default global variable for atlas """
        self.atlas_name = atlas
        self.atlas_dir = os.path.join(working_dir, 'lab_lucac/sebastiN/projects/OCDbaseline/utils/')
        self.atlas_cfg = pd.read_json(os.path.join(self.atlas_dir, 'atlas_config.json'))
        self.atlas_img = load_img(os.path.join(self.atlas_dir, self.atlas_name+atlas_suffix[atlas]))
        self.atlas_data = self.atlas_img.get_fdata()
        self.node_ids = np.array(self.atlas_cfg[self.atlas_name]['node_ids'])
        self.node_names = np.array(self.atlas_cfg[self.atlas_name]['node_names'])

    def get_roi_node_indices(self, roi):
        """ Get node indices of ROI based on its name """
        roi_node_ids = np.array([i for i,label in enumerate(self.node_names) if roi in label]).flatten()
        roi_node_ids = np.array(self.node_ids)[roi_node_ids.astype(int)]
        return roi_node_ids

    def get_rois_node_indices(self, rois):
        """ Get node indices of ROIs based on their names"""
        node_ids = np.concatenate([self.get_roi_node_indices(roi=reg) for reg in rois])
        return node_ids

    def get_roi_atlas_indices(self, roi):
        """ Get atlas data indices (i,j,k) of ROI """
        roi_node_ids = self.get_roi_node_indices(roi)
        roi_atlas_ids = np.where(np.any([self.atlas_data==roi_idx for roi_idx in roi_node_ids], axis=0))
        return np.array(roi_atlas_ids).T

    def get_roi_names(self, roi_node_ids):
        """ Get ROI names from their node indices """
        cfg_ids = [np.where(self.node_ids==idx)[0][0] for idx in roi_node_ids]
        names = [self.node_names[i] for i in cfg_ids]
        return names

    def save_new_atlas(new_atlas_data, new_atlas_name, out_dir, new_atlas_suffix='_MNI_lps_mni.nii.gz'):
        """ Save a new atlas """
        new_atlas_img = nib.Nifti1Image(new_atlas_data, self.atlas_img.affine, self.atlas_img.header)
        fname = new_atlas_name + new_atlas_suffix
        nib.save(new_atlas_img, os.path.join(out_dir, fname))
        print('New atlas {} saved in {}'.format(fname, out_dir))
        return fname

    def create_subatlas_img(self, rois):
        if type(rois)!=list:
            rois = [rois]
        new_data = np.zeros(self.atlas_data.shape)
        for roi in rois:
            roi_atlas_ids = self.get_roi_atlas_indices(roi)
            for idx in roi_atlas_ids:
                new_data[tuple(idx)] += self.atlas_data[tuple(idx)]
        new_img = nib.Nifti1Image(new_data, self.atlas_img.affine, self.atlas_img.header)
        return new_img

    def create_brain_map(self, list_node_ids, values):
        """ create a new niftii image with node_ids taking values
            (nodes_ids and values must have same length) """
        if len(list_node_ids)!=len(values):
            print('list_node_ids and values must have the same length')
            sys.exit()
        new_data = np.zeros(self.atlas_data.shape)
        for i,v in zip(list_node_ids, values):
            inds = np.where(self.atlas_data==i)
            new_data[inds] = v
        new_img = nib.Nifti1Image(new_data, self.atlas_img.affine, self.atlas_img.header)
        return new_img
