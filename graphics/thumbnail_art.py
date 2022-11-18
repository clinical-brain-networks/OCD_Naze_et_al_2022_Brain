import matplotlib
from matplotlib import pyplot as plt
import nilearn
import numpy as np
import os
import pyvista as pv
import pymeshfix as mf
import sys

proj_dir = '/home/sebastin/working/lab_lucac/sebastiN/projects/OCDbaseline'
code_dir = os.path.join(proj_dir, 'docs/code')
atlas_dir = os.path.join(proj_dir, 'utils')
fs_dir = '/usr/local/freesurfer/'

sys.path.insert(0, os.path.join(code_dir, 'utils'))
import atlaser

sys.path.insert(0, os.path.join(code_dir, 'graphics'))
import baseline_visuals


cortex = ['Right_Vis', 'Right_SomMot', 'Right_DorsAttn', 'Right_SalVentAttn', 'Right_Limbic', 'Right_Cont', 'Right_Default', 'Right_TempPar']
regions = ['Right_Hippocampus', 'Right_Amygdala', 'Right_PosteriorThalamus', 'Right_AnteriorThalamus', 'Right_NucleusAccumbens', 'Right_GlobusPallidus', 'Right_Putamen', 'Right_Caudate']

opts = {'Right_Putamen': {'color': 'RoyalBlue',
                          'show_edges': False,
                          'line_width': 1,
                          'opacity':1},
        'Right_NucleusAccumbens': {   'color': 'firebrick',
                                      'show_edges': False,
                                      'line_width': 1,
                                      'opacity':1},
        'Right_Hippocampus': {'color': 'orange',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity': 0.2,
                          'style': 'wireframe'},
        'Right_Amygdala': {'color': 'purple',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity':0.2,
                          'style': 'wireframe'},
        'Right_GlobusPallidus': {'color': 'gray',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity':0.2,
                          'style': 'wireframe'},
        'Right_AnteriorThalamus': {'color': 'black',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity':0.2,
                          'style': 'wireframe'},
        'Right_PosteriorThalamus': {'color': 'black',
                          'show_edges': True,
                          'line_width': 0.1,
                          'opacity':0.2,
                          'style': 'wireframe'},
        'Right_Caudate': {'color': 'green',
                          'show_edges': True,
                          'line_width': 0.5,
                          'opacity':0.5,
                          'style': 'wireframe'},
        'cortex':        {  'color': 'lightslategray',
                            'show_edges': True,
                            'line_width': 0.2,
                            'opacity': 0.2,
                            'style': 'wireframe'}
       }


def create_mesh(roi_img, alpha=1., n_iter=80):
    """ creates pyvista mesh from all non-zero entries of a niftii image """
    # create ROI point cloud
    ijk = np.array(np.where(roi_img.get_fdata()>0))
    xyz_coords = nilearn.image.coord_transform(ijk[0], ijk[1], ijk[2], affine=roi_img.affine)
    #roi_points = pv.PointSet(np.array(xyz_coords))
    roi = pv.PolyData(np.array(xyz_coords).T)

    # extract mesh from point cloud
    mesh = roi.delaunay_3d(alpha=alpha).extract_geometry().clean()

    # repair mesh for inconsistencies
    mesh = mf.MeshFix(mesh)
    mesh.repair()
    mesh = mesh.mesh

    # smooth mesh (use Taubin to keep volume)
    mesh.smooth_taubin(n_iter=n_iter, inplace=True)
    return mesh

# Additionally, all the modules other than ipygany and pythreejs require a framebuffer, which can be setup on a headless environment with pyvista.start_xvfb().
pv.start_xvfb()

atlazer = atlaser.Atlaser('schaefer100_tianS1')

pl = pv.Plotter(window_size=[800,600])
pl.background_color='white'

# subcortex
for region in regions:
    # get nifti img from region(s)
    roi_img = atlazer.create_subatlas_img(region)
    mesh = create_mesh(roi_img)
    # plot
    pl.add_mesh(mesh, **opts[region])

pl.show(jupyter_backend='panel')
