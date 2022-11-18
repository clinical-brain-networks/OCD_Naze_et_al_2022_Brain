#!/bin/bash

##########################################################################
#
#  Script:    qsiprep_parallel.pbs
#  Author:    Luke Hearne, Sebastien Naze
#  Created:   2021-08
#
##########################################################################

#PBS -N qsiprep_parallel
#PBS -l ncpus=16,mem=48gb,walltime=48:00:00
#PBS -m abe
#PBS -M sebastien.naze@qimrberghofer.edu.au
#PBS -o /working/lab_lucac/sebastiN/projects/OCDbaseline/.pbs_logs/
#PBS -e /working/lab_lucac/sebastiN/projects/OCDbaseline/.pbs_logs/

# add paths to code and subject list
export PATH="$PATH:/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/docs/code/"
export PATH="$PATH:/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/docs/code/preprocessing/"

#----.sh script
# qsiprep container
qsip_con=/mnt/lustre/working/lab_lucac/shared/x_qsiprep_versions/qsiprep-0.13.0.sif

# path
lukeH_project_dir=/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/
sebN_project_dir=/mnt/lustre/working/lab_lucac/sebastiN/projects/OCDbaseline/
bids_dir=${lukeH_project_dir}data/bids/
out_dir=${sebN_project_dir}data/derivatives/
work_dir=${sebN_project_dir}data/scratch/qsiprep/
atlas_dir=/mnt/lustre/working/lab_lucac/shared/parcellations/qsirecon_atlases_with_subcortex/

# load specific subject from subject list
mapfile -t subj_array < ${sebN_project_dir}docs/code/subject_list.txt
IDX=$((PBS_ARRAY_INDEX-1))  # double parenthesis needed for arithmetic operations
subj=${subj_array[$IDX]}
echo "Current subject: " ${subj}

# load modules
module load singularity/3.3.0
module load python/3.6.1

# run proxy script for internet access
#source ~/.proxy

# analysis pipeline recon file
recon_file=/mnt/lustre/working/lab_lucac/sebastiN/projects/OCDbaseline/docs/code/preprocessing/qsiprep_recon_file_100M_seeds.json
# run qsiprep
# - these commands are no longer in qsiprep
#--force-spatial-normalization \
#--combine-all-dwis \
#--denoise-before-combining \
# --custom_atlases ${atlas_dir} \  # removed for other reasons
# --prefer_dedicated_fmaps \

# First pass for probabilistic tractography

qsiprep-singularity ${bids_dir} ${out_dir} participant -i ${qsip_con} \
--participant-label ${subj} \
--dwi_denoise_window 5 \
--output-space T1w \
--output-resolution 2 \
--hmc-model eddy \
--fs-license-file /software/freesurfer/freesurfer-6.0.1/license.txt \
--custom_atlases ${atlas_dir} \
--recon-spec ${recon_file} \
--mem-mb 48000 \
--nthreads 32 \
--work-dir ${work_dir} \
--verbose \
| tee ${sebN_project_dir}data/scratch/.${subj}.log


# Second pass for DSI metrics
recon_file=dsi_studio_gqi

qsiprep-singularity ${bids_dir} ${out_dir} participant -i ${qsip_con} \
--participant-label ${subj} \
--dwi_denoise_window 5 \
--output-space T1w \
--output-resolution 2 \
--hmc-model eddy \
--fs-license-file /software/freesurfer/freesurfer-6.0.1/license.txt \
--custom_atlases ${atlas_dir} \
--recon-only \
--recon-input ${out_dir}qsiprep/ \
--recon-spec ${recon_file} \
--mem-mb 48000 \
--nthreads 32 \
--work-dir ${work_dir} \
--verbose
