%-----------------------------------------------------------------------
% Job saved on 17-Feb-2022 19:04:51 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

metric = 'detrend_gsr_filtered_scrubFD05';

% get subjects list
proj_dir = '/home/lucac/tmp_working_seb/projects/OCDbaseline/';
subjs = readcell([proj_dir, 'docs/code/subject_list_all.txt']);
%revoked = ["sub-patient16"];
%subjs = subjs(~contains(subjs, revoked));
n_subjs = numel(subjs);
n_scans = 880;

VOIs = ["vPutL"];

for j=1:numel(subjs)
    subj = subjs{j};
    spec_dir = [proj_dir, 'postprocessing/',subj,'/spm'];
    seeds_dir = [proj_dir, 'postprocessing/SPM/seeds_and_rois/']; 
    spm_file = [spec_dir, '/cluster/SPM.mat'];
    
    for i=1:numel(VOIs)
        VOI = VOIs{i};
        %% Matlabbatch 
        matlabbatch{i}.spm.util.voi.spmmat = {spm_file};
        matlabbatch{i}.spm.util.voi.adjust = 0;
        matlabbatch{i}.spm.util.voi.session = 1;
        matlabbatch{i}.spm.util.voi.name = VOI;
        %matlabbatch{i}.spm.util.voi.roi{1}.mask.image = {[spec_dir, '/masks/local_', VOI, '_', metric, '.nii',',1']};
        matlabbatch{i}.spm.util.voi.roi{1}.mask.image = {[seeds_dir, VOI, '_seed.nii',',1']};
        matlabbatch{i}.spm.util.voi.roi{1}.mask.threshold = 0.5;
        matlabbatch{i}.spm.util.voi.expression = 'i1';
    end
    
    try
        spm_jobman('run', matlabbatch);
    catch
        warning([subj, ' SPM first level did not pass!']);
    end
    clearvars matlabbatch;
end
