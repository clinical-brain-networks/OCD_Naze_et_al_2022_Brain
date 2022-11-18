%-----------------------------------------------------------------------
% Job saved on 17-Feb-2022 19:04:51 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

% get subjects list
proj_dir = '/home/lucac/tmp_working_seb/projects/OCDbaseline/';
subjs = readcell([proj_dir, 'docs/code/subject_list_extras.txt']);
revoked = ["sub-patient16"];
subjs = subjs(~contains(subjs,revoked));
n_subjs = size(subjs,1);
n_scans = 880;

for sub_i=1:n_subjs
    %subj = 'sub-control04';
    subj=subjs{sub_i};
    spec_dir = [proj_dir, 'postprocessing/',subj,'/spm'];

    % create scans list
    scan_files = cell(0,1);
    for i = 0:n_scans-1
      fname = [spec_dir, '/scans/detrend_gsr_smooth6mm/', subj, '_task-rest_space-MNI152NLin2009cAsym_desc-detrend_gsr_smooth-6mm', sprintf('%04i',i), '.nii,1'];
      scan_files{i+1,1} = fname;
    end

    %% Matlabbatch
    matlabbatch{1}.spm.stats.fmri_spec.dir = {spec_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'scans';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 0.81;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
    matlabbatch{1}.spm.stats.fmri_spec.sess.scans = scan_files;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
    matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
    %matlabbatch{3}.spm.stats.con.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    %matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'rest';
    %matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = ones(n_scans,1);
    %matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    %matlabbatch{3}.spm.stats.con.delete = 1;
    matlabbatch{3}.spm.util.voi.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    matlabbatch{3}.spm.util.voi.adjust = 0;
    matlabbatch{3}.spm.util.voi.session = 1;
    matlabbatch{3}.spm.util.voi.name = 'OFC';
    %matlabbatch{3}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/OFC_R.nii,1'};
    %matlabbatch{3}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/VOI_OFC_sphere.nii,1'};
    matlabbatch{3}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/VOI_OFC_R_sphere_5mm.nii,1'};
    matlabbatch{3}.spm.util.voi.roi{1}.mask.threshold = 0.5;
    matlabbatch{3}.spm.util.voi.expression = 'i1';
    matlabbatch{4}.spm.util.voi.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    matlabbatch{4}.spm.util.voi.adjust = 0;
    matlabbatch{4}.spm.util.voi.session = 1;
    matlabbatch{4}.spm.util.voi.name = 'AccR';
    %matlabbatch{4}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/AccR_seed.nii,1'};
    %matlabbatch{4}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/VOI_Acc_cluster.nii,1'};
    matlabbatch{4}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/VOI_AccR_sphere_5mm.nii,1'};
    matlabbatch{4}.spm.util.voi.roi{1}.mask.threshold = 0.5;
    matlabbatch{4}.spm.util.voi.expression = 'i1';
    matlabbatch{5}.spm.util.voi.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    matlabbatch{5}.spm.util.voi.adjust = 0;
    matlabbatch{5}.spm.util.voi.session = 1;
    matlabbatch{5}.spm.util.voi.name = 'dPutR';
    %matlabbatch{5}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/dPutR_seed.nii,1'};
    %matlabbatch{5}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/VOI_dPut_cluster.nii,1'};
    matlabbatch{5}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/VOI_dPutR_sphere_5mm.nii,1'};
    matlabbatch{5}.spm.util.voi.roi{1}.mask.threshold = 0.5;
    matlabbatch{5}.spm.util.voi.expression = 'i1';
    matlabbatch{6}.spm.util.voi.spmmat(1) = cfg_dep('Model estimation: SPM.mat File', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    matlabbatch{6}.spm.util.voi.adjust = 0;
    matlabbatch{6}.spm.util.voi.session = 1;
    matlabbatch{6}.spm.util.voi.name = 'PFC';
    %matlabbatch{6}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/PFC_R.nii,1'};
    %matlabbatch{6}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/VOI_PFC_sphere.nii,1'};
    matlabbatch{6}.spm.util.voi.roi{1}.mask.image = {'/home/lucac/tmp_working_seb/projects/OCDbaseline/postprocessing/SPM/seeds_and_rois/VOI_PFC_R_sphere_5mm.nii,1'};
    matlabbatch{6}.spm.util.voi.roi{1}.mask.threshold = 0.5;
    matlabbatch{6}.spm.util.voi.expression = 'i1';
    try
        spm_jobman('run', matlabbatch);
    catch
        warning([subj, ' SPM first level did not pass!']);
    end
    clearvars matlabbatch;
end
