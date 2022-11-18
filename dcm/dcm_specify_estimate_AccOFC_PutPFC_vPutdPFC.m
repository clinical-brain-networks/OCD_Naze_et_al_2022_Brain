% Script to specify and run DCM for each subject

% get subjects list
proj_dir = '/home/lucac/tmp_working_seb/projects/OCDbaseline/';
subjs = readcell([proj_dir, 'docs/code/subject_list_all.txt']);
n_subjs = size(subjs,1);

loop = 'AccOFC_PutPFC_vPutdPFC_full'; %'vPutdPFC'; %'AccOFC_PutPFC_vPutdPFC'; %'vPutdPFC'; %; %'AccOFC_PutPFC'; %or 'PutPFC' or 'AccOFC'

% load VOIs
if strcmp(loop,'AccOFC')
    VOIs = {'AccR', 'OFC'};
elseif strcmp(loop,'PutPFC')
    VOIs = {'dPutR', 'PFC'};
elseif strcmp(loop, 'vPutdPFC')
    VOIs = {'vPutL', 'dPFC_L'};
    % options
    s = struct();
    s.a = [0,1; 1,0;];
    s.b = [0,0; 0,0;];
    s.c = [0;0;];
    s.d = [];
    % s.u = zeros(t,1); % <-- let it figure out
    s.delays = [0.405; 0.405;];
elseif strcmp(loop, 'AccOFC_PutPFC')
    VOIs = {'AccR', 'OFC', 'dPutR', 'PFC'};
elseif strcmp(loop, 'AccOFC_PutPFC_vPutdPFC')
    VOIs = {'AccR', 'OFC', 'dPutR', 'PFC', 'vPutL', 'dPFC_L'};    
    % options
    s = struct();
    s.a = [0,1,0,0,0,0; 1,0,0,0,0,0; 0,0,0,1,0,0; 0,0,1,0,0,0; 0,0,0,0,0,1; 0,0,0,0,1,0];
    s.b = [0,0,0,0,0,0; 0,0,0,0,0,0; 0,0,0,0,0,0; 0,0,0,0,0,0; 0,0,0,0,0,0; 0,0,0,0,0,0];
    s.c = [0;0;0;0;0;0];
    s.d = [];
    % s.u = zeros(t,1); % <-- let it figure out
    s.delays = [0.405; 0.405; 0.405; 0.405; 0.405; 0.405];
elseif strcmp(loop, 'AccOFC_PutPFC_vPutdPFC_full')
    VOIs = {'AccR', 'OFC', 'dPutR', 'PFC', 'vPutL', 'dPFC_L'};    
    % options
    s = struct();
    s.a = ones(6);
    s.b = [0,0,0,0,0,0; 0,0,0,0,0,0; 0,0,0,0,0,0; 0,0,0,0,0,0; 0,0,0,0,0,0; 0,0,0,0,0,0];
    s.c = [0;0;0;0;0;0];
    s.d = [];
    % s.u = zeros(t,1); % <-- let it figure out
    s.delays = [0.405; 0.405; 0.405; 0.405; 0.405; 0.405];
end

s.TE = 0.03;
s.nonlinear  = false;
s.two_state  = false;
s.stochastic = true;
s.centre     = true;
s.induced    = 0; % <-- rest : not induced response

% Specify and estimate DCM
for s_i=1:n_subjs
    clear SPM DCM xY;
    subj = subjs{s_i};
    s.name = subj;
    disp(['Specify DCM for ' subj]);
    SPM_fname = [proj_dir, 'postprocessing/', subj, '/spm/cluster/SPM.mat'];
    for i=1:numel(VOIs)
        XY = load([proj_dir, 'postprocessing/', subj, '/spm/cluster/VOI_', VOIs{i}, '_1.mat']);
        xY(i) = XY.xY;
    end
    DCM = spm_dcm_specify(SPM_fname, xY, s);
    DCM.name = subj;
    fname = [proj_dir, 'postprocessing/', subj, '/spm/DCM_', loop, '.mat'];
    save(fname, '-struct', 'DCM');

    disp(['Estimate DCM for ' subj]);
    %DCM_fname = [proj_dir, 'postprocessing/', subj, '/spm/DCM_', subj,'.mat'];
    %DCM = load(DCM_fname);
    %DCM.fname
    DCM = spm_dcm_fmri_csd(DCM);

    DCM_outfname = [proj_dir, 'postprocessing/DCM/', loop, '/DCM_', subj,'_output.mat'];
    save(DCM_outfname, '-struct', 'DCM');

end
