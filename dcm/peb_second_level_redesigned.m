% Script using redesigned matrix centered on controls

% get subjects list
proj_dir = '/home/lucac/tmp_working_seb/projects/OCDbaseline/';
subjs = readcell([proj_dir, 'docs/code/subject_list_all.txt']);
revoked = ["sub-patient16"];
inds = find(~contains(subjs, revoked));
subjs = subjs(inds);
n_subjs = numel(subjs);
n_con = sum(contains(subjs, 'control')); % number of controls


fit_dcm = 0;
load_gcm = 0;
save_gcm = 1;

% set prereq
loop = 'AccOFC_PutPFC_vPutdPFC_full'; %'vPutdPFC'; %'PutPFC'; %'AccOFC_PutPFC_vPutdPFC';

if load_gcm
    tmp = load([proj_dir, 'postprocessing/DCM/', loop, '/GCM.mat']);
    GCM = tmp.GCM;
    clear tmp;
else
    % prepare GCM
    DCMs = {};
    for i=1:n_subjs
        subj = subjs{i};
        dcm_file = [proj_dir, 'postprocessing/DCM/', loop, '/DCM_', subj, '_output.mat'];
        DCM = load(dcm_file);
        DCMs{i} = DCM;
        GCM = DCMs;
        if save_gcm
            save([proj_dir, 'postprocessing/DCM/', loop, '/GCM.mat'], 'GCM', '-v7.3');
        end
    end
    if fit_dcm
        % Estimate each subject model
        GCM = spm_dcm_fit(DCMs);
        save([proj_dir, 'postprocessing/DCM/', loop, '/GCM.mat'], 'GCM', '-v7.3');
    end    
end


% Specify PEB model settings
% The 'all' option means the between-subject variability of each connection will
% be estimated individually
M   = struct();
M.Q = 'all';

% Specify design matrix for N subjects. It should start with a constant column
M.X = ones(n_subjs,2);
M.X(n_con+1:end,2) = -1;
%M.X(1:n_con,3) = 0;

% mean-centre regressor (commonalities = mean, differences =  +/- deviations from the mean)
M.X(:,2) = M.X(:,2) - mean(M.X(:,2));

% name regressors
M.Xnames = {'Mean', 'between-group'};

% Choose field
field = {'A'};

% Estimate model
PEB = spm_dcm_peb(GCM',M,field);

fname = [proj_dir, 'postprocessing/DCM/', loop, '/PEB_second_level_REREDESIGNED_all.mat'];
save(fname,'PEB');

% Bayesian Model Reduction/Averaging
BMA = spm_dcm_peb_bmc(PEB);

% Visualize outputs
spm_dcm_peb_review(BMA);