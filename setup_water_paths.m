function root=setup_water_paths()
% Add the method, shared-code and test directories independently of cwd.
root=fileparts(mfilename('fullpath'));
folders={'00_Legacy_PSO_BP','01_PSO_BP_Ensemble','02_Bayesian_BP',...
    '03_Gaussian_Process','04_Linear_Baselines','shared','tests'};
addpath(root);
for k=1:numel(folders),addpath(fullfile(root,folders{k}));end
end
