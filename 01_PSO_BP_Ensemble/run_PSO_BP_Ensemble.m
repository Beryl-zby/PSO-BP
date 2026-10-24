function R=run_PSO_BP_Ensemble(inputFile,makeFigures)
base=fileparts(fileparts(mfilename('fullpath')));addpath(base);setup_water_paths;
if nargin<1,inputFile=[];end
if nargin<2,makeFigures=true;end
R=water_run_method('pso',inputFile,makeFigures);
end
