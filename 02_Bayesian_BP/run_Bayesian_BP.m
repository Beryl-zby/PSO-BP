function R=run_Bayesian_BP(inputFile,makeFigures)
base=fileparts(fileparts(mfilename('fullpath')));addpath(base);setup_water_paths;
if nargin<1,inputFile=[];end
if nargin<2,makeFigures=true;end
R=water_run_method('bayes',inputFile,makeFigures);
end
