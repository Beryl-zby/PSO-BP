function R=run_BP_Ensemble(inputFile,makeFigures,outDir)
% 当前主模型：随机初始化、3-1-1正则BP、12成员集成；不运行PSO。
base=fileparts(fileparts(mfilename('fullpath')));addpath(base);setup_water_paths;
if nargin<1,inputFile=[];end
if nargin<2,makeFigures=true;end
if nargin<3,outDir=[];end
R=water_run_method('bp',inputFile,makeFigures,outDir);
end
