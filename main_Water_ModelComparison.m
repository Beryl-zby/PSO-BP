function R=main_Water_ModelComparison(inputFile,outDir,makeFigures,legacy)
% Default: regularized BP ensemble without PSO. Fourth argument true selects the archived model.
setup_water_paths;
if nargin<1,inputFile=[];end
if nargin<2,outDir=[];end
if nargin<3,makeFigures=true;end
if nargin<4,legacy=false;end
if legacy
    R=main_Water_LegacyComparison(inputFile,outDir,makeFigures);
else
    R=run_BP_Ensemble(inputFile,makeFigures,outDir);
end
end
