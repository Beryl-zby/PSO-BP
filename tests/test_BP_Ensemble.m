function V=test_BP_Ensemble()
% Exercise calibration, nested time selection and the saved forecast consumer.
root=fileparts(fileparts(mfilename('fullpath')));addpath(root);setup_water_paths;
D=water_suite_data;out=tempname;mkdir(out);guard=onCleanup(@()rmdir(out,'s'));
R=run_BP_Ensemble([],false,out);saved=load(fullfile(out,'result.mat'),'models');M=saved.models;
assert(strcmp(R.model_version,'regularized-bp-v2'));
assert(model_BP_Ensemble('check')<1e-6);
% A single large residual must not be passed through by the median correction.
raw=rmfield(M.final,'prediction_calibration');xx=D.X(end-2:end,:);
yy=model_BP_Ensemble('predict',raw,xx)+[1;1;90];
for item={'none',0;'last',90;'median3',1}'
    f=water_calibrate_model(raw,'model_BP_Ensemble',xx,yy,item{1});
    assert(abs(f.prediction_calibration.offset-item{2})<1e-10);
    pp=water_predict_method('model_BP_Ensemble',f,xx,xx(end,:),-999);
    assert(max(abs(pp-model_BP_Ensemble('predict',raw,xx)-item{2}))<1e-10);
end
old=water_predict_method('model_BP_Ensemble',raw,xx,xx(end,:),yy(end));
expected=yy(end)+model_BP_Ensemble('predict',raw,xx)-model_BP_Ensemble('predict',raw,xx(end,:));
assert(max(abs(old-expected))<1e-10);
% Perturb later targets: fixed predictions and earlier selections must not change.
[spec,cfg]=water_suite_setup('bp');cfg.seed=D.seed;y=D.Y;y(12:14)=y(12:14)+[40;60;80];
changed=water_method_backtest(D.X,y,D.years,spec,cfg);
assert(max(abs(changed.fixed_prediction-R.fixed_prediction))<1e-10);
assert(max(abs(changed.rolling_prediction(1:2)-R.rolling_prediction(1:2)))<1e-10);
assert(isequal(changed.selected(1:2),R.selected(1:2)));
assert(all(cell2mat(R.tuning_rows(:,4))<cell2mat(R.tuning_rows(:,3))));
assert(all(cell2mat(R.tuning_rows(:,3))<=cell2mat(R.tuning_rows(:,1))));
stages=[{M.fixed},M.rolling(2:4),{M.final}];counts=[10,11,12,13,14];
for k=1:5
    f=stages{k};assert(f.prediction_calibration.training_rows==counts(k));
    assert(strcmp(f.prediction_calibration.mode,R.selected{k}.calibration));
    assert(max(abs(f.xcenter-(min(D.X(1:counts(k),:))+max(D.X(1:counts(k),:)))/2))<1e-10);
    rows=R.tuning_rows(cell2mat(R.tuning_rows(:,1))==R.training_end_years(k),:);
    scores=zeros(1,numel(spec.candidates));
    for c=1:numel(scores)
        scores(c)=mean(cell2mat(rows(strcmp(rows(:,2),spec.candidates{c}.label),7)));
    end
    [value,best]=min(scores);assert(abs(value-R.inner_mape(k))<1e-10);
    assert(isequal(spec.candidates{best},R.selected{k}));
end
for mode=1:2
    if mode==1,p=R.fixed_prediction;stored=R.fixed_metrics;else,p=R.rolling_prediction;stored=R.rolling_metrics;end
    e=p-R.actual;assert(max(abs(stored-[mean(abs(e)./R.actual)*100,mean(abs(e)),sqrt(mean(e.^2)),mean(e.^2)]))<1e-10);
end
for s=1:5
    for j=1:3
        p=water_predict_method('model_BP_Ensemble',M.final,D.future_X{s,j},D.X(end,:),D.Y(end));
        assert(max(abs(p-R.future_prediction(:,s,j)))<1e-10);
    end
end
assert(all(isfinite(R.future_prediction),'all')&&R.future_checks_passed);
after=water_suite_data;assert(strcmp(D.source_sha256,after.source_sha256));
V=struct('passed',true,'prefix_isolation',true,'calibration_offsets',true,...
    'nested_selection',true,'saved_forecasts_reproduced',true,'input_unchanged',true);
disp(V);clear guard
end
