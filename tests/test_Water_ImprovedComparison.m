function V = test_Water_ImprovedComparison(outDir)
% Verify measured improvement, prefix isolation, numerical consistency and seeds.
base=fileparts(fileparts(mfilename('fullpath')));addpath(base);setup_water_paths;
if nargin<1||isempty(outDir),outDir=fullfile(base,'results','improved');end
data=load(fullfile(outDir,'comparison.mat'),'R','models');R=data.R;
cfg=R.settings;X=R.features;Y=R.historical_actual;years=R.historical_years;
assert(R.input_workbook_unchanged);
err=model_SmallSample_PSO_BP('check');assert(err<1e-6);
assert(R.improves_all_linear_metrics,'Network must beat all three linear baselines on MAPE/MAE/RMSE/MSE in both protocols.');
assert(R.future_positive&&R.future_water_saving_ordered,'Future direction/positivity check failed.');
assert(R.final_model_parameter_count==6,'Unexpected network architecture.');
assert(all(isfinite(R.fixed_prediction),'all')&&all(isfinite(R.rolling_prediction),'all'));
assert(all(cell2mat(R.tuning_rows(:,5))<cell2mat(R.tuning_rows(:,4))),'Validation labels entered training.');
assert(all(cell2mat(R.tuning_rows(:,4))<=cell2mat(R.tuning_rows(:,2))),'Tuning used a year beyond the current training prefix.');
recomputed=zeros(size(R.fixed_metrics));
for m=1:numel(R.methods)
    e=R.fixed_prediction(:,m)-R.actual;
    recomputed(m,:)=[mean(abs(e)./R.actual)*100,mean(abs(e)),sqrt(mean(e.^2)),mean(e.^2)];
end
assert(max(abs(recomputed-R.fixed_metrics),[],'all')<1e-10);
% New ridge results must reproduce the original workbook, independently of NN.
original=readmatrix(R.input_file,'Sheet','方法比较','UseExcel',false);
original=original(isfinite(original(:,2)),:);
assert(max(abs(R.fixed_metrics(2,:)-original(3,2:5)))<1e-8);
assert(max(abs(R.rolling_metrics(2,:)-original(3,6:9)))<1e-8);
% Perturb all labels from 2021 onward. Earlier forecasts and hyperparameters
% must stay identical. This catches suffix/global scaling and selection leakage.
changedY=Y;changedY(years>=2021)=changedY(years>=2021)+[40;60;80];
changedX=X;changedX(years>=2022,:)=changedX(years>=2022,:)*1.3;
perturbed=water_improved_backtest(changedX,changedY,years,cfg);
assert(max(abs(R.fixed_prediction(1:2,:)-perturbed.fixed_prediction(1:2,:)),[],'all')<1e-10);
assert(max(abs(R.rolling_prediction(1:2,:)-perturbed.rolling_prediction(1:2,:)),[],'all')<1e-10);
assert(isequal(R.lambda_by_training_end(1:2,:),perturbed.lambda_by_training_end(1:2,:)));
% All five predefined seeds are reported, never select the best seed.
seeds=cfg.seed+(0:4)*1000;seedMetrics=zeros(numel(seeds),2);
for k=1:numel(seeds)
    trialCfg=cfg;trialCfg.seed=seeds(k);trial=water_improved_backtest(X,Y,years,trialCfg);
    seedMetrics(k,:)=[trial.fixed_metrics(1,1),trial.rolling_metrics(1,1)];
    assert(trial.improves_all_linear_metrics,'Improvement did not survive a predefined seed.');
end
% Constant targets and constant columns remain finite and reproduce the level.
constantCfg=cfg;constantCfg.penalty=.01;
f=model_SmallSample_PSO_BP('train',ones(6,3),100*ones(6,1),constantCfg,cfg.seed);
p=model_SmallSample_PSO_BP('predict',f,[ones(1,3);2*ones(1,3)]);
assert(all(isfinite(p),'all')&&max(abs(p-100),[],'all')<1e-7);
for m=1:numel(data.models)
    if strcmp(data.models{m}.kind,'small_pso_bp')
        f=data.models{m};assert(f.max_loss_increase<=1e-10);
        assert(all([f.optimization.final_loss]<= [f.optimization.initial_loss]+1e-10));
        assert(~any(strcmp({f.optimization.stop},'max_iterations')),'A final member hit the iteration limit.');
    end
end
V=struct('passed',true,'jacobian_max_error',err,'prefix_isolation_passed',true,...
    'original_ridge_reproduced',true,'metric_recalculation_passed',true,...
    'future_checks_passed',true,'constant_input_target_passed',true,...
    'seeds',seeds,'seed_mape_fixed_rolling',seedMetrics,...
    'independent_holdout_available',false,...
    'scope','只验证现有开发回测的计算、前缀隔离、数值稳定性和相对基准改善；不证明独立泛化优势');
fid=fopen(fullfile(outDir,'verification.json'),'w','n','UTF-8');assert(fid>=0);
guard=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(V,'PrettyPrint',true));clear guard
disp('VERIFICATION PASSED');disp(array2table([seeds',seedMetrics],'VariableNames',{'Seed','Fixed_MAPE','Rolling_MAPE'}));
end
