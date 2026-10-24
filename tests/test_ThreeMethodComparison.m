function V=test_ThreeMethodComparison()
% Validate the comparison and the saved three distinct model implementations.
base=fileparts(fileparts(mfilename('fullpath')));addpath(base);setup_water_paths;D=water_suite_data;
saved=load(fullfile(base,'results','three_methods','comparison.mat'),'S');S=saved.S;
assert(strcmp(S.input_sha256,D.source_sha256),'Source workbook changed.');
families={'pso','bayes','gpr'};predictionError=zeros(3,1);observations=zeros(3,5);
for m=1:7
    R=S.results{m};
    for mode=1:2
        if mode==1,p=R.fixed_prediction;stored=R.fixed_metrics;else,p=R.rolling_prediction;stored=R.rolling_metrics;end
        e=p-R.actual;actual=[mean(abs(e)./R.actual)*100,mean(abs(e)),sqrt(mean(e.^2)),mean(e.^2)];
        assert(max(abs(actual-stored))<1e-9&&all(isfinite(p)));
    end
    assert(all(cell2mat(R.tuning_rows(:,4))<cell2mat(R.tuning_rows(:,3))));
    assert(all(cell2mat(R.tuning_rows(:,3))<=cell2mat(R.tuning_rows(:,1))));
end
for m=1:3
    [spec,cfg]=water_suite_setup(families{m});cfg.seed=D.seed;
    loaded=load(fullfile(spec.path,'results','result.mat'),'R','models');R=loaded.R;models=loaded.models;
    assert(strcmp(R.input_sha256,D.source_sha256)&&R.input_workbook_unchanged);
    assert(max(abs(R.fixed_prediction-S.fixed_prediction(:,m)))<1e-10);
    assert(max(abs(R.rolling_prediction-S.rolling_prediction(:,m)))<1e-10);
    stages=[{models.fixed},models.rolling(2:4),{models.final}];counts=[10,11,12,13,14];
    for k=1:5
        f=stages{k};observations(m,k)=f.training_rows;assert(f.training_rows==counts(k));
        center=f.xcenter;scale=f.xscale;
        expected=(min(D.X(1:counts(k),:))+max(D.X(1:counts(k),:)))/2;
        assert(max(abs(center-expected))<1e-10&&all(scale>0));
        if m==2
            assert(numel(f.networks)==cfg.members);
            assert(all(cellfun(@(net)strcmp(net.divideFcn,'dividetrain'),f.networks)));
        elseif m==3
            assert(f.gp.NumObservations==counts(k));
        end
    end
    f=models.fixed;mu=feval(spec.model,'predict',f,D.X(11:14,:));b=feval(spec.model,'predict',f,D.X(10,:));
    p=D.Y(10)+mu-b;predictionError(m)=max(abs(p-R.fixed_prediction));assert(predictionError(m)<1e-9);
    assert(all(isfinite(R.future_prediction),'all')&&all(isfinite(R.response_prediction),'all'));
    if m==3,assert(all(isfinite(R.gpr_raw_future_sd),'all')&&all(R.gpr_raw_future_sd>=0,'all'));end
end
% New shared protocol must reproduce the already verified PSO and linear baselines.
prior=load(fullfile(base,'results','improved','comparison.mat'),'R');P=prior.R;
indices=[1,4,5,6,7];
assert(max(abs(S.fixed_prediction(:,indices)-P.fixed_prediction),[],'all')<1e-8);
assert(max(abs(S.rolling_prediction(:,indices)-P.rolling_prediction),[],'all')<1e-8);
% Exercise prefix isolation once through the shared engine, not a second full
% expensive training of every family. All adapters receive these sliced rows.
[spec,cfg]=water_suite_setup('pso');cfg.seed=D.seed;y=D.Y;y(12:14)=y(12:14)+[40;60;80];
changed=water_method_backtest(D.X,y,D.years,spec,cfg);
assert(max(abs(changed.fixed_prediction-S.fixed_prediction(:,1)))<1e-10);
assert(max(abs(changed.rolling_prediction(1:2)-S.rolling_prediction(1:2,1)))<1e-10);
assert(isequal(changed.selected(1:2),S.results{1}.selected(1:2)));
V=struct('passed',true,'all_56_metrics_recomputed',true,...
    'saved_models_reproduce_predictions',true,'saved_prediction_max_error',predictionError,...
    'training_observation_counts',observations,'prefix_isolation_passed',true,...
    'prior_pso_and_linear_baselines_reproduced',true,'source_workbook_unchanged',true,...
    'gpr_raw_sd_finite',true,'independent_holdout_available',false);
fid=fopen(fullfile(base,'results','three_methods','verification.json'),'w','n','UTF-8');assert(fid>=0);
guard=onCleanup(@()fclose(fid));fprintf(fid,'%s',jsonencode(V,'PrettyPrint',true));clear guard
disp('THREE-METHOD VERIFICATION PASSED');
end
