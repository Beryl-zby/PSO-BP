function [R,models]=water_method_backtest(X,Y,years,spec,cfg)
% Shared time splits; BP selects regularization and calibration inside each prefix.
Y=Y(:);years=years(:);assert(isequal(years,(2010:2023)'));
v=find(years>2019);cache=containers.Map('KeyType','char','ValueType','any');
R=struct('method',spec.name,'family',spec.id,'description',spec.description,...
    'validation_years',years(v),'actual',Y(v),'fixed_prediction',zeros(4,1),...
    'rolling_prediction',zeros(4,1),'settings',cfg,'candidates',{spec.candidates},...
    'training_end_years',[years(v)-1;2023],'selected',{{}},'inner_mape',zeros(5,1),...
    'tuning_rows',{{}},'training_rows',{{}},...
    'validation_scope','同一2020—2023开发回测，已参与模型探索；不是新增独立测试集');
models=struct('fixed',[],'rolling',{{}},'final',[]);tic;
if isfield(spec.candidates{1},'calibration')
    R.model_version='regularized-bp-v2';
    R.calibration_policy='last / none / median3；与正则参数一起在训练末3年扩展验证选择；不读取当前检验目标';
end
for fold=1:5
    if fold<=4,n=v(fold)-1;else,n=numel(Y);end
    scores=zeros(numel(spec.candidates),1);
    for c=1:numel(scores)
        errors=zeros(3,1);
        for j=1:3
            at=n-3+j;f=getFit(at-1,c);
            p=water_predict_method(spec.model,f,X(at,:),X(at-1,:),Y(at-1));
            errors(j)=100*abs(p-Y(at))/Y(at);
            R.tuning_rows(end+1,:)={years(n),spec.candidates{c}.label,years(at),years(at-1),p,Y(at),errors(j)};
        end
        scores(c)=mean(errors);
    end
    [R.inner_mape(fold),best]=min(scores);R.selected{fold}=spec.candidates{best};
    f=getFit(n,best);
    if fold<=4
        R.rolling_prediction(fold)=water_predict_method(spec.model,f,X(v(fold),:),X(n,:),Y(n));models.rolling{fold}=f;
        if fold==1,R.fixed_prediction=water_predict_method(spec.model,f,X(v,:),X(n,:),Y(n));models.fixed=f;end
    else
        models.final=f;
    end
    trainP=feval(spec.model,'predict',f,X(1:n,:));e=trainP-Y(1:n);
    R.training_rows(end+1,:)={years(n),spec.candidates{best}.label,mean(abs(e)./Y(1:n))*100,sqrt(mean(e.^2))};
    fprintf('%s | train through %d | %s | inner MAPE %.4f%%\n',spec.id,years(n),spec.candidates{best}.label,R.inner_mape(fold));
end
R.fixed_metrics=metrics(R.actual,R.fixed_prediction);R.rolling_metrics=metrics(R.actual,R.rolling_prediction);
R.score=mean([R.fixed_metrics(1),R.rolling_metrics(1)]);R.training_seconds=toc;
R.metric_columns={'MAPE_percent','MAE','RMSE','MSE'};

    function model=getFit(n,c)
        candidate=spec.candidates{c};
        if isfield(candidate,'calibration'),key=sprintf('%d/lambda/%.17g',n,candidate.penalty);
        else,key=sprintf('%d/%d',n,c);end
        if isKey(cache,key),model=cache(key);
        else
            model=feval(spec.model,'train',X(1:n,:),Y(1:n),candidate,cfg);
            assert(model.training_rows==n);cache(key)=model;
        end
        if isfield(candidate,'calibration')
            model=water_calibrate_model(model,spec.model,X(1:n,:),Y(1:n),candidate.calibration);
        end
    end
end

function m=metrics(y,p)
e=p-y;m=[mean(abs(e)./abs(y))*100,mean(abs(e)),sqrt(mean(e.^2)),mean(e.^2)];
end
