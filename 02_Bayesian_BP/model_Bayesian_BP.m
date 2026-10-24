function result=model_Bayesian_BP(action,varargin)
% MATLAB trainbr: evidence-based weight/bias regularization, not a full BNN.
switch lower(action)
    case 'train'
        X=varargin{1};Y=varargin{2};candidate=varargin{3};cfg=varargin{4};Y=Y(:);
        result.kind='bayesian_regularized_bp';result.hidden=candidate.hidden;
        result.training_rows=size(X,1);result.xcenter=(min(X)+max(X))/2;
        result.xscale=max(X)-min(X);result.xscale(result.xscale<eps)=1;
        result.ycenter=mean(Y);result.yscale=max(max(Y)-min(Y),1e-8);
        z=(X-result.xcenter)./result.xscale;y=(Y-result.ycenter)/result.yscale;
        result.networks=cell(1,cfg.members);result.stops=cell(1,cfg.members);
        result.epochs=zeros(1,cfg.members);result.best_performance=zeros(1,cfg.members);
        old=rng;guard=onCleanup(@()rng(old));
        for m=1:cfg.members
            rng(mod(cfg.seed+m,2^32),'twister');net=fitnet(candidate.hidden,'trainbr');
            net.inputs{1}.processFcns={};net.outputs{2}.processFcns={};
            net.divideFcn='dividetrain';net.trainParam.showWindow=false;
            net.trainParam.showCommandLine=false;net.trainParam.epochs=cfg.bayes_epochs;
            [net,tr]=train(net,z',y');
            result.networks{m}=net;result.stops{m}=tr.stop;
            result.epochs(m)=tr.epoch(end);result.best_performance(m)=tr.best_perf;
        end
        clear guard
    case 'predict'
        model=varargin{1};X=varargin{2};z=(X-model.xcenter)./model.xscale;
        values=zeros(size(X,1),numel(model.networks));
        for m=1:numel(model.networks),values(:,m)=model.networks{m}(z')';end
        result=model.ycenter+model.yscale*mean(values,2);
    otherwise
        error('Unknown action: %s',action);
end
end
