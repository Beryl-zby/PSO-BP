function result=model_PSO_BP_Ensemble(action,varargin)
% Adapter to the verified PSO-BP implementation in this method directory.
switch lower(action)
    case 'train'
        X=varargin{1};Y=varargin{2};candidate=varargin{3};cfg=varargin{4};
        cfg.hidden=1;cfg.bias_penalty=1;cfg.penalty=candidate.penalty;
        result=model_SmallSample_PSO_BP('train',X,Y,cfg,cfg.seed);
        result.training_rows=size(X,1);
    case 'predict'
        result=mean(model_SmallSample_PSO_BP('predict',varargin{1},varargin{2}),2);
    otherwise
        error('Unknown action: %s',action);
end
end
