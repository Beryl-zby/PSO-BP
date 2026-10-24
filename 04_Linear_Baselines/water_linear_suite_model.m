function result=water_linear_suite_model(action,varargin)
switch lower(action)
    case 'train'
        X=varargin{1};Y=varargin{2};candidate=varargin{3};
        if isfield(candidate,'naive')
            result=struct('kind','persistence');
        else
            result=model_Ridge('train',X,Y,candidate.penalty);
        end
        result.training_rows=size(X,1);
    case 'predict'
        model=varargin{1};X=varargin{2};
        if strcmp(model.kind,'persistence'),result=zeros(size(X,1),1);
        else,result=model_Ridge('predict',model,X);end
    otherwise
        error('Unknown action.');
end
end
