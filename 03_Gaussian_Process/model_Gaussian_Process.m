function [result,sd]=model_Gaussian_Process(action,varargin)
% Exact GPR: training-only marginal-likelihood fit, prefix-CV kernel/basis choice.
sd=[];
switch lower(action)
    case 'train'
        X=varargin{1};Y=varargin{2};candidate=varargin{3};Y=Y(:);
        result.kind='gaussian_process';result.training_rows=size(X,1);
        result.xcenter=(min(X)+max(X))/2;
        result.xscale=max(X)-min(X);result.xscale(result.xscale<eps)=1;
        result.ycenter=mean(Y);result.yscale=max(max(Y)-min(Y),1e-8);
        z=(X-result.xcenter)./result.xscale;y=(Y-result.ycenter)/result.yscale;
        result.kernel=candidate.kernel;result.basis=candidate.basis;
        result.gp=fitrgp(z,y,'KernelFunction',candidate.kernel,...
            'BasisFunction',candidate.basis,'FitMethod','exact','PredictMethod','exact',...
            'Standardize',false,'Sigma',.1,'SigmaLowerBound',.01);
        result.kernel_information=result.gp.KernelInformation;
        result.noise_sigma=result.gp.Sigma*result.yscale;
    case 'predict'
        model=varargin{1};X=varargin{2};z=(X-model.xcenter)./model.xscale;
        [mu,s]=predict(model.gp,z);
        result=model.ycenter+model.yscale*mu;sd=model.yscale*s;
    otherwise
        error('Unknown action: %s',action);
end
end
