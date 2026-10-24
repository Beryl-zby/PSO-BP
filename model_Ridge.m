function result=model_Ridge(action,varargin)
% 岭回归子程序。训练期Min-Max归一化，截距不惩罚；正则化系数由主程序内验证选择。
switch lower(action)
 case 'train'
  X=varargin{1};Y=varargin{2};lambda=varargin{3};
  result.kind='ridge';result.xmin=min(X);result.xscale=max(X)-min(X);assert(all(result.xscale>0));
  result.ymin=min(Y);result.yscale=max(Y)-min(Y);
  xx=(X-result.xmin)./result.xscale;yn=(Y-result.ymin)/result.yscale;
  z=[ones(size(X,1),1),xx];n=size(X,1);
  result.theta=(z'*z/n+lambda*diag([0,1,1,1]))\(z'*yn/n);
 case 'predict'
  model=varargin{1};X=varargin{2};
  result=model.ymin+model.yscale*[ones(size(X,1),1),(X-model.xmin)./model.xscale]*model.theta;
 otherwise,error('未知操作：%s',action);
end
end
