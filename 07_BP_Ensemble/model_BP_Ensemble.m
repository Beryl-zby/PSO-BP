function result = model_BP_Ensemble(action,varargin)
% 正则BP集成：去掉PSO，仅随机初始化；其余网络、正则和局部求解保持一致。
% train(X,Y,candidate,cfg), predict(model,X), check().
% BP指反向传播解析Jacobian配合阻尼高斯牛顿；不是另换为SGD或trainbr。
% Fits/scales only the supplied training rows. No validation labels enter here.
switch lower(action)
    case 'train'
        X=varargin{1};Y=varargin{2};candidate=varargin{3};cfg=varargin{4};
        cfg.hidden=1;cfg.bias_penalty=1;cfg.penalty=candidate.penalty;
        result=trainNetwork(X,Y,cfg,cfg.seed);result.training_rows=size(X,1);
    case 'predict'
        model=varargin{1};X=varargin{2};
        assert(size(X,2)==3&&all(isfinite(X),'all'),'Expected finite N-by-3 inputs.');
        z=(X-model.xcenter)./model.xscale;
        values=zeros(size(X,1),size(model.theta,2));
        for m=1:size(model.theta,2)
            values(:,m)=model.ycenter+model.yscale*network(model.theta(:,m),z,model.hidden);
        end
        result=mean(values,2);
    case 'check'
        H=2;t=sin((1:5*H+1)');X=reshape(cos(1:15),5,3);
        [~,J]=network(t,X,H);fd=zeros(size(J));
        for j=1:numel(t)
            a=t;b=t;a(j)=a(j)+1e-6;b(j)=b(j)-1e-6;
            fd(:,j)=(network(a,X,H)-network(b,X,H))/2e-6;
        end
        result=max(abs(J-fd),[],'all');
        assert(result<1e-6,'Analytic Jacobian check failed.');
    otherwise
        error('Unknown action: %s',action);
end
end

function model=trainNetwork(X,Y,cfg,seed)
Y=Y(:);assert(size(X,2)==3&&size(X,1)==numel(Y)&&numel(Y)>=4);
assert(all(isfinite(X),'all')&&all(isfinite(Y)));
assert(cfg.hidden>=1&&cfg.members>=1&&cfg.penalty>0);
model.kind='regularized_bp';model.hidden=cfg.hidden;
model.xmin=min(X);model.xmax=max(X);model.xscale=model.xmax-model.xmin;
model.xscale(model.xscale<eps)=1;model.xcenter=(model.xmin+model.xmax)/2;
model.ycenter=mean(Y);model.yscale=max(max(Y)-min(Y),1e-8);
model.train_rows=size(X,1);model.penalty=cfg.penalty;model.seed=seed;
model.bias_penalty=cfg.bias_penalty;
z=(X-model.xcenter)./model.xscale;y=(Y-model.ycenter)/model.yscale;
P=5*cfg.hidden+1;model.theta=zeros(P,cfg.members);
model.optimization=repmat(struct('initial_loss',0,'final_loss',0,'iterations',0,...
    'gradient_inf',0,'stop',''),1,cfg.members);
old=rng;guard=onCleanup(@()rng(old));
for m=1:cfg.members
    rng(mod(seed+m,2^32),'twister');
    start=.3*randn(P,1);start(end)=mean(y); % 不进行群体搜索或挑选最优初值
    [model.theta(:,m),model.optimization(m)]=refine(start,z,y,cfg);
end
p=zeros(size(Y));
for m=1:cfg.members,p=p+network(model.theta(:,m),z,cfg.hidden)/cfg.members;end
e=model.ycenter+model.yscale*p-Y;
model.training_rmse=sqrt(mean(e.^2));
model.training_mape=mean(abs(e)./max(abs(Y),eps))*100;
model.max_loss_increase=max([model.optimization.final_loss]-[model.optimization.initial_loss]);
assert(model.max_loss_increase<=1e-10,'Local refinement increased its training objective.');
clear guard
end

function [t,info]=refine(t,X,Y,cfg)
[r,J]=residual(t,X,Y,cfg.hidden,cfg.penalty,cfg.bias_penalty);loss=r'*r;
info=struct('initial_loss',loss,'final_loss',loss,'iterations',0,'gradient_inf',0,'stop','max_iterations');
mu=1e-3;
for it=1:cfg.max_iterations
    grad=J'*r;info.iterations=it;
    if norm(grad,Inf)<1e-9,info.stop='gradient';break,end
    A=J'*J;accepted=false;
    for trial=1:16
        step=-(A+mu*eye(numel(t)))\grad;
        proposed=max(-5,min(5,t+step));
        [rr,JJ]=residual(proposed,X,Y,cfg.hidden,cfg.penalty,cfg.bias_penalty);next=rr'*rr;
        if next<loss
            previous=loss;t=proposed;r=rr;J=JJ;loss=next;
            mu=max(mu/3,1e-12);accepted=true;break
        end
        mu=min(mu*10,1e14);
    end
    if ~accepted,info.stop='no_descent';break,end
    if norm(step)<1e-9*(1+norm(t)),info.stop='step';break,end
    if previous-loss<1e-12*(1+previous),info.stop='objective';break,end
end
info.final_loss=loss;info.gradient_inf=norm(J'*r,Inf);
end

function [r,J]=residual(t,X,Y,H,lambda,biasPenalty)
[p,g]=network(t,X,H);mask=ones(size(t));
mask(3*H+1:4*H)=sqrt(biasPenalty);mask(end)=sqrt(biasPenalty);
pen=sqrt(lambda)*diag(mask);
r=[(p-Y)/sqrt(numel(Y));pen*t];J=[g/sqrt(numel(Y));pen];
end

function [p,J]=network(t,X,H)
W=reshape(t(1:3*H),H,3);b=t(3*H+1:4*H);v=t(4*H+1:5*H);c=t(end);
h=tanh(X*W'+b');p=h*v+c;
if nargout>1
    g=(1-h.^2).*v';J=zeros(size(X,1),5*H+1);
    for k=1:3,J(:,(k-1)*H+(1:H))=g.*X(:,k);end
    J(:,3*H+1:4*H)=g;J(:,4*H+1:5*H)=h;J(:,end)=1;
end
end
