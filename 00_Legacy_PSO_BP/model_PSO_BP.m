function result=model_PSO_BP(action,varargin)
% 普通PSO-BP：连接权重允许正负。
% 主程序统一提供训练样本、正则化系数、网络结构、PSO配置和随机种子。
% train返回模型；predict返回各初始化成员预测（由主程序等权集成并锚定）；check核验解析梯度。
switch lower(action)
 case 'train'
  X=varargin{1};Y=varargin{2};cfg=varargin{3};seed=varargin{4};cfg.constrained=false;
  result=fitEnsemble(X,Y,cfg,seed);result.kind='pso_bp';
 case 'predict'
  result=predictEnsemble(varargin{1},varargin{2});
 case 'check'
  checkGradient(varargin{1});result=true;
 otherwise,error('未知操作：%s',action);
end
end

function model=fitEnsemble(X,Y,cfg,seed)
model.xmin=min(X);model.xscale=max(X)-min(X);assert(all(model.xscale>0));
model.ymin=min(Y);model.yscale=max(Y)-min(Y);assert(model.yscale>0);
xn=(X-model.xmin)./model.xscale;yn=(Y-model.ymin)/model.yscale;
H=cfg.hidden;P=3*H+H+H+1;theta=zeros(P,cfg.members);
parfor m=1:cfg.members
 rng(seed+m,'twister');theta(:,m)=fitOne(xn,yn,cfg);
end
model.theta=theta;model.hidden=H;
end

function theta=fitOne(X,Y,cfg)
H=cfg.hidden;P=5*H+1;weights=[1:3*H,4*H+1:5*H];
lb=-5*ones(1,P);ub=5*ones(1,P);if cfg.constrained,lb(weights)=0;end
pop=lb+rand(cfg.pso_population,P).*(ub-lb);velocity=zeros(size(pop));
bestPersonal=pop;score=fitness(pop,X,Y,H,cfg.ridge_penalty);personalScore=score;
[bestScore,ix]=min(score);best=pop(ix,:);
for g=1:cfg.pso_generations
 velocity=.7*velocity+1.4*rand(size(pop)).*(bestPersonal-pop)+1.4*rand(size(pop)).*(best-pop);
 limit=.2*(ub-lb);velocity=max(-limit,min(limit,velocity));pop=max(lb,min(ub,pop+velocity));
 score=fitness(pop,X,Y,H,cfg.ridge_penalty);hit=score<personalScore;bestPersonal(hit,:)=pop(hit,:);personalScore(hit)=score(hit);
 [now,j]=min(score);if now<bestScore,bestScore=now;best=pop(j,:);end
end
opt=optimoptions('lsqnonlin','Display','off','SpecifyObjectiveGradient',true,'MaxIterations',200,'MaxFunctionEvaluations',1500,'FunctionTolerance',1e-10,'StepTolerance',1e-10);
theta=lsqnonlin(@(t)residual(t,X,Y,H,cfg.ridge_penalty),best',lb',ub',opt);
if cfg.constrained,assert(all(theta(weights)>=-1e-10));end
end

function score=fitness(pop,X,Y,H,lambda)
P=size(pop,1);W=reshape(pop(:,1:3*H)',H,3,P);B=reshape(pop(:,3*H+1:4*H)',H,1,P);
V=reshape(pop(:,4*H+1:5*H)',1,H,P);C=reshape(pop(:,end)',1,1,P);
h=tanh(pagemtimes(W,X')+B);p=pagemtimes(V,h)+C;
score=reshape(mean((p-Y').^2,2),P,1)+lambda*sum(pop(:,[1:3*H,4*H+1:5*H]).^2,2);
end

function [p,J]=network(theta,X,H)
W=reshape(theta(1:3*H),H,3);b=theta(3*H+1:4*H);v=theta(4*H+1:5*H);c=theta(end);
h=tanh(X*W'+b');p=h*v+c;
if nargout>1
 J=zeros(size(X,1),5*H+1);grad=(1-h.^2).*v';
 for k=1:3,J(:,(k-1)*H+(1:H))=grad.*X(:,k);end
 J(:,3*H+1:4*H)=grad;J(:,4*H+1:5*H)=h;J(:,end)=1;
end
end

function [r,J]=residual(theta,X,Y,H,lambda)
[p,g]=network(theta,X,H);ix=[1:3*H,4*H+1:5*H];pen=zeros(numel(ix),numel(theta));
for i=1:numel(ix),pen(i,ix(i))=sqrt(lambda);end
r=[(p-Y)/sqrt(numel(Y));sqrt(lambda)*theta(ix)];J=[g/sqrt(numel(Y));pen];
end

function p=predictEnsemble(model,X)
xx=(X-model.xmin)./model.xscale;p=zeros(size(X,1),size(model.theta,2));
for j=1:size(p,2),p(:,j)=model.ymin+model.yscale*network(model.theta(:,j),xx,model.hidden);end
end

function checkGradient(cfg)
rng(71);H=cfg.hidden;t=rand(5*H+1,1);x=rand(5,3);[~,J]=network(t,x,H);fd=J;
for i=1:numel(t),a=t;b=t;a(i)=a(i)+1e-6;b(i)=b(i)-1e-6;fd(:,i)=(network(a,x,H)-network(b,x,H))/2e-6;end
assert(max(abs(J-fd),[],'all')<1e-6,'反向传播Jacobian检查失败');
end

