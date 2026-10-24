function R=main_Water_ModelComparison(inputFile,outDir,makeFigures)
% 同输入、同划分、同锚定的普通PSO-BP/非负权重PSO-BP/岭回归对照。
% 两种PSO-BP同为3-4-1、200初始化等权；同一PSO及反向传播Jacobian训练。
% 唯一区别为连接权重是否非负；均不使用Bootstrap或检验标签筛选成员。
% 正则化只用每次训练集末3年的滚动内验证选择。2020—2023已在前期探索中见过，
% 本轮比较是模型选型证据，不声称全新的外部验证。数值全部由MATLAB计算。
base=fileparts(mfilename('fullpath'));addpath(base);
% 统一入口：输入与特征构造、训练期内选参、时间检验、未来情景、输出与作图。
% 三种方法的拟合与预测分别由同目录三个 model_*.m 子程序负责。
if nargin<1||isempty(inputFile),inputFile=fullfile(base,'需水预测模型对照.xlsx');end
if nargin<2||isempty(outDir),outDir=base;end
if nargin<3,makeFigures=true;end
if ~isfolder(outDir),mkdir(outDir);end
cfg=readSettings(inputFile);cfg.inner_members=20;
h=readmatrix(inputFile,'Sheet','历史数据','UseExcel',false);h=h(isfinite(h(:,1)),:);
assert(isequal(h(:,1),(2010:2023)')&&all(isfinite(h(:,1:14)),'all'));
years=h(:,1);Y=h(:,14);D=[h(:,8),h(:,10),h(:,2).*(1+h(:,3)/100)];E=h(:,[9,13]);X=features(D,E);
assert(max(abs(h(:,13)-h(:,11).*h(:,12)))<1e-8);
methods={'普通PSO-BP','受约束PSO-BP','岭回归'};scenarios={'SSP1','SSP2','SSP3','SSP4','SSP5'};
p=gcp('nocreate');if isempty(p),parpool('local',cfg.workers);end
model_PSO_BP('check',cfg);model_Constrained_PSO_BP('check',cfg);T=years<=2019;at0=find(T,1,'last');v=find(~T);actual=Y(v);
fixed=zeros(4,3);rolling=fixed;lambdaFixed=zeros(1,3);lambdaRolling=zeros(4,3);logs=cell(0,5);
for method=1:3
 fprintf('\nMETHOD %d/3 %s: FIXED ORIGIN\n',method,methods{method});
 [lam,tuningRecord]=choosePenalty(X(T,:),Y(T),years(T),method,cfg,cfg.seed+1000);logs=[logs;tuningRecord];lambdaFixed(method)=lam;
 model=fitModel(X(T,:),Y(T),method,lam,cfg,cfg.seed+10000);
 fixed(:,method)=anchoredPrediction(model,X(v,:),X(at0,:),Y(at0));
 for k=1:4
  at=v(k);fprintf('METHOD %d ROLLING %d\n',method,years(at));
  if k==1,lam=lambdaFixed(method);else
   [lam,tuningRecord]=choosePenalty(X(1:at-1,:),Y(1:at-1),years(1:at-1),method,cfg,cfg.seed+1000+100*k);logs=[logs;tuningRecord];
  end
  lambdaRolling(k,method)=lam;
  model=fitModel(X(1:at-1,:),Y(1:at-1),method,lam,cfg,cfg.seed+20000+1000*k);
  rolling(k,method)=anchoredPrediction(model,X(at,:),X(at-1,:),Y(at-1));
 end
end
naiveFixed=repmat(Y(at0),4,1);naiveRolling=Y(v-1);
fixedMetrics=zeros(3,4);rollingMetrics=fixedMetrics;
for method=1:3,fixedMetrics(method,:)=metrics(actual,fixed(:,method));rollingMetrics(method,:)=metrics(actual,rolling(:,method));end
% 未来驱动/三套强度路径与上版一致，不因模型不同改变。
r=readcell(inputFile,'Sheet','未来指标','UseExcel',false);r=r(2:end,:);yf=(2024:2030)';
b=[ones(14,1),years-years(1)]\log(E(:,2));decay=-b(2);df=cell(1,5);ep=cell(5,3);raw=cell(1,5);
for s=1:5
 a=cell2mat(r(strcmp(r(:,1),scenarios{s}),2:12));assert(isequal(a(:,1),yf));raw{s}=a;
 df{s}=[a(:,8),a(:,10),a(:,2).*(1+a(:,3)/100)];
 frozen=repmat(E(end,:),7,1);trend=[a(:,9),E(end,2)*exp(-decay*a(:,11).*(yf-2023))];policy=trend;
 cap=trend(2,2)*(1-cfg.industrial_drop).^((yf-2025)/5);ix=yf>2025;policy(ix,2)=min(trend(ix,2),cap(ix));
 ep(s,:)={frozen,trend,policy};
end
pred=zeros(3,5,7,3);derivativeRows=cell(0,5);lambdaFinal=zeros(1,3);models=cell(1,3);violation=zeros(3,4);pathReverse=zeros(3,1);
contrib=zeros(3,5,7,10);futureX=[];
for s=1:5,for j=1:3,futureX=[futureX;features(df{s},ep{s,j})];end,end
for method=1:3
 fprintf('METHOD %d FINAL FULL-HISTORY TRAINING\n',method);
 [lam,tuningRecord]=choosePenalty(X,Y,years,method,cfg,cfg.seed+9000);logs=[logs;tuningRecord];lambdaFinal(method)=lam;
 model=fitModel(X,Y,method,lam,cfg,cfg.seed+30000);models{method}=model;
 base=mean(predictModel(model,X(end,:)),2);f=@(x)Y(end)+mean(predictModel(model,x),2)-base;
 allPrediction=f(futureX);violation(method,4)=nnz(allPrediction<=0);
 for c=1:3
  down=futureX;down(:,c)=.99*down(:,c);change=allPrediction-f(down);violation(method,c)=nnz(change< -1e-7);
  derivativeRows(end+1,:)={methods{method},c,violation(method,c),numel(change),min(change)}; %#ok<AGROW>
 end
 for s=1:5
  development=df{s};frozen=ep{s,1};
  for j=1:3,pred(method,s,:,j)=reshape(f(features(development,ep{s,j})),1,1,7);end
  dev=zeros(7,8);
  for mask=0:7
   dd=repmat(D(end,:),7,1);for c=1:3,if bitget(mask,c),dd(:,c)=development(:,c);end,end
   dev(:,mask+1)=f(features(dd,frozen));
  end
  dc=zeros(7,3);for c=1:3,for mask=0:7,if ~bitget(mask,c)
   count=sum(bitget(mask,1:3));weight=factorial(count)*factorial(2-count)/6;
   dc(:,c)=dc(:,c)+weight*(dev(:,bitset(mask,c)+1)-dev(:,mask+1));
  end,end,end
  w00=reshape(pred(method,s,:,1),7,1);w11=reshape(pred(method,s,:,3),7,1);
  wh=reshape(pred(method,s,:,2),7,1);pathReverse(method)=pathReverse(method)+nnz(w00<wh-1e-7|wh<w11-1e-7);
  ea=frozen;ea(:,1)=ep{s,3}(:,1);ei=frozen;ei(:,2)=ep{s,3}(:,2);
  w10=f(features(development,ea));w01=f(features(development,ei));
  sa=.5*(w00-w10+w01-w11);si=.5*(w00-w01+w10-w11);
  growth=w00-Y(end);saved=w00-w11;offset=100*saved./growth;offset(growth<=1e-8)=NaN;
  extra=reshape(pred(method,s,:,2)-pred(method,s,:,3),7,1);
  assert(max(abs(sum(dc,2)-growth))<1e-6&&max(abs(sa+si-saved))<1e-6);
  contrib(method,s,:,:)=reshape([dc,growth,sa,si,saved,extra,offset,w11-Y(end)],1,1,7,10);
 end
end
% 主模型以两种时间检验MAPE均值作描述性排序；需水为正且农业/工业节水响应不反向。
% 人口驱动负响应单独报告，不将统计贡献解释为人口的因果作用。
score=mean([fixedMetrics(:,1),rollingMetrics(:,1)],2);eligible=violation(:,1)==0&violation(:,2)==0&violation(:,4)==0&pathReverse==0;
ranking=score;ranking(~eligible)=Inf;
if any(eligible),[~,selected]=min(ranking);else,[~,selected]=min(score);end
R=struct('methods',{methods},'scenarios',{scenarios},'future_years',yf,'historical_years',years,'historical_actual',Y,...
 'validation_years',years(v),'actual',actual,'fixed_prediction',fixed,'rolling_prediction',rolling,...
 'fixed_metrics',fixedMetrics,'rolling_metrics',rollingMetrics,'naive_fixed_metrics',metrics(actual,naiveFixed),...
 'naive_rolling_metrics',metrics(actual,naiveRolling),'prediction',pred,'contribution',contrib,...
 'direction_rows',{derivativeRows},'violations',violation,'path_reverse',pathReverse,'eligible',eligible,'score',score,'selected',selected,...
 'selected_name',methods{selected},'lambda_fixed',lambdaFixed,'lambda_rolling',lambdaRolling,'lambda_final',lambdaFinal,...
 'tuning_rows',{logs},'settings',cfg,'industrial_decay',decay,...
 'validation_scope','给定实际驱动的时间条件检验；同一2020—2023年已用于前期探索，不是新的外部验证',...
 'selection_rule','固定起点与滚动MAPE均值作描述性排序；过滤负需水及农业/工业节水反向的模型；不作显著性优劣结论');
slopes=models{3}.yscale*models{3}.theta(2:end)'./models{3}.xscale;
R.ridge_coefficients=[models{3}.ymin+models{3}.yscale*models{3}.theta(1)-sum(slopes.*models{3}.xmin),slopes];
exportResults(R,inputFile,outDir);if makeFigures,plotResults(R,outDir);end
disp('FIXED METRICS [MAPE MAE RMSE MSE]');disp(fixedMetrics);disp('ROLLING METRICS');disp(rollingMetrics);
disp('DIRECTION VIOLATIONS [AG INDUSTRY LIFE NEGATIVE_PRED]');disp(violation);
fprintf('Selected: %s; final lambdas %s\n',methods{selected},mat2str(lambdaFinal));
end

function [lambda,rows]=choosePenalty(X,Y,years,method,cfg,seed)
if method<=2,grid=[.0001,.001,.01];else,grid=[.00001,.0001,.001,.01,.1,1,10,100];end
scores=zeros(numel(grid),1);rows=cell(numel(grid),5);inner=cfg;inner.members=cfg.inner_members;
for g=1:numel(grid)
 e=zeros(3,1);
 for k=1:3
  at=size(X,1)-3+k;model=fitModel(X(1:at-1,:),Y(1:at-1),method,grid(g),inner,seed+100*k);
  pr=anchoredPrediction(model,X(at,:),X(at-1,:),Y(at-1));e(k)=abs(pr-Y(at))/Y(at)*100;
 end
 scores(g)=mean(e);rows(g,:)={method,years(end),grid(g),scores(g),sprintf('%d—%d逐年内验证',years(end-2),years(end))};
end
[~,best]=min(scores);lambda=grid(best);
end

function model=fitModel(X,Y,method,lambda,cfg,seed)
cfg.ridge_penalty=lambda;
switch method
 case 1,model=model_PSO_BP('train',X,Y,cfg,seed);
 case 2,model=model_Constrained_PSO_BP('train',X,Y,cfg,seed);
 case 3,model=model_Ridge('train',X,Y,lambda);
 otherwise,error('未知方法编号');
end
end

function p=predictModel(model,X)
switch model.kind
 case 'pso_bp',p=model_PSO_BP('predict',model,X);
 case 'constrained_pso_bp',p=model_Constrained_PSO_BP('predict',model,X);
 case 'ridge',p=model_Ridge('predict',model,X);
 otherwise,error('未知模型类型');
end
end

function p=anchoredPrediction(model,X,baseX,baseY)
p=baseY+mean(predictModel(model,X)-predictModel(model,baseX),2);
end

function cfg=readSettings(file)
r=readcell(file,'Sheet','模型设置','UseExcel',false);cfg=struct;
for i=2:size(r,1),if ischar(r{i,1})&&isnumeric(r{i,2}),cfg.(r{i,1})=r{i,2};end,end
assert(cfg.members>=2&&cfg.hidden>=1&&cfg.industrial_drop>=0&&cfg.industrial_drop<1);
assert(cfg.base_year==2023&&cfg.industrial_policy_base==2025&&cfg.policy_target_year==2030,'当前版本固定2023基期、2025政策比较基年与2030规划年');
end

function X=features(D,E)
% 千公顷×15000亩/千公顷×m3/亩÷1e8；亿元×m3/万元÷1e4。
X=[D(:,1).*E(:,1)*15/100000,D(:,2).*E(:,2)/10000,D(:,3)];
end


function m=metrics(y,p)
e=p(:)-y(:);m=[mean(abs(e)./abs(y(:)))*100,mean(abs(e)),sqrt(mean(e.^2)),mean(e.^2)];
end


function exportResults(R,inputFile,outDir)
file=fullfile(outDir,'需水预测模型对照.xlsx');tmp=[tempname(outDir),'.xlsx'];guard=onCleanup(@()removeTemporary(tmp));copyfile(inputFile,tmp);
rows=cell(3,12);
for m=1:3,rows(m,:)=[{R.methods{m}},num2cell([R.fixed_metrics(m,:),R.rolling_metrics(m,:),R.score(m)]),{R.eligible(m),m==R.selected}];end
rows(end+1,:)=[{'最近实际值基准'},num2cell([R.naive_fixed_metrics,R.naive_rolling_metrics,mean([R.naive_fixed_metrics(1),R.naive_rolling_metrics(1)])]),{NaN,false}];
writecell([{'方法','固定起点MAPE（%）','固定起点MAE','固定起点RMSE','固定起点MSE','滚动MAPE（%）','滚动MAE','滚动RMSE','滚动MSE','两种MAPE均值','节水响应准入','选为主模型'};rows],tmp,'Sheet','方法比较','UseExcel',false);
actual=R.actual;rows=cell(0,7);
for m=1:3,for k=1:4
 rows(end+1,:)={'固定起点',R.methods{m},R.validation_years(k),actual(k),R.fixed_prediction(k,m),R.fixed_prediction(k,m)-actual(k),100*abs(R.fixed_prediction(k,m)-actual(k))/actual(k)}; %#ok<AGROW>
 rows(end+1,:)={'逐年滚动',R.methods{m},R.validation_years(k),actual(k),R.rolling_prediction(k,m),R.rolling_prediction(k,m)-actual(k),100*abs(R.rolling_prediction(k,m)-actual(k))/actual(k)}; %#ok<AGROW>
end,end
writecell([{'检验方式','方法','年份','实际值（亿m³）','预测值（亿m³）','误差（亿m³）','绝对相对误差（%）'};rows],tmp,'Sheet','时间检验逐年值','UseExcel',false);
names={'主模型_效率保持','主模型_历史趋势','主模型_规划约束'};
for j=1:3
 a=zeros(8,6);a(:,1)=(2023:2030)';a(1,2:6)=R.historical_actual(end);
 for s=1:5,a(2:end,s+1)=reshape(R.prediction(R.selected,s,:,j),7,1);end
 writecell([{'年份'},R.scenarios;num2cell(a)],tmp,'Sheet',names{j},'UseExcel',false);
end
rows=cell(0,6);pathNames={'效率保持2023年','历史节水趋势','规划参考约束'};
for m=1:3,for s=1:5,for t=1:7
 rows(end+1,:)=[{R.methods{m},R.scenarios{s},R.future_years(t)},num2cell(reshape(R.prediction(m,s,t,:),1,3))]; %#ok<AGROW>
end,end,end
writecell([{'方法','情景','年份','效率保持2023年','历史节水趋势','规划参考约束'};rows],tmp,'Sheet','各模型未来预测','UseExcel',false);
rows=R.direction_rows;drivers={'农业驱动','工业驱动','人口城镇化驱动'};
for i=1:size(rows,1),rows{i,2}=drivers{rows{i,2}};end
for m=1:3
 rows(end+1,:)={R.methods{m},'节水三路径顺序',R.path_reverse(m),35,NaN}; %#ok<AGROW>
 rows(end+1,:)={R.methods{m},'非正需水预测',R.violations(m,4),105,NaN}; %#ok<AGROW>
end
writecell([{'方法','检查项','反向/异常个数','检查点个数','驱动降低1%时最小预测降低量'};rows],tmp,'Sheet','方向检查','UseExcel',false);
rows=R.tuning_rows;for i=1:size(rows,1),rows{i,1}=R.methods{rows{i,1}};end
writecell([{'方法','本次训练截止年','正则化参数','训练期内验证MAPE（%）','内验证范围'};rows],tmp,'Sheet','参数选择记录','UseExcel',false);
rows=cell(0,12);
for s=1:5,for t=1:7,rows(end+1,:)=[{R.scenarios{s},R.future_years(t)},num2cell(reshape(R.contribution(R.selected,s,t,:),1,10))];end,end %#ok<AGROW>
writecell([{'情景','年份','农业规模变化贡献','工业发展贡献','人口城镇化贡献','发展变化合计','农业效率情景贡献','工业效率情景贡献','总节水情景差值','规划额外差值','抵消率（%）','净变化'};rows],tmp,'Sheet','主模型贡献分解','UseExcel',false);
notes={
 '推荐主模型',R.selected_name;
 '结果单位','需水量、MAE和RMSE为亿m³，MSE为(亿m³)²，MAPE及抵消率为%；预测精度为给定实际驱动的条件精度，不包含人口、GDP等前级预测误差';
 '选择规则',R.selection_rule;
 '检验适用范围',R.validation_scope;
 '输入与锚定','相同三驱动、相同训练期内Min-Max变换；所有方法均以训练末年实际值锚定；检验标签不参与拟合、筛选成员或参数选择';
 '普通PSO-BP','同3-4-1结构，同PSO与反向传播Jacobian有界最小二乘训练，同L2与种子；无非负连接权重限制；数值边界统一为±5';
 '受约束PSO-BP','只增加连接权重非负约束；偏置仍可正可负';
 '集成','200次固定样本初始化，等权均值；不是Bootstrap；没有把重复训练当作新增观测';
 '参数选择','神经网络候选0.0001/0.001/0.01；岭回归候选0.00001到100；只在当前训练末3年作逐年滚动选择；神经网络内验证20成员、正式检验200成员';
 '固定起点','2010—2019训练，2020—2023检验；以2019实际值锚定';
 '逐年滚动','每次仅用该年以前数据，末3年内验证另行选参数；以前一年实际值锚定';
 '同一期多种检验','两个MAPE的平均只作为描述性排序，四个年份并未变成八个独立样本';
 '简单基准',sprintf('固定保持2019年：MAPE %.4f%%；滚动上一年实际值：MAPE %.4f%%',R.naive_fixed_metrics(1),R.naive_rolling_metrics(1));
 '方向准入','未来需求必须为正；农业/工业驱动降低1%不得增加预测需求；效率冻结≥历史趋势≥规划约束';
 '人口方向','人口城镇化负响应单独披露；统计预测允许相关系数为负，但不能据此解释人口增加的因果影响';
 '本次选型结论',sprintf('按声明规则暂定%s作总量预测主模型；其滚动MAPE %.4f%%，最近实际值基准 %.4f%%；不宣称显著优越',R.selected_name,R.rolling_metrics(R.selected,1),R.naive_rolling_metrics(1));
 '神经网络外推风险','2021年工业驱动低于截至2020年的训练范围；受约束模型在该年出现大偏差。约束保证方向，不保证外推误差小；本次不按检验结果回调参数';
 '与前版误差差异','本次三方法统一采用训练期内滚动选择正则化；前版固定参数。这里是完整选参流程的误差，不与前版固定参数误差混为一谈';
 '农业面积','播种面积按用户假设作为灌溉面积代理；不是实际有效灌溉面积';
 '工业强度','保留用户选定估算序列；价格已成对转换，但历年火电统计口径差异未消除，不宣称实际工业部门用水';
 '贡献分解','同一预测模型内的情景差值；不是部门实测节水量或政策因果效果。发展变化不增加需求时抵消率留空';
 '规划参考','2030工业强度较预测2025基值下降至少10%；为全国参考门槛，不是苏北直接分解目标；历史已达到时允许两路径重合';
 '正式网络成员数',R.settings.members;
 '正式随机种子',R.settings.seed;
 '固定起点正则化',mat2str(R.lambda_fixed);
 '最终正则化',mat2str(R.lambda_final);
 '岭回归原量纲系数[截距,农业,工业,人口城镇化]',mat2str(R.ridge_coefficients,8)
};
writecell([{'事项','取值或说明'};notes],tmp,'Sheet','方法与结论','UseExcel',false);
[ok,msg]=movefile(tmp,file,'f');assert(ok,'关闭结果Excel/WPS后再运行：%s',msg);clear guard
end

function removeTemporary(file)
if isfile(file),delete(file);end
end

function plotResults(R,outDir)
folder=fullfile(outDir,'图表');if ~isfolder(folder),mkdir(folder);end
colors=[.157,.467,.722;.84,.37,.1;0,.5,.36];styles={'-','--',':'};
modeNames={'固定起点','逐年滚动'};
for mode=1:2
 f=figure('Visible','off','Color','w','Position',[50,50,1150,700]);g=onCleanup(@()close(f));ax=axes(f);hold(ax,'on');
 plot(ax,R.validation_years,R.actual,'ko-','LineWidth',1.8,'DisplayName','实际值');
 for m=1:3
  if mode==1,v=R.fixed_prediction(:,m);else,v=R.rolling_prediction(:,m);end
  plot(ax,R.validation_years,v,'LineStyle',styles{m},'Color',colors(m,:),'LineWidth',1.8,'Marker','o','DisplayName',R.methods{m});
 end
 styleAxes(ax,2020,2023,'总需水量（亿m³）');legend(ax,'Location','northoutside','Orientation','horizontal','Box','off','FontName','宋体','FontSize',14);
 exportgraphics(f,fullfile(folder,sprintf('统一方法对照_%s检验.png',modeNames{mode})),'Resolution',300);clear g
end
f=figure('Visible','off','Color','w','Position',[30,30,1650,560]);g=onCleanup(@()close(f));tl=tiledlayout(f,1,3,'TileSpacing','loose','Padding','loose');
cs=[0,.545,.471;.157,.467,.722;.784,.482,.141;.549,.396,.659;.831,.310,.227];st={'--','-',':','-.','--'};
vv=R.prediction(:,:,:,3);allY=[R.historical_actual(end);vv(:)];yl=[floor(min(allY))-1,ceil(max(allY))+1];
for m=1:3
 ax=nexttile(tl);hold(ax,'on');for s=1:5,plot(ax,[2023;R.future_years],[R.historical_actual(end);reshape(R.prediction(m,s,:,3),7,1)],'Color',cs(s,:),'LineStyle',st{s},'LineWidth',1.8,'DisplayName',R.scenarios{s});end
 ylim(ax,yl);styleAxes(ax,2023,2030,'总需水量（亿m³）');title(ax,R.methods{m},'FontName','宋体','FontSize',17,'FontWeight','normal');
end
lg=legend(ax,'Orientation','horizontal','Box','off','FontName','Times New Roman','FontSize',14);lg.Layout.Tile='south';
exportgraphics(f,fullfile(folder,'三种模型_规划路径预测对照.png'),'Resolution',300);clear g
for start=[2023,2010]
 f=figure('Visible','off','Color','w','Position',[50,50,1150,700]);g=onCleanup(@()close(f));ax=axes(f);hold(ax,'on');
 if start==2010,plot(ax,R.historical_years,R.historical_actual,'k-','LineWidth',1.8,'DisplayName','历史实际值');end
 for s=1:5,plot(ax,[2023;R.future_years],[R.historical_actual(end);reshape(R.prediction(R.selected,s,:,3),7,1)],'Color',cs(s,:),'LineStyle',st{s},'LineWidth',1.8,'DisplayName',R.scenarios{s});end
 styleAxes(ax,start,2030,'总需水量（亿m³）');legend(ax,'Location','northoutside','Orientation','horizontal','Box','off','FontName','宋体','FontSize',14);
 exportgraphics(f,fullfile(folder,sprintf('推荐主模型_从%d年开始.png',start)),'Resolution',300);clear g
end
end

function styleAxes(ax,xmin,xmax,label)
set(ax,'FontName','Times New Roman','FontSize',15,'TickDir','in','LineWidth',.6,'Box','off','XGrid','off','YGrid','off','XLim',[xmin,xmax]);
if xmin==2023,xticks(ax,[2023,2025,2027,2030]);elseif xmin==2010,xticks(ax,[2010,2015,2020,2025,2030]);elseif xmax==2023,xticks(ax,2020:2023);end
ax.YAxis.Exponent=0;xl=xlim(ax);yl=ylim(ax);set(ax,'XLimMode','manual','YLimMode','manual');
line(ax,xl,[yl(2),yl(2)],'Color','k','LineWidth',.6,'HandleVisibility','off');line(ax,[xl(2),xl(2)],yl,'Color','k','LineWidth',.6,'HandleVisibility','off');
xlabel(ax,'年份','FontName','宋体','FontSize',18);ylabel(ax,label,'FontName','宋体','FontSize',18);
end
