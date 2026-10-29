function C=main_BP_vs_PSO_BP(root,outDir,makeFigures)
% 配对消融：唯一方法差异为是否先做PSO搜索；保留现有PSO-BP作为基准。
% 五组种子均报告，首组为主结果，不根据检验误差挑选种子。
here=fileparts(mfilename('fullpath'));addpath(here);
if nargin<1||isempty(root),root=fileparts(here);end
addpath(root);setup_water_paths;
if nargin<2||isempty(outDir),outDir=fullfile(here,'results');end
if nargin<3,makeFigures=true;end
if ~isfolder(outDir),mkdir(outDir);end
D=water_suite_data;[pso,cfg]=water_suite_setup('pso');cfg.seed=D.seed;
bp=pso;bp.id='bp';bp.name='正则BP集成（无PSO）';bp.model='model_BP_Ensemble';
bp.description='3-1-1，权重偏置联合L2，12次随机初始化，同解析Jacobian阻尼高斯牛顿';
specs={pso,bp};names={pso.name,bp.name};seeds=cfg.seed+(0:4)*1000;
gradientError=model_BP_Ensemble('check');assert(gradientError<1e-6);
% 各方法预热一次；计时不含数据读取、图表、模型保存和预热。
warm=cfg;warm.members=1;
for m=1:2,feval(specs{m}.model,'train',D.X(1:10,:),D.Y(1:10),pso.candidates{1},warm);end
runs=cell(2,5);primaryModels=cell(1,2);elapsed=zeros(2,5);
for k=1:5
 order=[1,2];if mod(k,2)==0,order=[2,1];end
 for m=order
  trial=cfg;trial.seed=seeds(k);fprintf('\nSEED %d METHOD %s\n',seeds(k),names{m});
  [r,models]=water_method_backtest(D.X,D.Y,D.years,specs{m},trial);
  elapsed(m,k)=r.training_seconds;runs{m,k}=r;
  if k==1,primaryModels{m}=models;end
 end
end
fm=zeros(2,4);rm=fm;forecast=zeros(7,5,3,2);direction=zeros(2,5);lambda=zeros(5,2);
for m=1:2
 r=runs{m,1};fm(m,:)=r.fixed_metrics;rm(m,:)=r.rolling_metrics;
 for k=1:5,lambda(k,m)=r.selected{k}.penalty;end
 model=primaryModels{m}.final;predict=@(xx)D.Y(end)+feval(specs{m}.model,'predict',model,xx)-feval(specs{m}.model,'predict',model,D.X(end,:));
 for s=1:5
  for j=1:3
   xx=D.future_X{s,j};pp=predict(xx);forecast(:,s,j,m)=pp;
   direction(m,4)=direction(m,4)+nnz(pp<=0);
   for c=1:3,down=xx;down(:,c)=.99*down(:,c);direction(m,c)=direction(m,c)+nnz(pp-predict(down)<-1e-7);end
  end
  q=reshape(forecast(:,s,:,m),7,3);direction(m,5)=direction(m,5)+nnz(q(:,1)<q(:,2)-1e-7|q(:,2)<q(:,3)-1e-7);
 end
end
% 复现既有PSO-BP结果，防止在本轮对照中悄然改变基准。
prior=load(fullfile(root,'01_PSO_BP_Ensemble','results','result.mat'),'R');
assert(max(abs(prior.R.fixed_prediction-runs{1,1}.fixed_prediction))<1e-8);
assert(max(abs(prior.R.rolling_prediction-runs{1,1}.rolling_prediction))<1e-8);
% 检查新BP训练/选参不读取当前训练前缀之外的目标。
y=D.Y;y(12:14)=y(12:14)+[40;60;80];perturbed=water_method_backtest(D.X,y,D.years,bp,cfg);
assert(max(abs(perturbed.fixed_prediction-runs{2,1}.fixed_prediction))<1e-10);
assert(max(abs(perturbed.rolling_prediction(1:2)-runs{2,1}.rolling_prediction(1:2)))<1e-10);
assert(isequal(perturbed.selected(1:2),runs{2,1}.selected(1:2)));
after=water_suite_data;assert(strcmp(D.source_sha256,after.source_sha256));
% 同参数诊断仅解释差异，不据此重新选择主结果参数。
matched=zeros(4,3);prefixDiagnostic=cell(0,7);
for g=1:numel(pso.candidates)
 for m=1:2
  model=feval(specs{m}.model,'train',D.X,D.Y,pso.candidates{g},cfg);
  fun=@(xx)D.Y(end)+feval(specs{m}.model,'predict',model,xx)-feval(specs{m}.model,'predict',model,D.X(end,:));
  ff=[];for s=1:5,for j=1:3,ff=[ff;fun(D.future_X{s,j})];end,end
  if m==1,reference=ff;lossOne=mean([model.optimization.final_loss]);else
   matched(g,:)=[pso.candidates{g}.penalty,max(abs(ff-reference)),mean([model.optimization.final_loss])-lossOne];
  end
  if ismember(pso.candidates{g}.penalty,[.03,.1])
   for n=11:13
    fit=feval(specs{m}.model,'train',D.X(1:n,:),D.Y(1:n),pso.candidates{g},cfg);
    pp=D.Y(n)+feval(specs{m}.model,'predict',fit,D.X(n+1,:))-feval(specs{m}.model,'predict',fit,D.X(n,:));
    prefixDiagnostic(end+1,:)={names{m},D.years(n),pso.candidates{g}.penalty,mean([fit.optimization.final_loss]),pp,D.Y(n+1),100*abs(pp-D.Y(n+1))/D.Y(n+1)};
   end
  end
 end
end
maxFutureDifference=max(abs(forecast(:,:,:,1)-forecast(:,:,:,2)),[],'all');
C=struct('methods',{names},'seeds',seeds,'runs',{runs},'training_seconds',elapsed,...
 'fixed_metrics',fm,'rolling_metrics',rm,'selected_penalty',lambda,...
 'future_years',D.future_years,'scenarios',{D.scenarios},'future_prediction',forecast,...
 'direction_counts',direction,'input_sha256',D.source_sha256,...
 'max_future_difference',maxFutureDifference,'matched_parameter_diagnostics',matched,'prefix_diagnostics',{prefixDiagnostic},'gradient_max_error',gradientError,...
 'pso_baseline_reproduced',true,'bp_prefix_isolation_passed',true,'input_unchanged',true,...
 'scope','同一已知2020—2023开发回测；非新增独立检验；随机种子不作为新增独立年份',...
 'bp_definition','随机小初值后直接用反向传播Jacobian的阻尼高斯牛顿求解；同PSO-BP局部求解器，未另换成SGD或trainbr',...
 'timing_scope','各方法预热后完整训练内选参+固定/滚动/最终拟合；交替运行顺序；排除数据读取与I/O，计时仅作本机参考');
save(fullfile(outDir,'comparison.mat'),'C','primaryModels');
writeText(fullfile(outDir,'comparison.json'),jsonencode(C,'PrettyPrint',true));
exportTables(C,outDir);writeReport(C,outDir);
if makeFigures,plotComparison(C,D,outDir);end
disp(array2table([fm(:,1),rm(:,1),median(elapsed,2)],'RowNames',names,'VariableNames',{'FixedMAPE','RollingMAPE','MedianSeconds'}));
fprintf('Max future difference %.12g; BASELINE / PREFIX / INPUT CHECKS PASSED\n',maxFutureDifference);
end

function exportTables(C,outDir)
file=fullfile(outDir,'BP与PSO_BP对照结果.xlsx');rows=cell(2,11);
for m=1:2,rows(m,:)=[C.methods(m),num2cell([C.fixed_metrics(m,:),C.rolling_metrics(m,:),median(C.training_seconds(m,:)),mean(C.training_seconds(m,:))])];end
writecell([{'方法','固定MAPE（%）','固定MAE（亿m³）','固定RMSE（亿m³）','固定MSE','滚动MAPE（%）','滚动MAE（亿m³）','滚动RMSE（亿m³）','滚动MSE','运行秒数中位数','运行秒数均值'};rows],file,'Sheet','方法对比','UseExcel',false);
rows=cell(0,6);
for k=1:5,for m=1:2,r=C.runs{m,k};rows(end+1,:)={C.seeds(k),C.methods{m},r.fixed_metrics(1),r.rolling_metrics(1),r.score,C.training_seconds(m,k)};end,end
writecell([{'随机种子','方法','固定MAPE（%）','滚动MAPE（%）','两项均值（%）','运行秒数'};rows],file,'Sheet','五组随机种子','UseExcel',false);
rows=cell(0,7);
for m=1:2,r=C.runs{m,1};for i=1:4,rows(end+1,:)={C.methods{m},r.validation_years(i),r.actual(i),r.fixed_prediction(i),r.rolling_prediction(i),r.fixed_prediction(i)-r.actual(i),r.rolling_prediction(i)-r.actual(i)};end,end
writecell([{'方法','年份','实际需水（亿m³）','固定预测','滚动预测','固定误差','滚动误差'};rows],file,'Sheet','逐年回测','UseExcel',false);
rows=cell(0,5);for m=1:2,for k=1:5,r=C.runs{m,k};for i=1:5,rows(end+1,:)={C.methods{m},C.seeds(k),r.training_end_years(i),r.selected{i}.penalty,r.inner_mape(i)};end,end,end
writecell([{'方法','随机种子','训练截止年','正则化参数','训练内验证MAPE（%）'};rows],file,'Sheet','训练内选参','UseExcel',false);
rows=cell(0,6);for m=1:2,for s=1:5,for i=1:7,rows(end+1,:)=[{C.methods{m},C.scenarios{s},C.future_years(i)},num2cell(reshape(C.future_prediction(i,s,:,m),1,3))];end,end,end
writecell([{'方法','情景','年份','效率冻结（亿m³）','历史趋势（亿m³）','规划参考（亿m³）'};rows],file,'Sheet','未来情景','UseExcel',false);
writecell([{'方法','农业反向','工业反向','人口反向','非正预测','路径反序'};[C.methods',num2cell(C.direction_counts)]],file,'Sheet','方向检查','UseExcel',false);
notes={'控制条件','3-1-1、12成员、权重和偏置联合L2、候选0.01/0.03/0.1/1、同归一化和同求解器；仅是否PSO不同';'BP定义',C.bp_definition;'初始参数','BP每个成员使用0.3×标准正态随机初值，输出偏置设为归一化训练均值；不搜索、不挑选初值';'主结果种子',C.seeds(1);'输入SHA256',C.input_sha256;'检验边界',C.scope;'计时范围',C.timing_scope;'最大未来预测差值（亿m³）',C.max_future_difference;'既有PSO基准复现',C.pso_baseline_reproduced;'BP训练前缀隔离',C.bp_prefix_isolation_passed;'数据文件未改变',C.input_unchanged};
writecell([{'事项','说明'};notes],file,'Sheet','设置与核验','UseExcel',false);
writecell([{'共同正则化参数','未来105点最大差值（亿m³）','BP减PSO最终平均训练目标'};num2cell(C.matched_parameter_diagnostics)],file,'Sheet','同参数诊断','UseExcel',false);
writecell([{'方法','训练截止年','共同正则化参数','平均训练目标','下一年预测','下一年实际','相对误差（%）'};C.prefix_diagnostics],file,'Sheet','内验证差异诊断','UseExcel',false);
end

function writeReport(C,outDir)
lines={'# 去掉PSO的配对对照','',C.bp_definition,'',C.scope,'',...
 '| 方法 | 固定MAPE（%） | 滚动MAPE（%） | 运行秒数中位数 |','|---|---:|---:|---:|'};
for m=1:2,lines{end+1}=sprintf('| %s | %.8f | %.8f | %.4f |',C.methods{m},C.fixed_metrics(m,1),C.rolling_metrics(m,1),median(C.training_seconds(m,:)));end
lines=[lines,{'','## 五组种子','','| 种子 | 方法 | 固定MAPE | 滚动MAPE | 运行秒数 |','|---|---|---:|---:|---:|'}];
for k=1:5,for m=1:2,r=C.runs{m,k};lines{end+1}=sprintf('| %d | %s | %.8f | %.8f | %.4f |',C.seeds(k),C.methods{m},r.fixed_metrics(1),r.rolling_metrics(1),C.training_seconds(m,k));end,end
lines=[lines,{'',sprintf('未来105输入点最大方法差值：%.10g 亿m³。',C.max_future_difference),'',...
 '两种方法重新执行相同训练期内选参协议，不使用待检验年份选择参数。首组种子为主结果，五组全部披露。',...
 '现有PSO-BP逐年结果复现、BP前缀隔离、解析梯度及输入文件不变检查均通过。',...
 C.timing_scope,'',...
 '比较的是去掉PSO后的正则BP集成，不是无正则单网络，也不是把局部求解器改成简单梯度下降。',...
 '人口负响应仍为统计关系；情景差值不代表部门实测或因果政策效应。'}];
lines=[lines,{'','## 相同参数诊断','','| 共同正则化参数 | 未来最大差值（亿m³） | BP减PSO平均训练目标 |','|---|---:|---:|'}];
for k=1:size(C.matched_parameter_diagnostics,1),lines{end+1}=sprintf('| %.4g | %.10g | %.10g |',C.matched_parameter_diagnostics(k,:));end
lines=[lines,{'','以上诊断不用于重选参数；主结果仍为各自训练期内验证自动选参。详情见Excel的内验证差异诊断。'}];
lines=[lines,{'','## 结果解释','',...
 sprintf('主种子全历史最终参数：PSO-BP lambda=%.4g；无PSO的BP lambda=%.4g。',C.selected_penalty(end,:)),...
 '两种方法的开发回测误差几乎一致。固定同一正则化参数后，未来预测也几乎一致；自动选参后的未来差异主要来自选参差异，不能把它理解为实际未来精度差异。',...
 '在lambda=0.1、训练截至2020年的诊断中，PSO-BP平均训练目标高于随机初始化BP，影响了末三年内验证的参数排名；PSO搜索未保证每次得到更低的训练目标。',...
 '当前小型强正则网络未显示PSO带来明显误差收益；可以优先考虑更简洁的正则BP集成，同时保留PSO-BP对照。两者都仍需要未见年份验证。'}];
writeText(fullfile(outDir,'BP与PSO_BP对照报告.md'),strjoin(lines,newline));
end

function plotComparison(C,D,outDir)
colors=[0,.5,.35;.15,.4,.8];styles={'-', '--'};
f=figure('Visible','off','Color','w','Position',[30,30,1350,590]);guard=onCleanup(@()close(f));t=tiledlayout(f,1,2,'Padding','loose');
for mode=1:2
 ax=nexttile(t);hold(ax,'on');plot(ax,2020:2023,D.Y(end-3:end),'ko-','LineWidth',1.8,'DisplayName','实际值');
 for m=1:2,r=C.runs{m,1};if mode==1,p=r.fixed_prediction;else,p=r.rolling_prediction;end
  plot(ax,2020:2023,p,'LineStyle',styles{m},'Color',colors(m,:),'LineWidth',1.8,'Marker','o','DisplayName',C.methods{m});end
 xticks(ax,2020:2023);xlabel(ax,'年份');ylabel(ax,'总需水量（亿m³）');legend(ax,'Location','best');
end
water_apply_plot_style(f);exportgraphics(f,fullfile(outDir,'BP与PSO_BP回测对照.png'),'Resolution',300);clear guard
f=figure('Visible','off','Color','w','Position',[30,30,1050,650]);guard=onCleanup(@()close(f));ax=axes(f);hold(ax,'on');
cs=lines(5);for s=1:5,for m=1:2,plot(ax,[2023;C.future_years],[D.Y(end);C.future_prediction(:,s,3,m)],'LineStyle',styles{m},'LineWidth',1.8,'Color',cs(s,:),'DisplayName',sprintf('%s %s',C.scenarios{s},C.methods{m}));end,end
xlabel(ax,'年份');ylabel(ax,'总需水量（亿m³）');legend(ax,'Location','eastoutside');water_apply_plot_style(f);
exportgraphics(f,fullfile(outDir,'BP与PSO_BP未来趋势对照.png'),'Resolution',300);clear guard
end

function writeText(file,txt)
fid=fopen(file,'w','n','UTF-8');assert(fid>=0);guard=onCleanup(@()fclose(fid));fprintf(fid,'%s',txt);clear guard
end
