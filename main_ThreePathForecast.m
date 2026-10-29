function T=main_ThreePathForecast(retrain)
% 五SSP三路径统一入口；参数与数据读取data/需水预测模型对照.xlsx。
% 近期效率固定于2021—2023均值；政策只作用农业工业；总量仅校核。
root=fileparts(mfilename('fullpath'));addpath(root);setup_water_paths;refreshPathSnapshot;
if nargin<1,retrain=true;end
if retrain,run_BP_Ensemble([],false);end
D=water_suite_data;S=load(fullfile(root,'07_BP_Ensemble','results','result.mat'),'R','models');R=S.R;models=S.models;
assert(isequal(R.actual,D.Y(11:14)),'历史目标与训练版本不一致');assert(strcmp(R.input_sha256,D.source_sha256),'训练文件指纹不一致，请重新训练');
limit=D.policy_settings.regional_water_limit;base=D.Y(end);f0=base+model_BP_Ensemble('predict',models.final,D.frozen_base_X)-model_BP_Ensemble('predict',models.final,D.X(end,:));baselineEffect=f0-base;
out=fullfile(root,'results','三路径预测_BP_最新');figdir=fullfile(out,'图件');if ~isfolder(figdir),mkdir(figdir);end
file=fullfile(out,'五SSP三路径需水预测_最新.xlsx');
headers={'情景','年份','近期平均效率维持','历史趋势延续','政策规划参考','发展变化_剔除效率基准差异','效率基准差异','历史趋势节水','额外政策节水','合计节水','政策相对2023净变化','节水抵消发展比例','2030总量参考上限','政策超参考上限'};
C=cell(35,14);k=0;summary=cell(5,7);wide=cell(8,16);wide(:,1)=num2cell((2023:2030)');labels={'年份'};
for s=1:5
 p=reshape(R.future_prediction(:,s,:),7,3);dev=p(:,1)-f0;hist=p(:,1)-p(:,2);extra=p(:,2)-p(:,3);saved=p(:,1)-p(:,3);net=p(:,3)-base;
 assert(max(abs(dev+baselineEffect-saved-net))<1e-9);
 assert(all(p(:,1)>=p(:,2)-1e-8 & p(:,2)>=p(:,3)-1e-8),'节水路径反序，应检查模型响应，不能强制排序');
 for i=1:7
  k=k+1;ratio=NaN;if dev(i)>1e-9,ratio=saved(i)/dev(i);end
  ref='';exceed='';if D.future_years(i)==2030,ref=limit;exceed=max(0,p(i,3)-limit);end
  C(k,:)={D.scenarios{s},D.future_years(i),p(i,1),p(i,2),p(i,3),dev(i),baselineEffect,hist(i),extra(i),saved(i),net(i),ratio,ref,exceed};
 end
 summary(s,:)={D.scenarios{s},p(end,1),p(end,2),p(end,3),extra(end),limit,max(0,p(end,3)-limit)};
 wide(:,(s-1)*3+(2:4))=num2cell([base base base;p]);for j=1:3,labels{end+1}=[D.scenarios{s},'_',D.paths{j}];end
 small=cell2table([num2cell((2023:2030)'),num2cell([base base base;p])],'VariableNames',{'年份','近期平均效率维持','历史趋势延续','政策规划参考'});
 writetable(small,file,'Sheet',D.scenarios{s},'WriteMode','overwritesheet','UseExcel',false);
 f=figure('Visible','off','Color','w','Position',[30 30 1000 620]);guard=onCleanup(@()close(f));hold on;
 colors=[.15 .55 .38;.15 .40 .73;.90 .48 .10];styles={'-','--',':'};
 legendNames={'近期平均效率维持','历史趋势延续','政策规划参考'};
 for j=1:3,plot(2023:2030,[base;p(:,j)],'LineStyle',styles{j},'Color',colors(j,:),'LineWidth',1.8,'DisplayName',legendNames{j});end
 % 总量参考仅在表中校核，趋势图不绘制，以免拉大纵轴。
 xlabel('年份');ylabel('总需水量（亿立方米）');xlim([2023 2030]);xticks(2023:2030);legend('Location','southoutside','NumColumns',2);water_apply_plot_style(f);exportgraphics(f,fullfile(figdir,[D.scenarios{s},'_三条路径.png']),'Resolution',250);clear guard
end
writecell([headers;C],file,'Sheet','逐年路径与节水','WriteMode','overwritesheet','UseExcel',false);
T=cell2table(summary,'VariableNames',{'情景','近期效率维持2030','历史趋势2030','政策路径2030','2030额外政策节水','2030参考上限','政策超参考上限'});writetable(T,file,'Sheet','2030汇总','WriteMode','overwritesheet','UseExcel',false);
writecell([labels;wide],file,'Sheet','15组合汇总','WriteMode','overwritesheet','UseExcel',false);
pathInput=readcell(D.file,'Sheet','节水路径输入','UseExcel',false);pathInput=clean(pathInput);writecell(pathInput,file,'Sheet','路径输入','WriteMode','overwritesheet','UseExcel',false);
history=clean(readcell(D.file,'Sheet','历史数据','UseExcel',false));writecell(history,file,'Sheet','历史输入','WriteMode','overwritesheet','UseExcel',false);
back=table(R.validation_years,R.actual,R.fixed_prediction,R.rolling_prediction,'VariableNames',{'年份','实际用水量','固定起点预测','滚动预测'});writetable(back,file,'Sheet','历史回测','WriteMode','overwritesheet','UseExcel',false);writetable(back,D.file,'Sheet','时间检验逐年值','WriteMode','overwritesheet','UseExcel',false);
metrics=cell2table([{'固定起点'},num2cell(R.fixed_metrics);{'滚动起点'},num2cell(R.rolling_metrics)],'VariableNames',{'检验','MAPE百分比','MAE','RMSE','MSE'});writetable(metrics,file,'Sheet','误差指标','WriteMode','overwritesheet','UseExcel',false);
for j=1:3
 v=squeeze(R.future_prediction(:,:,j));tab=array2table([(2023:2030)',[repmat(base,1,5);v]],'VariableNames',[{'年份'},D.scenarios]);writetable(tab,D.file,'Sheet',['主模型_',D.paths{j}],'WriteMode','overwritesheet','UseExcel',false);
end
writecell([headers;C],D.file,'Sheet','主模型贡献分解','WriteMode','overwritesheet','UseExcel',false);
notes={'项目','设置与含义';'预测模型','3-1-1正则BP、12成员；随机小初值直接局部训练，无PSO；其他训练设置沿用配对检验版本';'2022历史目标','194.3亿m3';'近期效率对照','2021—2023农业与工业2023价强度均值；不声称实际效率不变';'农业政策参考','省级eta2023=0.622线性至2030=0.69；亩均参考上限=m2023×eta2023/eta(t)，与趋势取小值';'工业政策参考','2000不变价、不含火电11；省级价格桥换算后与趋势取小值；现有火电与价格代理边界未完全统一，不认证正式达标';'生活政策','不额外设约束；沿用人口城镇化代理，无新增生活强度特征';'总量参考','600×165/571.4=173.258662933亿m3；研究折算而非正式苏北配额，仅2030检查，不截断';'效率基准差异',sprintf('近期均值效率下的2023反事实预测%.9f，相对2023实际差%.9f；独立输出以免计入发展效应',f0,baselineEffect);'分解恒等式','政策净变化=发展变化+效率基准差异−历史节水−额外政策节水；均为模型条件差异，不是政策因果效应';'回测性质','固定2010—2019预测2020—2023；逐年滚动；均为已用于开发的历史回测';'其他方法结果','本轮仅重跑当前正则BP集成；原其他方法排名与旧误差未同步重算，不能据其继续宣称主模型最优';'数据限制','分行业重构量是估计参考，未替换训练驱动；面积为播种代理；工业火电口径未完全统一'};
writecell(notes,file,'Sheet','参数与口径','WriteMode','overwritesheet','UseExcel',false);writecell(notes,D.file,'Sheet','方法与结论','WriteMode','overwritesheet','UseExcel',false);
f=figure('Visible','off','Color','w','Position',[30 30 1000 600]);g=onCleanup(@()close(f));plot(R.validation_years,[R.actual R.fixed_prediction R.rolling_prediction],'-o','LineWidth',1.7);legend({'实际值','固定起点','滚动起点'},'Location','southoutside','NumColumns',3);xticks(2020:2023);xlabel('年份');ylabel('总用水量（亿立方米）');water_apply_plot_style(f);exportgraphics(f,fullfile(figdir,'修正输入后的历史回测.png'),'Resolution',250);clear g
% Outputs inside input workbook change its full-file hash, but history and features must be identical.
after=water_suite_data;assert(isequal(after.X,D.X)&&isequal(after.Y,D.Y)&&isequal(after.future_X,D.future_X));
R.input_sha256=after.source_sha256;R.frozen_base_demand=f0;R.baseline_efficiency_effect=baselineEffect;R.regional_water_limit=limit;
save(fullfile(root,'07_BP_Ensemble','results','result.mat'),'R','models');fid=fopen(fullfile(root,'07_BP_Ensemble','results','result.json'),'w','n','UTF-8');fprintf(fid,'%s',jsonencode(R,'PrettyPrint',true));fclose(fid);
fid=fopen(fullfile(out,'latest_result.json'),'w','n','UTF-8');fprintf(fid,'%s',jsonencode(R,'PrettyPrint',true));fclose(fid);
node='C:/Users/zby/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe';
for script={'sync-research-input.mjs','cache-latest-matlab.mjs'}
 command=sprintf('"%s" "%s"',node,fullfile(root,'06_WaterLab_Web','scripts',script{1}));[code,msg]=system(command);assert(code==0,'工作台同步失败：%s',msg);
end
disp(T);fprintf('SAVED %s\n',file);
end
function refreshPathSnapshot()
D=water_suite_data;r=readcell(D.file,'Sheet','未来指标','UseExcel',false);r=r(2:end,:);rows=cell(105,11);k=0;
for s=1:5
 a=cell2mat(r(strcmp(r(:,1),D.scenarios{s}),2:12));
 for j=1:3
  xx=D.future_X{s,j};ag=xx(:,1)*100000./(a(:,8)*15);ind=xx(:,2)*10000./a(:,10);
  for t=1:7
   k=k+1;eta=D.policy_settings.irrigation_base+(D.policy_settings.irrigation_target-D.policy_settings.irrigation_base)*(D.future_years(t)-2023)/7;
   rows(k,:)={D.scenarios{s},D.paths{j},D.future_years(t),a(t,8),ag(t),a(t,10),ind(t),xx(t,1),xx(t,2),xx(t,3),eta};
  end
 end
end
old=clean(readcell(D.file,'Sheet','节水路径输入','UseExcel',false));new=[old(1,:);rows];if ~isequaln(old,new),writecell(new,D.file,'Sheet','节水路径输入','WriteMode','overwritesheet','UseExcel',false);end
end
function a=clean(a)
for i=1:numel(a),if any(ismissing(a{i}),'all'),a{i}='';end;end
end
