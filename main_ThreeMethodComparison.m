function S=main_ThreeMethodComparison(inputFile,makeFigures)
% Run all three requested folders, then compare against freshly fitted baselines.
base=setup_water_paths;
if nargin<1,inputFile=[];end
if nargin<2,makeFigures=true;end
families={'bp','bayes','gpr'};results=cell(1,7);
for m=1:3
    fprintf('\n=== METHOD %d/3: %s ===\n',m,families{m});
    results{m}=water_run_method(families{m},inputFile,makeFigures);
end
D=water_suite_data(inputFile);[~,cfg]=water_suite_setup('bp');cfg.seed=D.seed;
names={'岭回归（原网格）','岭回归（加密网格）','普通线性回归','最近实际值'};
grids={10.^(-5:2),logspace(-5,2,29),0,0};
for j=1:4
    candidates=cell(1,numel(grids{j}));
    for k=1:numel(candidates)
        candidates{k}=struct('label',sprintf('lambda=%.8g',grids{j}(k)),'penalty',grids{j}(k));
        if j==4,candidates{k}.naive=true;end
    end
    spec=struct('name',names{j},'id',sprintf('linear_%d',j),'model','water_linear_suite_model',...
        'description','相同数据与时间协议的线性/持平对照','candidates',{candidates});
    results{j+3}=water_method_backtest(D.X,D.Y,D.years,spec,cfg);
end
S=struct('methods',{{}},'fixed_metrics',zeros(7,4),'rolling_metrics',zeros(7,4),...
    'score',zeros(7,1),'validation_years',(2020:2023)','actual',D.Y(11:14),...
    'fixed_prediction',zeros(4,7),'rolling_prediction',zeros(4,7),...
    'results',{results},'input_file',D.file,'input_sha256',D.source_sha256,...
    'validation_scope','三个方法使用同一2020—2023开发回测；候选与代码是在已知数据上开发，不是新增独立测试');
for m=1:7
    R=results{m};S.methods{m}=R.method;S.fixed_metrics(m,:)=R.fixed_metrics;
    S.rolling_metrics(m,:)=R.rolling_metrics;S.score(m)=R.score;
    S.fixed_prediction(:,m)=R.fixed_prediction;S.rolling_prediction(:,m)=R.rolling_prediction;
end
[~,S.ranking]=sort(S.score);[~,S.best_of_three]=min(S.score(1:3));
S.best_of_three_name=S.methods{S.best_of_three};
out=fullfile(base,'results','three_methods');if ~isfolder(out),mkdir(out);end
save(fullfile(out,'comparison.mat'),'S');writeText(fullfile(out,'comparison.json'),jsonencode(S,'PrettyPrint',true));
writeSummary(S,fullfile(out,'三种方法对比报告.md'));
if makeFigures,plotSummary(S,out);end
disp(array2table([S.fixed_metrics(:,1),S.rolling_metrics(:,1),S.score],...
    'VariableNames',{'Fixed_MAPE','Rolling_MAPE','Mean_MAPE'},'RowNames',S.methods));
fprintf('Best of the three by mean MAPE: %s\n',S.best_of_three_name);
end

function writeSummary(S,file)
lines={'# 三种方法统一对比','',...
    '三种方法分别位于07_BP_Ensemble、02_Bayesian_BP、03_Gaussian_Process；每个目录都有模型代码、单独运行入口和独立结果。',...
    '同一三个输入、固定2010—2019训练/2020—2023检验、逐年扩展历史；BP在训练期末3年联合选择正则与预测校准，其他方法保留末年锚定；每次仅用训练期选配置。正则BP和贝叶斯正则网络均为12成员等权集成；GPR是单个精确概率模型。',...
    '**这是开发回测，不是新增独立测试。两种检验重复使用同四年，不是8个独立样本；两项MAPE均值仅用于描述性排序。**','',...
    '| 排名 | 方法 | 固定MAPE (%) | 滚动MAPE (%) | 均值 (%) | 固定RMSE | 滚动RMSE |',...
    '|---:|---|---:|---:|---:|---:|---:|'};
for rank=1:7
    m=S.ranking(rank);lines{end+1}=sprintf('| %d | %s | %.6f | %.6f | %.6f | %.6f | %.6f |',rank,S.methods{m},S.fixed_metrics(m,1),S.rolling_metrics(m,1),S.score(m),S.fixed_metrics(m,3),S.rolling_metrics(m,3)); %#ok<AGROW>
end
lines=[lines,{'',sprintf('在三种新方案中，两项MAPE均值最低的是：**%s**。实际逐项表现如上，不保证同一方案在两个检验、每一年都胜出。',S.best_of_three_name),'',...
    '## 三种方案的配置和未来方向检查','',...
    '| 方法 | 2019训练所选配置 | 2023全历史所选配置 | 农业反向 | 工业反向 | 人口反向 | 非正预测 | 路径反序 |',...
    '|---|---|---|---:|---:|---:|---:|---:|'}];
for m=1:3
    R=S.results{m};lines{end+1}=sprintf('| %s | %s | %s | %d | %d | %d | %d | %d |',R.method,R.selected{1}.label,R.selected{5}.label,R.direction_counts); %#ok<AGROW>
end
lines=[lines,{'','方向检查覆盖每种方法105个未来输入点；农业/工业反向或路径反序提示该模型的节水情景解释风险，按实际数值披露，不删除差结果。人口响应为统计关系，不作因果解释。','',...
    '## 逐年回测',''}];
for mode=1:2
    if mode==1,p=S.fixed_prediction;label='固定起点';else,p=S.rolling_prediction;label='逐年滚动';end
    lines=[lines,{['### ',label],'','| 年份 | 实际 | 正则BP集成 | 贝叶斯正则网络 | GPR | 普通线性回归 |','|---|---:|---:|---:|---:|---:|'}]; %#ok<AGROW>
    for k=1:4,lines{end+1}=sprintf('| %d | %.6f | %.6f | %.6f | %.6f | %.6f |',S.validation_years(k),S.actual(k),p(k,[1,2,3,6]));end %#ok<AGROW>
end
lines=[lines,{'','## 运行与结果','',...
    '在根目录运行 main_ThreeMethodComparison，然后运行 test_ThreeMethodComparison。各方法也能通过各自文件夹内run_*.m单独运行。',...
    '三个文件夹的results/result.mat保存固定、滚动、最终模型；result.json保存全量参数选择与预测；因素响应图和未来情景图揭示模型在不同驱动/节水路径下的变化。',...
    '贝叶斯正则网络使用Deep Learning Toolbox的trainbr；GPR使用Statistics and Machine Learning Toolbox的fitrgp；正则BP集成沿用已验证的基础MATLAB实现。',...
    '训练期归一化、内验证、时间锚定由共享代码统一执行。源Excel只读，文件SHA256记录在JSON。原14年数据的估算与统计口径限制仍然存在。'}];
writeText(file,strjoin(lines,newline));
end

function plotSummary(S,out)
font='Arial Unicode MS';colors=[0,.5,.35;.8,.3,.1;.2,.4,.8;.5,.5,.5;.6,.4,.7;.25,.25,.25];
f=figure('Visible','off','Color','w','Position',[30,30,1300,520]);guard=onCleanup(@()close(f));t=tiledlayout(f,1,2,'Padding','compact');
for mode=1:2
    ax=nexttile(t);hold(ax,'on');plot(ax,S.validation_years,S.actual,'ko-','LineWidth',2,'DisplayName','实际值');
    if mode==1,p=S.fixed_prediction;label='固定起点';else,p=S.rolling_prediction;label='逐年滚动';end
    for m=[1,2,3,6],plot(ax,S.validation_years,p(:,m),'-o','Color',colors(m,:),'LineWidth',1.5,'DisplayName',S.methods{m});end
    set(ax,'XTick',2020:2023,'FontName',font);title(ax,label,'FontName',font);ylabel(ax,'需水量（亿立方米）','FontName',font);legend(ax,'Location','best','FontName',font);grid(ax,'on');
end
title(t,'三种方法与普通线性回归：开发回测','FontName',font);
water_apply_plot_style(f);exportgraphics(f,fullfile(out,'三种方法预测对比.png'),'Resolution',160);clear guard
f=figure('Visible','off','Color','w','Position',[30,30,1100,520]);guard=onCleanup(@()close(f));
bar([S.fixed_metrics(1:6,1),S.rolling_metrics(1:6,1)]);
set(gca,'XTickLabel',S.methods(1:6),'FontName',font,'FontSize',10);xtickangle(15);
ylabel('MAPE（%）','FontName',font);title('三种方法与三种线性对照：误差越低越好','FontName',font);
legend({'固定起点','逐年滚动'},'FontName',font);grid on
water_apply_plot_style(f);exportgraphics(f,fullfile(out,'三种方法误差对比.png'),'Resolution',160);clear guard
end

function writeText(file,content)
fid=fopen(file,'w','n','UTF-8');assert(fid>=0);guard=onCleanup(@()fclose(fid));fprintf(fid,'%s',content);clear guard
end
