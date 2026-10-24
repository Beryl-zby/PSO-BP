function R=water_run_method(family,inputFile,makeFigures)
% Run one of the three requested methods and save it in its own folder.
if nargin<2,inputFile=[];end
if nargin<3,makeFigures=true;end
[spec,cfg]=water_suite_setup(family);D=water_suite_data(inputFile);cfg.seed=D.seed;
[R,models]=water_method_backtest(D.X,D.Y,D.years,spec,cfg);
R.input_file=D.file;R.input_sha256=D.source_sha256;
R.matlab_version=version;R.future_years=D.future_years;R.scenarios=D.scenarios;R.paths=D.paths;
R.future_prediction=zeros(7,5,3);R.direction_counts=zeros(1,5);
R.direction_columns={'农业负响应','工业负响应','人口负响应','非正需水','路径反序'};
baseX=D.X(end,:);baseY=D.Y(end);model=models.final;
for s=1:5
    for j=1:3
        xx=D.future_X{s,j};p=anchored(xx);R.future_prediction(:,s,j)=p;
        R.direction_counts(4)=R.direction_counts(4)+nnz(p<=0);
        for c=1:3
            down=xx;down(:,c)=.99*down(:,c);
            R.direction_counts(c)=R.direction_counts(c)+nnz(p-anchored(down)< -1e-7);
        end
        if strcmp(family,'gpr')
            [raw,sd]=feval(spec.model,'predict',model,xx);
            R.gpr_raw_future_mean(:,s,j)=raw;R.gpr_raw_future_sd(:,s,j)=sd;
        end
    end
    paths=reshape(R.future_prediction(:,s,:),7,3);
    R.direction_counts(5)=R.direction_counts(5)+nnz(paths(:,1)<paths(:,2)-1e-7|paths(:,2)<paths(:,3)-1e-7);
end
R.future_checks_passed=all(R.direction_counts([1,2,4,5])==0);
R.response_fraction=linspace(-.2,.2,81)';R.response_prediction=zeros(81,3);
R.feature_names=D.feature_names;
for c=1:3
    xx=repmat(baseX,81,1);xx(:,c)=baseX(c)*(1+R.response_fraction);
    R.response_prediction(:,c)=anchored(xx);
end
if strcmp(family,'bayes')
    R.final_training_stops=model.stops;R.final_training_epochs=model.epochs;
elseif strcmp(family,'gpr')
    R.final_kernel_information=model.kernel_information;R.final_noise_sigma=model.noise_sigma;
    R.uncertainty_scope='gpr_raw_future_sd是未锚定预测的模型标准差；统一回测使用锚定均值，未将此标准差直接当作锚定预测的95%区间';
end
after=water_suite_data(D.file);assert(strcmp(D.source_sha256,after.source_sha256));
R.input_workbook_unchanged=true;
out=fullfile(spec.path,'results');if ~isfolder(out),mkdir(out);end
save(fullfile(out,'result.mat'),'R','models');
writeText(fullfile(out,'result.json'),jsonencode(R,'PrettyPrint',true));
writeReport(R,fullfile(out,'结果报告.md'));
if makeFigures,makePlots(R,out);end
fprintf('SAVED %s | fixed %.6f%% | rolling %.6f%% | mean %.6f%%\n',spec.folder,R.fixed_metrics(1),R.rolling_metrics(1),R.score);

    function p=anchored(xx)
        p=baseY+feval(spec.model,'predict',model,xx)-feval(spec.model,'predict',model,baseX);
    end
end

function writeReport(R,file)
lines={['# ',R.method],'',R.description,'',...
    '**共同口径：** 2010—2019固定训练预测2020—2023；逐年滚动每次只用此前历史；训练期内缩放；实际末年锚定；末3年内验证选参。',...
    '**适用范围：** 已知数据开发回测，2020—2023已用于探索；不能作为新增独立测试或显著优越的证据。','',...
    '| 检验 | MAPE (%) | MAE | RMSE | MSE |','|---|---:|---:|---:|---:|',...
    sprintf('| 固定起点 | %.6f | %.6f | %.6f | %.6f |',R.fixed_metrics),...
    sprintf('| 逐年滚动 | %.6f | %.6f | %.6f | %.6f |',R.rolling_metrics),'',...
    '| 年份 | 实际值 | 固定预测 | 滚动预测 |','|---|---:|---:|---:|'};
for k=1:4,lines{end+1}=sprintf('| %d | %.6f | %.6f | %.6f |',R.validation_years(k),R.actual(k),R.fixed_prediction(k),R.rolling_prediction(k));end %#ok<AGROW>
lines=[lines,{'','## 训练期内参数选择','','| 训练截止年 | 所选配置 | 内验证MAPE (%) |','|---|---|---:|'}];
for k=1:5,lines{end+1}=sprintf('| %d | %s | %.6f |',R.training_end_years(k),R.selected{k}.label,R.inner_mape(k));end %#ok<AGROW>
lines=[lines,{'','## 情景与因素响应','',...
    sprintf('未来105点：农业负响应、工业负响应、人口负响应、非正预测、路径反序计数为 %s。前三项为驱动降低1%%却增加需求的点数。',mat2str(R.direction_counts)),...
    '方向检查是结果诊断，不是所有模型都具有的结构性保证；不据此改动测试预测或删除差模型。人口响应不解释为因果效应。',...
    '因素响应图把其他驱动固定在2023基值，单个驱动上下扰动20%；可能包含训练范围外输入，用于揭示外推形状。未来情景与原项目相同。','',...
    'result.mat保存固定、滚动和最终模型；result.json保存完整逐折选参、误差、未来情景和响应曲线。根目录三模型对比报告提供全部模型及线性基准排名。'}];
if strcmp(R.family,'bayes')
    lines=[lines,{'','贝叶斯正则使用MATLAB trainbr，自动平衡数据误差与权重/偏置惩罚。它不是完整的贝叶斯神经网络，12次初始化差异不解释为后验预测区间。',...
        '技术来源：https://www.mathworks.com/help/deeplearning/ref/trainbr.html'}];
elseif strcmp(R.family,'gpr')
    lines=[lines,{'',R.uncertainty_scope,...
        '技术来源：https://www.mathworks.com/help/stats/fitrgp.html'}];
end
writeText(file,strjoin(lines,newline));
end

function makePlots(R,out)
font='Arial Unicode MS';f=figure('Visible','off','Color','w','Position',[30,30,1100,460]);guard=onCleanup(@()close(f));
t=tiledlayout(f,1,2,'Padding','compact');
for j=1:2
    ax=nexttile(t);plot(ax,R.validation_years,R.actual,'ko-','LineWidth',1.8);hold(ax,'on');
    if j==1,p=R.fixed_prediction;name='固定起点';else,p=R.rolling_prediction;name='逐年滚动';end
    plot(ax,R.validation_years,p,'o-','Color',[0,.45,.65],'LineWidth',1.8);
    set(ax,'XTick',2020:2023,'FontName',font,'FontSize',12);title(ax,name,'FontName',font);
    ylabel(ax,'需水量（亿立方米）','FontName',font);legend(ax,{'实际值',R.method},'Location','best','FontName',font);grid(ax,'on');
end
title(t,[R.method,'：开发回测'],'FontName',font);
exportgraphics(f,fullfile(out,'时间检验.png'),'Resolution',150);clear guard
f=figure('Visible','off','Color','w','Position',[30,30,1100,390]);guard=onCleanup(@()close(f));t=tiledlayout(f,1,3,'Padding','compact');
for c=1:3
    ax=nexttile(t);plot(ax,100*R.response_fraction,R.response_prediction(:,c),'LineWidth',1.8);
    title(ax,R.feature_names{c},'FontName',font);xlabel(ax,'相对2023基值变化（%）','FontName',font);
    ylabel(ax,'预测需水量（亿立方米）','FontName',font);set(ax,'FontName',font);grid(ax,'on');
end
title(t,'模型条件响应：其余输入固定；包含外推探测','FontName',font);
exportgraphics(f,fullfile(out,'因素响应.png'),'Resolution',150);clear guard
f=figure('Visible','off','Color','w','Position',[30,30,900,460]);guard=onCleanup(@()close(f));hold on
for s=1:5,plot([2023;R.future_years],[R.actual(end);R.future_prediction(:,s,3)],'-o','LineWidth',1.5,'DisplayName',R.scenarios{s});end
set(gca,'FontName',font);xlabel('年份','FontName',font);ylabel('需水量（亿立方米）','FontName',font);
title([R.method,'：规划约束路径'],'FontName',font);legend('Location','best');grid on
exportgraphics(f,fullfile(out,'未来情景.png'),'Resolution',150);clear guard
end

function writeText(file,content)
fid=fopen(file,'w','n','UTF-8');assert(fid>=0);guard=onCleanup(@()fclose(fid));fprintf(fid,'%s',content);clear guard
end
