function R = main_Water_ImprovedComparison(inputFile,outDir,makeFigures,cfg)
% New small-sample PSO-BP comparison; runs with base MATLAB R2024a.
% Saves a separate JSON/MAT report and figures. Never rewrites source Excel.
base=fileparts(fileparts(mfilename('fullpath')));addpath(base);setup_water_paths;
if nargin<1||isempty(inputFile),inputFile=fullfile(base,'data','需水预测模型对照.xlsx');end
if nargin<2||isempty(outDir),outDir=fullfile(base,'results','improved');end
if nargin<3||isempty(makeFigures),makeFigures=true;end
if nargin<4||isempty(cfg),cfg=defaultSettings(inputFile);end
if ~isfolder(outDir),mkdir(outDir);end
sourceHash=fileHash(inputFile);
h=readmatrix(inputFile,'Sheet','历史数据','UseExcel',false);h=h(isfinite(h(:,1)),:);
years=h(:,1);Y=h(:,14);D=[h(:,8),h(:,10),h(:,2).*(1+h(:,3)/100)];E=h(:,[9,13]);
X=features(D,E);assert(isequal(years,(2010:2023)'));
assert(max(abs(h(:,13)-h(:,11).*h(:,12)))<1e-8,'Industrial price conversion mismatch.');
gradientError=model_SmallSample_PSO_BP('check');
[R,models]=water_improved_backtest(X,Y,years,cfg);
R.jacobian_max_error=gradientError;
R.input_file=inputFile;R.input_sha256=sourceHash;
R.input_workbook_unchanged=strcmp(sourceHash,fileHash(inputFile));
assert(R.input_workbook_unchanged,'Source workbook changed during the run.');
R.generated_at=char(datetime('now','TimeZone','Europe/Amsterdam','Format','yyyy-MM-dd HH:mm:ss Z'));
R.matlab_version=version;
[R.future_prediction,R.future_direction,R.future_scenarios,R.future_years]=futureScenarios(inputFile,X,E,Y,years,models,cfg);
R.future_direction_columns={'农业负响应数','工业负响应数','人口负响应数','非正预测数','节水路径反序数'};
R.future_positive=all(R.future_prediction(1,:,:,:)>0,'all');
R.future_water_saving_ordered=all(R.future_direction(1,[1,2,4,5])==0);
R.final_model_parameter_count=size(models{1}.theta,1);
ablationCfg=cfg;ablationCfg.bias_penalty=0;
ablation=water_improved_backtest(X,Y,years,ablationCfg);
R.ablation_weights_only=struct('fixed_metrics',ablation.fixed_metrics(1,:),...
    'rolling_metrics',ablation.rolling_metrics(1,:),'score',ablation.score(1),...
    'lambda',ablation.lambda_by_training_end(:,1));
save(fullfile(outDir,'comparison.mat'),'R','models');
writeText(fullfile(outDir,'comparison.json'),jsonencode(R,'PrettyPrint',true));
writeReport(R,fullfile(outDir,'模型改进报告.md'));
if makeFigures,plotComparison(R,outDir);end
disp('METRICS [fixed_MAPE fixed_MAE fixed_RMSE rolling_MAPE rolling_MAE rolling_RMSE]');
disp(array2table([R.fixed_metrics(:,1:3),R.rolling_metrics(:,1:3)],'RowNames',R.methods));
fprintf('Improves original ridge in both MAPEs: %d; improves dense ridge combined MAPE: %d\n',R.improves_original_both,R.improves_dense_combined);
fprintf('Improves all three linear baselines in all eight error measures: %d\n',R.improves_all_linear_metrics);
fprintf('Future water-saving checks pass: %d\nResults: %s\n',R.future_water_saving_ordered,outDir);
end

function cfg=defaultSettings(inputFile)
% Development choices are explicit; these are not claimed to be preregistered.
% Six parameters (3-1-1), weights AND biases regularized, no global monotonicity.
cfg=struct('hidden',1,'members',12,'pso_population',24,'pso_generations',40,...
    'max_iterations',300,'seed',20261024,'nn_penalties',[.01,.03,.1,1],'bias_penalty',1,...
    'industrial_drop',.1,'feature_version','same-three-drivers-as-original');
settings=readcell(inputFile,'Sheet','模型设置','UseExcel',false);
for i=2:size(settings,1)
    if ischar(settings{i,1})&&isnumeric(settings{i,2})&&ismember(settings{i,1},{'seed','industrial_drop'})
        cfg.(settings{i,1})=settings{i,2};
    end
end
assert(cfg.industrial_drop>=0&&cfg.industrial_drop<1&&cfg.seed>=0);
end

function [pred,viol,scenarios,yf]=futureScenarios(file,X,E,Y,years,models,cfg)
r=readcell(file,'Sheet','未来指标','UseExcel',false);r=r(2:end,:);
scenarios={'SSP1','SSP2','SSP3','SSP4','SSP5'};yf=(2024:2030)';
b=[ones(numel(years),1),years-years(1)]\log(E(:,2));decay=-b(2);
M=numel(models);pred=zeros(M,5,7,3);viol=zeros(M,5);
for s=1:5
    a=cell2mat(r(strcmp(r(:,1),scenarios{s}),2:12));assert(isequal(a(:,1),yf));
    df=[a(:,8),a(:,10),a(:,2).*(1+a(:,3)/100)];
    frozen=repmat(E(end,:),7,1);
    trend=[a(:,9),E(end,2)*exp(-decay*a(:,11).*(yf-2023))];policy=trend;
    cap=trend(2,2)*(1-cfg.industrial_drop).^((yf-2025)/5);ix=yf>2025;
    policy(ix,2)=min(trend(ix,2),cap(ix));paths={frozen,trend,policy};
    for m=1:M
        for j=1:3
            xx=features(df,paths{j});p=anchored(models{m},xx,X(end,:),Y(end));
            pred(m,s,:,j)=reshape(p,1,1,7);viol(m,4)=viol(m,4)+nnz(p<=0);
            for c=1:3
                down=xx;down(:,c)=.99*down(:,c);
                change=p-anchored(models{m},down,X(end,:),Y(end));
                viol(m,c)=viol(m,c)+nnz(change< -1e-7);
            end
        end
        p=reshape(pred(m,s,:,:),7,3);
        viol(m,5)=viol(m,5)+nnz(p(:,1)<p(:,2)-1e-7|p(:,2)<p(:,3)-1e-7);
    end
end
end

function p=anchored(model,X,baseX,baseY)
switch model.kind
    case 'small_pso_bp'
        p=baseY+mean(model_SmallSample_PSO_BP('predict',model,X),2)-mean(model_SmallSample_PSO_BP('predict',model,baseX),2);
    case 'persistence'
        p=repmat(baseY,size(X,1),1);
    otherwise
        p=baseY+model_Ridge('predict',model,X)-model_Ridge('predict',model,baseX);
end
end

function X=features(D,E)
X=[D(:,1).*E(:,1)*15/100000,D(:,2).*E(:,2)/10000,D(:,3)];
end

function writeReport(R,file)
lines={'# 需水预测模型改进结果','',...
    '本轮将原3-4-1网络改为3-1-1小样本正则PSO-BP，21参数降至6参数。PSO初始化后用解析Jacobian阻尼高斯牛顿训练，直接在基础MATLAB运行。',...
    '训练集内居中并缩放；连接权重和偏置共同L2正则；候选0.01、0.03、0.1、1。保留与原模型一致的三个输入、时间划分和实际值锚定。每次仅在训练期末3年选参，内验证与正式预测均使用12个固定初始化成员。','',...
    '**验证边界：这是已知数据上的开发回测。2020—2023在此前探索及本轮模型开发中已被查看；不能将改善称为新增独立测试、显著优势或未来精度保证。**','',...
    '| 方法 | 固定MAPE (%) | 滚动MAPE (%) | 两种MAPE均值 (%) | 固定RMSE | 滚动RMSE |',...
    '|---|---:|---:|---:|---:|---:|'};
for m=1:numel(R.methods)
    lines{end+1}=sprintf('| %s | %.6f | %.6f | %.6f | %.6f | %.6f |',R.methods{m},R.fixed_metrics(m,1),R.rolling_metrics(m,1),R.score(m),R.fixed_metrics(m,3),R.rolling_metrics(m,3)); %#ok<AGROW>
end
lines=[lines,{'',sprintf('相对原岭回归，固定/滚动MAPE分别下降 %.2f%% / %.2f%%；相对加密网格岭回归，两种MAPE均值下降 %.2f%%。',R.relative_mape_reduction_original,R.relative_score_reduction_dense),...
    '加密网格岭回归使用29个候选（10^-5到10^2），原岭回归使用8个候选。另列普通最小二乘线性回归和最近实际值基准；逐项结果完整报告。','',...
    '## 逐年预测（亿立方米）','',...
    '| 年份 | 实际值 | 网络固定 | 原岭固定 | 加密岭固定 | 普通线性固定 | 网络滚动 | 原岭滚动 | 加密岭滚动 | 普通线性滚动 |',...
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'}];
for k=1:numel(R.validation_years)
    lines{end+1}=sprintf('| %d | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |',R.validation_years(k),R.actual(k),R.fixed_prediction(k,1:4),R.rolling_prediction(k,1:4)); %#ok<AGROW>
end
lines=[lines,{'','## 参数与未来情景','',...
    '| 训练截止年 | 网络正则 | 原岭正则 | 加密岭正则 |','|---|---:|---:|---:|'}];
for k=1:numel(R.training_end_years)
    lines{end+1}=sprintf('| %d | %.8g | %.8g | %.8g |',R.training_end_years(k),R.lambda_by_training_end(k,1:3)); %#ok<AGROW>
end
lines=[lines,{'',sprintf('消融对照（同一3-1-1结构、同一候选网格，仅不惩罚偏置）：固定MAPE %.6f%%，滚动MAPE %.6f%%。完整模型共同约束权重与偏置后，取得上表所示的更均衡回测。',R.ablation_weights_only.fixed_metrics(1),R.ablation_weights_only.rolling_metrics(1)),...
    '',sprintf('网络未来105个情景检查点：农业反向、工业反向、人口反向、非正预测、节水路径反序计数依次为 %s。人口响应单独披露，不作因果解释。',mat2str(R.future_direction(1,:))),...
    '未来使用2023全历史模型，按训练期内验证另选正则；不把2019模型的回测参数硬套到最终预测。未来路径与原程序相同，结果见comparison.json或comparison.mat。','',...
    '## 复现与数据限制','',...
    '运行 main_Water_ImprovedComparison，再运行 test_Water_ImprovedComparison。comparison.mat保存网络权重、每个成员的训练目标/停止原因及最终线性模型；comparison.json保存完整选参逐折记录、逐年预测和未来情景。',...
    '原始Excel及其已有结果/图表保持不变；改进结果单独写入results/improved。工业强度含估算值、历年火电口径未统一、农业面积为代理，这些原数据限制仍然存在。',...
    '本轮先探索了较小直接网络、残差网络、贝叶斯正则网络和多步内验证；直接照搬原弱正则网格仍外推失真。仅正则连接权重的小网络优于原岭回归，但固定起点不及普通线性回归。最终采用单隐藏节点、权重和偏置共同正则与较强候选网格。方案本身经过已知开发回测选择，新增未见数据仍需独立检验。',''}];
writeText(file,strjoin(lines,newline));
end

function plotComparison(R,outDir)
font='Arial Unicode MS';colors=[0,.45,.32;.15,.42,.72;.85,.45,.12;.55,.35,.65];styles={'-','--',':','-.'};
f=figure('Visible','off','Color','w','Position',[50,50,1300,520]);guard=onCleanup(@()close(f));
t=tiledlayout(f,1,2,'TileSpacing','compact','Padding','compact');
names={'固定起点：2010—2019训练','逐年滚动：只使用此前历史'};
for mode=1:2
    ax=nexttile(t);hold(ax,'on');plot(ax,R.validation_years,R.actual,'ko-','LineWidth',2,'DisplayName','实际值');
    if mode==1,p=R.fixed_prediction;else,p=R.rolling_prediction;end
    for m=1:4,plot(ax,R.validation_years,p(:,m),'Marker','o','LineStyle',styles{m},'Color',colors(m,:),'LineWidth',1.8,'DisplayName',R.methods{m});end
    title(ax,names{mode},'FontName',font);xlabel(ax,'年份','FontName',font);ylabel(ax,'总需水量（亿立方米）','FontName',font);
    set(ax,'FontName',font,'FontSize',12,'XTick',2020:2023);box(ax,'off');grid(ax,'on');
    legend(ax,'Location','best','FontName',font,'FontSize',10);
end
title(t,'开发回测：同输入、同时间划分、同实际值锚定','FontName',font,'FontSize',16);
exportgraphics(f,fullfile(outDir,'改进网络与线性模型对照.png'),'Resolution',180);clear guard
f=figure('Visible','off','Color','w','Position',[50,50,900,540]);guard=onCleanup(@()close(f));
bar([R.fixed_metrics(1:4,1),R.rolling_metrics(1:4,1)]);set(gca,'XTickLabel',R.methods(1:4),'FontName',font,'FontSize',12);
ylabel('MAPE（%）','FontName',font);legend({'固定起点','逐年滚动'},'FontName',font,'Location','northeast');
title('开发回测误差（越低越好）','FontName',font);box off
exportgraphics(f,fullfile(outDir,'改进网络误差对比.png'),'Resolution',180);clear guard
end

function writeText(file,content)
fid=fopen(file,'w','n','UTF-8');assert(fid>=0,'Cannot write %s',file);
guard=onCleanup(@()fclose(fid));fprintf(fid,'%s',content);clear guard
end

function h=fileHash(file)
fid=fopen(file,'r');assert(fid>=0);guard=onCleanup(@()fclose(fid));bytes=fread(fid,Inf,'*uint8');clear guard
digest=java.security.MessageDigest.getInstance('SHA-256');digest.update(bytes);
h=lower(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[]));
end
