function T = main_JiangsuPolicyComparison()
% 独立政策对照：复用已训练模型，不改历史数据、不重新挑选模型。
% 本文件放在需水模型根目录运行。价格转换与省级效率均为区域代理。
root=fileparts(mfilename('fullpath')); setup_water_paths;
D=water_suite_data; S=load(fullfile(root,'07_BP_Ensemble','results','result.mat'),'R','models');
assert(isequal(D.Y(11:14),S.R.actual),'历史目标改变，请先重跑原正则BP集成主程序。'); % 后面另核对历史趋势完全复现
model=S.models.final; baseX=D.X(end,:); baseY=D.Y(end);
h=readmatrix(D.file,'Sheet','历史数据','UseExcel',false); h=h(isfinite(h(:,1)),:);
r=readcell(D.file,'Sheet','未来指标','UseExcel',false); r=r(2:end,:);
% 政策原文：苏政发〔2012〕27号，2030年，省级范围。
targetYear=2030; eta0=.622; etaTarget=.69; industrialTarget2000=11; provincialLimit=600;
% 江苏统计年鉴2024表2-1、2-4：全部工业增加值；不是规上工业指数。
priceYears=(2001:2023)';
index=[111.5;113.9;117.8;116.5;116.8;116.5;116.8;113.7;111.8;113.3;112.4;111.3;110.5;108.2;108.1;107;107.5;106.6;105.7;104.1;112.9;101.8;106.3];
nominal2000=3848.52; nominal2023=49244.63;
real2000=nominal2000*prod(index/100); deflator=nominal2023/real2000;
industrialCap2023=industrialTarget2000/deflator;
years=D.future_years; eta=eta0+(etaTarget-eta0)*(years-2023)/(targetYear-2023);
agCap=h(end,9)*eta0./eta;
rows=cell(35,16); k=0; summary=cell(5,10); P=zeros(7,5); H=zeros(7,5);
for s=1:5
 a=cell2mat(r(strcmp(r(:,1),D.scenarios{s}),2:12)); assert(isequal(a(:,1),years));
 trendX=D.future_X{s,2}; trendAG=a(:,9); trendIN=trendX(:,2)*10000./a(:,10);
 % 净亩均灌溉需求不变的参考约束；与历史强度取小值，避免重复叠加节水。
 policyAG=min(trendAG,agCap);
 % 2000价上限向2023价转换；平滑衔接仅为研究路径，非政策规定的年度目标。
 inCap=h(end,13)*(industrialCap2023/h(end,13)).^((years-2023)/7);
 policyIN=min(trendIN,inCap);
 policyX=trendX; policyX(:,1)=a(:,8).*policyAG*15/100000; policyX(:,2)=a(:,10).*policyIN/10000;
 H(:,s)=water_predict_method('model_BP_Ensemble',model,trendX,baseX,baseY);
 P(:,s)=water_predict_method('model_BP_Ensemble',model,policyX,baseX,baseY);
 assert(max(abs(H(:,s)-S.R.future_prediction(:,s,2)))<1e-8,'历史趋势没有复现。');
 assert(all(P(:,s)<=H(:,s)+1e-8),'政策路径出现反向响应。');
 for i=1:7
  k=k+1; rows(k,:)={D.scenarios{s},years(i),H(i,s),P(i,s),H(i,s)-P(i,s),...
   trendAG(i),policyAG(i),agCap(i),eta(i),trendIN(i),policyIN(i),inCap(i),...
   trendIN(i)*deflator,industrialTarget2000,trendAG(i)>agCap(i)+1e-10,trendIN(i)>inCap(i)+1e-10};
 end
 summary(s,:)={D.scenarios{s},H(end,s),P(end,s),H(end,s)-P(end,s),...
  sum(H(:,s)-P(:,s)),trendAG(end),policyAG(end),trendIN(end),trendIN(end)*deflator,industrialCap2023};
end
T=cell2table(summary,'VariableNames',{'情景','历史趋势2030_亿立方米','政策参考2030_亿立方米','2030额外节水_亿立方米',...
 '2024至2030累计额外节水_亿立方米','历史农业强度2030','政策农业强度2030','历史工业强度2023价','历史工业强度2000价代理','工业上限2023价代理'});
out=fullfile(root,'results','江苏政策约束对照'); if ~isfolder(out),mkdir(out);end
file=fullfile(out,'江苏政策约束与历史趋势对照.xlsx');
writetable(T,file,'Sheet','2030对照','UseExcel',false);
detail=cell2table(rows,'VariableNames',{'情景','年份','历史趋势需水量_亿立方米','政策参考需水量_亿立方米','额外节水_亿立方米',...
 '历史亩均强度','政策亩均强度','农业参考上限','省级效率参考路径','历史工业强度2023价','政策工业强度2023价',...
 '工业年度上限2023价代理','历史工业强度2000价代理','工业2030上限2000价','农业约束触发','工业参考约束触发'});
writetable(detail,file,'Sheet','逐年对照','UseExcel',false);
policyURL='https://www.jiangsu.gov.cn/art/2012/3/15/art_46757_2680159.html';
sources={ '项目','数值或处理','适用范围与假设','来源';
 '全省2030用水总量上限',provincialLimit,'单位亿m3；全省，未分解至苏北，不截断苏北预测',policyURL;
 '工业2030强度上限2000价',industrialTarget2000,'m3/万元，2000年不变价、不含火电；计算以边界11作上限近似',policyURL;
 '灌溉有效利用系数2030',etaTarget,'省级目标作为苏北研究参考；非苏北已核实分解指标',policyURL;
 '灌溉有效利用系数2023',eta0,'江苏实测省级值；不称为苏北实测值','https://jswater.jiangsu.gov.cn/module/download/downfile.jsp?classid=0&filename=4fe2b40b2dc24027a9c29e6f3628eeec.pdf';
 '农业映射','m_cap(t)=m_2023*eta_2023/eta_policy(t)','净亩均灌溉需求保持不变；面积沿用播种面积代理；min(历史趋势,参考上限)不重复压减','本研究转换假设';
 '效率年度路径','2023至2030线性提高','为连接端点的研究假设，非官方年度分解','本研究假设';
 '工业名义增加值2000',nominal2000,'亿元，全省全部工业','https://tj.jiangsu.gov.cn/2024/nj02/nj0201.htm';
 '工业名义增加值2023',nominal2023,'亿元，全省全部工业','https://tj.jiangsu.gov.cn/2024/nj02/nj0201.htm';
 '2000至2023工业价格桥',deflator,'全省全部工业链式隐含平减代理；不是苏北不含火电专用平减指数','https://tj.jiangsu.gov.cn/2024/nj02/nj0204.htm';
 '工业上限2023价代理',industrialCap2023,'11/价格桥；当前未来规模沿用2023价GDP平减代理，仅作参考','本程序计算';
 '工业火电口径','未统一','现有强度部分估算且含火电不完全一致；不得据此认定正式工业达标','原输入口径与政策依据工作表';
 '总量与强度关系','未套用600至苏北','缺苏北分配指标及省内其他地区预测，不能完成全省达标核验','区域边界限制';
 '模型比较','同一已训练正则BP集成、同一2023锚点','不重新训练、筛选或校准；差值是模型条件情景差，不是因果效应','07_BP_Ensemble/results/result.mat';
 '源输入SHA256',D.source_sha256,'历史数据、原输入文件、原结果和工作台均未覆盖',D.file;
 '目标版本','苏政发〔2012〕27号情景试算','按用户指定文件，未确认后续承接或调整，不宣称最新正式考核标准',policyURL};
writecell(sources,file,'Sheet','参数与口径','UseExcel',false);
writetable(table(priceYears,index,'VariableNames',{'年份','全省全部工业可比价指数_上年100'}),file,'Sheet','工业价格桥输入','UseExcel',false);
% 沿用现有MATLAB图件标准：宋体/Times New Roman、内刻度、四边细框、无网格/总图题。
figdir=fullfile(out,'图件');if ~isfolder(figdir),mkdir(figdir);end
f=figure('Visible','off','Color','w','Position',[50,50,1350,680]); g=onCleanup(@()close(f));
colors=lines(5);hold on;
for s=1:5
 plot([2023;years],[baseY;H(:,s)],'-','Color',colors(s,:),'LineWidth',1.6,'DisplayName',[D.scenarios{s},' 历史趋势']);
 plot([2023;years],[baseY;P(:,s)],'--','Color',colors(s,:),'LineWidth',1.6,'DisplayName',[D.scenarios{s},' 政策参考']);
end
xlabel('年份');ylabel('总需水量（亿立方米）');xticks(2023:2030);xlim([2023,2030]);
legend('Location','southoutside','NumColumns',5);water_apply_plot_style(f);
exportgraphics(f,fullfile(figdir,'五情景政策参考与历史趋势.png'),'Resolution',220);clear g;
assert(isequal(size(readtable(file,'Sheet','逐年对照','VariableNamingRule','preserve')),[35,16]));
fprintf('农业上限2030: %.9f; 2000至2023价格桥: %.9f; 工业上限2023价代理: %.9f\n',agCap(end),deflator,industrialCap2023);
disp(T);fprintf('SAVED %s\n',file);
end
