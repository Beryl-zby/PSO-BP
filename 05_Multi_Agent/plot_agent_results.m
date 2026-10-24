function plot_agent_results()
root=fileparts(fileparts(mfilename('fullpath')));out=fullfile(root,'05_Multi_Agent','results');
R=jsondecode(fileread(fullfile(out,'result.json')));font='Arial Unicode MS';
f=figure('Visible','off','Color','w','Position',[30,30,1300,520]);guard=onCleanup(@()close(f));
t=tiledlayout(f,1,2,'Padding','compact');
S=load(fullfile(root,'results','three_methods','comparison.mat'),'S');S=S.S;
for mode=1:2
    ax=nexttile(t);hold(ax,'on');plot(ax,R.validation_years,R.actual,'ko-','LineWidth',2,'DisplayName','实际');
    if mode==1,key='fixed_prediction';label='固定起点';else,key='rolling_prediction';label='逐年滚动';end
    for m=1:2,plot(ax,R.validation_years,R.results(m).(key),'-o','LineWidth',1.5,'DisplayName',R.results(m).method);end
    plot(ax,R.validation_years,S.(key)(:,1),'--o','LineWidth',1.5,'DisplayName','正则PSO-BP');
    plot(ax,R.validation_years,S.(key)(:,6),':o','LineWidth',1.5,'DisplayName','普通线性回归');
    set(ax,'FontName',font,'XTick',2020:2023);title(ax,label,'FontName',font);ylabel(ax,'需水量（亿立方米）','FontName',font);
    legend(ax,'Location','best','FontName',font);grid(ax,'on');
end
title(t,'本地千问多智能体：同一已知四年的开发回测','FontName',font);
exportgraphics(f,fullfile(out,'agents预测对比.png'),'Resolution',150);clear guard
f=figure('Visible','off','Color','w','Position',[30,30,1100,500]);guard=onCleanup(@()close(f));
values=zeros(numel(R.ranking),2);names=cell(numel(R.ranking),1);
for k=1:numel(R.ranking)
    values(k,:)=[R.ranking(k).fixed_metrics(1),R.ranking(k).rolling_metrics(1)];names{k}=R.ranking(k).method;
end
bar(values);set(gca,'XTick',1:numel(names),'XTickLabel',names,'FontName',font,'FontSize',10);xtickangle(25);
ylabel('MAPE（%）','FontName',font);legend({'固定起点','逐年滚动'},'FontName',font);grid on
exportgraphics(f,fullfile(out,'agents误差对比.png'),'Resolution',150);clear guard
end
