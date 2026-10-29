function water_apply_plot_style(f)
% 中文宋体，数字Times New Roman；四边细框，左下向内刻度，无网格和图题。
axesList=findall(f,'Type','axes');
for k=1:numel(axesList)
 ax=axesList(k);set(ax,'FontName','Times New Roman','FontSize',15,'TickDir','in',...
  'LineWidth',.6,'Box','off','XGrid','off','YGrid','off','XMinorGrid','off','YMinorGrid','off',...
  'XAxisLocation','bottom','YAxisLocation','left');
 ax.Title.String='';ax.XLabel.FontName='宋体';ax.YLabel.FontName='宋体';
 ax.XLabel.FontSize=18;ax.YLabel.FontSize=18;
 labels=strjoin(cellstr(string(ax.XTickLabel)),'');
 if ~isempty(regexp(labels,'[一-龥]','once')),ax.XAxis.FontName='宋体';end
 xl=xlim(ax);yl=ylim(ax);set(ax,'XLimMode','manual','YLimMode','manual');
 delete(findall(ax,'Tag','WaterStyleFrame'));
 line(ax,xl,[yl(2),yl(2)],'Color','k','LineWidth',.6,'HandleVisibility','off','Tag','WaterStyleFrame');
 line(ax,[xl(2),xl(2)],yl,'Color','k','LineWidth',.6,'HandleVisibility','off','Tag','WaterStyleFrame');
end
legends=findall(f,'Type','legend');set(legends,'FontName','宋体','FontSize',14,'Box','off');
layouts=findall(f,'-isa','matlab.graphics.layout.TiledChartLayout');
for k=1:numel(layouts),layouts(k).Title.String='';end
end
