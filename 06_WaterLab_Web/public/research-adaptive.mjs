import {importTable,finalizeForecast,selectCandidate} from './adaptive-engine.mjs';
import {prepareResearch} from './research-engine.mjs';
export function researchAdaptiveData(bundle,scenario='SSP2',path='policy'){
 const prepared=prepareResearch(bundle),selected=prepared.paths.find(c=>c.scenario===scenario&&c.path===path);
 if(!selected)throw Error('SSP或节水路径无效');
 const matrix=[...prepared.rows.map(r=>[r.year,r.y,...r.x]),...selected.points.map(r=>[r.year,'',...r.x])];
 return importTable(['年份','苏北总需水量','农业驱动','工业驱动','人口城镇化驱动'],matrix,{time:0,target:1,features:[2,3,4],unit:'亿m³',name:'苏北五市真实数据 · '+scenario+' · '+selected.name});
}
export function researchPlanningContext(scenario,path){return {
 research:'苏北五市调水受水区总需水预测；评估发展增加的需水能否被效率改善抵消',
 history:'2010—2023年度数据；统计总用水量为总需水代理',future:'2024—2030，SSP1—SSP5×三条节水路径',selectedScenario:scenario,selectedPath:path,
 drivers:['农业=播种面积代理灌溉面积×亩均用水强度×单位换算','工业=2023价格代理工业增加值×同口径强度×单位换算','生活代理=人口×(1+城镇化率/100)'],
 assumptions:['未来驱动由既有SSP预测给定，不能由Agent改写','工业火电口径未完全统一，指标为代理','农业按江苏灌溉效率0.69目标换算参考上限；工业按2000价11及价格桥作参考约束；均与历史趋势取小值','年度数据不默认季节周期','规划Agent只看开发期统计和验证反馈，不接收末段目标或其统计','目的为试算优化，不覆盖原主模型，也不保证优于固定方案'],
 };}
export function finalizeResearchAdaptive(bundle,candidates,settings={}){
 const prepared=prepareResearch(bundle),selected=selectCandidate(candidates,settings),combinations=prepared.paths.map(c=>{
  const data=researchAdaptiveData(bundle,c.scenario,c.path),r=finalizeForecast(data,candidates,{...settings,horizon:7,season:0,driverStrategy:'provided'});
  return {scenario:c.scenario,path:c.path,pathName:c.name,rows:r.future};
 });
 const referenceData=researchAdaptiveData(bundle,'SSP2','frozen');
 const base=prepared.base,eff=prepared.frozenEfficiency;
 const x=[base.area*eff.agriculturalIntensity*15/100000,base.industrialVA*eff.industrialIntensity/10000,base.population*(1+base.urbanization/100)];
 referenceData.future=referenceData.future.map(r=>({...r,x:[...x]}));
 const referenceRows=finalizeForecast(referenceData,candidates,{...settings,horizon:7,season:0,driverStrategy:'provided'}).future;
 return {referenceRows,referenceDefinition:"各预测年份发展规模保持2023年，农业工业强度维持近期均值；同模型递推。相对实际值的基准差异含拟合及递推偏差。",model:selected.name,candidate:selected.candidate,usesFutureDrivers:selected.candidate.useDrivers,source:bundle.source,combinations,scope:'同一验证冻结模型，15种给定未来驱动；无驱动模型各组合相同，不代表SSP没有影响'};
}
