import {SSP_NAMES,PATHS} from '../public/research-engine.mjs';

export function validateMultiResearch(result){
 if(result?.schema!=='waterlab-ssp-result-v1'||result.combinations?.length!==15)throw Error('Multi-agent需要当前论文模型的15组SSP路径结果');
 const ids=new Set();
 for(const c of result.combinations){
  const id=c.scenario+'.'+c.path;
  if(!SSP_NAMES.includes(c.scenario)||!PATHS.some(p=>p.id===c.path)||ids.has(id))throw Error('SSP与路径重复或不完整');
  ids.add(id);
  if(c.points.length!==7||c.points.some((p,i)=>p.year!==2024+i||!Number.isFinite(p.demand)))throw Error('预测结果须含2024—2030逐年有限需水值');
 }
 if(result.history.at(-1)?.year!==2023||result.baseDemand!==result.history.at(-1)?.y)throw Error('2023历史锚点不一致');
 return result;
}
export function researchMultiEvidence(result,scenario='SSP2',path='policy'){
 validateMultiResearch(result);
 if(!SSP_NAMES.includes(scenario)||!PATHS.some(p=>p.id===path))throw Error('未知SSP或节水路径');
 const facts={'observed.last':result.baseDemand};if(result.metrics.agent){facts['metrics.agent.rmse']=result.metrics.agent.rmse;facts['metrics.agent.mae']=result.metrics.agent.mae;}else{facts['metrics.rolling.mape']=result.metrics.rolling.mape;facts['metrics.fixed.rmse']=result.metrics.fixed.rmse;}
 for(const c of result.combinations){facts['terminal.'+c.scenario+'.'+c.path]=c.points.at(-1).demand;}
 for(const c of result.combinations.filter(c=>c.scenario===scenario))for(const p of c.points)facts['annual.'+c.path+'.'+p.year]=p.demand;
 for(const row of result.comparisons.filter(r=>r.scenario===scenario&&r.year===2030))for(const [key,value] of Object.entries(row))if(typeof value==='number'&&Number.isFinite(value)&&key!=='year')facts['comparison.'+row.year+'.'+key]=value;
 return {task:'ssp-water-demand',model:result.provenance?.method,modelSelection:result.provenance?.selectionPolicy,taskAssessment:result.provenance?.taskAssessment,researchBackground:result.researchBackground,validation:result.provenance?.validation,scoringFields:result.metrics.agent?{calibration:['metrics.agent.rmse','terminal.SSP2.policy','observed.last'],heldout:['metrics.agent.mae','terminal.SSP5.frozen']}:null,unit:'亿m³',source:result.source?.sha256,selection:{scenario,path},historyYears:'2010—2023',futureYears:'2024—2030',scenarios:SSP_NAMES,paths:Object.fromEntries(PATHS.map(p=>[p.id,p.name])),facts,
  selectedTerminal:result.comparisons.find(r=>r.scenario===scenario&&r.year===2030),
  policy:{...result.policy,identicalTrendPolicyPoints:result.diagnostics.identicalTrendPolicyPoints},
  definitions:{development:'development=未来效率维持需水－同模型基期效率维持反事实需水；baselineEfficiencyEffect是近期平均效率与2023实测效率的基期衔接差，不是新增节水。historicalSaving=frozen－trend；additionalPolicySaving=trend－policy；totalSaving=frozen－policy；offsetRatio仅在发展增需为正时定义。比较totalSaving和development，判断抵消程度；不得把基期衔接差当作节水。',frozen:'农业及工业强度保持2021—2023均值，发展规模按SSP变化',trend:'农业沿输入趋势；工业历史指数节水速度按SSP倍数外推',policy:'江苏2012年政策参考；农业按省级效率0.622至0.69换算强度上限；工业按2000价11及省级价格桥换算上限；两者分别取trend与上限较小值；苏北173.26亿m³为2030份额折算参考，仅作超限检查，不截断；全省600不套苏北，不认定正式达标'},
  scope:'苏北五市总需水；15组终期值及当前SSP逐年值可核对；其他SSP逐年分析请切换情景。面积为播种面积代理；工业采用既有价格与强度代理。生活未另设节水路径。历史回测已参与开发。节水差值不是因果效果；重合按实值说明。'};
}
export function researchMultiPlotData(result,options={}){
 validateMultiResearch(result);
 return {task:'ssp-water-demand',modelName:result.provenance?.method,decompositionLabel:result.decompositionLabel||'Efficiency baseline',unitLabel:'100 million m3',sourceUnit:'亿m³',source:result.source?.sha256,baseYear:result.baseYear,baseDemand:result.baseDemand,
  history:result.history.map(r=>({year:r.year,y:r.y})),
  combinations:result.combinations.map(c=>({scenario:c.scenario,path:c.path,name:c.name,points:c.points.map(p=>({year:p.year,demand:p.demand}))})),
  comparisons:result.comparisons.map(r=>({...r})),
  scenarioNames:SSP_NAMES,pathNames:Object.fromEntries(PATHS.map(p=>[p.id,p.name])),
  plotKind:options.plotKind||'paths',fullHistory:!!options.fullHistory,scenario:options.scenario||'SSP2',path:options.path||'policy',
  assumption:'5 SSP x 3 efficiency paths; conditional scenario differences, not confidence intervals or causal policy effects.'};
}
export const researchCalibration=['metrics.rolling.mape','terminal.SSP2.policy','observed.last'];
export const researchHeldout=['metrics.fixed.rmse','terminal.SSP5.frozen'];
