import {researchMultiEvidence} from '../lib/research-multi.mjs';
import {FIGURE_TYPES,FIGURE_SSPS} from './research-figures.mjs';

export const COMPARISON_FIGURES=[
 {id:'paths',name:'需水趋势对比',note:''},
 {id:'terminal',name:'2030年需水量对比',note:''},
 {id:'backtest',name:'共同年份回测对比',note:'BP采用固定起点预测；Agent采用冻结模型末段预测，验证口径不同。'},
];
export function availableResultFigures(result,comparison=null){
 return comparison?COMPARISON_FIGURES.filter(f=>f.id!=='backtest'||comparisonRows(result,comparison,'SSP2','policy','backtest').length>0):FIGURE_TYPES.filter(f=>!(f.id==='waterfall'&&result.hasDevelopmentReference===false));
}
function value(r,s,p,year){const v=r.combinations.find(c=>c.scenario===s&&c.path===p)?.points.find(v=>v.year===year)?.demand;if(!Number.isFinite(v))throw Error('所选记录缺少预测值');return v;}
export function comparisonRows(bp,agent,scenario,path,kind='paths'){
 if(kind==='terminal')return FIGURE_SSPS.map(s=>({name:s,bp:value(bp,s,path,2030),agent:value(agent,s,path,2030)}));
 if(kind==='backtest')return bp.backtest.flatMap(b=>{
  const a=agent.backtest.find(a=>a.year===b.year),p=b.fixed??b.rolling,x=a?.agent??a?.predicted;
  return a&&[p,x,b.actual].every(Number.isFinite)?[{year:b.year,actual:b.actual,bp:p,agent:x}]:[];
 });
 return [{year:2023,bp:bp.baseDemand,agent:agent.baseDemand},...Array.from({length:7},(_,i)=>({year:2024+i,bp:value(bp,scenario,path,2024+i),agent:value(agent,scenario,path,2024+i)}))];
}
export function resultViewEvidence({record,bpRecord,agentRecord,scenario,path,kind,background}){
 if(record){const terminal=record.result.comparisons.find(r=>r.scenario===scenario&&r.year===2030);return {...researchMultiEvidence(record.result,scenario,path),accountingInterpretation:terminal?{netChangeDirection:terminal.netChange>0?'较2023年增加':terminal.netChange<0?'较2023年减少':'较2023年不变',identity:'相对2023净变化＝发展增需＋基期衔接差－总节水；节水大于发展增需不必然意味着总量低于2023年。'}:null,researchBackground:record.background||background,view:{record:record.id,source:record.source.sha256,model:record.label,scenario,path,kind,figureOptions:availableResultFigures(record.result),allowedPlotKinds:availableResultFigures(record.result).map(f=>f.id)}};}
 const bp=bpRecord.result,agent=agentRecord.result,rows=comparisonRows(bp,agent,scenario,path),facts={};
 for(const row of rows){facts['annual.bp.'+row.year]=row.bp;facts['annual.agent.'+row.year]=row.agent;facts['annual.agent_minus_bp.'+row.year]=row.agent-row.bp;}
 for(const row of comparisonRows(bp,agent,scenario,path,'terminal')){facts['terminal.bp.'+row.name]=row.bp;facts['terminal.agent.'+row.name]=row.agent;}
 return {task:'model-result-comparison',comparisonToBase:{bpNetChange:rows.at(-1).bp-bp.baseDemand,agentNetChange:rows.at(-1).agent-agent.baseDemand,bpDirection:rows.at(-1).bp>bp.baseDemand?'较2023年增加':'较2023年减少或不变',agentDirection:rows.at(-1).agent>agent.baseDemand?'较2023年增加':'较2023年减少或不变'},researchBackground:background,unit:'亿m³',facts,view:{records:[bpRecord.id,agentRecord.id],models:[bpRecord.label,agentRecord.label],scenario,path,kind,figureOptions:availableResultFigures(bp,agent),allowedPlotKinds:availableResultFigures(bp,agent).map(f=>f.id)},protocols:[bpRecord.protocol,agentRecord.protocol],scope:'只比较用户选中的BP和Agent记录；输入版本已核验一致。验证口径不一致时，不能据回测误差排名；未来情景差值不代表真实误差或因果效应。'};
}
export function resolvePlotRequest(proposed,allowed){
 const mapped=proposed==='cumulative'?'savings':proposed;
 return allowed.includes(mapped)?{kind:mapped,cumulative:proposed==='cumulative'}:null;
}

// Explicit user figure requests take precedence over model suggestions.
export function resolveUserPlotRequest(question,modelKind,allowed){
 if(!/画|绘|出图|换.*图|切换.*图|改成.*图|看.*图|给.*图|要.*图|plot|draw|show.*chart|generate.*figure|visuali/i.test(question))return null;
 const requests=[[/瀑布|抵消|贡献|waterfall/i,'waterfall'],[/累计|累积|cumulative/i,'cumulative'],[/回测|误差|backtest/i,'backtest'],[/强度|intensity/i,'intensity'],[/节水量|saving/i,'savings'],[/同一路径|同一条路径/i,'ssps'],[/总览|全部情景|所有情景/i,'all-paths'],[/2030|终期|柱状|bar/i,'terminal'],[/趋势|折线|trend|line chart/i,'paths']];
 for(const [pattern,kind] of requests)if(pattern.test(question))return resolvePlotRequest(kind,allowed);
 return resolvePlotRequest(modelKind,allowed);
}
