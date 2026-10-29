import {prepareResearch} from './research-engine.mjs';
export function psoRecord(result,bundle){const kind=['regularized-bp-v1','regularized-bp-v2'].includes(result.provenance?.modelVersion)?'bp':'pso';return {id:kind+'-'+result.createdAt+'-'+result.source.sha256,kind,label:result.provenance?.method||'PSO-BP',createdAt:result.createdAt,source:result.source,bundle,result,protocol:{name:result.provenance?.validation||'2020—2023开发回测',testYears:result.backtest.map(r=>r.year),type:'fixed-and-rolling-development'}};}
export function agentRecord(run,bundle){
 if(run.source?.sha256!==bundle.source.sha256||run.researchBatch?.combinations?.length!==15)throw Error('Agent记录与输入不匹配，或缺少15组预测');
 const prepared=prepareResearch(bundle),baseDemand=prepared.base.demand;
 const combinations=prepared.paths.map(c=>({...c,points:c.points.map((p,i)=>({...p,demand:run.researchBatch.combinations.find(v=>v.scenario===c.scenario&&v.path===c.path).rows[i].value}))}));
 const comparisons=[];let identical=0;
 for(const scenario of ['SSP1','SSP2','SSP3','SSP4','SSP5'])for(let i=0;i<7;i++){
  const [a,b,c]=['frozen','trend','policy'].map(path=>combinations.find(v=>v.scenario===scenario&&v.path===path).points[i].demand);
  const reference=run.researchBatch.referenceRows?.[i]?.value;
  comparisons.push({scenario,year:2024+i,frozen:a,trend:b,policy:c,development:Number.isFinite(reference)?a-reference:null,baselineEfficiencyEffect:Number.isFinite(reference)?reference-baseDemand:null,historicalSaving:a-b,additionalPolicySaving:b-c,totalSaving:a-c,netChange:c-baseDemand,offsetRatio:Number.isFinite(reference)&&a-reference>1e-9?(a-c)/(a-reference):null});
  if(Math.abs(b-c)<1e-9)identical++;
 }
 const r=run.result,backtest=r.test.rows.map(p=>({year:Number(p.time),actual:p.actual,agent:p.predicted,baseline:p.baseline}));
 const result={schema:'waterlab-ssp-result-v1',source:bundle.source,history:prepared.rows,combinations,comparisons,baseYear:2023,baseDemand,createdAt:r.createdAt,policy:bundle.policy,provenance:{method:(run.calls?'Agent选择：':'规则候选选择（未调用Agent）：')+r.name,selectionPolicy:run.selectionProtocol||null,agentArchitecture:run.agentArchitecture||null,workflowSearch:run.workflowSearch||null,orchestration:run.orchestration||null,taskAssessment:run.assessment||null,validation:'初训2010—2019；验证2020—2021；末段2022—2023；历史已参与开发'},metrics:{agent:r.test.metrics},backtest,diagnostics:{identicalTrendPolicyPoints:identical},decompositionLabel:'基准与递推差异',hasDevelopmentReference:!!run.researchBatch.referenceRows};
 return {id:'agent-'+r.createdAt+'-'+bundle.source.sha256,kind:'agent',label:(run.calls?'Agent · ':'规则对照 · ')+r.name,createdAt:r.createdAt,source:bundle.source,bundle,result,raw:run,protocol:{name:result.provenance.validation,testYears:backtest.map(p=>p.year),type:'frozen-selection-heldout'}};
}
export function comparisonStatus(a,b){
 if(!a||!b)return {paths:false,errors:false,message:'请选择两条已完成的预测记录。'};
 const same=a.source.sha256===b.source.sha256;
 const sameYears=JSON.stringify(a.protocol.testYears)===JSON.stringify(b.protocol.testYears);
 const sameProtocol=a.protocol.type===b.protocol.type;
 return {paths:same,errors:same&&sameYears&&sameProtocol,message:!same?'输入版本不同，已禁止直接对比。':!sameYears||!sameProtocol?'输入一致，可比较未来路径；验证年份或协议不同，不能据误差高低判断模型优劣。':'输入与验证协议一致，可并列比较误差及未来路径。'};
}
