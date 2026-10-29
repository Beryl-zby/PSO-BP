import {runWorkflow} from './langgraph-workflow.mjs';
import {selectCandidate} from '../public/adaptive-engine.mjs';
import {initialTaskCandidates,appendTaskProposals} from '../public/task-policy.mjs';
export const FIXED_FLOW={id:'fixed',audit:false,feedback:true,style:'diagnostic'};
export const FLOW_VERSION='three-agent-aflow-bp-v2';
export function normalizeFlow(v){
 if(!v||typeof v.audit!=='boolean'||typeof v.feedback!=='boolean'||!['direct','diagnostic'].includes(v.style))throw Error('工作流结构无效');
 return {audit:v.audit,feedback:v.feedback,style:v.style,id:`${v.audit?'audit':'plain'}-${v.feedback?'feedback':'once'}-${v.style}`};
}
export const signature=c=>JSON.stringify(Object.fromEntries(Object.entries(c).filter(([k])=>!['id','origin','name'].includes(k)).sort(([a],[b])=>a.localeCompare(b))));
export function selectTreeNode(tree){const total=tree.reduce((n,v)=>n+v.visits,0);let chosen=0,best=-Infinity;tree.forEach((v,i)=>{const score=v.visits?v.reward/v.visits+Math.sqrt(2*Math.log(Math.max(2,total))/v.visits):Infinity;if(score>best){chosen=i;best=score;}});return chosen;}
// Only development summaries and validation feedback enter this search. No finalize tool exists here.
export async function searchForecastFlows({profile,assessment,settings,plan,review,authorize,evaluate,propose,signal,progress=(message)=>{},mode='fixed',expansions=3}){
 const cache=new Map(),tree=[],notes=[];let calls=0,tokens=0,trainingEvaluations=0;
 const account=p=>{calls+=p.calls||0;tokens+=p.tokens||0;return p;};
 async function run(flow){
  let candidates=initialTaskCandidates(assessment),results=[],pending=[],reason=[],trace;
  const initial=async()=>{const p=account(await plan(profile,0,[],flow));reason.push(p.reason);const extra=appendTaskProposals(candidates,p.candidates,assessment,0);notes.push(...extra.notes);pending=extra.candidates;};
  const audit=async()=>{const p=account(await review(profile,pending,flow));reason.push(p.reason);const extra=appendTaskProposals(candidates,p.candidates,assessment,0);pending=extra.candidates;notes.push(...extra.notes);};
  const execute=async()=>{candidates=[...candidates,...pending];pending=[];const a=account(await authorize('evaluate',candidates,flow));if(a.action!=='evaluate')throw Error('预测执行Agent未授权验证工具');
   const missing=candidates.filter(c=>!cache.has(signature(c)));if(missing.length){const computed=await evaluate(missing);for(let i=0;i<missing.length;i++)cache.set(signature(missing[i]),computed[i]);trainingEvaluations+=missing.length;}
   results=candidates.map(c=>({...cache.get(signature(c)),candidate:{...cache.get(signature(c)).candidate,id:c.id}}));};
  const feedback=async()=>{const p=account(await plan(profile,1,results,flow));reason.push(p.reason);const extra=appendTaskProposals(candidates,p.candidates,assessment,1);pending=extra.candidates;notes.push(...extra.notes);};
  const stages=[{name:'model_planning',run:initial}];
  if(flow.audit)stages.push({name:'configuration_review',run:audit});
  stages.push({name:'validation_execution',run:execute});
  if(flow.feedback)stages.push({name:'feedback_planning',run:feedback},{name:'feedback_execution',run:execute});
  let selected;stages.push({name:'freeze_candidate',run:async()=>{selected=selectCandidate(results,settings);}});
  const before=calls;trace=await runWorkflow(stages,{signal,onStep:n=>progress('LangGraph · '+n)});
  return {flow,results,selected,validationScore:selected.score,reason,trace,calls:calls-before};
 }
 async function evaluateFlow(flow,parent){const start=performance.now();let row;try{row=await run(flow);row={...row,valid:true,reward:1/(1+Math.max(0,row.validationScore))};}catch(e){signal?.throwIfAborted();row={flow,valid:false,error:String(e.message||e),reward:0};progress('候选工作流失败：'+row.error);}
  const index=tree.length;tree.push({...row,parent,visits:0,rewardSum:0,reward:0,localReward:row.reward,seconds:(performance.now()-start)/1000});
  let i=index;while(i>=0){tree[i].visits++;tree[i].reward+=row.reward;i=tree[i].parent;}
 }
 await evaluateFlow({...FIXED_FLOW,id:'fixed'},-1);
 if(mode==='aflow')for(let i=0;i<expansions;i++){
  signal?.throwIfAborted();const parent=selectTreeNode(tree);progress('AFlow · 搜索扩展 '+(i+1)+'/'+expansions);
  // Explicit whitelist excludes predictions, held-out metrics and target labels.
  const history=tree.map(t=>({flow:t.flow,valid:t.valid,validationScore:t.validationScore,error:t.error,seconds:t.seconds,calls:t.calls}));
  const p=account(await propose({profile,parent:tree[parent].flow,history}));let flow=normalizeFlow(p.flow);
  if(tree.some(t=>normalizeFlow(t.flow).id===flow.id)){
   const unseen=[false,true].flatMap(a=>[false,true].flatMap(f=>['direct','diagnostic'].map(style=>normalizeFlow({audit:a,feedback:f,style})))).find(v=>!tree.some(t=>normalizeFlow(t.flow).id===v.id));
   if(!unseen)break;flow=unseen;notes.push('LLM重复提案，程序采用未搜索的合法结构。');
  }
  await evaluateFlow(flow,parent);
 }
 const valid=tree.filter(t=>t.valid);if(!valid.length)throw Error('没有工作流通过验证');
 valid.sort((a,b)=>a.validationScore-b.validationScore||a.calls-b.calls);const best=valid[0];
 return {best,results:best.results,reasons:best.reason,notes,calls,tokens,report:{version:FLOW_VERSION,method:mode==='aflow'?'AFlow-inspired constrained workflow search':'fixed',scope:'工作流结构与提示策略搜索；非原版AFlow完整复现',criterion:'验证误差优先，相同时较少模型调用；不使用测试指标',selected:best.flow,trainingEvaluations,cacheEntries:cache.size,expansions:tree.length-1,tree:tree.map(({results,selected,reason,...r})=>r),numericResultsChangedBy:'候选流程可能提出不同模型配置；非框架本身改变数值'}};
}
