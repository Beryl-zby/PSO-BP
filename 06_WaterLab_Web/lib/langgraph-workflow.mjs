import { Annotation, StateGraph, START, END } from '../vendor/langgraph-runtime/node_modules/@langchain/langgraph/dist/index.js';

// Explicit program routing; no LLM decides whether held-out labels can be read.
// State is per invocation. This migration does not enable durable resume.
/** @param {Array<{name:string,run:()=>Promise<unknown>,next?:()=>string,targets?:Record<string,string>}>} stages
 * @param {{signal?:AbortSignal,onStep?:(name:string)=>void}} options */
export async function runWorkflow(stages, {signal, onStep=()=>{}}={}) {
  if (!stages.length) throw Error('工作流不能为空');
  const names=stages.map(s=>s.name);
  if(new Set(names).size!==names.length) throw Error('工作流节点名重复');
  const State=Annotation.Root({trace:Annotation({reducer:(a,b)=>a.concat(b),default:()=>[]})});
  const builder=new StateGraph(State);
  for (const stage of stages) builder.addNode(stage.name,async()=>{
    signal?.throwIfAborted();
    onStep(stage.name);
    await stage.run();
    signal?.throwIfAborted();
    return {trace:[stage.name]};
  });
  builder.addEdge(START,names[0]);
  stages.forEach((stage,i)=>{
    if(stage.next) builder.addConditionalEdges(stage.name,stage.next,stage.targets);
    else builder.addEdge(stage.name,names[i+1]||END);
  });
  const graph=builder.compile();
  const result=await graph.invoke({trace:[]},{signal,recursionLimit:64});
  return {framework:'LangGraph',runtime:'@langchain/langgraph',version:1,trace:result.trace};
}
