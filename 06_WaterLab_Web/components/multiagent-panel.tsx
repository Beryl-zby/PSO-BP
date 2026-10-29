'use client';
import {useEffect,useRef,useState} from 'react';
import {researchMultiEvidence,researchMultiPlotData} from '../lib/research-multi.mjs';
import {makeEvidence} from '../lib/multiagent';
import {generatePlot,plotData,type PlotArtifact} from '../lib/matplot';
import type {Dataset,Forecast} from '../lib/types';
function save(name:string,text:string,type='application/json'){const u=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),10000);}
export function MultiAgentPanel({dataset,result,enabled,research,selectedScenario,onScenarioChange}:{dataset:Dataset;result:Forecast;enabled:boolean;research?:any;selectedScenario?:string;onScenarioChange?:(s:string)=>void}){
 const [scenario,setScenario]=useState(selectedScenario||'SSP2'),[path,setPath]=useState('policy'),[plotKind,setPlotKind]=useState('paths'),[fullHistory,setFullHistory]=useState(false),[question,setQuestion]=useState(''),[answer,setAnswer]=useState(''),[busy,setBusy]=useState(false),[error,setError]=useState(''),[events,setEvents]=useState<string[]>([]),[plot,setPlot]=useState<PlotArtifact|null>(null),[code,setCode]=useState(false);
 const abort=useRef<AbortController|null>(null),generation=useRef(0);
 useEffect(()=>{if(selectedScenario)setScenario(selectedScenario);},[selectedScenario]);
 useEffect(()=>{generation.current++;abort.current?.abort();setBusy(false);setAnswer('');setPlot(null);setError('');setEvents([]);},[research,scenario,path]);
 useEffect(()=>()=>{generation.current++;abort.current?.abort();},[]);
 function evidence(){return research?researchMultiEvidence(research,scenario,path):makeEvidence(dataset,result);}
 function progress(s:string){setEvents(v=>[...v,s]);}
 async function draw(kind:string,signal:AbortSignal,id:number){
  if(research?.hasDevelopmentReference===false&&kind==='waterfall')throw Error('旧记录缺少反事实参考，请重新预测。');
  const data=research?researchMultiPlotData(research,{scenario,path,plotKind:kind,fullHistory}):plotData(dataset,result);
  const v=await generatePlot(question.trim()||'绘制当前选择的预测结果',data,signal,progress);if(id===generation.current)setPlot(v);
 }
 async function execute(analyze:boolean){if(busy)return;const id=++generation.current;abort.current?.abort();const c=new AbortController();abort.current=c;setBusy(true);setError('');setEvents([]);try{
  let kind=plotKind;
  if(analyze){progress('结果分析与可视化智能体 · 读取所选记录');const r=await fetch('/api/analyze-results',{method:'POST',signal:c.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({question:question.trim()||'分析当前SSP下三条路径差异，说明节水能否抵消发展增需，并绘制三路径趋势图。',evidence:evidence()})});const v:any=await r.json();if(!r.ok)throw Error(v.error||'分析失败');if(id!==generation.current)return;setAnswer(v.answer);progress('程序已核验结构化数值引用；'+v.calls+'次模型调用');if(v.plotKind){kind=v.plotKind;setPlotKind(kind);}else return;
  }
  await draw(kind,c.signal,id);
 }catch(e){if(id===generation.current)setError(String(e instanceof Error?e.message:e));}finally{if(id===generation.current)setBusy(false);}}
 function stop(){generation.current++;abort.current?.abort();setBusy(false);}
 return <section className="multi-panel">
 <div className="multi-heading"><div><div className="eyebrow">AGENT 03 · ANALYSIS & VISUALIZATION</div><h2>结果分析与可视化智能体</h2><p>共用Qwen · 解释所选预测记录 · 图表读取原始计算结果</p></div></div>
 <p className="multi-disclosure">AFlow预测工作流搜索位于第二模块。这里不再搜索多个分析角色，不重新训练或修改预测。SSP图使用固定Python模板；智能体负责解释及选择图型，程序检查数值引用。</p>
 {research&&<div className="research-controls"><label>分析SSP<select value={scenario} disabled={busy} onChange={e=>{setScenario(e.target.value);onScenarioChange?.(e.target.value);}}>{['SSP1','SSP2','SSP3','SSP4','SSP5'].map(v=><option key={v}>{v}</option>)}</select></label><label>分析路径<select value={path} disabled={busy} onChange={e=>setPath(e.target.value)}>{[['frozen','近期平均效率维持'],['trend','历史节水趋势延续'],['policy','江苏政策参考约束']].map(([v,n])=><option value={v} key={v}>{n}</option>)}</select></label><label>研究图型<select aria-label="研究图型" value={plotKind} disabled={busy} onChange={e=>setPlotKind(e.target.value)}>{[['paths','各SSP三路径趋势'],['ssps','各路径五SSP趋势'],['terminal','2030分组柱状图'],['savings','逐年节水量'],['waterfall','发展增需与节水抵消'],['cumulative','累计节水量']].map(([v,n])=><option value={v} key={v}>{n}</option>)}</select></label><label><input type="checkbox" checked={fullHistory} disabled={busy} onChange={e=>setFullHistory(e.target.checked)}/>包含完整历史</label></div>}
 <label>分析问题或绘图需求<textarea aria-label="结果分析与绘图需求" rows={3} value={question} onChange={e=>setQuestion(e.target.value)} placeholder="例如：比较五种SSP下节水能否抵消发展增需，并画图。"/></label>
 <div className="multi-actions"><button className="primary" disabled={busy||!enabled} onClick={()=>execute(true)}>智能分析与按需出图</button><button className="secondary" disabled={busy} onClick={()=>execute(false)}>按所选图型直接绘图</button>{busy&&<button className="secondary" onClick={stop}>停止</button>}</div>
 {error&&<p className="error-box" role="alert">{error}</p>}{answer&&<div className="details-card stage-card"><h3>分析结果</h3><p style={{whiteSpace:'pre-wrap'}}>{answer}</p></div>}
 {plot&&<div className="matplot-card"><img className="generated-plot" src={plot.image} alt="所选预测结果图"/><div className="multi-actions"><button className="secondary" onClick={()=>{const a=document.createElement('a');a.href=plot.image;a.download='需水预测结果.png';a.click();}}>下载PNG</button><button onClick={()=>save('需水预测绘图.py',plot.code,'text/x-python')}>下载Python代码</button><button onClick={()=>setCode(!code)}>{code?'收起':'查看'}代码</button></div><p>{plot.review}</p>{code&&<pre className="python-code">{plot.code}</pre>}</div>}
 {events.length>0&&<details open={busy}><summary>本次执行记录</summary><ol>{events.map((v,i)=><li key={i}>{v}</li>)}</ol></details>}
 </section>;
}
