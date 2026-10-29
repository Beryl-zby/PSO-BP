'use client';
import {useEffect,useRef,useState} from 'react';
import {ResearchFigureGallery} from './research-figure-gallery';
import {ResultRecordCard} from './result-record-card';
import {PATHS,SSP_NAMES} from '../public/research-engine.mjs';
import {comparisonStatus} from '../public/workspace-records.mjs';
import {availableResultFigures,comparisonRows,resultViewEvidence,resolveUserPlotRequest} from '../public/result-view.mjs';

function download(record:any){const u=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='预测记录_'+record.kind+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),10000);}
const fmt=(v:number)=>Number.isFinite(v)?v.toFixed(3):'—';
export function ResultAnalysisPanel({records,view,onViewChange,requestedRecord,currentSource,background,connected,scenario,onScenarioChange}:{records:any[];view:string;onViewChange:(v:string)=>void;requestedRecord:string;currentSource:string; background:string;connected:boolean;scenario:string;onScenarioChange:(s:string)=>void}){
 const [bpId,setBpId]=useState(''),[agentId,setAgentId]=useState(''),[path,setPath]=useState('policy'),[kind,setKind]=useState('paths'),[cumulative,setCumulative]=useState(false);
 const [question,setQuestion]=useState(''),[answer,setAnswer]=useState(''),[asking,setAsking]=useState(false),[error,setError]=useState('');
 const abort=useRef<AbortController|null>(null),generation=useRef(0),plotRevision=useRef(0);
 const bps=records.filter(r=>r.kind==='bp'),agents=records.filter(r=>r.kind==='agent');
 const bp=bps.find(r=>r.id===bpId)||bps[0],agent=agents.find(r=>r.id===agentId)||agents[0],compare=view==='compare';
 const active=view==='agent'?agent:bp,check=comparisonStatus(bp,agent),ready=compare?!!bp&&!!agent&&check.paths:!!active;
 const allowed=ready?availableResultFigures(active.result,compare?agent.result:null).map((f:any)=>f.id):['paths'];
 const shownKind=allowed.includes(kind)?kind:'paths';
 const snapshot=[view,compare?bp?.id:active?.id,compare?agent?.id:'',scenario,path].join('|');
 useEffect(()=>{const r=records.find(r=>r.id===requestedRecord);if(r?.kind==='bp')setBpId(r.id);if(r?.kind==='agent')setAgentId(r.id);},[requestedRecord]);
 useEffect(()=>{generation.current++;abort.current?.abort();setAsking(false);setAnswer('');setError('');},[snapshot]);
 useEffect(()=>()=>{generation.current++;abort.current?.abort();},[]);
 function chooseKind(v:string){plotRevision.current++;setKind(v);setCumulative(false);}
 async function analyze(){
  if(!ready||asking)return;
  const id=++generation.current,revision=plotRevision.current,c=new AbortController();abort.current?.abort();abort.current=c;
  setAsking(true);setError('');setAnswer('');
  try{
   const evidence=resultViewEvidence({record:compare?null:active,bpRecord:bp,agentRecord:agent,scenario,path,kind:shownKind,background});
   const response=await fetch('/api/analyze-results',{method:'POST',signal:c.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({question:question.trim()||'解读当前所选结果的主要变化。',evidence})});
   const v:any=await response.json();if(!response.ok)throw Error(v.error||'分析失败');if(id!==generation.current)return;
   setAnswer(v.answer);const request=resolveUserPlotRequest(question,v.plotKind,allowed);
   // Only the shared figure can change. Record, model view, SSP and path are user-owned.
   if(request&&revision===plotRevision.current){setKind(request.kind);setCumulative(request.cumulative);}
  }catch(e){if(id===generation.current&&!c.signal.aborted)setError(String(e instanceof Error?e.message:e));}
  finally{if(id===generation.current)setAsking(false);}
 }
 function stop(){generation.current++;abort.current?.abort();setAsking(false);}
 function selector(label:string,value:string,onChange:(v:string)=>void,items:any[]){return <label>{label}<select aria-label={label} value={value} onChange={e=>onChange(e.target.value)} disabled={!items.length}>{!items.length&&<option value="">暂无记录</option>}{items.map(r=><option key={r.id} value={r.id}>{new Date(r.createdAt).toLocaleString('zh-CN')} · {r.label}</option>)}</select></label>;}
 return <>
  <section className="adaptive-intro results-clean-heading"><h2>结果分析与绘图</h2></section>
  <section className="results-selection-card" data-view={view}>
   <div className="result-view-tabs" role="group" aria-label="选择预测结果">{[['bp','BP结果'],['agent','Agent结果'],['compare','两者对比']].map(([v,label])=><button key={v} type="button" aria-pressed={view===v} className={view===v?'is-selected':''} onClick={()=>onViewChange(v)}>{label}</button>)}</div>
   <div className="research-controls results-record-selectors">
    {(view==='bp'||compare)&&selector('BP预测记录',bp?.id||'',setBpId,bps)}
    {(view==='agent'||compare)&&selector('Agent预测记录',agent?.id||'',setAgentId,agents)}
    <label>SSP情景<select aria-label="结果分析SSP" value={scenario} onChange={e=>onScenarioChange(e.target.value)}>{SSP_NAMES.map((s:string)=><option key={s}>{s}</option>)}</select></label>
    {compare&&<label>节水路径<select aria-label="对比节水路径" value={path} onChange={e=>setPath(e.target.value)}>{PATHS.map((p:any)=><option key={p.id} value={p.id}>{p.name}</option>)}</select></label>}
   </div>
  </section>
  {!ready?<p className="research-note">{compare?(bp&&agent?'两条记录的输入版本不同，请选择相同输入版本的记录。':'请先完成BP和Agent预测。'):view==='agent'?'暂无Agent结果，请先在第二模块运行预测。':'暂无BP结果，请先在第三模块运行预测。'}</p>:<>
   {!compare&&<ResultRecordCard record={active} currentSource={currentSource} onExport={()=>download(active)}/>}
   {compare&&!check.errors&&<p className="results-validation-note">两种方法的验证口径不同，回测误差不可直接排名。</p>}
   <ResearchFigureGallery result={active.result} historical={active.bundle.historical} scenario={scenario} comparisonResult={compare?agent.result:undefined} kind={shownKind} onKindChange={chooseKind} path={path} onPathChange={setPath} cumulative={cumulative} onCumulativeChange={setCumulative}/>
   <details className="results-data-table"><summary>{compare?'查看逐年数值':'查看2030年数值'}</summary><div className="table-scroll">{compare?<table><thead><tr><th>年份</th><th>BP</th><th>Agent</th><th>Agent－BP</th></tr></thead><tbody>{comparisonRows(bp.result,agent.result,scenario,path).map((r:any)=><tr key={r.year}><td>{r.year}</td><td>{fmt(r.bp)}</td><td>{fmt(r.agent)}</td><td>{fmt(r.agent-r.bp)}</td></tr>)}</tbody></table>:<table><thead><tr><th>SSP</th>{PATHS.map((p:any)=><th key={p.id}>{p.name}</th>)}</tr></thead><tbody>{SSP_NAMES.map((s:string)=><tr key={s}><td>{s}</td>{PATHS.map((p:any)=><td key={p.id}>{fmt(active.result.combinations.find((c:any)=>c.scenario===s&&c.path===p.id).points.at(-1).demand)}</td>)}</tr>)}</tbody></table>}</div></details>
   <section className="results-ai-panel" aria-label="AI结果解读"><h3>AI解读</h3><div className="results-ai-composer"><textarea aria-label="AI解读问题" value={question} onChange={e=>setQuestion(e.target.value)} rows={2} placeholder="例如：节水能否抵消发展增需？请画图说明。"/><button className="primary" disabled={!connected||asking} onClick={analyze}>{asking?'分析中…':'分析当前结果'}</button>{asking&&<button className="secondary" onClick={stop}>停止</button>}</div>{!connected&&<p className="results-ai-status">AI服务未连接</p>}{error&&<p className="error-box" role="alert">{error}</p>}{answer&&<div className="results-ai-answer" aria-live="polite">{answer}</div>}</section>
  </>}
 </>;
}
