'use client';
import {useEffect,useRef,useState} from 'react';
import {Download,FileSpreadsheet,Loader2,Play,RefreshCw,Send,Square,Upload} from 'lucide-react';
import {Legend,Line,LineChart,ReferenceArea,ResponsiveContainer,Tooltip,XAxis,YAxis} from 'recharts';
import {parseResearchSheets,PATHS,prepareResearch,SSP_NAMES} from '../public/research-engine.mjs';

type Bundle=ReturnType<typeof parseResearchSheets>&{source:Record<string,any>};
type ResearchResult=any;
const COLORS=['#15976b','#2271b6','#e88b24','#8762ab','#d74b43'];
const DASHES=['','7 4','2 3'];
const format=(v:number,d=3)=>Number.isFinite(v)?v.toLocaleString('zh-CN',{maximumFractionDigits:d,minimumFractionDigits:d}):'—';
function saveFile(name:string,data:string,type='application/json'){
 const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
}
function csvCell(v:unknown){return '"'+String(v??'').replaceAll('"','""')+'"';}
function downloadCSV(name:string,rows:unknown[][]){saveFile(name,'\uFEFF'+rows.map(r=>r.map(csvCell).join(',')).join('\r\n'),'text/csv;charset=utf-8');}

export function ResearchWorkbench({connected}:{connected:boolean}){
 const [bundle,setBundle]=useState<Bundle|null>(null),[result,setResult]=useState<ResearchResult|null>(null);
 const [busy,setBusy]=useState(false),[loading,setLoading]=useState(false),[error,setError]=useState(''),[status,setStatus]=useState('正在读取当前论文模型输入…');
 const [scenario,setScenario]=useState('SSP2'),[path,setPath]=useState('policy'),[view,setView]=useState('paths'),[fullHistory,setFullHistory]=useState(false);
 const [question,setQuestion]=useState(''),[answer,setAnswer]=useState(''),[asking,setAsking]=useState(false),[inputTab,setInputTab]=useState('history');
 const workerRef=useRef<Worker|null>(null),fileRef=useRef<HTMLInputElement>(null),abortRef=useRef<AbortController|null>(null),generation=useRef(0),svgRef=useRef<HTMLDivElement>(null);
 const prepared=bundle?prepareResearch(bundle):null;
 function cancel(){generation.current++;workerRef.current?.terminate();workerRef.current=null;abortRef.current?.abort();setBusy(false);setAsking(false);}
 function load(b:Bundle){prepareResearch(b);cancel();setBundle(b);setResult(null);setAnswer('');setError('');setStatus('已读取14年历史、35行SSP指标；可运行15种情景与节水路径组合。');}
 async function reload(){cancel();setLoading(true);setError('');try{const r=await fetch('/research-input.json',{cache:'no-store'});if(!r.ok)throw Error('读取内置论文输入失败，请运行启动脚本同步输入。');load(await r.json());}catch(e){setError(String(e instanceof Error?e.message:e));}finally{setLoading(false);}}
 useEffect(()=>{void reload();return ()=>{generation.current++;workerRef.current?.terminate();abortRef.current?.abort();};},[]);
 async function upload(file:File){
  cancel();setLoading(true);setError('');
  try{
   if(file.size>25*1024*1024||!/\.xlsx$/i.test(file.name))throw Error('请选择25MB以内的原模型XLSX输入工作簿。');
   const bytes=await file.arrayBuffer(),XLSX=await import('xlsx'),book=XLSX.read(bytes,{type:'array'}),sheets:Record<string,any[][]>={};
   for(const name of ['历史数据','未来指标','模型设置','口径与政策依据'])if(book.Sheets[name])sheets[name]=XLSX.utils.sheet_to_json(book.Sheets[name],{header:1,defval:'',blankrows:false});
   const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(v=>v.toString(16).padStart(2,'0')).join('');
   load(parseResearchSheets(sheets,{file:file.name,sha256:hash,syncedAt:new Date().toISOString(),uploaded:true}));
  }catch(e){setError(String(e instanceof Error?e.message:e));}finally{setLoading(false);}
 }
 function execute(){
  if(!bundle||busy)return;cancel();const id=generation.current;setBusy(true);setError('');setResult(null);setAnswer('');setStatus('训练当前主模型并进行时间回测…');
  const w=new Worker('/research-worker.js',{type:'module'});workerRef.current=w;
  w.onmessage=e=>{if(id!==generation.current)return;if(e.data.progress){setStatus(e.data.progress.label);return;}w.terminate();workerRef.current=null;setBusy(false);if(e.data.error){setError(e.data.error);setStatus('预测未完成。');}else{setResult(e.data.result);setStatus('已完成105个年度预测值：5种SSP × 3条路径 × 7年。');}};
  w.onerror=()=>{if(id!==generation.current)return;w.terminate();workerRef.current=null;setBusy(false);setError('预测线程执行失败，请刷新页面重试。');};w.postMessage({bundle});
 }
 function summaryCSV(){
  if(!result)return;const labels=SSP_NAMES.flatMap(s=>PATHS.map(p=>s+'_'+p.name));
  const rows:unknown[][]=[['年份',...labels],...Array.from({length:8},(_,i)=>[2023+i,...SSP_NAMES.flatMap(s=>PATHS.map(p=>i===0?result.baseDemand:result.combinations.find((c:any)=>c.scenario===s&&c.path===p.id).points[i-1].demand))])];
  downloadCSV('苏北需水预测_五SSP三路径_亿立方米.csv',rows);
 }
 function detailCSV(){
  if(!result)return;downloadCSV('SSP路径输入与需水预测明细.csv',[
   ['SSP','路径','年份','人口_万人','城镇化率_%','GDP_亿元2023价代理','第一产业','第二产业','第三产业','播种灌溉代理_千公顷','亩均灌溉强度_m3每亩','工业增加值_亿元2023价代理','工业强度_m3每万元2023价代理','节水速度倍数','农业驱动','工业驱动','人口城镇化驱动','预测总需水_亿m3','工业参考门槛是否生效'],
   ...result.combinations.flatMap((c:any)=>c.points.map((r:any)=>[c.scenario,c.name,r.year,r.population,r.urbanization,r.gdp,r.primary,r.secondary,r.tertiary,r.area,r.agriculturalIntensity,r.industrialVA,r.industrialIntensity,r.savingMultiplier,...r.x,r.demand,r.referenceBinding?'是':'否'])),
  ]);
 }
 function comparisonCSV(){if(!result)return;downloadCSV('发展新增与节水抵消对比.csv',[
  ['SSP','年份','效率保持需水','历史趋势需水','规划参考需水','发展变化_亿m3','历史趋势节水_亿m3','额外规划节水_亿m3','合计节水_亿m3','规划净变化_亿m3','抵消比例_小数'],
  ...result.comparisons.map((r:any)=>[r.scenario,r.year,r.frozen,r.trend,r.policy,r.development,r.historicalSaving,r.additionalPolicySaving,r.totalSaving,r.netChange,r.offsetRatio]),
 ]);}
 async function downloadPlot(){
  try{const svg=svgRef.current?.querySelector('svg.recharts-surface') as SVGSVGElement|null;if(!svg)throw Error('趋势图尚未就绪');const clone=svg.cloneNode(true) as SVGSVGElement,rect=svg.getBoundingClientRect();clone.setAttribute('xmlns','http://www.w3.org/2000/svg');clone.setAttribute('width',String(rect.width));clone.setAttribute('height',String(rect.height));clone.setAttribute('style','font-family:Times New Roman,SimSun,serif;background:white');const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml'}));try{const image=new Image();image.src=url;await image.decode();const canvas=document.createElement('canvas');canvas.width=Math.round(rect.width*3);canvas.height=Math.round(rect.height*3);const ctx=canvas.getContext('2d')!;ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);const a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download='苏北需水_'+(view==='paths'?scenario:path)+'_'+(fullHistory?'历史起点':'2023起点')+'.png';a.click();}finally{URL.revokeObjectURL(url);}}catch(e){setError(String(e instanceof Error?e.message:e));}
 }
 async function ask(){
  if(!result||!question.trim()||asking)return;const id=generation.current,controller=new AbortController();abortRef.current=controller;setAsking(true);setError('');
  try{
   const evidence={task:'ssp-water-demand',unit:'亿m³',baseYear:2023,baseDemand:result.baseDemand,model:result.provenance.method,historyYears:'2010—2023',futureYears:'2024—2030',scenario,policy:result.policy,policyFacts:{policyStarts:2026,all35PointsIdentical:result.diagnostics.identicalTrendPolicyPoints===35,explanation:result.diagnostics.identicalTrendPolicyPoints===35?'五种SSP在2024—2030全部35个点均重合；2024—2025未额外施加门槛，2026—2030是趋势已达到参考门槛，额外规划节水均为0':'按逐年结果判断是否重合'},metrics:result.metrics,diagnostics:result.diagnostics,annual:result.comparisons.filter((r:any)=>r.scenario===scenario),terminal:result.comparisons.filter((r:any)=>r.year===2030),pathDefinitions:{frozen:'农业与工业强度保持2023；发展规模按各SSP变化',trend:'农业采用输入趋势；工业由2010—2023强度拟合指数速度，按SSP倍数外推',policy:'农业同历史趋势；工业2025后取趋势与2030按policy.industrialDrop参考下降比例形成路径的较小值'},scope:['面积是播种面积代理灌溉面积','工业含火电口径未完全统一，强度为模型代理','2025工业基值是趋势预测，不是实测','情景差值不是因果政策效果；不输出实测部门用水量','回测已参与模型开发，不是新增独立测试']};
   const r=await fetch('/api/agent',{method:'POST',signal:controller.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:[{role:'user',content:question}],evidence:JSON.parse(JSON.stringify(evidence,(_,v)=>typeof v==='number'?Math.round(v*1e6)/1e6:v))})});const v:any=await r.json();if(!r.ok)throw Error(v.error||'分析失败');if(id===generation.current)setAnswer(v.answer);
  }catch(e){if(id===generation.current)setError(String(e instanceof Error?e.message:e));}finally{if(id===generation.current)setAsking(false);}
 }
 const selected=result?.comparisons.filter((r:any)=>r.scenario===scenario),terminal=selected?.at(-1);
 const lines=view==='paths'?PATHS.map((p,i)=>({key:p.id,label:p.name,color:COLORS[i],dash:DASHES[i]})):SSP_NAMES.map((s,i)=>({key:s,label:s,color:COLORS[i],dash:i===1?'':'7 4'}));
 const chartRows=result?[
  ...result.history.filter((r:any)=>fullHistory||r.year===2023).map((r:any)=>({year:r.year,...Object.fromEntries(lines.map(l=>[l.key,r.y]))})),
  ...Array.from({length:7},(_,i)=>({year:2024+i,...Object.fromEntries(lines.map(l=>[l.key,result.combinations.find((c:any)=>view==='paths'?c.scenario===scenario&&c.path===l.key:c.scenario===l.key&&c.path===path).points[i].demand]))})),
 ]:[];
 const numbers=chartRows.flatMap(r=>lines.map(l=>r[l.key])),min=numbers.length?Math.min(...numbers):0,max=numbers.length?Math.max(...numbers):1,pad=Math.max((max-min)*.1,.3),domain=[Math.floor((min-pad)*10)/10,Math.ceil((max+pad)*10)/10];
 return <div className="research-workbench">
  <section className="adaptive-intro"><div className="eyebrow">SU BEI · SSP1—SSP5</div><h2>苏北五市 · 五种SSP与三条节水路径</h2><p>采用当前正则PSO-BP主模型，统一训练与锚定，对比2024—2030年15种组合。数据来自论文现有输入工作簿。</p></section>
  <section className="data-card research-source">
   <div className="dataset-row"><div className="dataset-icon"><FileSpreadsheet size={22}/></div><div className="dataset-name"><strong>{bundle?.source.file||'正在读取输入工作簿'}</strong><span>苏北五市合计 · 2010—2023历史 · 2024—2030预测 · 单位：亿m³</span></div></div>
   {bundle&&<p className="research-hash">输入指纹：{bundle.source.sha256}<br/>载入时间：{new Date(bundle.source.syncedAt).toLocaleString('zh-CN')}</p>}
   <div className="research-actions"><button className="secondary" disabled={busy||loading||asking} onClick={reload}><RefreshCw size={14}/>重新读取内置输入</button><button className="secondary" disabled={busy||loading||asking} onClick={()=>fileRef.current?.click()}><Upload size={14}/>上传同格式输入Excel</button><button className="primary" disabled={!bundle||busy||loading||asking} onClick={execute}>{busy?<Loader2 size={15} className="spin"/>:<Play size={15}/>}运行15种组合</button>{(busy||asking)&&<button className="secondary" onClick={()=>{cancel();setStatus('已停止当前计算。');}}><Square size={12}/>停止</button>}<input id="research-file-input" ref={fileRef} type="file" accept=".xlsx" hidden onChange={e=>{if(e.target.files?.[0])void upload(e.target.files[0]);e.target.value='';}}/></div>
   <p className="adaptive-limit">启动脚本每次从模型data文件夹同步输入。修改原Excel后，请重新启动工作台，或直接上传新文件；“重新读取”只加载最近同步的内置副本。人口、产业、GDP等原值全部保留；模型实际输入为农业、工业和人口城镇化三个合成驱动。</p>
  </section>
  <div className="adaptive-status" role="status">{(busy||loading||asking)&&<Loader2 size={15} className="spin"/>}{status}</div>{error&&<div className="error-box" role="alert">{error}</div>}
  {bundle&&prepared&&<details className="research-inputs"><summary>查看历史基础数据、未来输入与路径定义</summary>
   <div className="research-definitions"><p><b>效率保持：</b>农业和工业用水强度均保持2023年水平，其他发展指标按各SSP变化。</p><p><b>历史趋势：</b>农业使用原输入的指数衰减结果；工业统一价格口径后拟合2010—2023年指数速度 k={format(prepared.industrialDecay,6)}，SSP1/5乘1.2，SSP2/4乘1，SSP3乘0.8（研究假设）。</p><p><b>规划参考约束：</b>农业与历史趋势相同；工业2025年后取历史趋势与参考门槛较小值，2030较本情景预测2025基值下降至少{format(bundle.policy.industrialDrop*100,0)}%。不是直接下达苏北的考核值。</p><p><b>三个驱动：</b>面积×农业强度×15/100000；工业增加值×工业强度/10000；人口×(1+城镇化率/100)。面积单位千公顷，人口万人，工业增加值亿元。保留原代理特征含义。</p></div>
   <div className="research-actions"><button className="text-button" onClick={()=>setInputTab('history')}>历史输入（14年）</button><button className="text-button" onClick={()=>setInputTab('future')}>五SSP未来输入（35行）</button><button className="text-button" onClick={()=>setInputTab('paths')}>路径强度与合成驱动（105行）</button></div>
   <div className="table-scroll"><table><thead><tr>{(inputTab==='history'?['年份','人口(万人)','城镇化率(%)','GDP(亿元)','第一产业','第二产业','第三产业','面积(千公顷)','农业强度','工业增加值','工业强度(原)','省平减代理','工业强度(2023价)','实际需水']:inputTab==='future'?['SSP','年份','人口','城镇化率','GDP','第一产业','第二产业','第三产业','面积','农业趋势强度','工业增加值','节水速度倍数']:['SSP','路径','年份','农业强度','工业强度','农业驱动','工业驱动','人口城镇化驱动']).map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{(inputTab==='history'?bundle.historical.map((r:any)=>[r.year,r.population,r.urbanization,r.gdp,r.primary,r.secondary,r.tertiary,r.area,r.agriculturalIntensity,r.industrialVA,r.industrialRawIntensity,r.deflator,r.industrialIntensity,r.demand]):inputTab==='future'?bundle.future.map((r:any)=>[r.scenario,r.year,r.population,r.urbanization,r.gdp,r.primary,r.secondary,r.tertiary,r.area,r.agriculturalIntensity,r.industrialVA,r.savingMultiplier]):prepared.paths.flatMap(c=>c.points.map((r:any)=>[c.scenario,c.name,r.year,r.agriculturalIntensity,r.industrialIntensity,...r.x]))).map((row:any[],i:number)=><tr key={i}>{row.map((v:any,j:number)=><td key={j}>{typeof v==='number'?(Number.isInteger(v)?v:format(v)):v}</td>)}</tr>)}</tbody></table></div>
  </details>}
  {result&&<>
   <div className="research-controls"><label>SSP<select value={scenario} onChange={e=>{cancel();setScenario(e.target.value);setAnswer('');}}>{SSP_NAMES.map(s=><option key={s}>{s}</option>)}</select></label><label>图表对比<select value={view} onChange={e=>setView(e.target.value)}><option value="paths">当前SSP的三条节水路径</option><option value="ssps">同一路径下的五种SSP</option></select></label>{view==='ssps'&&<label>路径<select value={path} onChange={e=>setPath(e.target.value)}>{PATHS.map(p=><option value={p.id} key={p.id}>{p.name}</option>)}</select></label>}<label className="research-check"><input type="checkbox" checked={fullHistory} onChange={e=>setFullHistory(e.target.checked)}/>从2010年历史开始画</label></div>
   <section className="adaptive-stats"><div><span>{scenario} · 2030效率保持</span><strong>{format(terminal.frozen,2)}</strong><small>亿m³</small></div><div><span>历史趋势</span><strong>{format(terminal.trend,2)}</strong><small>亿m³</small></div><div><span>规划参考约束</span><strong>{format(terminal.policy,2)}</strong><small>亿m³</small></div><div><span>节水对发展新增需求抵消比例</span><strong>{terminal.offsetRatio===null?'不适用':format(terminal.offsetRatio*100,1)+'%'}</strong><small>{terminal.offsetRatio===null?'效率保持路径下发展变化≤0':'(效率保持−规划约束)÷(效率保持−2023)'}</small></div></section>
   {result.diagnostics.identicalTrendPolicyPoints>0&&<div className="research-note">当前{result.diagnostics.identicalTrendPolicyPoints}/35个SSP年度点中，历史节水趋势已达到参考门槛，两条路径结果相同；图中后画的曲线可能覆盖前一条。额外规划节水量如实记为0。</div>}
   <section className="chart-card" id="analysis-chart"><div className="card-heading"><div><h2>{view==='paths'?scenario+' · 三条节水路径':'五种SSP · '+PATHS.find(p=>p.id===path)?.name}</h2><p>苏北五市预测总需水量 · 亿m³ · 2023各路径为同一历史值</p></div><button className="text-button" onClick={downloadPlot}><Download size={14}/>PNG</button></div><div ref={svgRef} className="research-chart"><ResponsiveContainer width="100%" height={360}><LineChart data={chartRows} margin={{top:18,right:28,bottom:20,left:14}}><XAxis height={48} label={{value:"年份",position:"insideBottom",offset:-8,fontFamily:"SimSun",fontSize:16}} dataKey="year" type="number" domain={['dataMin','dataMax']} allowDecimals={false} tick={{fontFamily:'Times New Roman',fontSize:15}} tickSize={-4} axisLine={{stroke:'#444',strokeWidth:.6}}/><YAxis label={{value:"需水量（亿m³）",angle:-90,position:"insideLeft",fontFamily:"SimSun",fontSize:16,dx:-9}} domain={domain} tick={{fontFamily:'Times New Roman',fontSize:15}} tickFormatter={v=>format(v,1)} width={88} tickSize={-4} axisLine={{stroke:'#444',strokeWidth:.6}}/><ReferenceArea x1={chartRows[0].year} x2={2030} y1={domain[0]} y2={domain[1]} stroke="#444" strokeWidth={.6} fillOpacity={0}/><Tooltip formatter={(v:any)=>format(Number(v),4)}/><Legend wrapperStyle={{fontFamily:'SimSun,宋体,serif',fontSize:15,paddingTop:16}}/>{lines.map(l=><Line key={l.key} type="linear" dataKey={l.key} name={l.label} stroke={l.color} strokeDasharray={l.dash} strokeWidth={1.8} dot={false} isAnimationActive={false}/>)}</LineChart></ResponsiveContainer></div></section>
   <section className="details-card"><div className="card-heading"><h2>{scenario} · 逐年预测与对照</h2></div><div className="table-scroll"><table><thead><tr>{['年份','效率保持','历史趋势','规划约束','发展变化','历史节水','额外规划节水','规划净变化'].map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{selected.map((r:any)=><tr key={r.year}>{[r.year,r.frozen,r.trend,r.policy,r.development,r.historicalSaving,r.additionalPolicySaving,r.netChange].map((v,j)=><td key={j}>{j===0?v:format(v)}</td>)}</tr>)}</tbody></table></div><p className="adaptive-limit">变化均相对2023年实际值或同SSP效率保持路径。情景差值是模型条件比较，不是政策因果效果，也不是实际部门用水量。</p></section>
   <section className="research-export"><h3>导出本次计算</h3><div className="research-actions"><button className="secondary" onClick={summaryCSV}><Download size={14}/>15组合汇总CSV</button><button className="secondary" onClick={detailCSV}>路径输入与预测明细CSV</button><button className="secondary" onClick={comparisonCSV}>发展与节水对照CSV</button><button className="secondary" onClick={()=>saveFile('WaterLab_SSP完整实验.json',JSON.stringify({bundle,result},null,2))}>完整实验JSON</button></div><p className="adaptive-limit">CSV可用Excel打开；汇总为年份纵向、15种组合横向，包含2023历史基值。完整JSON保留原始输入、指纹、逐折选参、回测和全部路径。</p></section>
   <details className="research-inputs"><summary>回测、模型配置与数值检查</summary><p>3-1-1网络，12成员，PSO种群24、40代；L2正则λ={result.config.lambda}，只按训练前缀内验证选择。未使用输入表旧版200成员/4隐节点设置。</p><p>固定起点MAPE {format(result.metrics.fixed.mape,4)}%；滚动MAPE {format(result.metrics.rolling.mape,4)}%。2020—2023已参与模型开发，是给定实际驱动的开发回测，不称新增独立测试。</p><p>105个预测点：非正值{result.diagnostics.nonpositive}，驱动超出历史范围{result.diagnostics.outside}，路径反序{result.diagnostics.reversed}；农业/工业/人口驱动负响应检查计数 {result.diagnostics.negativeResponses.join(' / ')}。这些是检查结果，不是模型硬约束。浏览器随机数与MATLAB不同，结果不保证逐位一致。</p><div className="table-scroll"><table><thead><tr>{['年份','实际值','固定预测','滚动预测'].map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{result.backtest.map((r:any)=><tr key={r.year}><td>{r.year}</td><td>{format(r.actual)}</td><td>{format(r.fixed)}</td><td>{format(r.rolling)}</td></tr>)}</tbody></table></div></details>
   <section className="adaptive-analysis"><h3>本地千问 · 当前SSP结果分析</h3><p>依据当前计算证据解读，不修改输入和预测值。AI文字可能出错，数值与路径定义以表格为准。</p><form onSubmit={e=>{e.preventDefault();void ask();}}><textarea aria-label="SSP预测分析问题" value={question} maxLength={2000} onChange={e=>setQuestion(e.target.value)} rows={3} placeholder="为什么历史趋势与规划约束重合？节水能抵消发展新增需水吗？"/><button className="primary" disabled={!connected||asking||busy||!question.trim()}><Send size={14}/>{asking?'正在分析':'分析当前结果'}</button></form>{answer&&<div className="multi-answer">{answer}</div>}</section>
  </>}
  <p className="research-scope">口径说明：播种面积代理灌溉面积；工业增加值及强度使用2023价格代理，历史火电口径未完全统一；工业2030门槛是全国目标参考，不是苏北考核达标认定。生活只采用人口城镇化驱动，未另设生活节水路径。</p>
 </div>;
}
