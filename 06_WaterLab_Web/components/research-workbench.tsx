'use client';
import {useEffect,useRef,useState} from 'react';
import {Download,FileSpreadsheet,Loader2,Play,RefreshCw,Send,Square,Upload} from 'lucide-react';
import {ResearchFigureGallery} from './research-figure-gallery';
import {MultiAgentPanel} from './multiagent-panel';
import {parseResearchSheets,PATHS,prepareResearch,SSP_NAMES} from '../public/research-engine.mjs';

type Bundle=ReturnType<typeof parseResearchSheets>&{source:Record<string,any>};
type ResearchResult=any;
const format=(v:number,d=3)=>Number.isFinite(v)?v.toLocaleString('zh-CN',{maximumFractionDigits:d,minimumFractionDigits:d}):'—';
function saveFile(name:string,data:string,type='application/json'){
 const url=URL.createObjectURL(new Blob([data],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
}
function csvCell(v:unknown){return '"'+String(v??'').replaceAll('"','""')+'"';}
function downloadCSV(name:string,rows:unknown[][]){saveFile(name,'\uFEFF'+rows.map(r=>r.map(csvCell).join(',')).join('\r\n'),'text/csv;charset=utf-8');}

export function ResearchWorkbench({connected,multi=false}:{connected:boolean;multi?:boolean}){
 const [bundle,setBundle]=useState<Bundle|null>(null),[result,setResult]=useState<ResearchResult|null>(null);
 const [busy,setBusy]=useState(false),[loading,setLoading]=useState(false),[error,setError]=useState(''),[status,setStatus]=useState('正在读取当前论文模型输入…');
 const [scenario,setScenario]=useState('SSP2');
 const [question,setQuestion]=useState(''),[answer,setAnswer]=useState(''),[asking,setAsking]=useState(false),[inputTab,setInputTab]=useState('history');
 const workerRef=useRef<Worker|null>(null),fileRef=useRef<HTMLInputElement>(null),abortRef=useRef<AbortController|null>(null),generation=useRef(0);
 const autoSource=useRef<string|null>(null);
 const prepared=bundle?prepareResearch(bundle):null;
 function cancel(){generation.current++;workerRef.current?.terminate();workerRef.current=null;abortRef.current?.abort();setBusy(false);setAsking(false);}
 function load(b:Bundle){prepareResearch(b);cancel();setBundle(b);setResult(null);setAnswer('');setError('');setStatus('已读取14年历史、35行SSP指标；可运行15种情景与节水路径组合。');}
 async function reload(){cancel();setLoading(true);setError('');try{const r=await fetch('/research-input.json',{cache:'no-store'});if(!r.ok)throw Error('读取内置论文输入失败，请运行启动脚本同步输入。');const b=await r.json() as Bundle;load(b);const cached=await fetch('/research-latest-result.json',{cache:'no-store'});if(cached.ok){const v:any=await cached.json();if(v.source?.sha256===b.source.sha256){setResult(v);setStatus('已载入最新MATLAB重算结果：2022年总量已纠正，15组路径已同步。');}}}catch(e){setError(String(e instanceof Error?e.message:e));}finally{setLoading(false);}}
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
 useEffect(()=>{const key=bundle?.source.sha256;if(multi&&bundle&&!result&&!busy&&!loading&&autoSource.current!==key){autoSource.current=key;execute();}},[multi,bundle,result,busy,loading]);
 function summaryCSV(){
  if(!result)return;const labels=SSP_NAMES.flatMap(s=>PATHS.map(p=>s+'_'+p.name));
  const rows:unknown[][]=[['年份',...labels],...Array.from({length:8},(_,i)=>[2023+i,...SSP_NAMES.flatMap(s=>PATHS.map(p=>i===0?result.baseDemand:result.combinations.find((c:any)=>c.scenario===s&&c.path===p.id).points[i-1].demand))])];
  downloadCSV('苏北需水预测_五SSP三路径_亿立方米.csv',rows);
 }
 function detailCSV(){
  if(!result)return;downloadCSV('SSP路径输入与需水预测明细.csv',[
   ['SSP','路径','年份','人口_万人','城镇化率_%','GDP_亿元2023价代理','第一产业','第二产业','第三产业','播种灌溉代理_千公顷','亩均灌溉强度_m3每亩','工业增加值_亿元2023价代理','工业强度_m3每万元2023价代理','节水速度倍数','农业驱动','工业驱动','人口城镇化驱动','预测总需水_亿m3','农业参考约束是否生效','工业参考约束是否生效'],
   ...result.combinations.flatMap((c:any)=>c.points.map((r:any)=>[c.scenario,c.name,r.year,r.population,r.urbanization,r.gdp,r.primary,r.secondary,r.tertiary,r.area,r.agriculturalIntensity,r.industrialVA,r.industrialIntensity,r.savingMultiplier,...r.x,r.demand,r.agriculturalReferenceBinding?'是':'否',r.industrialReferenceBinding?'是':'否'])),
  ]);
 }
 function comparisonCSV(){if(!result)return;downloadCSV('发展新增与节水抵消对比.csv',[
  ['SSP','年份','效率保持需水','历史趋势需水','规划参考需水','发展变化_不含效率基准差异_亿m3','历史趋势节水_亿m3','额外规划节水_亿m3','合计节水_亿m3','规划净变化_亿m3','抵消比例_小数','近期均值效率基准影响_亿m3','2030政策路径超参考上限_亿m3'],
  ...result.comparisons.map((r:any)=>[r.scenario,r.year,r.frozen,r.trend,r.policy,r.development,r.historicalSaving,r.additionalPolicySaving,r.totalSaving,r.netChange,r.offsetRatio,r.baselineEfficiencyEffect,r.policyExceedance]),
 ]);}
 async function ask(){
  if(!result||!question.trim()||asking)return;const id=generation.current,controller=new AbortController();abortRef.current=controller;setAsking(true);setError('');
  try{
   const evidence={task:'ssp-water-demand',unit:'亿m³',baseYear:2023,baseDemand:result.baseDemand,model:result.provenance.method,historyYears:'2010—2023',futureYears:'2024—2030',scenario,policy:result.policy,policyFacts:{policyStarts:2024,all35PointsIdentical:result.diagnostics.identicalTrendPolicyPoints===35,explanation:'农业按省级效率0.622至0.69构造参考强度上限；工业按2000价11及价格桥构造参考上限；是否生效看逐年计算'},metrics:result.metrics,diagnostics:result.diagnostics,annual:result.comparisons.filter((r:any)=>r.scenario===scenario),terminal:result.comparisons.filter((r:any)=>r.year===2030),pathDefinitions:{frozen:'农业与工业强度保持2021—2023平均；发展规模按各SSP变化',trend:'农业采用输入趋势；工业由2010—2023强度拟合指数速度，按SSP倍数外推',policy:'江苏政策参考：农业强度取历史趋势与m2023×eta2023/eta(t)较小值，效率从0.622线性到0.69；工业取趋势与2000价11换算参考上限路径的较小值；苏北173.26亿m³为2023份额折算参考上限，仅检查2030超限，不截断预测；全省600不直接套苏北'},scope:['面积是播种面积代理灌溉面积','工业含火电口径未完全统一，强度为模型代理','灌溉效率和工业价格桥均为省级代理，不能认定苏北正式达标','情景差值不是因果政策效果；不输出实测部门用水量','回测已参与模型开发，不是新增独立测试']};
   const r=await fetch('/api/agent',{method:'POST',signal:controller.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:[{role:'user',content:question}],evidence:JSON.parse(JSON.stringify(evidence,(_,v)=>typeof v==='number'?Math.round(v*1e6)/1e6:v))})});const v:any=await r.json();if(!r.ok)throw Error(v.error||'分析失败');if(id===generation.current)setAnswer(v.answer);
  }catch(e){if(id===generation.current)setError(String(e instanceof Error?e.message:e));}finally{if(id===generation.current)setAsking(false);}
 }
 const selected=result?.comparisons.filter((r:any)=>r.scenario===scenario),terminal=selected?.at(-1);
 return <div className="research-workbench">
  <section className="adaptive-intro"><div className="eyebrow">SU BEI · SSP1—SSP5</div><h2>{multi?'Multi-agent · 五种SSP与三条节水路径':'苏北五市 · 五种SSP与三条节水路径'}</h2><p>{multi?'复用论文模型全部15组结果；多角色核对数值与节水抵消关系，按真实数据绘图。':'采用当前正则BP主模型，统一训练与锚定，对比2024—2030年15种组合。数据来自论文现有输入工作簿。'}</p></section>
  <section className="data-card research-source">
   <div className="dataset-row"><div className="dataset-icon"><FileSpreadsheet size={22}/></div><div className="dataset-name"><strong>{bundle?.source.file||'正在读取输入工作簿'}</strong><span>苏北五市合计 · 2010—2023历史 · 2024—2030预测 · 单位：亿m³</span></div></div>
   {bundle&&<details><summary>数据来源与输入指纹</summary><p className="research-hash">输入指纹：{bundle.source.sha256}<br/>载入时间：{new Date(bundle.source.syncedAt).toLocaleString('zh-CN')}</p></details>}
   <div className="research-actions"><button className="secondary" disabled={busy||loading||asking} onClick={reload}><RefreshCw size={14}/>重新读取内置输入</button><button className="secondary" disabled={busy||loading||asking} onClick={()=>fileRef.current?.click()}><Upload size={14}/>上传同格式输入Excel</button><button className="primary" disabled={!bundle||busy||loading||asking} onClick={execute}>{busy?<Loader2 size={15} className="spin"/>:<Play size={15}/>}运行15种组合</button>{(busy||asking)&&<button className="secondary" onClick={()=>{cancel();setStatus('已停止当前计算。');}}><Square size={12}/>停止</button>}<input id="research-file-input" ref={fileRef} type="file" accept=".xlsx" hidden onChange={e=>{if(e.target.files?.[0])void upload(e.target.files[0]);e.target.value='';}}/></div>
   <p className="adaptive-limit">启动脚本每次从模型data文件夹同步输入。修改原Excel后，请重新启动工作台，或直接上传新文件；“重新读取”只加载最近同步的内置副本。人口、产业、GDP等原值全部保留；模型实际输入为农业、工业和人口城镇化三个合成驱动。</p>
  </section>
  {result&&bundle&&<p className="adaptive-limit">2030年苏北总量参考上限 {format(bundle.policy.regionalWaterLimit,2)} 亿m³（按2023年用水份额折算，非正式配额）；生活不加政策约束。当前SSP政策路径2030年预测 {format(result.comparisons.find((r:any)=>r.scenario===scenario&&r.year===2030)?.policy??NaN)} 亿m³，超参考上限 {format(Math.max(0,(result.comparisons.find((r:any)=>r.scenario===scenario&&r.year===2030)?.policy??0)-bundle.policy.regionalWaterLimit))} 亿m³。</p>}
  <div className="adaptive-status" role="status">{(busy||loading||asking)&&<Loader2 size={15} className="spin"/>}{status}</div>{error&&<div className="error-box" role="alert">{error}</div>}
  {bundle&&prepared&&<details className="research-inputs"><summary>查看历史基础数据、未来输入与路径定义</summary>
   <div className="research-definitions"><p><b>效率保持：</b>农业和工业用水强度均保持2021—2023年平均水平（计算对照，不表示历史效率不变），其他发展指标按各SSP变化。</p><p><b>历史趋势：</b>农业使用原输入的指数衰减结果；工业统一价格口径后拟合2010—2023年指数速度 k={format(prepared.industrialDecay,6)}，SSP1/5乘1.2，SSP2/4乘1，SSP3乘0.8（研究假设）。</p><p><b>江苏政策参考约束：</b>以苏政发〔2012〕27号为依据，灌溉效率由2023年省级{format(bundle.policy.irrigationBase,3)}线性提高到2030年{format(bundle.policy.irrigationTarget,3)}，农业强度取历史趋势与换算上限较小值。工业采用2000年不变价、不含火电的{format(bundle.policy.industrialTarget2000,0)}m³/万元目标，折算至2023价代理为{format(bundle.policy.industrialTarget2000/bundle.policy.industrialPriceFactor,3)}；同样取较小值。苏北2030年总量参考上限为{format(bundle.policy.regionalWaterLimit,2)}亿m³，按2023年份额折算，仅作超限校核，不截断预测。全省{format(bundle.policy.provincialWaterLimit,0)}亿m³上限不直接套给苏北。</p><p><b>三个驱动：</b>面积×农业强度×15/100000；工业增加值×工业强度/10000；人口×(1+城镇化率/100)。面积单位千公顷，人口万人，工业增加值亿元。保留原代理特征含义。</p></div>
   <div className="research-actions"><button className="text-button" onClick={()=>setInputTab('history')}>历史输入（14年）</button><button className="text-button" onClick={()=>setInputTab('future')}>五SSP未来输入（35行）</button><button className="text-button" onClick={()=>setInputTab('paths')}>路径强度与合成驱动（105行）</button></div>
   <div className="table-scroll"><table><thead><tr>{(inputTab==='history'?['年份','人口(万人)','城镇化率(%)','GDP(亿元)','第一产业','第二产业','第三产业','面积(千公顷)','农业强度','工业增加值','工业强度(原)','省平减代理','工业强度(2023价)','实际需水']:inputTab==='future'?['SSP','年份','人口','城镇化率','GDP','第一产业','第二产业','第三产业','面积','农业趋势强度','工业增加值','节水速度倍数']:['SSP','路径','年份','农业强度','工业强度','农业驱动','工业驱动','人口城镇化驱动']).map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{(inputTab==='history'?bundle.historical.map((r:any)=>[r.year,r.population,r.urbanization,r.gdp,r.primary,r.secondary,r.tertiary,r.area,r.agriculturalIntensity,r.industrialVA,r.industrialRawIntensity,r.deflator,r.industrialIntensity,r.demand]):inputTab==='future'?bundle.future.map((r:any)=>[r.scenario,r.year,r.population,r.urbanization,r.gdp,r.primary,r.secondary,r.tertiary,r.area,r.agriculturalIntensity,r.industrialVA,r.savingMultiplier]):prepared.paths.flatMap(c=>c.points.map((r:any)=>[c.scenario,c.name,r.year,r.agriculturalIntensity,r.industrialIntensity,...r.x]))).map((row:any[],i:number)=><tr key={i}>{row.map((v:any,j:number)=><td key={j}>{typeof v==='number'?(Number.isInteger(v)?v:format(v)):v}</td>)}</tr>)}</tbody></table></div>
  </details>}
  {multi&&result&&<MultiAgentPanel key={result.createdAt} dataset={{name:bundle?.source.file||'苏北五市论文输入',unit:'亿m³',featureNames:['农业驱动','工业驱动','人口城镇化驱动'],rows:result.history}} result={result as any} research={result} enabled={connected&&!busy}/> }
  {result&&<details className="research-inputs" open={!multi}><summary>15组预测数据、趋势与导出</summary><>
   <div className="research-controls"><label>SSP<select aria-label="研究SSP情景" value={scenario} onChange={e=>{cancel();setScenario(e.target.value);setAnswer('');}}>{SSP_NAMES.map(s=><option key={s}>{s}</option>)}</select></label></div>
   <section className="adaptive-stats"><div><span>{scenario} · 2030效率保持</span><strong>{format(terminal.frozen,2)}</strong><small>亿m³</small></div><div><span>历史趋势</span><strong>{format(terminal.trend,2)}</strong><small>亿m³</small></div><div><span>规划参考约束</span><strong>{format(terminal.policy,2)}</strong><small>亿m³</small></div><div><span>节水对发展新增需求抵消比例</span><strong>{terminal.offsetRatio===null?'不适用':format(terminal.offsetRatio*100,1)+'%'}</strong><small>{terminal.offsetRatio===null?'效率保持路径下发展变化≤0':'(近期效率维持−政策)÷发展变化；已剔除效率基准差异'}</small></div></section>
   {result.diagnostics.identicalTrendPolicyPoints>0&&<div className="research-note">当前{result.diagnostics.identicalTrendPolicyPoints}/35个SSP年度点中，历史节水趋势与政策参考约束给出相同结果；图中后画的曲线可能覆盖前一条。额外规划节水量如实记为0。</div>}
   <ResearchFigureGallery key={result.source?.sha256||result.createdAt} result={result} historical={bundle?.historical||[]} scenario={scenario}/>
   <section className="details-card"><div className="card-heading"><h2>{scenario} · 逐年预测与对照</h2></div><div className="table-scroll"><table><thead><tr>{['年份','效率保持','历史趋势','规划约束','发展变化','历史节水','额外规划节水','规划净变化'].map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{selected.map((r:any)=><tr key={r.year}>{[r.year,r.frozen,r.trend,r.policy,r.development,r.historicalSaving,r.additionalPolicySaving,r.netChange].map((v,j)=><td key={j}>{j===0?v:format(v)}</td>)}</tr>)}</tbody></table></div><p className="adaptive-limit">发展变化相对近期平均效率下的2023年反事实基值；总量净变化仍相对2023实际值，效率基准差异单独保留在导出数据。情景差值是模型条件比较，不是政策因果效果，也不是实际部门用水量。</p></section>
   <section className="research-export"><h3>导出本次计算</h3><div className="research-actions"><button className="secondary" onClick={summaryCSV}><Download size={14}/>15组合汇总CSV</button><button className="secondary" onClick={detailCSV}>路径输入与预测明细CSV</button><button className="secondary" onClick={comparisonCSV}>发展与节水对照CSV</button><button className="secondary" onClick={()=>saveFile('WaterLab_SSP完整实验.json',JSON.stringify({bundle,result},null,2))}>完整实验JSON</button></div><p className="adaptive-limit">CSV可用Excel打开；汇总为年份纵向、15种组合横向，包含2023历史基值。完整JSON保留原始输入、指纹、逐折选参、回测和全部路径。</p></section>
   <details className="research-inputs"><summary>回测、模型配置与数值检查</summary><p>3-1-1网络，12成员，随机小初值，无PSO搜索；L2正则λ={result.config.lambda}，只按训练前缀内验证选择。未使用输入表旧版200成员/4隐节点设置。</p><p>固定起点MAPE {format(result.metrics.fixed.mape,4)}%；滚动MAPE {format(result.metrics.rolling.mape,4)}%。2020—2023已参与模型开发，是给定实际驱动的开发回测，不称新增独立测试。</p><p>105个预测点：非正值{result.diagnostics.nonpositive}，驱动超出历史范围{result.diagnostics.outside}，路径反序{result.diagnostics.reversed}；农业/工业/人口驱动负响应检查计数 {result.diagnostics.negativeResponses.join(' / ')}。这些是检查结果，不是模型硬约束。内置结果为最新MATLAB重算；点击运行可作浏览器复算，随机数不同，数值不保证逐位一致。</p><div className="table-scroll"><table><thead><tr>{['年份','实际值','固定预测','滚动预测'].map(h=><th key={h}>{h}</th>)}</tr></thead><tbody>{result.backtest.map((r:any)=><tr key={r.year}><td>{r.year}</td><td>{format(r.actual)}</td><td>{format(r.fixed)}</td><td>{format(r.rolling)}</td></tr>)}</tbody></table></div></details>
   <section className="adaptive-analysis"><h3>本地千问 · 当前SSP结果分析</h3><p>依据当前计算证据解读，不修改输入和预测值。AI文字可能出错，数值与路径定义以表格为准。</p><form onSubmit={e=>{e.preventDefault();void ask();}}><textarea aria-label="SSP预测分析问题" value={question} maxLength={2000} onChange={e=>setQuestion(e.target.value)} rows={3} placeholder="为什么历史趋势与规划约束重合？节水能抵消发展新增需水吗？"/><button className="primary" disabled={!connected||asking||busy||!question.trim()}><Send size={14}/>{asking?'正在分析':'分析当前结果'}</button></form>{answer&&<div className="multi-answer">{answer}</div>}</section>
  </></details>}
  <p className="research-scope">口径说明：播种面积代理灌溉面积；工业增加值及强度使用2023价格代理，历史火电口径未完全统一；采用江苏2012年政策作参考试算；省级灌溉效率与工业价格桥不代表苏北实测，不能认定正式政策达标。生活只采用人口城镇化驱动，未另设生活节水路径。</p>
 </div>;
}
