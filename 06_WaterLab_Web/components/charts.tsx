'use client';
import {Area,Bar,CartesianGrid,ComposedChart,Legend,Line,ReferenceLine,ResponsiveContainer,Tooltip,XAxis,YAxis} from 'recharts';
import type {Forecast,Dataset,ChartKind} from '../lib/types';
const colors={actual:'#66899c',base:'#118a7a',saving:'#78b9a3',growth:'#c99c5c',ols:'#9b94bf'};
const fmt=(v:number)=>Number(v).toLocaleString('en-US',{maximumFractionDigits:2});
function Tip({active,payload,label}:any){if(!active||!payload?.length)return null;return <div className="chart-tooltip"><strong>{label}</strong>{payload.filter((p:any)=>p.value!==null&&p.value!==undefined).map((p:any)=><div key={p.dataKey}><i style={{background:p.color}}/>{p.name}<b>{fmt(p.value)}</b></div>)}</div>;}
export function ForecastChart({result,dataset,kind}:{result:Forecast;dataset:Dataset;kind:ChartKind}){
 const last=result.rows.at(-1)!;
 const forecastRows=result.rows.map((r,i)=>({year:r.year,actual:r.y,...(i===result.rows.length-1?{base:r.y,saving:r.y,growth:r.y}:{})})).concat(result.forecasts.map(r=>({year:r.year,base:r.base,saving:r.saving,growth:r.growth})) as any);
 const response=result.sensitivity[0].map((r,i)=>({change:r.change.toFixed(0),a:r.value,b:result.sensitivity[1][i].value,c:result.sensitivity[2][i].value}));
 return <div className="chart-canvas" id="main-chart" role="img" aria-label={kind==='trend'?'历史需水与三种假设情景预测图':kind==='backtest'?'实测与模型回测对比图':'驱动变化与预测响应图'}>
  <ResponsiveContainer width="100%" height="100%">
  {kind==='trend'?<ComposedChart data={forecastRows} margin={{top:20,right:15,left:8,bottom:5}}>
   <defs><linearGradient id="waterFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#91b7c0" stopOpacity={.17}/><stop offset="100%" stopColor="#91b7c0" stopOpacity={.015}/></linearGradient></defs>
   <CartesianGrid stroke="#edf1f2" vertical={false}/><XAxis dataKey="year" tickLine={false} axisLine={false} minTickGap={26} tick={{fontSize:11,fill:'#94a1a7'}} dy={10}/><YAxis domain={['auto','auto']} tickLine={false} axisLine={false} width={48} tickFormatter={v=>fmt(v)} tick={{fontSize:11,fill:'#94a1a7'}}/>
   <Tooltip content={<Tip/>}/><ReferenceLine x={last.year} stroke="#c9d6d9" strokeDasharray="4 5" label={{value:'预测起点',position:'insideTopRight',fill:'#8b9fa3',fontSize:10}}/>
   <Area type="monotone" dataKey="actual" name="历史实测" stroke={colors.actual} strokeWidth={2.5} fill="url(#waterFill)" dot={false} activeDot={{r:4}}/>
   <Line type="monotone" dataKey="growth" name="农工驱动 +5%" stroke={colors.growth} strokeWidth={1.8} strokeDasharray="5 4" dot={false}/>
   <Line type="monotone" dataKey="saving" name="农工驱动 −5%" stroke={colors.saving} strokeWidth={1.8} strokeDasharray="5 4" dot={false}/>
   <Line type="monotone" dataKey="base" name="基准情景" stroke={colors.base} strokeWidth={2.7} dot={{r:2,strokeWidth:0}}/>
  </ComposedChart>:kind==='backtest'?<ComposedChart data={result.backtest} margin={{top:20,right:15,left:8,bottom:5}}>
   <CartesianGrid stroke="#edf1f2" vertical={false}/><XAxis dataKey="year" tickLine={false} axisLine={false} tick={{fontSize:11,fill:'#94a1a7'}} dy={10}/><YAxis domain={['auto','auto']} tickLine={false} axisLine={false} width={48} tick={{fontSize:11,fill:'#94a1a7'}}/><Tooltip content={<Tip/>}/>
   <Bar dataKey="actual" name="实测" fill="#d5e4e7" barSize={28} radius={[4,4,0,0]}/>
   <Line dataKey="rolling" name="PSO-BP 滚动回测" stroke={colors.base} strokeWidth={2.5} dot={{r:4}}/>
   <Line dataKey="fixed" name="PSO-BP 固定起点" stroke={colors.growth} strokeWidth={2} dot={{r:3}}/>
   <Line dataKey="ols" name="线性模型滚动对照" stroke={colors.ols} strokeWidth={1.5} strokeDasharray="4 4" dot={{r:3}}/>
  </ComposedChart>:<ComposedChart data={response} margin={{top:20,right:15,left:8,bottom:5}}>
   <CartesianGrid stroke="#edf1f2" vertical={false}/><XAxis dataKey="change" tickLine={false} axisLine={false} minTickGap={30} tickFormatter={v=>v+'%'} tick={{fontSize:11,fill:'#94a1a7'}} dy={10}/><YAxis domain={['auto','auto']} tickLine={false} axisLine={false} width={48} tick={{fontSize:11,fill:'#94a1a7'}}/><Tooltip content={<Tip/>}/>
   <ReferenceLine y={last.y} stroke="#ccd9dc" strokeDasharray="3 3"/>
   <Line dataKey="a" name={dataset.featureNames[0]} stroke={colors.base} strokeWidth={2.3} dot={false}/><Line dataKey="b" name={dataset.featureNames[1]} stroke={colors.growth} strokeWidth={2.3} dot={false}/><Line dataKey="c" name={dataset.featureNames[2]} stroke={colors.ols} strokeWidth={2.3} dot={false}/>
  </ComposedChart>}
  </ResponsiveContainer>
 </div>;
}
export function Sparkline({values,color='#118a7a'}:{values:number[];color?:string}){const min=Math.min(...values),max=Math.max(...values),d=values.map((v,i)=>(i*72/(values.length-1||1)).toFixed(1)+','+(25-(v-min)/(max-min||1)*22).toFixed(1)).join(' ');return <svg viewBox="0 0 74 30" width="74" height="30" aria-hidden="true"><polyline fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" points={d}/></svg>;}
