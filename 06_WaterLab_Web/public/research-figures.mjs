export const FIGURE_TYPES=[
 {id:'paths',name:'三条路径需水趋势',note:'同一SSP下比较三条路径；2023年显示同一历史实际值。'},
 {id:'all-paths',name:'五SSP路径趋势总览',note:'五个子图采用相同坐标范围，比较发展情景与节水路径。'},
 {id:'ssps',name:'同一路径的五SSP趋势',note:'固定节水路径，比较五种SSP情景。'},
 {id:'terminal',name:'2030年需水量分组柱状图',note:'各组分别为近期平均效率维持、历史趋势和政策参考路径；柱状图从零起画。'},
 {id:'savings',name:'节水量分解堆叠柱状图',note:'历史趋势节水＝维持−趋势；额外政策节水＝趋势−政策。两项之和为相对维持路径的总节水。'},
 {id:'waterfall',name:'发展增需与节水抵消瀑布图',note:'保留预测校准与近期平均效率相对2023实际值的基期衔接差。贡献是模型条件情景差，不是部门用水量或因果效应；纵轴局部放大。'},
 {id:'backtest',name:'历史回测与逐年误差',note:'上图比较实际与预测，下图误差＝预测−实际。2020—2023已参与模型开发，不是新增独立测试。'},
 {id:'intensity',name:'农业与工业强度路径',note:'强度是输入假设，不是神经网络预测的部门需水量；农业面积、工业价格与火电口径仍采用现有代理。'},
];
export const FIGURE_PATHS=[{id:'frozen',name:'近期平均效率维持'},{id:'trend',name:'历史趋势延续'},{id:'policy',name:'政策规划参考'}];
export const FIGURE_SSPS=['SSP1','SSP2','SSP3','SSP4','SSP5'];
function point(result,scenario,path,year){const p=result.combinations.find(c=>c.scenario===scenario&&c.path===path)?.points.find(r=>r.year===year);if(!p)throw Error(`缺少${scenario}/${path}/${year}预测`);return p;}
export function trendRows(result,scenario,path='policy',compare='paths',fullHistory=false){
 const keys=compare==='paths'?FIGURE_PATHS.map(p=>p.id):FIGURE_SSPS;
 return [...result.history.filter(r=>fullHistory||r.year===2023).map(r=>({year:r.year,...Object.fromEntries(keys.map(k=>[k,r.y]))})),...Array.from({length:7},(_,i)=>({year:2024+i,...Object.fromEntries(keys.map(k=>[k,point(result,compare==='paths'?scenario:k,compare==='paths'?k:path,2024+i).demand]))}))];
}
export function terminalRows(result){return FIGURE_SSPS.map(s=>({name:s,...Object.fromEntries(FIGURE_PATHS.map(p=>[p.id,point(result,s,p.id,2030).demand]))}));}
export function savingRows(result,cumulative=false){return FIGURE_SSPS.map(name=>{const rows=result.comparisons.filter(r=>r.scenario===name&&(cumulative||r.year===2030));if(rows.length!==(cumulative?7:1))throw Error('节水分解年份不完整');return {name,historical:rows.reduce((s,r)=>s+r.historicalSaving,0),additional:rows.reduce((s,r)=>s+r.additionalPolicySaving,0),total:rows.reduce((s,r)=>s+r.totalSaving,0)};});}
export function waterfallRows(result,scenario){
 const c=result.comparisons.find(r=>r.scenario===scenario&&r.year===2030);if(!c)throw Error('缺少2030贡献数据');
 const adjustment=c.baselineEfficiencyEffect??result.baselineEfficiencyEffect??(c.frozen-result.baseDemand-c.development);
 const items=[['2023实际',result.baseDemand,'total'],['发展变化',c.development,'increase'],['基期衔接差',adjustment,'adjustment'],['历史趋势节水',-c.historicalSaving,'saving'],['额外政策节水',-c.additionalPolicySaving,'policy'],['2030政策',c.policy,'total']];
 let running=result.baseDemand;const rows=items.map(([name,value,kind],i)=>{const start=i===0||i===items.length-1?0:running;const end=kind==='total'?Number(value):running+Number(value);if(kind!=='total')running=end;return {name,value:Number(value),kind,start,end};});
 if(Math.abs(running-c.policy)>1e-7)throw Error('瀑布图贡献加和与2030结果不一致');return rows;
}
export function backtestRows(result){return result.backtest.map(r=>({...r,fixedError:r.fixed-r.actual,rollingError:r.rolling-r.actual}));}
export function intensityRows(result,historical,scenario){
 const b=historical.find(r=>r.year===2023);if(!b)throw Error('缺少2023强度基值');
 return ['agriculturalIntensity','industrialIntensity'].map((key,j)=>({key,unit:j?'m³/万元（2023价代理）':'m³/亩',name:j?'工业用水强度':'农业灌溉强度',rows:[{year:2023,frozen:b[key],trend:b[key],policy:b[key],reference:b[key]},...Array.from({length:7},(_,i)=>{const y=2024+i;const p=point(result,scenario,'policy',y);return {year:y,...Object.fromEntries(FIGURE_PATHS.map(c=>[c.id,point(result,scenario,c.id,y)[key]])),reference:j?p.industrialReferenceCap:p.agriculturalReferenceCap};})]}));
}
