import {runForecast,predict} from './engine.mjs';

export const SSP_NAMES=['SSP1','SSP2','SSP3','SSP4','SSP5'];
export const PATHS=[
 {id:'frozen',name:'近期平均效率维持'},
 {id:'trend',name:'历史节水趋势延续'},
 {id:'policy',name:'江苏政策参考约束'},
];
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const range=(a,b)=>Array.from({length:b-a+1},(_,i)=>a+i);
function number(v,label){if(v===''||v===null||v===undefined||!Number.isFinite(Number(v)))throw Error(label+'缺失或不是有限数值');return Number(v);}
function assertYears(rows,years,label){if(rows.length!==years.length||rows.some((r,i)=>r.year!==years[i]))throw Error(label+'年份缺失、重复或不连续');}
export function features(r,agriculturalIntensity=r.agriculturalIntensity,industrialIntensity=r.industrialIntensity){
 return [r.area*agriculturalIntensity*15/100000,r.industrialVA*industrialIntensity/10000,r.population*(1+r.urbanization/100)];
}
export function parseResearchSheets(sheets,source={}){
 for(const name of ['历史数据','未来指标','模型设置'])if(!Array.isArray(sheets[name]))throw Error('工作簿缺少“'+name+'”工作表');
 const headers=sheets['历史数据'][0]||[],fheaders=sheets['未来指标'][0]||[];
 for(const [i,re] of [[0,/年份/],[1,/人口/],[2,/城镇化/],[7,/面积/],[8,/灌溉/],[9,/工业增加值/],[12,/工业强度.*2023/],[13,/总需水/]])if(!re.test(String(headers[i])))throw Error('历史数据第'+(i+1)+'列名称与模型约定不一致');
 for(const [i,re] of [[0,/情景/],[1,/年份/],[8,/面积/],[9,/农业强度/],[10,/工业增加值/],[11,/速度/]])if(!re.test(String(fheaders[i])))throw Error('未来指标第'+(i+1)+'列名称与模型约定不一致');
 const hkeys=['year','population','urbanization','gdp','primary','secondary','tertiary','area','agriculturalIntensity','industrialVA','industrialRawIntensity','deflator','industrialIntensity','demand'];
 const fkeys=['year','population','urbanization','gdp','primary','secondary','tertiary','area','agriculturalIntensity','industrialVA','savingMultiplier'];
 const historical=sheets['历史数据'].slice(1).filter(r=>r.some(v=>v!==''&&v!=null)).map((r,i)=>Object.fromEntries(hkeys.map((k,j)=>[k,number(r[j],'历史第'+(i+2)+'行第'+(j+1)+'列')]))).sort((a,b)=>a.year-b.year);
 const future=sheets['未来指标'].slice(1).filter(r=>r.some(v=>v!==''&&v!=null)).map((r,i)=>({scenario:String(r[0]),...Object.fromEntries(fkeys.map((k,j)=>[k,number(r[j+1],'未来第'+(i+2)+'行第'+(j+2)+'列')]))}));
 const settings=Object.fromEntries(sheets['模型设置'].slice(1).filter(r=>r[0]).map(r=>[String(r[0]),r[1]]));
 const get=(k,fallback)=>settings[k]===undefined?fallback:number(settings[k],k);
 const bundle={schema:'waterlab-ssp-v1',source,historical,future,policy:{mode:String(settings.policy_mode||''),baseYear:get('base_year',2023),targetYear:get('policy_target_year',2030),irrigationBase:get('irrigation_base',NaN),irrigationTarget:get('irrigation_target',NaN),industrialTarget2000:get('industrial_target_2000',NaN),industrialPriceFactor:get('industrial_price_factor',NaN),provincialWaterLimit:get('provincial_water_limit',NaN),regionalWaterLimit:get('regional_water_limit',600*165/571.4),regionalLimitYear:get('policy_target_year',2030),frozenStartYear:get('frozen_start_year',2021),frozenEndYear:get('frozen_end_year',2023)},seed:get('seed',20261024),notes:(sheets['口径与政策依据']||[]).slice(1).filter(r=>r[0]).map(r=>({topic:String(r[0]),definition:String(r[1]||''),scope:String(r[2]||''),source:String(r[3]||'')}))};
 validateResearchBundle(bundle);return bundle;
}
export function validateResearchBundle(b){
 if(b?.schema!=='waterlab-ssp-v1')throw Error('不是支持的论文SSP输入格式');
 if(!Number.isInteger(b.seed)||b.seed<0)throw Error('随机种子必须为非负整数');
 if(b.policy.mode!=='jiangsu-2012'||b.policy.baseYear!==2023||b.policy.targetYear!==2030)throw Error('请使用含江苏政策参数的最新版输入（2023基期、2030目标年）');
 if(!finite(b.policy.irrigationBase)||!finite(b.policy.irrigationTarget)||b.policy.irrigationBase<=0||b.policy.irrigationTarget<b.policy.irrigationBase||b.policy.irrigationTarget>1)throw Error('灌溉效率应满足0<基期≤目标≤1');
 for(const key of ['industrialTarget2000','industrialPriceFactor','provincialWaterLimit'])if(!finite(b.policy[key])||b.policy[key]<=0)throw Error('江苏政策参数必须大于0：'+key);
 assertYears(b.historical,range(2010,2023),'历史');
 const common=['population','urbanization','gdp','primary','secondary','tertiary','area','agriculturalIntensity','industrialVA'];
 for(const r of [...b.historical,...b.future]){
  if(common.some(k=>!finite(r[k])||r[k]<0)||r.urbanization>100||r.agriculturalIntensity<=0)throw Error('输入指标应为非负有限数，城镇化率为0—100%，农业强度大于0');
  if(Math.abs(r.primary+r.secondary+r.tertiary-r.gdp)>Math.max(.1,r.gdp*1e-4))throw Error(r.year+'年三产业增加值之和与GDP不一致');
  if(r.industrialVA>r.secondary+Math.max(.1,r.secondary*1e-4))throw Error(r.year+'年工业增加值超过第二产业');
 }
 for(const r of b.historical)if(!finite(r.industrialIntensity)||r.industrialIntensity<=0||!finite(r.demand)||r.demand<=0||!finite(r.deflator)||r.deflator<=0)throw Error('历史强度、平减指数和需水量应大于0');
 for(const r of b.future)if(!SSP_NAMES.includes(r.scenario)||!finite(r.savingMultiplier)||r.savingMultiplier<=0)throw Error('情景名称或节水速度倍数无效');
 if(b.future.length!==35)throw Error('五种SSP应各有2024—2030年7行，共35行');
 for(const s of SSP_NAMES)assertYears(b.future.filter(r=>r.scenario===s).sort((a,b)=>a.year-b.year),range(2024,2030),s);
 return b;
}
export function prepareResearch(b){
 validateResearchBundle(b);
 const t=b.historical.map(r=>r.year-2010),z=b.historical.map(r=>Math.log(r.industrialIntensity)),tm=mean(t),zm=mean(z);
 const decay=-t.reduce((sum,v,i)=>sum+(v-tm)*(z[i]-zm),0)/t.reduce((sum,v)=>sum+(v-tm)**2,0);
 const base=b.historical.at(-1),rows=b.historical.map(r=>({year:r.year,x:features(r),y:r.demand})),paths=[];
 const recent=b.historical.filter(r=>r.year>=b.policy.frozenStartYear&&r.year<=b.policy.frozenEndYear);
 if(recent.length!==3)throw Error('近期平均效率需要2021—2023完整三年输入');
 const frozenEfficiency={agriculturalIntensity:mean(recent.map(r=>r.agriculturalIntensity)),industrialIntensity:mean(recent.map(r=>r.industrialIntensity))};
 for(const scenario of SSP_NAMES){
  const future=b.future.filter(r=>r.scenario===scenario).sort((a,c)=>a.year-c.year);
  const industrialTrend=r=>base.industrialIntensity*Math.exp(-decay*r.savingMultiplier*(r.year-b.policy.baseYear));
  const terminalCap=b.policy.industrialTarget2000/b.policy.industrialPriceFactor;
  for(const path of PATHS){
   const points=future.map(r=>{
    const trend=industrialTrend(r),fraction=(r.year-b.policy.baseYear)/(b.policy.targetYear-b.policy.baseYear);
    const eta=b.policy.irrigationBase+(b.policy.irrigationTarget-b.policy.irrigationBase)*fraction;
    const agCap=base.agriculturalIntensity*b.policy.irrigationBase/eta;
    const cap=base.industrialIntensity*(terminalCap/base.industrialIntensity)**fraction;
    const ai=path.id==='frozen'?frozenEfficiency.agriculturalIntensity:path.id==='policy'?Math.min(r.agriculturalIntensity,agCap):r.agriculturalIntensity;
    const ii=path.id==='frozen'?frozenEfficiency.industrialIntensity:path.id==='policy'?Math.min(trend,cap):trend;
    const agriculturalReferenceBinding=path.id==='policy'&&agCap<r.agriculturalIntensity-1e-10;
    const industrialReferenceBinding=path.id==='policy'&&cap<trend-1e-10;
    return {...r,path:path.id,pathName:path.name,agriculturalIntensity:ai,industrialIntensity:ii,industrialTrend:trend,irrigationReference:eta,agriculturalReferenceCap:agCap,industrialReferenceCap:cap,agriculturalReferenceBinding,industrialReferenceBinding,referenceBinding:agriculturalReferenceBinding||industrialReferenceBinding,x:features(r,ai,ii)};
   });
   paths.push({scenario,path:path.id,name:path.name,points});
  }
 }
 return {rows,paths,industrialDecay:decay,base,frozenEfficiency};
}
export function runResearchForecast(bundle,progress=()=>{}){
 const started=performance.now(),prepared=prepareResearch(bundle);
 // Same fitted model, scaling and anchor for all 15 combinations. No future labels enter tuning.
 const fitResult=runForecast(prepared.rows,{holdout:4,horizon:7,future:prepared.paths[0].points.map(r=>({year:r.year,x:r.x})),seed:bundle.seed,includeModel:true},progress);
 const model=fitResult.finalModel,anchor=predict(model,[prepared.rows.at(-1).x])[0],baseDemand=prepared.base.demand;
 const frozenBaseDemand=baseDemand+predict(model,[features(prepared.base,prepared.frozenEfficiency.agriculturalIntensity,prepared.frozenEfficiency.industrialIntensity)])[0]-anchor;
 const baselineEfficiencyEffect=frozenBaseDemand-baseDemand;
 let nonpositive=0,outside=0;const negativeResponses=[0,0,0];
 const combinations=prepared.paths.map(path=>{
  const raw=predict(model,path.points.map(r=>r.x));
  const rows=path.points.map((r,i)=>{
   const demand=baseDemand+raw[i]-anchor;
   if(demand<=0)nonpositive++;
   if(r.x.some((v,j)=>v<model.xmin[j]||v>model.xmax[j]))outside++;
   for(let j=0;j<3;j++){const down=[...r.x];down[j]*=.99;if(demand-(baseDemand+predict(model,[down])[0]-anchor)<-1e-7)negativeResponses[j]++;}
   return {...r,demand};
  });
  return {...path,points:rows};
 });
 let reversed=0,identical=0;const comparisons=[];
 for(const scenario of SSP_NAMES){
  const [frozen,trend,policy]=PATHS.map(p=>combinations.find(c=>c.scenario===scenario&&c.path===p.id));
  for(let i=0;i<7;i++){
   const a=frozen.points[i].demand,b=trend.points[i].demand,c=policy.points[i].demand,development=a-frozenBaseDemand,historicalSaving=a-b,additionalPolicySaving=b-c,totalSaving=a-c;
   if(a<b-1e-7||b<c-1e-7)reversed++;
   if(Math.abs(b-c)<1e-9)identical++;
   comparisons.push({scenario,year:frozen.points[i].year,frozen:a,trend:b,policy:c,development,baselineEfficiencyEffect,frozenBaseDemand,historicalSaving,additionalPolicySaving,totalSaving,netChange:c-baseDemand,regionalReferenceLimit:bundle.policy.regionalWaterLimit,regionalLimitApplicable:frozen.points[i].year===bundle.policy.targetYear,policyExceedance:frozen.points[i].year===bundle.policy.targetYear?Math.max(0,c-bundle.policy.regionalWaterLimit):null,offsetRatio:development>1e-9?totalSaving/development:null});
  }
 }
 progress({label:'五种SSP × 三条节水路径已完成',progress:100});
 return {schema:'waterlab-ssp-result-v1',source:bundle.source,history:prepared.rows,combinations,comparisons,metrics:fitResult.metrics,backtest:fitResult.backtest,selections:fitResult.selections,config:fitResult.config,industrialDecay:prepared.industrialDecay,policy:bundle.policy,diagnostics:{nonpositive,outside,negativeResponses,reversed,identicalTrendPolicyPoints:identical,points:105},baseYear:2023,baseDemand,frozenBaseDemand,baselineEfficiencyEffect,createdAt:new Date().toISOString(),seconds:(performance.now()-started)/1000,provenance:{modelVersion:'regularized-bp-v1',method:'正则BP集成',scope:'苏北五市合计，2024—2030，5 SSP × 3效率路径',featureDefinition:['面积×亩均强度×15/100000','工业增加值×工业强度/10000','人口×(1+城镇化率/100)'],validation:'2020—2023固定起点及滚动开发回测；给定实际驱动；该时期已用于模型探索',prng:'浏览器mulberry32，与MATLAB Twister不同，不承诺逐位一致',comparison:'同模型内条件情景差值，不是政策因果效果；面积与工业强度使用现有代理口径'}};
}
