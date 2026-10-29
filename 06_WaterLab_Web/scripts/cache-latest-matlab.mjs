import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {runResearchForecast} from '../public/research-engine.mjs';
const b=JSON.parse(fs.readFileSync(new URL('../public/research-input.json',import.meta.url),'utf8'));
const m=JSON.parse(fs.readFileSync(new URL('../../07_BP_Ensemble/results/result.json',import.meta.url),'utf8'));
if(b.source.sha256!==m.input_sha256)throw Error('MATLAB result hash does not match current synchronized workbook');
const r=runResearchForecast(b);let maxBrowserGap=0;
for(const c of r.combinations){const s=Number(c.scenario.at(-1))-1,j={frozen:0,trend:1,policy:2}[c.path];for(const [i,p] of c.points.entries()){maxBrowserGap=Math.max(maxBrowserGap,Math.abs(p.demand-m.future_prediction[i][s][j]));p.demand=m.future_prediction[i][s][j];}}
r.frozenBaseDemand=m.frozen_base_demand;r.baselineEfficiencyEffect=m.baseline_efficiency_effect;
for(const c of r.comparisons){const s=Number(c.scenario.at(-1))-1,i=c.year-2024;const [f,t,p]=m.future_prediction[i][s];Object.assign(c,{frozen:f,trend:t,policy:p,development:f-r.frozenBaseDemand,baselineEfficiencyEffect:r.baselineEfficiencyEffect,frozenBaseDemand:r.frozenBaseDemand,historicalSaving:f-t,additionalPolicySaving:t-p,totalSaving:f-p,netChange:p-r.baseDemand,offsetRatio:f-r.frozenBaseDemand>1e-9?(f-p)/(f-r.frozenBaseDemand):null,policyExceedance:c.year===2030?Math.max(0,p-b.policy.regionalWaterLimit):null});}
for(const [key,a] of [['fixed',m.fixed_metrics],['rolling',m.rolling_metrics]])Object.assign(r.metrics[key],{mape:a[0],mae:a[1],rmse:a[2],mse:a[3]});
r.backtest=m.validation_years.map((year,i)=>({year,actual:m.actual[i],fixed:m.fixed_prediction[i],rolling:m.rolling_prediction[i]}));
r.config.lambda=m.selected.at(-1).penalty;r.selections=m.selected;r.createdAt=new Date().toISOString();r.seconds=m.training_seconds;
r.diagnostics.negativeResponses=m.direction_counts.slice(0,3);r.diagnostics.nonpositive=m.direction_counts[3];r.diagnostics.reversed=m.direction_counts[4];
r.provenance.modelVersion='regularized-bp-v1';r.config.initialization='bp';r.config.population=0;r.config.generations=0;r.provenance.method='MATLAB 正则BP集成';r.provenance.prng='MATLAB Twister，内置结果为当前重算；浏览器复算使用mulberry32';r.provenance.frozen='2021—2023平均农业/工业效率；基准效率差异单独列示';r.provenance.regionalLimit='173.258662933亿m³：2023份额折算2030参考，非正式配额，不截断预测';r.provenance.maxBrowserRecalculationGap=maxBrowserGap;
const path=new URL('../public/research-latest-result.json',import.meta.url);fs.writeFileSync(path,JSON.stringify(r,null,2));console.log(JSON.stringify({file:fileURLToPath(path),maxBrowserGap,terminal:r.comparisons.filter(c=>c.year===2030),fixedMAPE:r.metrics.fixed.mape,rollingMAPE:r.metrics.rolling.mape}));
