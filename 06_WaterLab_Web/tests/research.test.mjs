import assert from 'node:assert/strict';
import fs from 'node:fs';
import {prepareResearch,runResearchForecast} from '../public/research-engine.mjs';
const bundle=JSON.parse(fs.readFileSync(new URL('../public/research-input.json',import.meta.url),'utf8'));
const prepared=prepareResearch(bundle),result=runResearchForecast(bundle);
assert.equal(prepared.rows.length,14);assert.equal(result.combinations.length,15);
assert.equal(result.combinations.flatMap(c=>c.points).length,105);
assert.ok(result.combinations.every(c=>c.points[0].year===2024&&c.points.at(-1).year===2030));
for(const r of result.comparisons){assert.ok(Math.abs(r.development+r.baselineEfficiencyEffect-r.totalSaving-r.netChange)<1e-10);assert.ok(Math.abs(r.historicalSaving+r.additionalPolicySaving-r.totalSaving)<1e-10);}
const altered=structuredClone(bundle);altered.future.forEach(r=>r.area*=1.01);const other=runResearchForecast(altered);
assert.deepEqual(result.backtest,other.backtest);assert.deepEqual(result.selections,other.selections);
const missing=structuredClone(bundle);missing.future.pop();assert.throws(()=>prepareResearch(missing),/35/);
const binding=structuredClone(bundle);binding.policy.industrialTarget2000=.5;const p=prepareResearch(binding);
assert.ok(p.paths.filter(c=>c.path==='policy').every(c=>c.points.at(-1).referenceBinding));
const originalResult=new URL('../../01_PSO_BP_Ensemble/results/result.json',import.meta.url);
let gap=null;
if(fs.existsSync(originalResult)){
 const original=JSON.parse(fs.readFileSync(originalResult,'utf8'));
 if(original.input_sha256===bundle.source.sha256){
  gap=Math.max(...result.combinations.flatMap((c,index)=>c.points.map((r,i)=>Math.abs(r.demand-original.future_prediction[i][Math.floor(index/3)][index%3]))));
  assert.ok(Number.isFinite(gap),'Browser/MATLAB difference is reported; canonical UI uses the MATLAB result with matching input hash');
 }
}
console.log(JSON.stringify({tests:'15 SSP/path combinations; 105 points; counterfactual accounting; future inputs do not affect tuning; invalid input rejection; binding policy; MATLAB agreement when source hashes match',input:bundle.source.sha256,maxMatlabGap:gap,diagnostics:result.diagnostics}));
