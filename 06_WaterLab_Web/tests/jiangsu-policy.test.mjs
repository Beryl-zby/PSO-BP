import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as XLSX from 'xlsx';
import {prepareResearch,runResearchForecast} from '../public/research-engine.mjs';
const b=JSON.parse(fs.readFileSync(new URL('../public/research-input.json',import.meta.url),'utf8'));
assert.equal(b.policy.mode,'jiangsu-2012');
const prepared=prepareResearch(b),r=runResearchForecast(b);
assert.equal(r.diagnostics.identicalTrendPolicyPoints,0);
assert.equal(r.diagnostics.reversed,0);
const book=XLSX.read(fs.readFileSync(new URL('../../data/需水预测模型对照.xlsx',import.meta.url)),{type:'buffer'});
const snapshot=XLSX.utils.sheet_to_json(book.Sheets['节水路径输入'],{header:1}).slice(1);
assert.equal(snapshot.length,105);
for(const path of prepared.paths)for(const p of path.points){
 const row=snapshot.find(v=>v[0]===path.scenario&&v[2]===p.year&&v[1]===({frozen:'效率保持',trend:'历史趋势',policy:'规划约束'}[path.path]));
 assert.ok(row);
 for(const [j,v] of [[4,p.agriculturalIntensity],[6,p.industrialIntensity],[7,p.x[0]],[8,p.x[1]],[9,p.x[2]]])assert.ok(Math.abs(row[j]-v)<1e-8);
}
const loose=structuredClone(b);loose.policy.irrigationTarget=loose.policy.irrigationBase;
const lr=runResearchForecast(loose);
assert.equal(lr.diagnostics.identicalTrendPolicyPoints,35,'无农业额外目标且工业不触发时应回到趋势；不能人为强制分开');
const strong=structuredClone(b);strong.policy.irrigationTarget=.72;
const sr=runResearchForecast(strong);
for(let i=0;i<35;i++)assert.ok(sr.comparisons[i].policy<=r.comparisons[i].policy+1e-9);
const invalid=structuredClone(b);invalid.policy.irrigationTarget=1.1;
assert.throws(()=>prepareResearch(invalid),/效率/);
const matlab=JSON.parse(fs.readFileSync(new URL('../../01_PSO_BP_Ensemble/results/result.json',import.meta.url),'utf8'));
assert.equal(b.historical.find(v=>v.year===2022).demand,194.3);
assert.ok(Math.abs(b.policy.regionalWaterLimit-600*165/571.4)<1e-9);
assert.ok(Math.abs(prepared.frozenEfficiency.agriculturalIntensity-(b.historical.slice(-3).reduce((s,r)=>s+r.agriculturalIntensity,0)/3))<1e-10);
const terminal=r.comparisons.filter(v=>v.year===2030);
const gap=Math.max(...terminal.map((v,i)=>Math.abs(v.policy-matlab.future_prediction[6][i][2])));
for(const v of r.comparisons){
 assert.ok(Math.abs(v.development+v.baselineEfficiencyEffect-v.totalSaving-v.netChange)<1e-9);
 if(v.year===2030)assert.equal(v.policyExceedance,Math.max(0,v.policy-b.policy.regionalWaterLimit));else assert.equal(v.policyExceedance,null);
}
console.log(JSON.stringify({inputSnapshotPoints:105,maxMatlabGap:gap,terminal:terminal.map(v=>({scenario:v.scenario,frozen:v.frozen,trend:v.trend,policy:v.policy})),checks:'105输入一致；近期平均效率；政策参数响应；宽松目标允许重合；2022修正；2030超限只报告；含效率基准差异的恒等式'}));
