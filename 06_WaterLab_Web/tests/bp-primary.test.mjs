import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const base=new URL('../',import.meta.url).href;
const {fit,predict}=await import(base+'public/engine.mjs');
const {prepareResearch,runResearchForecast}=await import(base+'public/research-engine.mjs');
const {researchAdaptiveData}=await import(base+'public/research-adaptive.mjs');
const {profileData,evaluateCandidates,normalizeCandidate}=await import(base+'public/adaptive-engine.mjs');
const {assessForecastTask,initialTaskCandidates}=await import(base+'public/task-policy.mjs');
const {psoRecord}=await import(base+'public/workspace-records.mjs');
const bundle=JSON.parse(readFileSync(new URL(base+'public/research-input.json'),'utf8'));
test('default engine trains without a PSO population, deterministic and descent',()=>{
 const X=Array.from({length:9},(_,i)=>[i,i%3,i*.3]),Y=X.map(x=>10+x[0]*.4);
 const a=fit(X,Y,{population:0,generations:0,members:3}),b=fit(X,Y,{population:0,generations:0,members:3});
 assert.equal(a.initialization,'bp');assert.deepEqual(a.theta,b.theta);assert.ok(a.optimization.every(o=>o.finalLoss<=o.initialLoss));assert.ok(predict(a,X).every(Number.isFinite));
 const old=fit(X,Y,{initialization:'pso',population:8,generations:3,members:2});assert.equal(old.initialization,'pso');
});
test('fresh Agent policies execute BP and do not silently rename historical PSO candidates',()=>{
 const data=researchAdaptiveData(bundle),settings={metric:'rmse',season:0,horizon:7,driverStrategy:'provided'};
 const a=assessForecastTask(data,profileData(data,settings),true),core=initialTaskCandidates(a);
 assert.ok(core.some(c=>c.kind==='bp'));assert.ok(!a.policy.allowedKinds.includes('pso'));
 const r=evaluateCandidates(data,core,{...settings,taskPolicy:a.policy}).find(v=>v.candidate.kind==='bp');assert.ok(r.valid);assert.equal(r.name,'正则BP集成');
 assert.equal(normalizeCandidate({kind:'pso'}).kind,'pso');
});
test('fixed BP version produces all 105 finite points with decomposition and distinct record identities',()=>{
 const r=runResearchForecast(bundle);assert.equal(r.provenance.modelVersion,'regularized-bp-v1');assert.equal(r.config.initialization,'bp');assert.equal(r.config.population,0);
 assert.equal(r.combinations.flatMap(c=>c.points).length,105);assert.ok(r.combinations.every(c=>c.points.every(p=>Number.isFinite(p.demand))));
 for(const row of r.comparisons)assert.ok(Math.abs(row.development+row.baselineEfficiencyEffect-row.totalSaving-row.netChange)<1e-9);
 const fresh=psoRecord(r,bundle),old=structuredClone(r);delete old.provenance.modelVersion;old.provenance.method='MATLAB 正则PSO-BP集成';
 const historic=psoRecord(old,bundle);assert.equal(fresh.kind,'bp');assert.equal(historic.kind,'pso');assert.notEqual(fresh.id,historic.id);assert.match(historic.label,/PSO/);
});
