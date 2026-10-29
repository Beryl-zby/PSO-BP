import assert from 'node:assert/strict';
import fs from 'node:fs';
import {evaluateCandidates,finalizeForecast} from '../public/adaptive-engine.mjs';
import {researchAdaptiveData,finalizeResearchAdaptive} from '../public/research-adaptive.mjs';
import {psoRecord,agentRecord,comparisonStatus} from '../public/workspace-records.mjs';
import {waterfallRows,terminalRows} from '../public/research-figures.mjs';
import {researchMultiEvidence,researchMultiPlotData} from '../lib/research-multi.mjs';
const bundle=JSON.parse(fs.readFileSync(new URL('../public/research-input.json',import.meta.url)));
const original=JSON.stringify(bundle),data=researchAdaptiveData(bundle),settings={horizon:7,season:0,metric:'rmse',driverStrategy:'provided'};
const candidates=evaluateCandidates(data,[{id:'ridge-test',kind:'ridge',lags:1,useDrivers:true,alpha:10}],settings);
const run={result:finalizeForecast(data,candidates,settings),researchBatch:finalizeResearchAdaptive(bundle,candidates,settings),source:bundle.source};
const agent=agentRecord(run,bundle),pso=psoRecord(JSON.parse(fs.readFileSync(new URL('../public/research-latest-result.json',import.meta.url))),bundle);
assert.equal(agent.result.combinations.length,15);assert.equal(agent.result.combinations.flatMap(c=>c.points).length,105);
assert.equal(agent.result.backtest.length,2);assert.ok(agent.result.metrics.agent);assert.equal(agent.result.metrics.fixed,undefined);
for(const scenario of ['SSP1','SSP2','SSP3','SSP4','SSP5']){
 waterfallRows(agent.result,scenario);
 const e=researchMultiEvidence(agent.result,scenario);assert.equal(e.model,agent.result.provenance.method);assert.equal(e.facts['metrics.agent.rmse'],run.result.test.metrics.rmse);assert.equal(e.facts['metrics.rolling.mape'],undefined);
 researchMultiPlotData(agent.result,{scenario,plotKind:'waterfall'});
}
assert.deepEqual(terminalRows(agent.result).map(r=>r.policy),run.researchBatch.combinations.filter(c=>c.path==='policy').map(c=>c.rows.at(-1).value));
assert.equal(comparisonStatus(agent,pso).paths,true);assert.equal(comparisonStatus(agent,pso).errors,false);
const changed=structuredClone(pso);changed.source={sha256:'different'};assert.equal(comparisonStatus(agent,changed).paths,false);
const old=structuredClone(run);delete old.researchBatch.referenceRows;assert.equal(agentRecord(old,bundle).result.hasDevelopmentReference,false);
assert.equal(JSON.stringify(bundle),original);console.log('PASS: shared inputs unchanged; 105 exact Agent values; development identity; truthful Agent metrics; comparison guards; old-record migration.');
