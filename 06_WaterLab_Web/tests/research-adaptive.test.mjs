import assert from 'node:assert/strict';
import fs from 'node:fs';
import {researchAdaptiveData,researchPlanningContext,finalizeResearchAdaptive} from '../public/research-adaptive.mjs';
import {prepareResearch} from '../public/research-engine.mjs';
import {profileData,evaluateCandidates,ruleCandidates} from '../public/adaptive-engine.mjs';

const bundle=JSON.parse(fs.readFileSync(new URL('../public/research-input.json',import.meta.url),'utf8'));
const settings={season:0,horizon:7,metric:'rmse',driverStrategy:'provided'};
const data=researchAdaptiveData(bundle),profile=profileData(data,settings);
assert.equal(data.rows.length,14);assert.equal(data.future.length,7);
assert.equal(profile.initialTraining,10);assert.equal(profile.validationRows,2);assert.equal(profile.heldoutRows,2);
const changed=structuredClone(data);changed.rows.slice(-2).forEach(r=>r.y+=1000);
assert.deepEqual(profileData(changed,settings),profile,'Held-out targets cannot affect the planner profile');
const context=researchPlanningContext('SSP2','policy');
assert.equal(context.selectedScenario,'SSP2');assert.equal(context.selectedPath,'policy');
assert.ok(!Object.hasOwn(context,'heldoutTargets'));
for(const path of prepareResearch(bundle).paths){
 const d=researchAdaptiveData(bundle,path.scenario,path.path);
 assert.deepEqual(d.rows,data.rows);
 assert.deepEqual(d.future.map(r=>r.x),path.points.map(r=>r.x));
}
const candidates=evaluateCandidates(data,ruleCandidates(profile),settings);
const result=finalizeResearchAdaptive(bundle,candidates,settings);
assert.equal(result.combinations.length,15);
assert.equal(result.combinations.flatMap(c=>c.rows).length,105);
const naive=evaluateCandidates(data,[{kind:'naive'}],settings);
const noDrivers=finalizeResearchAdaptive(bundle,naive,settings);
assert.equal(noDrivers.usesFutureDrivers,false);
assert.ok(noDrivers.combinations.every(c=>JSON.stringify(c.rows)===JSON.stringify(noDrivers.combinations[0].rows)));
console.log('Research-to-Agent mapping, 10/2/2 split, held-out profile isolation, 15 frozen-model cases and explicit no-driver behavior passed.');
