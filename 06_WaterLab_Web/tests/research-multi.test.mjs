import assert from 'node:assert/strict';
import fs from 'node:fs';
import {runResearchForecast} from '../public/research-engine.mjs';
import {researchMultiEvidence,researchMultiPlotData,validateMultiResearch,researchCalibration,researchHeldout} from '../lib/research-multi.mjs';
import {researchPlotCode} from '../lib/research-plot.mjs';
const bundle=JSON.parse(fs.readFileSync(new URL('../public/research-input.json',import.meta.url),'utf8'));
const result=runResearchForecast(bundle);
const original=JSON.stringify(result);
for(const scenario of ['SSP1','SSP2','SSP3','SSP4','SSP5']){
 for(const path of ['frozen','trend','policy']){
  const evidence=researchMultiEvidence(result,scenario,path);
  assert.equal(evidence.task,'ssp-water-demand');assert.equal(evidence.selection.scenario,scenario);assert.equal(evidence.selection.path,path);
  assert.equal(evidence.source,bundle.source.sha256);
  for(const c of result.combinations){
   assert.equal(evidence.facts['terminal.'+c.scenario+'.'+c.path],c.points.at(-1).demand);
   if(c.scenario===scenario)for(const r of c.points)assert.equal(evidence.facts['annual.'+c.path+'.'+r.year],r.demand);
  }
  for(const key of [...researchCalibration,...researchHeldout])assert.ok(Number.isFinite(evidence.facts[key]));
 }
}
assert.ok(researchCalibration.every(v=>!researchHeldout.includes(v)));
for(const kind of ['paths','ssps','terminal','savings','waterfall','cumulative']){
 const data=researchMultiPlotData(result,{plotKind:kind});
 assert.equal(data.combinations.length,15);assert.equal(data.combinations.flatMap(c=>c.points).length,105);
 assert.equal(data.history.at(-1).y,result.baseDemand);
 assert.ok(data.history.every(r=>!('x' in r)));
 for(const c of data.combinations){const actual=result.combinations.find(v=>v.scenario===c.scenario&&v.path===c.path);assert.deepEqual(c.points,actual.points.map(r=>({year:r.year,demand:r.demand})));}
 assert.equal(data.plotKind,kind);
 for(const s of data.scenarioNames){const rows=data.comparisons.filter(r=>r.scenario===s);assert.equal(rows.length,7);for(const r of rows)assert.ok(Math.abs(r.frozen-r.historicalSaving-r.additionalPolicySaving-r.policy)<1e-10);}
}
const invalid=structuredClone(result);invalid.combinations[1]=invalid.combinations[0];assert.throws(()=>validateMultiResearch(invalid),/重复/);
assert.throws(()=>researchMultiEvidence(result,'SSP6'),/未知/);
assert.equal(JSON.stringify(result),original,'Bridge must not mutate predictions');
if(process.env.RESEARCH_PLOT_OUTPUT){const directory=process.env.RESEARCH_PLOT_OUTPUT;fs.mkdirSync(directory,{recursive:true});fs.writeFileSync(directory+'/data.json',JSON.stringify(researchMultiPlotData(result)));fs.writeFileSync(directory+'/plot.py',researchPlotCode);}
console.log('Research Multi-agent: all 15 evidence selections, exact 105 plotting values, common historical anchor, accounting identities, disjoint workflow fields and invalid combination rejection passed.');
