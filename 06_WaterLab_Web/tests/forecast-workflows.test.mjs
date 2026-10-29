import test from 'node:test';
import assert from 'node:assert/strict';
import {searchForecastFlows,normalizeFlow} from '../lib/forecast-workflows.mjs';
import {importTable,profileData,evaluateCandidates,selectCandidate,normalizeCandidate} from '../public/adaptive-engine.mjs';
import {assessForecastTask,initialTaskCandidates,appendTaskProposals} from '../public/task-policy.mjs';
const data=importTable(['year','y','x'],Array.from({length:30},(_,i)=>[1990+i,20+i*.3+Math.sin(i),10+i*.2]),{time:0,target:1,features:[2]});
const settings={metric:'rmse',season:0,horizon:3,driverStrategy:'hold'};
const profile=profileData(data,settings),assessment=assessForecastTask(data,profile,false);
settings.taskPolicy=assessment.policy;
const candidate=n=>normalizeCandidate({id:'a'+n,kind:'ridge',useDrivers:true,lags:0,alpha:n?10:1});
function options(extra={}){return {profile,assessment,settings,plan:async(p,n,feedback,flow)=>{assert.equal(p,profile);assert.ok(feedback.every(r=>!('test' in r)));return {reason:'fixture',candidates:[candidate(n)],calls:1,tokens:10};},review:async(p,cs)=>({reason:'review',candidates:cs,calls:1,tokens:10}),authorize:async(action)=>{assert.equal(action,'evaluate');return {action,calls:1,tokens:10};},evaluate:async cs=>evaluateCandidates(data,cs,settings),propose:async(input)=>{assert.ok(input.history.every(r=>!('test' in r)&&!('results' in r)));return {flow:{audit:true,feedback:false,style:'direct'},calls:1,tokens:10};},...extra};}
test('fixed graph preserves legacy two-round validation candidates',async()=>{
 const out=await searchForecastFlows(options());let cs=initialTaskCandidates(assessment);
 cs=[...cs,...appendTaskProposals(cs,[candidate(0)],assessment,0).candidates];cs=[...cs,...appendTaskProposals(cs,[candidate(1)],assessment,1).candidates];
 const expected=selectCandidate(evaluateCandidates(data,cs,settings),settings);
 assert.deepEqual(out.best.selected.candidate,expected.candidate);assert.equal(out.best.validationScore,expected.score);
 assert.deepEqual(out.best.trace.trace,['model_planning','validation_execution','feedback_planning','feedback_execution','freeze_candidate']);
});
test('AFlow explores distinct executable structures, caches duplicate training, never finalizes during search',async()=>{
 let trained=0;const out=await searchForecastFlows(options({mode:'aflow',evaluate:async cs=>{trained+=cs.length;return evaluateCandidates(data,cs,settings);}}));
 assert.equal(out.report.tree.length,4);assert.equal(new Set(out.report.tree.map(t=>normalizeFlow(t.flow).id)).size,4);
 assert.equal(trained,out.report.trainingEvaluations);assert.ok(trained<out.report.tree.length*out.results.length);
 assert.ok(out.report.tree.every(t=>!('test' in t)));assert.equal(out.best.validationScore,Math.min(...out.report.tree.filter(t=>t.valid).map(t=>t.validationScore)));
 assert.ok(out.report.tree.some(t=>t.trace.trace.includes('configuration_review')));
 assert.ok(out.report.tree.some(t=>!t.trace.trace.includes('feedback_planning')));
});
test('invalid workflow schemas and cancellation fail closed',async()=>{
 assert.throws(()=>normalizeFlow({audit:'yes',feedback:true,style:'direct'}));
 const c=new AbortController();c.abort();await assert.rejects(searchForecastFlows(options({signal:c.signal})));
});
