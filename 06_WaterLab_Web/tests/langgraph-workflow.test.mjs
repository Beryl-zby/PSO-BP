import test from 'node:test';
import assert from 'node:assert/strict';
import {runWorkflow} from '../lib/langgraph-workflow.mjs';
import {importTable,profileData,ruleCandidates,evaluateCandidates,finalizeForecast} from '../public/adaptive-engine.mjs';

test('failure prevents held-out evaluation and later output',async()=>{
 const seen=[];
 await assert.rejects(runWorkflow([{name:'validation',run:async()=>{seen.push('validation');throw Error('failed');}},{name:'heldout',run:async()=>seen.push('heldout')}]),/failed/);
 assert.deepEqual(seen,['validation']);
});
test('cancellation stops graph before subsequent tool',async()=>{
 const c=new AbortController(),seen=[];
 await assert.rejects(runWorkflow([{name:'planning',run:async()=>{seen.push('planning');c.abort();}},{name:'execute',run:async()=>seen.push('execute')}],{signal:c.signal}));
 assert.deepEqual(seen,['planning']);
});
test('graph preserves numeric evaluation, selection and forecasts',async()=>{
 const data=importTable(['year','y'],Array.from({length:30},(_,i)=>[1990+i,50+i*.4+Math.sin(i)]),{time:0,target:1,features:[],unit:'test',name:'fixture'});
 const settings={horizon:3,metric:'rmse',season:0,driverStrategy:'hold'};
 const candidates=ruleCandidates(profileData(data,settings));
 const compute=()=>evaluateCandidates(data,candidates,settings);
 const expected=finalizeForecast(data,compute(),settings);
 let results,actual;
 const record=await runWorkflow([{name:'validation',run:async()=>{results=compute();}},{name:'freeze_and_finalize',run:async()=>{actual=finalizeForecast(data,results,settings);}}]);
 const numeric=v=>JSON.parse(JSON.stringify(v,(k,x)=>['seconds','createdAt'].includes(k)?undefined:x));
 assert.deepEqual(numeric(actual),numeric(expected));
 assert.deepEqual(record.trace,['validation','freeze_and_finalize']);
});
