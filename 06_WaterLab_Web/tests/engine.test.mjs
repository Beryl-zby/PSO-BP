import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {jacobian,network,refine,fit,predict,runForecast,validateRows,metrics} from '../public/engine.mjs';
const t=[.2,-.4,.3,.1,.7,-.2],x=[.1,.8,-.4],J=jacobian(t,x);
for(let j=0;j<6;j++){const a=[...t],b=[...t];a[j]+=1e-6;b[j]-=1e-6;assert.ok(Math.abs(J[j]-(network(a,x)-network(b,x))/2e-6)<1e-8);}
const X=Array.from({length:9},(_,i)=>[i,i%3,2*i]),Y=X.map(r=>2+r[0]*.5);
const model=fit(X,Y,{members:3,population:8,generations:8});
assert.ok(model.optimization.every(o=>o.finalLoss<=o.initialLoss));
assert.ok(predict(model,X).every(Number.isFinite));
const constant=fit(Array.from({length:5},()=>[1,1,1]),[2,2,2,2,2],{members:2,population:4,generations:2});
assert.ok(predict(constant,[[1,1,1]]).every(Number.isFinite));
assert.throws(()=>validateRows([{year:2000,x:[1,2,3],y:4}]),/9/);
const demo=JSON.parse(readFileSync(new URL('../data/demo.json',import.meta.url),'utf8'));
assert.throws(()=>validateRows([...demo.rows.slice(0,9),demo.rows[0]]),/重复/);
const historical=runForecast(demo.rows,{horizon:7,holdout:4,calibrationModes:['last']});
assert.ok(Math.abs(historical.metrics.fixed.mape-.805875)<1e-4);
assert.ok(Math.abs(historical.metrics.rolling.mape-1.035417)<1e-4);
const result=runForecast(demo.rows,{horizon:7,holdout:4});
assert.equal(result.forecasts.length,7);assert.equal(result.backtest.length,4);
assert.ok(result.forecasts.every(r=>[r.base,r.saving,r.growth].every(Number.isFinite)));
assert.ok(result.selections.every(s=>s.scores.every(v=>Number.isFinite(v.mape))));
assert.deepEqual(metrics(result.backtest.map(r=>r.actual),result.backtest.map(r=>r.rolling)),result.metrics.rolling);
// Future test labels must not influence the first fixed-origin prediction or its selected lambda.
const changed=demo.rows.map((r,i)=>({...r,x:[...r.x],y:i<10?r.y:r.y+50}));
const changedResult=runForecast(changed,{horizon:2,holdout:4});
assert.equal(result.backtest[0].fixed,changedResult.backtest[0].fixed);
assert.equal(result.selections[0].lambda,changedResult.selections[0].lambda);
console.log(JSON.stringify({tests:'Jacobian, loss descent, constant inputs, validation, temporal isolation, metric recomputation',rolling:result.metrics.rolling,fixed:result.metrics.fixed,seconds:result.seconds}));
