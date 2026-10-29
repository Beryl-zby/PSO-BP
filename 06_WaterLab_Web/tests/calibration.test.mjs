import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fit,predict,calibrate,predictCalibrated,runForecast,metrics} from '../public/engine.mjs';

test('calibration resists one large residual and is applied exactly once',()=>{
 const X=Array.from({length:7},(_,i)=>[i,i%3,2*i]),Y=X.map(r=>100+r[0]);
 const raw=fit(X,Y,{members:2}),xx=X.slice(-3),yy=predict(raw,xx).map((v,i)=>v+[1,1,90][i]);
 for(const [mode,offset] of [['none',0],['last',90],['median3',1]]){
  const f=calibrate(raw,xx,yy,mode);assert.equal(f.predictionCalibration.offset,offset);
  assert.deepEqual(f.theta,raw.theta);assert.equal(f.predictionCalibration.trainingRows,3);
  assert.deepEqual(predictCalibrated(f,xx),predict(raw,xx).map(v=>v+offset));
 }
 assert.equal(raw.predictionCalibration,undefined);
 assert.throws(()=>calibrate(raw,xx,yy,'unknown'),/calibration/);
});

test('joint lambda and calibration selection uses only the current training prefix',()=>{
 const {rows}=JSON.parse(readFileSync(new URL('../data/demo.json',import.meta.url),'utf8'));
 const input=structuredClone(rows);input.at(-2).y=194.3;
 const options={holdout:4,horizon:2,includeModel:true};
 const result=runForecast(input,options),changed=structuredClone(input);
 changed.slice(-3).forEach(r=>r.y+=70);
 const other=runForecast(changed,options);
 assert.deepEqual(result.backtest.map(r=>r.fixed),other.backtest.map(r=>r.fixed));
 assert.deepEqual(result.backtest.slice(0,2).map(r=>r.rolling),other.backtest.slice(0,2).map(r=>r.rolling));
 assert.deepEqual(result.selections.slice(0,2),other.selections.slice(0,2));
 for(const s of result.selections){
  assert.equal(s.scores.length,12);
  assert.equal(s.mape,Math.min(...s.scores.map(v=>v.mape)));
  assert.ok(s.scores.some(v=>v.lambda===s.lambda&&v.calibration===s.calibration&&v.mape===s.mape));
 }
 assert.deepEqual(metrics(result.backtest.map(r=>r.actual),result.backtest.map(r=>r.rolling)),result.metrics.rolling);
 assert.deepEqual(predictCalibrated(result.finalModel,result.forecasts.map(r=>r.x)),result.forecasts.map(r=>r.base));
 assert.equal(result.config.calibration.mode,result.selections.at(-1).calibration);
 assert.throws(()=>runForecast(input,{...options,calibrationModes:[]}),/calibration/);
});
