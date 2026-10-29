import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {comparisonRows,resultViewEvidence,availableResultFigures,resolvePlotRequest,resolveUserPlotRequest} from '../public/result-view.mjs';
const result=JSON.parse(readFileSync(new URL('../public/research-latest-result.json',import.meta.url),'utf8'));
const bp={id:'bp-fixture',label:'BP集成',source:result.source,result,protocol:{type:'fixed',testYears:[2020,2021,2022,2023]}};
const ar=structuredClone(result);ar.metrics={agent:{rmse:2,mae:1}};ar.backtest=result.backtest.slice(-2).map(r=>({year:r.year,actual:r.actual,agent:r.rolling,baseline:r.actual}));
const agent={id:'agent-fixture',label:'Agent选择的候选模型',source:result.source,result:ar,protocol:{type:'heldout',testYears:[2022,2023]}};
test('comparison uses exact selected records, scenario and efficiency path',()=>{
 for(const s of ['SSP1','SSP2','SSP3','SSP4','SSP5'])for(const path of ['frozen','trend','policy']){
  const rows=comparisonRows(bp.result,agent.result,s,path);assert.equal(rows.length,8);
  for(const r of rows.slice(1)){const value=result.combinations.find(c=>c.scenario===s&&c.path===path).points.find(p=>p.year===r.year).demand;assert.equal(r.bp,value);assert.equal(r.agent,value);}
 }
 assert.equal(comparisonRows(result,ar,'SSP2','policy','terminal').length,5);
 assert.deepEqual(comparisonRows(result,ar,'SSP2','policy','backtest').map(r=>r.year),[2022,2023]);
});
test('AI sees explicit user-selected record identity; only allowed plot kinds are actionable',()=>{
 const evidence=resultViewEvidence({record:bp,bpRecord:bp,agentRecord:agent,scenario:'SSP3',path:'trend',kind:'paths',background:'fixture'});
 assert.equal(evidence.view.record,bp.id);assert.equal(evidence.view.scenario,'SSP3');assert.equal(evidence.view.path,'trend');
 const comparison=resultViewEvidence({record:null,bpRecord:bp,agentRecord:agent,scenario:'SSP2',path:'policy',kind:'paths',background:'fixture'});
 assert.deepEqual(comparison.view.records,[bp.id,agent.id]);assert.equal(comparison.facts['annual.agent_minus_bp.2030'],0);
 const allowed=availableResultFigures(result,ar).map(f=>f.id);assert.deepEqual(allowed,['paths','terminal','backtest']);
 assert.equal(resolvePlotRequest('waterfall',allowed),null);assert.equal(resolvePlotRequest('switch_to_agent',allowed),null);assert.equal(resolvePlotRequest(null,allowed),null);
 assert.deepEqual(resolvePlotRequest('cumulative',['savings']),{kind:'savings',cumulative:true});
 assert.deepEqual(resolvePlotRequest('waterfall',['waterfall']),{kind:'waterfall',cumulative:false});
});

test('explicit drawing request wins over model response and never selects a method',()=>{
 assert.deepEqual(resolveUserPlotRequest('请画2030年需水量对比图',null,['paths','terminal']),{kind:'terminal',cumulative:false});
 assert.deepEqual(resolveUserPlotRequest('请画发展增需与节水抵消瀑布图','terminal',['paths','waterfall','terminal']),{kind:'waterfall',cumulative:false});
 assert.equal(resolveUserPlotRequest('解释一下2030年的结果','terminal',['terminal']),null);
 assert.equal(resolveUserPlotRequest('画瀑布图','terminal',['paths','terminal']),null);
});
