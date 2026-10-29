import assert from 'node:assert/strict';
import fs from 'node:fs';
import {trendRows,terminalRows,savingRows,waterfallRows,backtestRows,intensityRows} from '../public/research-figures.mjs';
const r=JSON.parse(fs.readFileSync(new URL('../public/research-latest-result.json',import.meta.url),'utf8'));
const b=JSON.parse(fs.readFileSync(new URL('../public/research-input.json',import.meta.url),'utf8'));
assert.equal(terminalRows(r).length,5);
for(const scenario of ['SSP1','SSP2','SSP3','SSP4','SSP5']){
 const rows=trendRows(r,scenario);assert.equal(rows.length,8);assert.equal(rows[0].year,2023);
 for(const p of ['frozen','trend','policy'])assert.equal(rows.at(-1)[p],r.combinations.find(c=>c.scenario===scenario&&c.path===p).points.at(-1).demand);
 const wf=waterfallRows(r,scenario);assert.equal(wf.length,6);assert.ok(Math.abs(wf[0].end+wf.slice(1,-1).reduce((s,v)=>s+v.value,0)-wf.at(-1).end)<1e-8);
 const ip=intensityRows(r,b.historical,scenario);assert.equal(ip.length,2);assert.equal(ip[0].rows.length,8);assert.equal(ip[1].rows.at(-1).policy,r.combinations.find(c=>c.scenario===scenario&&c.path==='policy').points.at(-1).industrialIntensity);
}
for(const cumulative of [false,true])for(const v of savingRows(r,cumulative)){assert.ok(Math.abs(v.historical+v.additional-v.total)<1e-8);const a=r.comparisons.filter(c=>c.scenario===v.name&&(cumulative||c.year===2030));assert.equal(v.total,a.reduce((s,c)=>s+c.totalSaving,0));}
assert.equal(trendRows(r,'SSP2','policy','ssps',true).length,21);
for(const v of backtestRows(r)){assert.equal(v.fixedError,v.fixed-v.actual);assert.equal(v.rollingError,v.rolling-v.actual);}
assert.equal(backtestRows(r).find(v=>v.year===2022).actual,194.3);
console.log('PASS: all five SSP values, 15 terminal totals, waterfall identity including baseline efficiency, cumulative savings, two intensity paths, full history and signed backtest errors.');
