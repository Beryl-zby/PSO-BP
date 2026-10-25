import {runForecast} from './engine.mjs';
self.onmessage=event=>{try{const result=runForecast(event.data.rows,event.data.options,p=>self.postMessage({type:'progress',...p}));self.postMessage({type:'result',result});}catch(e){self.postMessage({type:'error',message:e.message||'预测失败，请检查数据。'});}};
