import {finalizeResearchAdaptive} from './research-adaptive.mjs';
import {evaluateCandidates,finalizeForecast} from './adaptive-engine.mjs';
self.onmessage=e=>{const {id,action,data,candidates,settings}=e.data;try{const result=action==='research-finalize'?finalizeResearchAdaptive(e.data.bundle,candidates,settings):action==='evaluate'?evaluateCandidates(data,candidates,settings):finalizeForecast(data,candidates,settings);self.postMessage({id,result});}catch(error){self.postMessage({id,error:error.message||'预测任务失败'});}};
