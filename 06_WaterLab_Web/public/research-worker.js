import {runResearchForecast} from './research-engine.mjs';
self.onmessage=e=>{try{const result=runResearchForecast(e.data.bundle,p=>self.postMessage({progress:p}));self.postMessage({result});}catch(error){self.postMessage({error:error.message});}};
