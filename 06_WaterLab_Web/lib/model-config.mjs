// Shared server configuration; this module contains no credentials.
export function modelConfig(input={}){
 const explicit=(input.QWEN_BASE_URL||'').trim().replace(/\/+$/,'');
 const baseURL=explicit||'https://dashscope.aliyuncs.com/compatible-mode/v1';
 let loopback=false;try{loopback=['localhost','127.0.0.1','[::1]'].includes(new URL(baseURL).hostname);}catch{throw Error('QWEN_BASE_URL 必须是完整的 HTTP(S) 地址');}
 if(!/^https?:\/\//.test(baseURL))throw Error('QWEN_BASE_URL 必须是 HTTP(S) 地址');
 const local=input.QWEN_PROVIDER==='local'||loopback;
 const model=(input.QWEN_MODEL||(local?'':'qwen3.5-flash')).trim();
 // A cloud key inherited by the shell must never be sent to a local model server.
 const apiKey=(input.QWEN_API_KEY||(!local?input.DASHSCOPE_API_KEY:'')||'').trim();
 const timeout=Number(input.QWEN_TIMEOUT_MS)|| (local?180000:60000);
 return {baseURL,model,local,apiKey,configured:Boolean(model&&(local?explicit:apiKey)),timeout:Math.max(1000,Math.min(timeout,600000)),jsonMode:input.QWEN_JSON_MODE!=='off',reasoningEffort:local?(input.QWEN_REASONING_EFFORT||'').trim():''};
}
export function requestOptions(config,messages,maxTokens,schema){return {method:'POST',headers:{'Content-Type':'application/json',...(config.apiKey?{Authorization:'Bearer '+config.apiKey}:{})},body:JSON.stringify({model:config.model,messages,temperature:.15,max_tokens:maxTokens,...(!config.local?{enable_thinking:false}:{}),...(config.reasoningEffort?{reasoning_effort:config.reasoningEffort}:{}),...(config.jsonMode?{response_format:config.local&&schema?{type:'json_schema',json_schema:{name:'waterlab_response',strict:true,schema}}:{type:'json_object'}}:{})})};}
export function parseModelJSON(content){let text=String(content||'').replace(/<think>[\s\S]*?<\/think>/g,'').trim();if(text.startsWith('```'))text=text.replace(/^```(?:json)?\s*/i,'').replace(/\s*```$/,'');return JSON.parse(text);}
