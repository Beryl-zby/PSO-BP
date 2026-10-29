import {parseModelJSON} from './model-config.mjs';

// JSON syntax alone does not guarantee the fields required by a caller.
// Retry once with the rejected response and the actual validation errors.
export async function structuredResponse(messages, maxTokens, invoke, validate=v=>v, retry=true, signal){
 const started=performance.now();let calls=0,tokens=0,current=messages;
 for(let attempt=0;attempt<(retry?2:1);attempt++){
  signal?.throwIfAborted();
  const data=await invoke(current,Math.min(10000,maxTokens*(attempt?2:1)));
  calls++;tokens+=Number(data.usage?.total_tokens||0);
  const content=data.choices?.[0]?.message?.content;
  try{
   const value=validate(parseModelJSON(content));
   return {value,calls,tokens,seconds:(performance.now()-started)/1000};
  }catch(error){
   signal?.throwIfAborted();
   const detail=Array.isArray(error?.issues)?error.issues.map(i=>i.path.join('.')+': '+i.message).join('; '):String(error?.message||error);
   if(attempt||!retry)throw Error('模型返回格式仍不符合要求（已校验）：'+detail.slice(0,500));
   current=[...messages,{role:'assistant',content:String(content||'').slice(0,16000)},{role:'user',content:'上一次响应未通过格式校验：'+detail.slice(0,1200)+'。请按系统指定结构重新输出一个完整JSON对象，包含所有必填字段；不得编造数据或引用。无数值引用时claims仍须为[]。不要输出解释、Markdown或思考过程。'}];
  }
 }
 throw Error("Structured response attempts exhausted");
}
