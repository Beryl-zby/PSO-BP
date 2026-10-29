// Task screening uses initial-training features and user-provided future inputs,
// never held-out target values. These are explicit engineering eligibility rules.
export const TASK_POLICY_VERSION='data-task-screen-bp-v3';
const finite=v=>typeof v==='number'&&Number.isFinite(v);
export function assessForecastTask(data,profile,scenario=false){
 const train=data.rows.slice(0,profile.initialTraining),p=data.featureNames.length;
 const missing=train.reduce((n,r)=>n+r.x.filter(v=>!finite(v)).length,0);
 const unusable=data.featureNames.filter((_,j)=>!train.some(r=>finite(r.x[j])));
 const constants=data.featureNames.filter((_,j)=>{const a=train.map(r=>r.x[j]).filter(finite);return a.length&&Math.max(...a)===Math.min(...a);});
 const futureComplete=!!data.future?.length&&data.future.every(r=>r.x.length===p&&r.x.every(finite));
 const maxLag=data.rows.length<40?1:Math.min(12,Math.max(1,Math.floor(profile.initialTraining/4)));
 const ridgeOK=p>0||profile.allowedRidgeLags.length>0;
 const bpOK=profile.eligibleBP&&p===3&&!unusable.length;
 const families=[
  {kind:'ridge',name:'岭回归',role:ridgeOK?'primary':'excluded',reason:ridgeOK?(p?'可使用外部驱动；正则化控制参数规模，优先0—'+maxLag+'阶滞后。':'可使用目标滞后；通过时间验证选择正则强度。'):'缺少可用驱动或历史滞后。'},
  {kind:'bp',name:'轻量化正则BP集成',role:bpOK?'primary':'excluded',reason:bpOK?'当前实现适配3个非负驱动与正目标，可作为非线性候选；效果由验证结果决定。':'当前BP实现要求3个非负驱动、正目标且记录不超过400条。'},
  {kind:'naive',name:'最近实际值',role:scenario?'baseline':'primary',reason:scenario?'不读取外部驱动，仅作预测误差对照。':'简单持平基线，可参与普通总量预测选优。'},
  {kind:'drift',name:'线性漂移',role:scenario?'baseline':'primary',reason:scenario?'不读取外部驱动，不能承担SSP与节水路径分析，仅作对照。':'延续历史平均变化，可参与普通总量预测选优。'},
  {kind:'ses',name:'简单指数平滑',role:scenario?'baseline':'primary',reason:scenario?'不读取外部驱动，仅作时序对照。':'少参数时序候选。'},
  {kind:'seasonal',name:'季节持平',role:profile.eligibleSeasonal?(scenario?'baseline':'primary'):'excluded',reason:profile.eligibleSeasonal?'训练段至少覆盖两个已指定季节周期。':'当前没有适用的季节周期，或训练期不足两个完整周期。'},
 ];
 if(scenario&&!p)families[0]={...families[0],role:'excluded',reason:'情景主模型需要外部驱动，当前没有驱动列。'};
 const issues=[];
 if(scenario&&!p)issues.push('情景预测必须提供外部驱动。');
 if(scenario&&!futureComplete)issues.push('情景预测必须提供完整未来驱动，不能用持平或趋势代替给定路径。');
 if(unusable.length&&scenario)issues.push('初始训练期这些驱动全缺失：'+unusable.join('、'));
 const allowedKinds=families.filter(f=>f.role!=='excluded').map(f=>f.kind);
 const policy={version:TASK_POLICY_VERSION,requireDrivers:scenario,allowedKinds,maxRidgeLag:maxLag,initialTraining:profile.initialTraining};
 return {version:TASK_POLICY_VERSION,task:scenario?'五SSP×三节水路径情景预测':'普通时间序列预测',requireDrivers:scenario,rows:data.rows.length,initialTraining:profile.initialTraining,validationRows:profile.validationRows,heldoutRows:profile.heldoutRows,featureCount:p,frequency:scenario?'年度':profile.frequency,missingTrainingFeatures:missing,unusable,constants,futureComplete,smallSample:data.rows.length<40,families,issues,policy,budget:{rounds:2,perRound:2,additionalMax:4},scope:'样本规模与滞后上限是公开的工程筛选规则，不证明某个模型一定更好；任务要求先于验证误差排序。'};
}
export function candidateTaskRole(c,policy){return policy?.requireDrivers?(['ridge','bp'].includes(c.kind)&&c.useDrivers===true?'primary':'baseline'):'primary';}
export function taskCandidateError(c,policy){
 if(!policy)return null;
 if(!policy.allowedKinds.includes(c.kind))return '数据与任务筛选未纳入该模型类型。';
 if(c.kind==='ridge'&&c.lags>policy.maxRidgeLag)return `当前样本规模下滞后阶数最多${policy.maxRidgeLag}，该配置未通过筛选。`;
 return null;
}
function signature(c){return JSON.stringify([c.kind,c.kind==='ridge'?c.lags??3:0,['ridge','bp'].includes(c.kind)?c.alpha??1:1,c.window??0,c.kind==='ses'?c.smoothing??.3:.3,c.kind==='bp'||c.kind==='ridge'&&c.useDrivers===true,c.missing??'median']);}
export function initialTaskCandidates(assessment){
 const allowed=k=>assessment.policy.allowedKinds.includes(k),list=[{id:'baseline-naive',kind:'naive'},{id:'baseline-drift',kind:'drift'},{id:'baseline-ses',kind:'ses',smoothing:.3}];
 if(allowed('seasonal'))list.push({id:'baseline-seasonal',kind:'seasonal'});
 if(allowed('ridge'))list.push({id:'core-ridge',kind:'ridge',lags:assessment.featureCount?0:1,useDrivers:assessment.featureCount>0,alpha:1});
 if(allowed('bp'))list.push({id:'core-bp',kind:'bp',alpha:.03,useDrivers:true});
 return list.filter(c=>allowed(c.kind)).map(c=>({...c,origin:'固定基础候选'}));
}
export function appendTaskProposals(existing,proposals,assessment,round){
 const selected=[],notes=[],seen=new Set(existing.map(signature));
 for(const [i,c] of proposals.entries()){
  if(selected.length>=assessment.budget.perRound){notes.push('本轮预算已满，后续建议未执行。');break;}
  if(!c.invalidReason&&seen.has(signature(c))){notes.push('重复配置未重新训练：'+c.kind);continue;}
  seen.add(signature(c));selected.push({...c,id:c.id||`agent-${round}-${i}`,origin:`Agent第${round+1}轮建议`});
 }
 return {candidates:selected,notes};
}
export function ruleTaskCandidates(assessment){
 const core=initialTaskCandidates(assessment),extras=[];
 if(assessment.policy.allowedKinds.includes('ridge'))for(const [alpha,lags] of [[10,0],[1,1],[.1,0],[100,0]])extras.push({kind:'ridge',useDrivers:assessment.featureCount>0,lags:assessment.featureCount?lags:1,alpha});
 if(assessment.policy.allowedKinds.includes('bp'))extras.splice(2,2,{kind:'bp',alpha:.1,useDrivers:true},{kind:'bp',alpha:1,useDrivers:true});
 return [...core,...extras.slice(0,assessment.budget.additionalMax).map((c,i)=>({...c,id:'rule-task-'+i,origin:'固定规则配置'}))];
}
