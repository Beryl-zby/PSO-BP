'use client';
import {useEffect,useRef} from 'react';
export function ResultRecordCard({record,currentSource,onExport}:{record:any;currentSource:string;onExport:()=>void}){
 const same=record.source.sha256===currentSource;
 const details=useRef<HTMLDetailsElement|null>(null);
 useEffect(()=>{if(details.current)details.current.open=false;},[record.id]);
 return <section className="result-record-card" aria-label="当前预测结果信息">
  <div className="result-record-top">
   <div className="result-record-identity"><span className="result-record-caption">当前结果</span><h3>{record.label}</h3></div>
   <button className="secondary result-record-export" onClick={onExport}>导出结果记录 ↓</button>
  </div>
  <div className="result-record-meta"><span>生成时间 <time dateTime={record.createdAt}>{new Date(record.createdAt).toLocaleString('zh-CN')}</time></span><span className={'result-input-status '+(same?'is-current':'is-historical')}>{same?'输入与当前设置一致':'历史输入版本'}</span></div>
  <details ref={details} className="result-record-details"><summary>验证与版本信息</summary><dl><div><dt>验证方式</dt><dd>{record.protocol.name}</dd></div><div><dt>输入版本</dt><dd><code>{record.source.sha256}</code></dd></div></dl></details>
 </section>;
}
