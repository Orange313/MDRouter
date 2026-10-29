import common as C,numpy as np,json,pathlib
C.start('A2/trace_diagnostics_001');raw=[];summary=[]
for d in C.DOM:
 for mode in ['main','aa']:
  rr=C.load(C.PREV/f'E4/workers/{d}_{mode}_001/request_rows.json');rs=[r for r in rr if r['method']=='Full-Visual' and r['window']==1 and r['concurrency']==1 and r['load_index']==1 and r['cohort']=='UNPROFILED_PRIMARY'];rs=sorted(rs,key=lambda r:r['position']);assert len(rs)==64
  first=rs[0]['completed_time']-rs[0]['arrival_to_completion_s'];last=max(r['completed_time'] for r in rs)
  for r in rs:
   err=abs(r['arrival_to_completion_s']-r['executor_queue_s']-r['service_wall_s']-r['window_and_planning_wait_s']);assert err<1e-9
   raw.append(dict(domain=d,mode=mode,position=r['position'],query_id=r['query_id'],arrival_offset_s=r['completed_time']-r['arrival_to_completion_s']-first,queue_s=r['executor_queue_s'],service_s=r['service_wall_s'],latency_s=r['arrival_to_completion_s'],planning_wait_s=r['window_and_planning_wait_s'],identity_error_s=err,offered_QPS=r['QPS_offered'],active_worker=None,inflight=None,GPU_clock=None,CPU_preemption=None,missing_reason='not recorded in old trace; not imputed'))
  mean=lambda k:np.mean([r[k] for r in rs]);summary.append(dict(domain=d,source='closure003_raw',mode=mode,offered_QPS=rs[0]['QPS_offered'],completed_QPS=len(rs)/(last-first),service_mean_s=mean('service_wall_s'),queue_mean_s=mean('executor_queue_s'),latency_mean_s=mean('arrival_to_completion_s'),P50_s=np.quantile([r['arrival_to_completion_s'] for r in rs],.5),P95_s=np.quantile([r['arrival_to_completion_s'] for r in rs],.95),estimated_utilization=rs[0]['QPS_offered']*mean('service_wall_s'),requests=len(rs),status='INVALID_FOR_STREAMING_SUPERIORITY'))
oldaa=C.load('reports/icde_final_gate_004/S1/aa_per_request.csv');prior=[]
for d in C.DOM:
 for qps in [2.,5.]:
  z=[r for r in oldaa if r['domain']==d and float(r['QPS_offered'])==qps]
  if not z:continue
  v=lambda k:np.array([float(r[k]) for r in z]);prior.append(dict(domain=d,QPS=qps,requests=len(z),service_mean_s=v('service_wall_s').mean(),queue_mean_s=v('executor_queue_s').mean(),dispatch_mean_s=v('arrival_dispatch_lateness_s').mean(),latency_mean_s=v('arrival_to_completion_s').mean(),P50_s=np.quantile(v('arrival_to_completion_s'),.5),P95_s=np.quantile(v('arrival_to_completion_s'),.95),max_worker=max(float(r['active_worker']) for r in z),max_inflight=max(float(r['inflight_at_start']) for r in z),threadpool='persistent actual worker warm20/process+4/cell',scope='PRIOR004_NOT_NEW005_MEASUREMENT'))
C.table(C.R/'old_queue_service_breakdown.csv',summary);C.table(C.R/'old_raw_AA_requests.csv',raw);C.table(C.R/'prior004_light_load_summary.csv',prior)
C.prose(C.R/'failure_regime.md','# FAIL旧近饱和A/A；新轻负载尚待独立验证\n\n本轮从旧003原始逐请求JSON重算512条FullV配对请求，latency=queue+service+planning恒等误差<1e-9。估算offered utilization临界或超载，HR服务36.05→45.03ms经排队放大成40.55→340.41ms总延迟。队列解释算术放大，不确定造成服务差异的唯一原因。旧active/inflight/clock/CPU抢占缺失不可补0。\n\n另重新汇总004轻负载5248条，保留HR/Pharma2/5QPS通过以及其他域不足的原状态。本轮新A/A与旧通过结果分开，尚无新请求时只能NOT_RUN/BLOCKED。新协议固定同域物理GPU、CPU affinity、常驻模型和索引、同一query序列/到达、warm20+4、固定ABBA和所有慢请求。当前空闲守卫与全过程PID/功耗/时钟/队列遥测不得省略。相同到达QPS下吞吐受arrival限制，不是最大容量；Goodput使用预登记200ms阈值，不含正式质量保证。\n')
plt=C.plots();fig,ax=plt.subplots(1,4,figsize=(14,3.8))
for i,d in enumerate(C.DOM):
 for mode in ['main','aa']:
  z=[r for r in raw if r['domain']==d and r['mode']==mode];ax[i].plot([r['position'] for r in z],[r['queue_s']*1000 for r in z],label=mode)
 ax[i].set(title=C.NAMES[i],xlabel='Old request position',ylabel='Queue wait(ms)');ax[i].legend()
fig.tight_layout();fig.savefig(C.R/'old_queue_amplification.png',dpi=170);fig.savefig(C.R/'old_queue_amplification.pdf');plt.close(fig)
C.write(C.A/'complete.json',{'status':'PASS_TRACE_REANALYSIS_ONLY','raw_old_AA_requests':len(raw),'prior004_light_requests':len(oldaa),'time_bj':C.stamp(),'new_GPU_measurements_in_this_script':0});print(json.dumps({'status':'TRACE_DIAGNOSTICS_COMPLETE','old512':len(raw),'prior_light':len(oldaa),'time_bj':C.stamp()}),flush=True)
