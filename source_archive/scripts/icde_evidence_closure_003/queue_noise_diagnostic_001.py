"""Describe failed streaming controls from existing requests; no reruns or retuning."""
import common as C
import numpy as np,json,sys,os
C.start('E4/queue_noise_diagnostic_001');C.STAGE='E4_QUEUE_CONTROL_DIAGNOSTIC';C.load(C.ART/'E4/manifest.json','IMMUTABLE_PRIMARY_E4_MANIFEST');rows=[];series=[]
for d in ['Pharmaceuticals']+C.DOM:
 for mode in ['main','aa']:
  raw=C.load(C.ART/'E4/workers'/f'{d}_{mode}_001/request_rows.json','ALREADY_MEASURED_REQUESTS');z=[r for r in raw if r['method']=='Full-Visual' and r['window']==1 and r['concurrency']==1 and r['load_index']==1];z.sort(key=lambda r:r['position']);assert len(z)==64
  err=max(abs(r['arrival_to_completion_s']-r['window_and_planning_wait_s']-r['executor_queue_s']-r['service_wall_s']) for r in z);assert err<1e-6,'E2E path accounting does not conserve wall time'
  service=np.array([r['service_wall_s'] for r in z]);rows.append({'domain':d,'process':mode,'offered_QPS':z[0]['QPS_offered'],'mean_arrival_latency_s':np.mean([r['arrival_to_completion_s'] for r in z]),'mean_service_s':service.mean(),'mean_executor_queue_s':np.mean([r['executor_queue_s'] for r in z]),'mean_window_planning_wait_s':np.mean([r['window_and_planning_wait_s'] for r in z]),'first_request_service_s':service[0],'remaining_requests_service_mean_s':service[1:].mean(),'observed_service_mean_times_offered_QPS':service.mean()*z[0]['QPS_offered'],'max_path_accounting_error_s':err,'status':'EXPLORATORY_DIAGNOSTIC_NO_REQUEST_EXCLUDED'})
  series.extend([{'domain':d,'process':mode,**{k:r[k] for k in ['position','query_id','family','arrival_to_completion_s','service_wall_s','executor_queue_s','window_and_planning_wait_s']}} for r in z])
C.table(C.R/'AA_service_queue_decomposition.csv',rows);C.table(C.R/'AA_by_request.csv',series)
sys.path.insert(0,str(C.ROOT/'cache/report_dependencies/prereg_v2_round_001'));os.environ['MPLCONFIGDIR']=str(C.A/'mpl_cache')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(4,2,figsize=(12,12))
for i,d in enumerate(['Pharmaceuticals']+C.DOM):
 for mode in ['main','aa']:
  z=[r for r in series if r['domain']==d and r['process']==mode];ax[i,0].plot([r['position'] for r in z],[r['arrival_to_completion_s']*1000 for r in z],label=mode);ax[i,1].plot([r['position'] for r in z],[r['service_wall_s']*1000 for r in z],label=mode)
 ax[i,0].set(title=d,ylabel='Arrival latency ms');ax[i,1].set(title='Measured service (incl first request)',ylabel='Service ms');ax[i,0].legend()
fig.tight_layout();fig.savefig(C.R/'AA_queue_service.png',dpi=180);plt.close(fig)
facts='4域既有A/A各64请求进行service/queue/window分解，全部E2E时钟路径恒等校验PASS，未丢弃首请求、未重跑。'+json.dumps(C.clean(rows),ensure_ascii=False)
decision='所有流式A/A仍未通过，不能宣称稳定部署优势。接近饱和负载的队列放大、线程池冷初始化及共享CPU是候选解释，现有单对过程不能确定唯一原因；没有以排除首请求或改QPS修复主结果。'
C.prose(C.R/'queue_reliability_diagnostic.md','# FAILED_RELIABILITY_GATE / EXPLORATORY_DIAGNOSTIC\n\n'+facts+'\n\n'+decision+'\n');C.finish('PASS_ACCOUNTING_CHECK_FAIL_STREAM_RELIABILITY',facts,decision)
