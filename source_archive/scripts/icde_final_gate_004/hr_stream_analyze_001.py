import common as C
import numpy as np,json,collections,itertools
import frontier_math as F
C.A=C.ART/'S2';C.R=C.REPORT/'S2';C.STAGE='S2';__import__('sys').addaudithook(C.protected);d='vidore_v3_hr';methods=C.PLANS+['Q_regret','QP_regret','RandomMix_Q_regret','RandomMix_QP_regret'];rows=[];cells=[];invalid=[]
for rep in range(3):
 p=C.A/f'stream_workers/{d}_routing_b{rep}_R_008'
 if not (p/'resource_guard.json').exists() or not (p/'cell_rows.json').exists():
  invalid.append({'rep':rep,'path':str(p),'status':'INCOMPLETE_OR_NOT_RUN'});continue
 g=C.load(p/'resource_guard.json')
 if not g['valid']:invalid.append({'rep':rep,'path':str(p),'status':'FAIL_POLLUTION'});continue
 z=C.load(p/'request_rows.json');rows.extend([{**r,'independent_process':rep} for r in z]);cells.extend([{**r,'independent_process':rep} for r in C.load(p/'cell_rows.json')])
C.table(C.R/'HR_stream_all_requests.csv',rows);C.table(C.R/'HR_stream_cells.csv',cells);out=[];compar=[]
if len({r['independent_process'] for r in rows})==3:
 for qps in [2.,5.]:
  r0=[r for r in rows if r['QPS_offered']==qps];qids=sorted({r['query_id'] for r in r0});groups=np.array([next(r['family'] for r in r0 if r['query_id']==q) for q in qids]);N=len(qids);T=np.zeros((N,3,8));R=np.zeros_like(T);S=np.zeros_like(T);CPU=np.zeros_like(T)
  for qi,q in enumerate(qids):
   for rep in range(3):
    for mi,m in enumerate(methods):
     z=[r for r in r0 if r['query_id']==q and r['independent_process']==rep and r['method']==m];assert len(z)==2;T[qi,rep,mi]=np.mean([r['arrival_to_completion_s'] for r in z]);R[qi,rep,mi]=np.mean([r['r'] for r in z]);S[qi,rep,mi]=np.mean([r['service_wall_s'] for r in z]);CPU[qi,rep,mi]=np.mean([r['service_thread_CPU_s'] for r in z])
  ixs=F.boots(groups,2000,C.SEED);rng=np.random.default_rng(C.SEED);ps=rng.integers(0,3,(2000,3));BT=np.array([T[ix][:,p].mean((0,1)) for ix,p in zip(ixs,ps)]);BR=np.array([R[ix][:,p].mean((0,1)) for ix,p in zip(ixs,ps)]);ref=BT[:,0];tm=T.mean((0,1));rm=R.mean((0,1));st=S.mean((0,1));cpu=CPU.mean((0,1))
  for mi,m in enumerate(methods):
   ev,w=F.envelope(rm[:4],tm[:4],rm[mi]);be,_=F.envelope(BR[:,:4],BT[:,:4],BR[:,mi]);inc=100*(be-BT[:,mi])/ref;lo,hi=np.quantile(inc,[.025,.975]);lat=np.array([r['arrival_to_completion_s'] for r in r0 if r['method']==m]);cc=[r for r in cells if r['QPS_offered']==qps and r['method']==m];out.append({'domain':d,'QPS':qps,'method':m,'Dev_n':N,'families':len(set(groups)),'independent_processes':3,'actual_requests':len(lat),'mean_r':rm[mi],'mean_arrival_E2E_s':tm[mi],'P50_s':np.quantile(lat,.5),'P95_s':np.quantile(lat,.95),'P99_s':np.quantile(lat,.99),'mean_service_s':st[mi],'mean_thread_CPU_s':cpu[mi],'mean_completed_QPS':np.mean([r['completed_QPS'] for r in cc]),'mean_goodput_latency_le200ms_QPS':np.mean([r['goodput_QPS_le200ms'] for r in cc]),'matched_risk_static_random_E2E_s':ev[0],'matched_risk_increment_pp':100*(ev[0]-tm[mi])/tm[0],'paired95_increment_low_pp':lo,'paired95_increment_high_pp':hi,'status':'AA_GATED_DEV_EXPLORATORY_UNCERTIFIED','cost_scope':'actual arrival-to-completion, W1/C1, fixed low offered rate; not maximum throughput; quality not certified; no business net claim'})
  for method,mix in [('Q_regret','RandomMix_Q_regret'),('QP_regret','RandomMix_QP_regret')]:
   j,k=methods.index(method),methods.index(mix);dr=BR[:,j]-BR[:,k];dt=100*(BT[:,k]-BT[:,j])/ref;ci=np.quantile(dt,[.025,.975]);ri=np.quantile(dr,[.025,.975]);compar.append({'domain':d,'QPS':qps,'method':method,'frozen_random_comparator':mix,'risk_diff':rm[j]-rm[k],'paired95_risk_diff_low':ri[0],'paired95_risk_diff_high':ri[1],'latency_saving_pp_of_FullV':100*(tm[k]-tm[j])/tm[0],'paired95_latency_diff_low_pp':ci[0],'paired95_latency_diff_high_pp':ci[1],'risk_match_margin':.005,'point_risk_matched':abs(rm[j]-rm[k])<=.005,'status':'DESCRIPTIVE_FIXED_GRID_NO_NEW_THRESHOLD'})
else:out=[{'domain':d,'status':'INSUFFICIENT_INCOMPLETE_3_PROCESS_STREAM','completed_processes':len({r['independent_process'] for r in rows}),'reason':invalid,'performance_superiority_claim':False}]
C.table(C.R/'HR_stream_matched_risk.csv',out);C.table(C.R/'HR_stream_vs_frozen_random.csv',compar);C.write(C.A/'hr_stream_analysis_complete.json',{'time_bj':C.stamp(),'requests':len(rows),'completed_processes':len({r['independent_process'] for r in rows}),'invalid':invalid,'primary':out,'comparisons':compar});print(json.dumps({'status':'HR_STREAM_ANALYZED','requests':len(rows),'models':[r for r in out if r.get('method') in ['Q_regret','QP_regret']],'invalid':invalid,'time_bj':C.stamp()}),flush=True)
