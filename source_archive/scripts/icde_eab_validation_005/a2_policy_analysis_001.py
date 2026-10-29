import common as C,frontier_math as F,json,numpy as np,pathlib,collections
C.start('A2/policy_analysis_001');done=C.load(C.ART/'A2/combined_complete_004.json');schedule=done['schedule'];valid=[];allcells=[];energy=[];setups=[];status=[]
for s in schedule:
 if s.get('status')!='PASS_WORKER_COMPLETE' or 'path' not in s:continue
 p=pathlib.Path(s['path']);g=C.load(p/'resource_guard.json');assert g['valid'];rows=C.load(p/'request_rows.json');cells=C.load(p/'cell_rows.json');setup=C.load(p/'setup.json');setups.append({'path':str(p),**setup});samples=sorted([x for x in g['samples'] if x['power_W'] is not None],key=lambda x:x['monotonic']);t=np.array([x['monotonic'] for x in samples]);w=np.array([x['power_W'] for x in samples]);valid+=rows;allcells+=cells
 for cell in cells:
  rr=[x for x in rows if x['cell_id']==cell['method']+'_'+str(cell['load_index'])];left=min(x['scheduled_arrival_ts'] for x in rr);right=max(x['complete_ts'] for x in rr);joule=None
  if len(t)>=2 and t[0]<=left and t[-1]>=right:
   tt=np.r_[left,t[(t>left)&(t<right)],right];ww=np.interp(tt,t,w);joule=np.trapezoid(ww,tt)
  energy.append(dict(domain=s['domain'],phase=s['phase'],block=s['block'],method=cell['method'],QPS=cell['QPS_offered'],requests=len(rr),board_energy_J=joule,board_energy_J_per_arrival=joule/len(rr) if joule is not None else None,scope='NVML board power integral including idle interarrival time, no host energy or query-active attribution; NOT GPU-active seconds',valid_exclusive_compute_PID=True))
for d in C.DOM:
 procs=sorted({int(x['block']) for x in valid if x['domain']==d and x['phase']=='routing'});status.append(dict(domain=d,complete_routing_processes=len(procs),status='PASS_MEASUREMENT_DEV_EXPLORATORY' if len(procs)==3 else 'INSUFFICIENT_NO_COMPLETE_3PROCESS_COMPARISON'))
C.table(C.R/'whole_board_energy_by_cell.csv',energy);C.write(C.A/'setups_all_valid.json',setups);C.table(C.R/'routing_comparison_status.csv',status);comp=[]
for d in C.DOM:
 rr=[x for x in valid if x['domain']==d and x['phase']=='routing'];procs=sorted({int(x['block']) for x in rr})
 if len(procs)!=3:continue
 for qps in [2.,5.]:
  z=[x for x in rr if float(x['QPS_offered'])==qps];methods=sorted({x['method'] for x in z});ids=sorted({x['query_id'] for x in z});groups=np.array([next(x['family'] for x in z if x['query_id']==q) for q in ids]);N=len(ids);T=np.empty((N,3,len(methods)));Y=np.empty_like(T);S=np.empty_like(T)
  for qi,q in enumerate(ids):
   for bi,b in enumerate(procs):
    for mi,m in enumerate(methods):
     zz=[x for x in z if x['query_id']==q and int(x['block'])==b and x['method']==m];assert len(zz)==2;T[qi,bi,mi]=np.mean([x['arrival_to_completion_s'] for x in zz]);Y[qi,bi,mi]=np.mean([x['r'] for x in zz]);S[qi,bi,mi]=np.mean([x['service_wall_s'] for x in zz])
  ix=F.boots(groups,2000,C.SEED+int(qps));rng=np.random.default_rng(C.SEED+int(qps));bs=rng.integers(0,3,(2000,3));bt=np.array([T[j][:,b].mean((0,1)) for j,b in zip(ix,bs)]);by=np.array([Y[j][:,b].mean((0,1)) for j,b in zip(ix,bs)]);v=methods.index('Full-Visual');static=[methods.index(m) for m in ['Full-Visual','BM25-R100','BM25']];meanT=T.mean((0,1));meanY=Y.mean((0,1))
  for mi,m in enumerate(methods):
   env,_=F.envelope(meanY[static],meanT[static],meanY[mi]);e,_=F.envelope(by[:,static],bt[:,static],by[:,mi]);inc=100*(e-bt[:,mi])/bt[:,v];lo,hi=np.quantile(inc,[.025,.975]);r0=[x for x in z if x['method']==m];comp.append(dict(domain=d,QPS=qps,method=m,mean_r=meanY[mi],mean_arrival_E2E_s=meanT[mi],mean_service_s=S[:,:,mi].mean(),P50_s=np.quantile([x['arrival_to_completion_s'] for x in r0],.5),P95_s=np.quantile([x['arrival_to_completion_s'] for x in r0],.95),ideal_descriptive_same_risk_static_random_E2E_s=env[0],increment_pp=100*(env[0]-meanT[mi])/meanT[v],paired_family_process95_low_pp=lo,paired_family_process95_high_pp=hi,actual_inference_scope='measured open-loop within provenlightregion; interpolated envelope is descriptive NOT deployed mixture latency',guarantee='EXPLORATORY_UNCERTIFIED',interval='2000 paired family+process draws; repeated requests no newqueries, pointwise'))
C.table(C.R/'new_stream_matched_risk_comparison.csv',comp,fields=list(comp[0]) if comp else ['domain','QPS','method','mean_r','mean_arrival_E2E_s','increment_pp','paired_family_process95_low_pp','paired_family_process95_high_pp','status']);C.write(C.A/'complete.json',{'time_bj':C.stamp(),'status':'ANALYSIS_COMPLETED_WITH_EXPLICIT_MISSING','valid_all_requests':len(valid),'valid_cells':len(allcells),'policy_contrasts':len(comp),'new_kernel_GPU_active_seconds':'MISSING_UNPROFILED_PRIMARY','whole_board_energy':'measured where bracketingNVMLsamples available, no queryactive claim'});print(json.dumps({'status':'POLICY_ANALYSIS_COMPLETE','policy_rows':len(comp),'valid_requests':len(valid),'time_bj':C.stamp()}),flush=True)
