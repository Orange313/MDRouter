import common as C
import json,numpy as np
C.A=C.ART/'S2';C.R=C.REPORT/'S2';C.STAGE='S2';__import__('sys').addaudithook(C.protected)
for d,attempt in [('vidore_v3_hr','008'),('Pharmaceuticals','009')]:
 complete=[];cells=[];partial=[];bad=[]
 for rep in range(3):
  p=C.A/f'stream_workers/{d}_routing_b{rep}_R_{attempt}'
  if not (p/'resource_guard.json').exists():bad.append({'process':rep,'status':'NOT_RUN_AFTER_RESOURCE_ABORT','path':str(p)});continue
  g=C.load(p/'resource_guard.json')
  if not g['valid']:
   z=C.load(p/'all_completed_invokes.json') if (p/'all_completed_invokes.json').exists() else [];partial.extend([{**r,'independent_process':rep,'whole_process_valid':False,'failure':'FOREIGN_GPU_PID','foreign_pids':g['foreign_pids']} for r in z if not r['warmup']]);bad.append({'process':rep,'status':'FAIL_RESOURCE_GUARD_POLLUTION','foreign_pids':g['foreign_pids'],'path':str(p),'partial_measured_requests_preserved':sum(not r['warmup'] for r in z)});continue
  assert (p/'request_rows.json').exists();complete.extend([{**r,'independent_process':rep,'whole_process_valid':True} for r in C.load(p/'request_rows.json')]);cells.extend([{**r,'independent_process':rep} for r in C.load(p/'cell_rows.json')])
 nproc=len({r['independent_process'] for r in complete});assert nproc<3,'Use full predeclared analyzer if all3processes completed';summary={'domain':d,'status':'INSUFFICIENT_RESOURCE_GUARD_ABORT','completed_processes':nproc,'valid_full_process_requests':len(complete),'invalid_partial_requests_preserved':len(partial),'all_three_process_comparisons':'NOT_RUN','performance_superiority_claim':False}
 C.table(C.R/(d+'_stream_all_requests.csv'),complete);C.table(C.R/(d+'_stream_invalid_partial_requests.csv'),partial);C.table(C.R/(d+'_stream_cells.csv'),cells);C.table(C.R/(d+'_stream_matched_risk.csv'),[summary]);C.write(C.A/(d+'_stream_analysis_complete.json'),{'time_bj':C.stamp(),'requests':len(complete),'completed_processes':nproc,'invalid':bad,'partial_invalid_requests':len(partial),'primary':[summary],'comparisons':[]});print(json.dumps(summary),flush=True)
