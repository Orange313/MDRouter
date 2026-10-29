"""Resume ONLY never-run slots. Correct wait-vs-execution bookkeeping; no outcome retries."""
import common as C,pathlib,json,time,os
source=pathlib.Path(__file__).parent/'a2_driver_003.py';C.A=C.ART/'A2';C.R=C.REPORT/'A2'
C.write(C.A/'continuation_preexecution_lock_005.json',{'time_bj':C.stamp(),'status':'POST_HOC_ENGINEERING_SCHEDULER_FIX_NOT_SCIENTIFIC_DESIGN_CHANGE','reason':'Original idle-wait45min deadline also counted healthy execution wall. Continue fixed never-run slots; no new seeds/loads/policies/process counts, no successful or failed measured slot rerun.','protocol_SHA256':C.sha(C.A/'streaming_protocol_v3.json'),'code_SHA256':C.sha(__file__),'no_gate_results_used_for_change':True})
while True:
 try:original=json.loads((C.A/'driver_complete.json').read_text());break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(5)
env={'__file__':str(source)};prefix=source.read_text().split("for d in spec['domains']:")[0];prefix=prefix.replace("{s}_001'","{s}_005'").replace("s,'001'","s,'005'");exec(compile(prefix,str(source),'exec'),env)
env['deadline']=time.monotonic()+45*60
schedule=original['schedule'].copy();gates=original['AA_checks'].copy();env['schedule']=schedule;env['gates']=gates
for d in env['spec']['domains']:
 paths={};bad=False;existing=[x for x in schedule if x.get('domain')==d and x.get('GPU') is not None];fixed=existing[0]['GPU'] if existing else None
 for b in range(2):
  for slot in ['A1','B1','B2','A2']:
   p=C.A/'workers'/f'{d}_light_b{b}_{slot}_001'
   if p.exists():
    if not (p/'cell_rows.json').exists() or not (p/'resource_guard.json').exists() or not json.loads((p/'resource_guard.json').read_text())['valid']:bad=True;break
    paths[b,slot]=p;continue
   g=env['idle_gpu'](fixed)
   if g is None:schedule.append(dict(domain=d,phase='light',block=b,slot=slot,status='BLOCKED_CONTINUATION_NO_IDLE_GPU'));bad=True;break
   fixed=g;ok,p=env['run'](d,g,'light',b,slot);paths[b,slot]=p
   if not ok:bad=True;break
  if bad:break
 if bad:continue
 gp=C.A/f'{d}_AA_gate.json'
 good=json.loads(gp.read_text())['status']=='PASS' if gp.exists() else env['gate'](d,paths)
 if not good:continue
 for b in range(3):
  p=C.A/'policy_workers'/f'{d}_routing_b{b}_R_001'
  if p.exists():
   if not (p/'cell_rows.json').exists() or not (p/'resource_guard.json').exists() or not json.loads((p/'resource_guard.json').read_text())['valid']:break
   continue
  g=env['idle_gpu'](fixed)
  if g is None:schedule.append(dict(domain=d,phase='routing',block=b,status='BLOCKED_CONTINUATION_NO_IDLE_GPU'));break
  ok,p=env['run'](d,g,'routing',b,'R')
  if not ok:break
C.table(C.R/'execution_schedule_continuation_004.csv',schedule);C.write(C.A/'combined_complete_004.json',{'time_bj':C.stamp(),'status':'FIXED_SCHEDULE_CONTINUED_WITHOUT_REPEATS','schedule':schedule,'AA_checks':gates,'original_complete_SHA256':C.sha(C.A/'driver_complete.json'),'new_successful_or_failed_slot_reruns':0});print(json.dumps({'status':'CONTINUATION_FINISHED','time_bj':C.stamp(),'scheduled_events':len(schedule),'gate_checks':gates}),flush=True)
