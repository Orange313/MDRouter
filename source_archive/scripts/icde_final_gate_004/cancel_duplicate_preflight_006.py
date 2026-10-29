import common as C
import pathlib,os,signal,json
root=C.ART/'S1/workers/vidore_v3_finance_en_light_b1_A2_002';assert (root/'cell_rows.json').exists();assert json.loads((root/'resource_guard.json').read_text())['valid'];pid=1513707;p=pathlib.Path('/proc')/str(pid)
if p.exists():
 cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'stream_worker_004.py vidore_v3_finance_en 1 light 1 A2 005' in cmd;assert p.stat().st_uid==os.getuid();dest=C.ART/'S1/workers/vidore_v3_finance_en_light_b1_A2_005';assert not (dest/'requests_partial.jsonl').exists(),'Never discard completed measured requests';C.write(C.ART/'S1/duplicate_launch_preflight_006.json',{'time_bj':C.stamp(),'status':'CANCEL_OWN_PREMEASUREMENT_DUPLICATE','pid':pid,'cmd':cmd,'reason':'recovery scanner ran while original Finance A2 still active; original completed valid; no repeat allowed; cancel owned redundant process before measured requests','prior_002_retained':True});os.kill(pid,signal.SIGINT)
else:C.write(C.ART/'S1/duplicate_launch_preflight_006.json',{'time_bj':C.stamp(),'status':'PROCESS_ALREADY_EXITED_CHECK_RETAIN_LOGS','pid':pid})
