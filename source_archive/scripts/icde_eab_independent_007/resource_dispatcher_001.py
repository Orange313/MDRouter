"""Bounded resource monitoring and dependency-only dispatch; no scientific selection."""
import common as C
import subprocess,sys,os,time,json,csv,io,pwd
from pathlib import Path
op=C.ROOT/'artifacts/icde_eab_independent_007_operations_001';op.mkdir(exist_ok=True)
logs=C.ROOT/'artifacts/icde_eab_independent_007_logs_001';logs.mkdir(exist_ok=True)
start=time.monotonic();jobs={};done=set();failed=set();last=None
def event(data):
    data={'time_bj':C.stamp(),**data}
    with (op/'dispatch_events.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
    print(json.dumps(data),flush=True)
def snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[tuple(int(v.strip()) for v in r) for r in csv.reader(io.StringIO(raw))]
    owner=[]
    procs=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    for r in csv.reader(io.StringIO(procs)):
        if len(r)!=2:continue
        pid=int(r[0]);account=pwd.getpwuid(os.stat(f'/proc/{pid}').st_uid).pw_name if Path(f'/proc/{pid}').exists() else 'exited';owner.append({'pid':pid,'account':account,'memory_MiB':int(r[1])})
    s={'GPU':rows,'GPU3_owner':owner}
    with (op/'resource_samples.jsonl').open('a') as f:f.write(json.dumps({'time_bj':C.stamp(),**s})+'\n')
    return s
def ready_jobs():
    if not (op/'orange_T3_pilot_receipt.json').exists():return []
    todo=[];idx=C.ROOT/'cache/indexes/icde_eab_independent_007_biomedical_colqwen_001/index_manifest.json'
    for p in range(3):
        k=f'Bio_p{p}'
        if k not in done|failed and k not in jobs and (p==0 or idx.exists()):todo.append((k,'t2_biomedical_exact_004.py',[str(p)]))
    if not (op/'orange_T2_receipt.json').exists():return todo
    for d in ['vidore_v3_finance_en','Pharmaceuticals','vidore_v3_hr','vidore_v3_industrial']:
        for p in range(3):
            k=f'ColPali_{d}_p{p}';idx=C.ROOT/f'cache/indexes/{C.RUN}_colpali_{d}_002/index_manifest.json'
            if k not in done|failed and k not in jobs and (p==0 or idx.exists()):todo.append((k,'t3_colpali_expansion_004.py',[d,str(p)]))
    return todo
event({'event':'MONITOR_STARTED','interval_s':30,'max_duration_h':6,'GPU_priority':[3,5,0,1,2,4],'max_parallel_jobs':2,'reservation':'user coordinated GPUs3/5; no dummy allocation, no process termination'})
try:
    while time.monotonic()-start<6*3600:
        for k,j in list(jobs.items()):
            code=j['p'].poll()
            if code is not None:
                j['log'].close();(done if code==0 else failed).add(k);event({'event':'JOB_COMPLETED' if code==0 else 'JOB_FAILED','job':k,'gpu':j['gpu'],'returncode':code,'log':str(j['path'])});del jobs[k]
        if all(f'Bio_p{p}' in done for p in range(3)) and not (op/'biomedical_all_completed.json').exists():C.write(op/'biomedical_all_completed.json',{'time_bj':C.stamp(),'status':'PASS_ALL_THREE_PROCESSES','jobs':sorted(done)})
        s=snapshot();state=[(x['pid'],x['account']) for x in s['GPU3_owner']]
        if state!=last:event({'event':'GPU3_OWNERSHIP_CHANGED','owners':s['GPU3_owner'],'GPU':s['GPU']});last=state
        idle={r[0] for r in s['GPU'] if r[1]<128 and r[2]==0};idle -= {x['gpu'] for x in jobs.values()}
        queue=ready_jobs()
        for gpu in [3,5,0,1,2,4]:
            if gpu not in idle or len(jobs)>=2 or not queue:continue
            k,script,args=queue.pop(0);path=logs/(k+'_dispatcher_001.log');f=path.open('x');p=subprocess.Popen([sys.executable,str(Path(__file__).with_name(script)),str(gpu),*args],stdout=f,stderr=subprocess.STDOUT);jobs[k]={'p':p,'gpu':gpu,'log':f,'path':path};event({'event':'JOB_STARTED','job':k,'gpu':gpu,'pid':p.pid,'script':script,'note':'worker still must pass its three idle checks and ResourceGuard'})
        if (op/'orange_T2_receipt.json').exists() and len(done|failed)>=15 and not jobs:break
        if (op/'stop_monitor_requested.json').exists() and not jobs:break
        time.sleep(30)
finally:
    C.write(op/'monitor_end.json',{'time_bj':C.stamp(),'done':sorted(done),'failed':sorted(failed),'active_jobs':[{'job':k,'pid':j['p'].pid,'gpu':j['gpu']} for k,j in jobs.items()],'no_foreign_process_killed':True})
