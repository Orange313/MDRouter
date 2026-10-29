import common as C
import time,subprocess,json
out=C.ART/'X1/driver_002';out.mkdir(exist_ok=False);monitor=C.ART/'X1/resource_monitor_001'
while True:
    if (monitor/'idle_observed.json').exists():break
    if (monitor/'deadline_reached.json').exists():C.write(out/'result.json',{'status':'BLOCKED_GPU_BUSY','time_bj':C.stamp()});raise SystemExit(0)
    time.sleep(10)
gpu=json.loads((monitor/'idle_observed.json').read_text())['candidates'][0]
C.write(out/'selected_GPU.json',{'GPU':gpu,'time_bj':C.stamp(),'physical_GPU_sticky':True,'before_each_process':'three idle checks + ResourceGuard'})
for domain in [C.DOM[0],C.DOM[2]]:
    for process in range(3):
        log=C.ROOT/'artifacts/icde_eab_main_006_logs_001'/f'X1_{domain}_p{process}_002.log'
        with log.open('x') as f:r=subprocess.run([str(C.ROOT/'.venv/bin/python'),'-u',str(C.ROOT/'scripts'/C.RUN/'x1_gpu_pilot_002.py'),str(gpu),domain,str(process)],stdout=f,stderr=subprocess.STDOUT)
        with (out/'execution.jsonl').open('a') as f:f.write(json.dumps({'domain':domain,'process':process,'returncode':r.returncode,'time_bj':C.stamp(),'log':str(log)})+'\n')
        if r.returncode!=0:C.write(out/'result.json',{'status':'BLOCKED_OR_FAILED_PILOT_STOP_EXPANSION','time_bj':C.stamp(),'domain':domain,'process':process});raise SystemExit(0)
C.write(out/'result.json',{'status':'PASS_SMALL_PILOT_PROCESSES_COMPLETED','time_bj':C.stamp(),'expanded_beyond_pilot':False})
