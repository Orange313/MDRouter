import common as C
import subprocess,time,json
out=C.ART/'X1/resource_monitor_001';out.mkdir(exist_ok=False)
deadline=time.monotonic()+2700
while time.monotonic()<deadline:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    rows=[r.split(',') for r in raw.strip().splitlines()];free=[int(r[0]) for r in rows if int(r[1].split()[0])<128 and int(r[2].split()[0])==0]
    with (out/'polls.jsonl').open('a') as f:f.write(json.dumps({'time_bj':C.stamp(),'raw':raw,'idle_candidates':free})+'\n')
    if free:
        C.write(out/'idle_observed.json',{'time_bj':C.stamp(),'candidates':free,'not_authorization_to_skip_three_checks':True});print(json.dumps({'idle':free,'time_bj':C.stamp()}),flush=True);break
    time.sleep(20)
else:C.write(out/'deadline_reached.json',{'status':'BLOCKED_GPU_BUSY','time_bj':C.stamp()})
