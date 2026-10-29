import common as C
import subprocess,time,json
def idle(gpu):
 rows=[];consecutive=0
 # Same strict idle threshold; waiting replaces abort-on-first-busy before CUDA only.
 while consecutive<3:
  raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();z=raw.split(',');ok=int(z[2].split()[0])<128 and int(z[3].split()[0])==0;consecutive=consecutive+1 if ok else 0;rows.append({'time_bj':C.stamp(),'raw':raw,'idle':ok,'consecutive':consecutive});time.sleep(1)
 C.write(C.A/'three_idle_checks.json',rows)
