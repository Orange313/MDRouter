import common as C
import subprocess,os,signal
lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();targets=[]
for line in lines:
    bits=line.strip().split(None,1)
    if len(bits)==2 and bits[1]=='.venv/bin/python scripts/icde_eab_independent_007/t1_audit_001.py':targets.append(int(bits[0]))
assert len(targets)==1,'Expected exactly one own named audit process'
C.write(C.ART/'T1/serial_audit_abort_001.json',{'time_bj':C.stamp(),'pid':targets[0],'reason':'Runtime-only switch from serial page decoding to four bounded CPU workers; identical data/model/thresholds, prior query candidates retained; no retrieval outcomes yet','old_artifacts':'retained','replacement':'T1/audit_002'})
os.kill(targets[0],signal.SIGTERM)
print({'stopped_own_pid':targets[0]})
