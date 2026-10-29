import common as C
import collections,json
import numpy as np
quality=C.load(C.REPORT/'T2/process_1_001/quality.csv','NEW_FIXED_POLICY_INTERMEDIATE');raw=[]
for p in range(3):raw += C.load(C.REPORT/f'T2/process_{p}_001/requests.csv','NEW_UNPROFILED_TIMINGS')
out=[]
for m in dict.fromkeys(x['method'] for x in quality):
    q=[r for r in quality if r['method']==m];t=[r for r in raw if r['method']==m];out.append({'method':m,'nDCG10':np.mean([float(r['nDCG10']) for r in q]),'r':np.mean([float(r['r']) for r in q]),'harm_rate':np.mean([float(r['harm']) for r in q]),'E2E_mean_ms':1000*np.mean([float(r['E2E_s']) for r in t]),'timed_requests':len(t)})
print(json.dumps({'status':'INTERMEDIATE_PENDING_GPU_PROFILE_GUARD_AND_CI','time_bj':C.stamp(),'unique_queries':160,'values':C.clean(out)}))
