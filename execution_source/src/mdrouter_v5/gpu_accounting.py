"""Process-owned Kineto activities; device-busy union is distinct from CUDA span."""
import ctypes as ct
import json
import os
from pathlib import Path
import threading
import time

class ProcessInfo(ct.Structure):
    _fields_=[('pid',ct.c_uint),('memory',ct.c_ulonglong),('gpu_instance',ct.c_uint),('compute_instance',ct.c_uint)]

class ResourceGuard:
    def __init__(self,gpu,output,period=.1):
        self.lib=ct.CDLL('libnvidia-ml.so.1')
        if self.lib.nvmlInit_v2()!=0:raise RuntimeError('NVML init failed')
        self.handle=ct.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(ct.c_uint(gpu),ct.byref(self.handle))!=0:raise RuntimeError('NVML device failed')
        self.output=Path(output);self.period=period;self.samples=[];self.errors=[];self.foreign=set();self.stop=threading.Event()
        initial=self.sample()
        if initial['pids']:raise RuntimeError(f'GPU occupied before model load: {initial["pids"]}')
        self.samples.append(initial)
        self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
    def sample(self):
        count=ct.c_uint(64);procs=(ProcessInfo*64)()
        code=self.lib.nvmlDeviceGetComputeRunningProcesses_v2(self.handle,ct.byref(count),procs)
        if code!=0:raise RuntimeError(f'NVML compute process query failed: {code}')
        power=ct.c_uint();temp=ct.c_uint()
        pc=self.lib.nvmlDeviceGetPowerUsage(self.handle,ct.byref(power))
        tc=self.lib.nvmlDeviceGetTemperature(self.handle,ct.c_uint(0),ct.byref(temp))
        return {'monotonic':time.perf_counter(),'pids':[int(procs[i].pid) for i in range(count.value)],
            'power_W':power.value/1000 if pc==0 else None,'temperature_C':temp.value if tc==0 else None}
    def monitor(self):
        while not self.stop.wait(self.period):
            try:
                r=self.sample();self.samples.append(r);self.foreign.update(p for p in r['pids'] if p!=os.getpid())
            except Exception as e:self.errors.append(str(e));return
    def check(self):
        r=self.sample();self.samples.append(r);self.foreign.update(p for p in r['pids'] if p!=os.getpid())
        if self.foreign or self.errors:raise RuntimeError(f'GPU ownership contaminated: foreign={sorted(self.foreign)} errors={self.errors}')
    def close(self):
        self.stop.set();self.thread.join()
        try:self.check()
        finally:
            self.output.parent.mkdir(parents=True,exist_ok=True)
            with self.output.open('x') as f:json.dump({'pid':os.getpid(),'sample_period':self.period,'samples':self.samples,
               'foreign_pids':sorted(self.foreign),'errors':self.errors,'valid':not self.foreign and not self.errors},f)

def union_duration(intervals):
    total=0.;end=None
    for a,b in sorted(intervals):
        if b<a:raise ValueError('Negative activity duration')
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total

def summarize_trace(path,require_kernels=True):
    """Chrome/Kineto GPU categories only; never sum nested CPU operator timers."""
    raw=json.loads(Path(path).read_text())
    events=[e for e in raw['traceEvents'] if e.get('ph')=='X' and e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']]
    kernels=[e for e in events if e.get('cat')=='kernel']
    if require_kernels and not kernels:raise RuntimeError('No actual GPU kernel events: CUPTI activity capture unavailable; no CUDA-span substitution')
    intervals=[(float(e['ts']),float(e['ts'])+float(e['dur'])) for e in events]
    devices={e.get('args',{}).get('device') for e in events if 'device' in e.get('args',{})}
    if len(devices)>1:raise RuntimeError('Multiple GPU devices in single-device accounting')
    return {'gpu_active_union_seconds':union_duration(intervals)/1e6,
        'gpu_kernel_sum_seconds':sum(float(e['dur']) for e in kernels)/1e6,
        'gpu_memory_activity_sum_seconds':sum(float(e['dur']) for e in events if e.get('cat')!='kernel')/1e6,
        'kernel_events':len(kernels),'activity_events':len(events),'devices':sorted(devices),
        'definition':'union of process-owned kernel/memcpy/memset durations; profiler instrumentation present',
        'cpu_time_and_cuda_elapsed_span_are_separate':True,'device_power_is_not_query_energy':True}

def profile_call(function,path,require_kernels=True):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,profile_memory=False,with_stack=False) as prof:
        result=function()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    return result,summarize_trace(path,require_kernels)
