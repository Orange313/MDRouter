"""Kineto GPU activity union, checked against the existing Chrome-trace parser."""
import common as C
def profile(fn,path,validate=False,require_kernels=False):
    import torch
    from mdrouter_v5.gpu_accounting import union_duration,summarize_trace
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as p:
        result=fn();torch.cuda.synchronize()
    events=[]
    for e in p.events():
        if e.device_type!=torch.autograd.DeviceType.CUDA:continue
        name=e.name;cat='gpu_memcpy' if 'memcpy' in name.lower() else 'gpu_memset' if 'memset' in name.lower() else 'kernel'
        events.append({'cat':cat,'name':name,'start_us':float(e.time_range.start),'end_us':float(e.time_range.end),'device':e.device_index})
    kernels=[x for x in events if x['cat']=='kernel'];assert events,'NO_GPU_EVENTS_NOT_ZERO'
    if require_kernels:assert kernels,'NO_KERNEL_EVENTS'
    u=union_duration([(x['start_us'],x['end_us']) for x in events])/1e6
    stats={'gpu_active_union_seconds':u,'kernel_events':len(kernels),'activity_events':len(events),'gpu_kernel_sum_seconds':sum(x['end_us']-x['start_us'] for x in kernels)/1e6,'definition':'Kineto process-owned CUDA kernel/memcpy/memset union; profiler cohort, not unprofiled wall'}
    if validate:
        p.export_chrome_trace(str(path.with_suffix('.chrome.json')));original=summarize_trace(path.with_suffix('.chrome.json'),require_kernels=require_kernels)
        assert abs(original['gpu_active_union_seconds']-u)<=2e-6,'LEAN_PARSER_MISMATCH_STOP_PROFILED_COST'
        stats['validated_against_original_parser']=original;stats['absolute_union_error_s']=abs(original['gpu_active_union_seconds']-u)
    C.write(path,{'stats':stats,'GPU_events':events});return result,stats
