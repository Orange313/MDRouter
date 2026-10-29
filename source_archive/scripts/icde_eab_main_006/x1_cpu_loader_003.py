import common as C
import os,time,json,traceback
from pathlib import Path
from colpali_loader_003 import prepare_overlay,validate_model
out=C.ART/'X1/CPU_loader_003';out.mkdir(exist_ok=False)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
try:
    import torch
    from colpali_engine.models import ColPali,ColPaliProcessor
    torch.set_num_threads(1);overlay,mapping=prepare_overlay(C,out/'adapter_overlay');t=time.perf_counter()
    model=ColPali.from_pretrained(str(overlay),torch_dtype=torch.bfloat16,device_map='cpu',local_files_only=True,attn_implementation='eager').eval();processor=ColPaliProcessor.from_pretrained(str(overlay),local_files_only=True)
    active=model.active_adapters();assert active
    check=validate_model(model,overlay,mapping);C.write(out/'loaded_tensor_checks.json',check)
    with torch.inference_mode():
        q=model(**processor.process_queries(['Synthetic compatibility query, not a benchmark question.']));assert torch.isfinite(q).all()
    size={k:getattr(processor.image_processor.size,k) for k in ['height','width'] if getattr(processor.image_processor.size,k,None) is not None}
    result={'status':'PASS_TRAINED_ADAPTER_254_TENSORS_EXACT','active_adapters':active,'checked_adapter_tensors':len(check),'parameters':sum(p.numel() for p in model.parameters()),'load_and_synthetic_query_wall_s':time.perf_counter()-t,'image_size':size,'GPU_used':False,'benchmark_queries_executed':0,'time_bj':C.stamp()};C.write(out/'result.json',result);print(json.dumps(result),flush=True)
except Exception as e:
    C.write(out/'failure.json',{'status':'FAIL_CPU_LOADER_COMPATIBILITY','exception':str(e),'traceback':traceback.format_exc(),'GPU_used':False,'time_bj':C.stamp()});raise
