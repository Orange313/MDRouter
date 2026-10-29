import common as C
import os,sys,time,json
from pathlib import Path
C.A=C.ART/'X1';out=C.A/'CPU_loader_001';out.mkdir(exist_ok=False)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
prep=json.loads((C.A/'preparation.json').read_text());base=Path(prep['model_path']['vidore/colpaligemma-3b-pt-448-base']);adapter=Path(prep['model_path']['vidore/colpali-v1.3']);overlay=out/'adapter_overlay';overlay.mkdir()
for p in adapter.iterdir():
    if p.name!='adapter_config.json' and p.is_file():(overlay/p.name).symlink_to(p.resolve())
cfg=json.loads((adapter/'adapter_config.json').read_text());cfg['base_model_name_or_path']=str(base);C.write(overlay/'adapter_config.json',cfg)
try:
    import torch
    from colpali_engine.models import ColPali,ColPaliProcessor
    torch.set_num_threads(1);t=time.perf_counter()
    model=ColPali.from_pretrained(str(overlay),torch_dtype=torch.bfloat16,device_map='cpu',local_files_only=True,attn_implementation='eager').eval();processor=ColPaliProcessor.from_pretrained(str(adapter),local_files_only=True)
    active=model.active_adapters();assert active
    C.write(out/'result.json',{'status':'PASS_TRAINED_MODEL_CPU_LOAD_ONLY','active_adapters':active,'parameters':sum(p.numel() for p in model.parameters()),'load_wall_s':time.perf_counter()-t,'image_size':processor.image_processor.size,'GPU_used':False,'quality_queries_executed':0,'time_bj':C.stamp()})
except Exception as e:
    import traceback
    C.write(out/'result.json',{'status':'FAIL_CPU_LOADER_COMPATIBILITY','exception':str(e),'traceback':traceback.format_exc(),'GPU_used':False,'quality_queries_executed':0,'time_bj':C.stamp()});raise
