"""Transformers v5 adapter-key compatibility; tensor values never changed."""
from pathlib import Path
import json,hashlib
def prepare_overlay(common,out):
    C=common;out=Path(out);out.mkdir(exist_ok=False)
    prep=json.loads((C.ROOT/'artifacts/icde_eab_main_006/X1/preparation.json').read_text());base=Path(prep['model_path']['vidore/colpaligemma-3b-pt-448-base']);adapter=Path(prep['model_path']['vidore/colpali-v1.3'])
    from safetensors.torch import load_file,save_file
    original=load_file(adapter/'adapter_model.safetensors',device='cpu');converted={};mapping=[]
    for old,tensor in original.items():
        new=old.replace('model.language_model.model.','model.model.language_model.')
        assert new not in converted;converted[new]=tensor
        mapping.append({'source_name':old,'target_name':new,'tensor_shape':list(tensor.shape),'tensor_dtype':str(tensor.dtype),'tensor_value_SHA256':hashlib.sha256(tensor.view(__import__('torch').uint8).numpy().tobytes()).hexdigest()})
    save_file(converted,str(out/'adapter_model.safetensors'))
    for p in adapter.iterdir():
        if p.name not in ['adapter_model.safetensors','adapter_config.json'] and p.is_file():(out/p.name).symlink_to(p.resolve())
    cfg=json.loads((adapter/'adapter_config.json').read_text());cfg['base_model_name_or_path']=str(base);C.write(out/'adapter_config.json',cfg);C.write(out/'tensor_mapping.json',mapping)
    check=load_file(out/'adapter_model.safetensors',device='cpu')
    assert all(__import__('torch').equal(original[m['source_name']],check[m['target_name']]) for m in mapping)
    return out,mapping
def validate_model(model,overlay,mapping):
    from safetensors.torch import load_file
    import torch
    source=load_file(Path(overlay)/'adapter_model.safetensors',device='cpu');state=model.state_dict();records=[]
    for row in mapping:
        k=row['target_name'].removeprefix('base_model.model.').replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
        assert k in state,'missing target tensor '+k
        expected=source[row['target_name']].to(dtype=state[k].dtype,device=state[k].device)
        ok=torch.equal(state[k],expected);records.append({'key':k,'exact_equal_after_declared_inference_dtype':ok});assert ok,'tensor not loaded '+k
    return records
