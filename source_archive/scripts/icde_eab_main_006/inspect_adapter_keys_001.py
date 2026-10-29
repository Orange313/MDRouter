from pathlib import Path
from safetensors import safe_open
p=Path('/path/to/user/.cache/huggingface/hub/models--vidore--colpali-v1.3/snapshots/1b5c8929330df1a66de441a9b5409a878f0de5b0/adapter_model.safetensors')
with safe_open(p,framework='pt',device='cpu') as f:
 k=list(f.keys());print('TENSORS',len(k));print('\n'.join(k[:12]+k[-8:]))
