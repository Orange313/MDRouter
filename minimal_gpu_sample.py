"""Standalone clean-checkout four-plan smoke; no labels or new scientific claim."""
from pathlib import Path
import argparse,os,json,time,subprocess,hashlib,types,ast,collections,math
p=argparse.ArgumentParser(description=__doc__)
for n in ['adapter','base','corpus','queries','ocr','output']:p.add_argument('--'+n,required=True)
p.add_argument('--gpu',type=int,required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
pkg=Path(__file__).resolve().parent;import sys;sys.path.insert(0,str(pkg/'execution_source'));sys.path.insert(0,str(pkg/'execution_source/src'))
checks=[]
for _ in range(3):
    z=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip().split(',');ok=int(z[0].split()[0])<128 and int(z[1].split()[0])==0;checks.append({'memory':z[0],'utilization':z[1],'idle':ok});assert ok,'BUSY_GPU_NO_CUDA';time.sleep(1)
(out/'three_idle_checks.json').write_text(json.dumps(checks,indent=2))
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call
from mdrouter_v5.graded_metrics import stable_ranking
guard=ResourceGuard(a.gpu,out/'resource_guard.json',period=.1)
try:
    import torch,numpy as np,pyarrow.parquet as pq
    from PIL import Image
    import io
    from colpali_engine.models import ColPali,ColPaliProcessor
    from colpali_loader_003 import prepare_overlay,validate_model
    torch.set_num_threads(1);torch.manual_seed(202610281450);torch.backends.cuda.matmul.allow_tf32=False
    def write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(value,f,indent=2)
    prep=out/'runtime/artifacts/icde_eab_main_006/X1/preparation.json';write(prep,{'model_path':{'vidore/colpali-v1.3':a.adapter,'vidore/colpaligemma-3b-pt-448-base':a.base}})
    overlay,mapping=prepare_overlay(types.SimpleNamespace(ROOT=out/'runtime',write=write),out/'adapter_overlay')
    start=time.perf_counter();model=ColPali.from_pretrained(str(overlay),torch_dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True,attn_implementation='eager').eval();processor=ColPaliProcessor.from_pretrained(a.adapter,local_files_only=True);torch.cuda.synchronize();loadwall=time.perf_counter()-start;loaded=validate_model(model,overlay,mapping);assert len(loaded)==254
    rows=next(pq.ParquetFile(a.corpus).iter_batches(batch_size=16,columns=['corpus-id','image'])).to_pylist();ids=[int(x['corpus-id']) for x in rows]
    qrow=min(pq.read_table(a.queries).to_pylist(),key=lambda x:int(x['query-id']));text=qrow['query'];qid=int(qrow['query-id']);pages=[x for x in json.loads(Path(a.ocr).read_text()) if int(x['corpus_id']) in ids];pages.sort(key=lambda x:ids.index(int(x['corpus_id'])))
    from aaber.common.text import tokenize
    src=(pkg/'execution_source/cross_domain_four_plan_001.py').read_text();nodes=[n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='BM25'];cfg=json.loads((pkg/'config/physical_plan_config_original.json').read_text());scope={'CFG':cfg,'Counter':collections.Counter,'defaultdict':collections.defaultdict,'np':np,'math':math,'tokenize':tokenize,'stable_ranking':stable_ranking};exec(compile(ast.Module(nodes,type_ignores=[]),'frozen_BM25_AST','exec'),scope);bm=scope['BM25'](pages)
    embeddings=[]
    def build():
        for i in range(0,len(rows),2):
            guard.check();im=[Image.open(io.BytesIO(x['image']['bytes'])).convert('RGB') for x in rows[i:i+2]];inp=processor.process_images(im).to('cuda:0');v=model(**inp);embeddings.extend(x[m.bool()].float() for x,m in zip(v,inp['attention_mask']))
    with torch.inference_mode():
        start=time.perf_counter();_,setup_gpu=profile_call(build,out/'index_gpu.json');setupwall=time.perf_counter()-start
        lengths=torch.tensor([len(x) for x in embeddings],device='cuda:0');passage=torch.nn.utils.rnn.pad_sequence(embeddings,batch_first=True)
        def score(q,cand):
            if not cand:return []
            ix=torch.tensor(cand,device='cuda:0');v=passage.index_select(0,ix);logits=torch.matmul(q[None],v.transpose(1,2));mask=torch.arange(v.shape[1],device='cuda:0')[None,:]>=lengths.index_select(0,ix)[:,None];values=logits.masked_fill(mask[:,None,:],-torch.inf).max(2).values.sum(1).cpu().tolist();order=stable_ranking([ids[i] for i in cand],values);return [ids[cand[i]] for i in order[:10]]
        def execute(plan):
            if plan=='BM25':return [ids[i] for i in bm.search(text)[0][:10]]
            cand=list(range(len(ids))) if plan=='Full-Visual' else bm.search(text)[0][:int(plan[6:])]
            q=model(**processor.process_queries([text]).to('cuda:0'))[0].float();return score(q,cand)
        plans=['Full-Visual','BM25-R50','BM25-R100','BM25'];reference=execute('Full-Visual');timings=[]
        for plan in plans:
            for _ in range(3):execute(plan)
        work=[(plan,i) for plan in plans for i in range(3)];np.random.default_rng(202610281450).shuffle(work)
        for plan,i in work:
            guard.check();torch.cuda.synchronize();t=time.perf_counter();ct=time.process_time();result=execute(plan);torch.cuda.synchronize();timings.append({'plan':plan,'repeat':i,'E2E_ms':1000*(time.perf_counter()-t),'CPU_ms':1000*(time.process_time()-ct),'ranking':result})
        profiles=[]
        for plan in plans:
            _,g=profile_call(lambda:execute(plan),out/(plan+'_gpu.json'),require_kernels=plan!='BM25');profiles.append({'plan':plan,**g})
    write(out/'minimal_e2e_result.json',{'status':'PASS_TECHNICAL_SMOKE_ONLY','query_id':qid,'pages':len(ids),'trained_adapter_tensors_verified':len(loaded),'model_load_wall_s':loadwall,'index_build_wall_s':setupwall,'index_GPU_profile':setup_gpu,'timings':timings,'GPU_profiles':profiles,'reference':reference,'label_reads':0,'scope':'one fixed query/16 pages; standalone execution from clean checkout with explicit official external assets, not an independent benchmark sample or certificate'})
finally:guard.close()
print(json.dumps({'status':'PASS_TECHNICAL_SMOKE_ONLY','label_reads':0,'pages':16,'timed_requests':12,'output':str(out)}))
