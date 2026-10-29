import common as C
import numpy as np,os,time,sys,json,collections,ast,math,subprocess,types
sys.path.insert(0,str(C.ROOT/'src'))
def idle(gpu):
 rows=[]
 for _ in range(3):
  raw=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();z=raw.split(',');ok=int(z[2].split()[0])<128 and int(z[3].split()[0])==0;rows.append({'time_bj':C.stamp(),'raw':raw,'idle':ok})
  if not ok:C.write(C.A/'three_idle_checks.json',rows);raise RuntimeError('GPU not idle; CUDA not started')
  time.sleep(1)
 C.write(C.A/'three_idle_checks.json',rows)
def backend():
 import torch
 from transformers import AutoModel,AutoTokenizer
 from colpali_engine.models import ColQwen2,ColQwen2Processor
 from aaber.common.text import tokenize
 from mdrouter_v5.graded_metrics import stable_ranking
 torch.set_num_threads(1);torch.manual_seed(C.SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 models=C.load('manifests/hr_models_verified.json');cfg=C.load('configs/hr_pilot_sparse_004.json');source=C.load('scripts/cross_domain_four_plan_001.py');nodes=[x for x in ast.parse(source).body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in ['BM25','load_visual','load_dense','encode_dense']]
 env={'MODELS':models,'CFG':cfg,'torch':torch,'AutoModel':AutoModel,'AutoTokenizer':AutoTokenizer,'ColQwen2':ColQwen2,'ColQwen2Processor':ColQwen2Processor,'Counter':collections.Counter,'defaultdict':collections.defaultdict,'np':np,'math':math,'tokenize':tokenize,'stable_ranking':stable_ranking};exec(compile(ast.Module(nodes,type_ignores=[]),'frozen_backend_AST','exec'),env);return env
def probe_features(bm,text,order,scores):
 from aaber.common.text import tokenize
 order=order[:50];v=np.array([scores[i] for i in order]);top=lambda k:float(v[k-1]) if len(v)>=k else 0.;norm=v/v.sum() if v.sum()>0 else np.array([]);entropy=-float(np.sum(norm*np.log(norm))) if len(v)>1 else 0.;words=tokenize(text)
 return np.array([top(1),top(2),float(v[:5].mean()) if len(v) else 0.,float(v[:10].mean()) if len(v) else 0.,float(v.mean()) if len(v) else 0.,float((scores>0).sum()),0. if top(1)<=0 else (top(1)-top(2))/top(1),0. if top(1)<=0 else (top(1)-top(5))/top(1),0. if top(1)<=0 else (top(1)-top(10))/top(1),entropy/max(1,np.log(max(2,len(v)))),float(v.std()/v.mean()) if len(v) and v.mean()>0 else 0.,sum(t in bm.contributions for t in words)/max(1,len(words)),float(len(order)==0)])
class Engine:
 def __init__(self,index,guard):
  self.env=backend();self.torch=self.env['torch'];self.guard=guard;self.ip=C.ROOT/'cache/indexes'/index;self.setup={};t=time.perf_counter();ct=time.process_time();self.pages=C.load(self.ip/'pages.json','UNLABELED_SHARED_CORPUS');self.ids=[int(x['corpus_id']) for x in self.pages];self.pos={x:i for i,x in enumerate(self.ids)};self.bm=self.env['BM25'](self.pages);self.setup['BM25_CPU_s']=time.process_time()-ct;self.setup['BM25_wall_s']=time.perf_counter()-t
  t=time.perf_counter();self.model,self.processor=self.env['load_visual']();self.torch.cuda.synchronize();self.setup['model_load_wall_s']=time.perf_counter()-t;manifest=C.load(self.ip/'index_manifest.json','UNLABELED_INDEX_METADATA');vv={};t=time.perf_counter()
  for p in sorted((self.ip/'visual_parts').glob('*.pt')):
   guard.check();digest=C.sha(p);assert digest==manifest['files'][str(p.relative_to(self.ip))];C.INPUT[str(p.relative_to(C.ROOT))]={'sha256':digest,'role':'UNLABELED_VISUAL_INDEX'};part=self.torch.load(p,weights_only=True);vv.update(zip(part['ids'],part['embeddings']))
  self.page_lengths=[len(vv[i]) for i in self.ids];self.lengths=self.torch.tensor(self.page_lengths,device='cuda:0');self.passage=self.torch.nn.utils.rnn.pad_sequence([vv[i].float() for i in self.ids],batch_first=True).cuda();self.torch.cuda.synchronize();self.setup['index_residency_wall_s']=time.perf_counter()-t
  from aaber.router.fast_query_router import FastLexicalFeatureEncoder
  self.qencoder=FastLexicalFeatureEncoder();self.qencoder.fit_corpus([types.SimpleNamespace(searchable_text=p['markdown'] or '') for p in self.pages])
 def Q(self,text):
  f=self.qencoder.transform(types.SimpleNamespace(question=text));return np.array([f[n] for n in self.qencoder.feature_names])
 def encode(self,text):return self.model(**self.processor.process_queries([text]).to('cuda:0'))[0].float()
 def score(self,q,indices):
  torch=self.torch;out=[]
  if not indices:return []
  for start in range(0,len(indices),64):
   sub=indices[start:start+64];block=self.passage.index_select(0,torch.tensor(sub,device='cuda:0'))
   if len(sub)<64:block=torch.nn.functional.pad(block,(0,0,0,0,0,64-len(sub)))
   logits=torch.matmul(q.unsqueeze(0).expand(64,-1,-1),block.transpose(1,2));lengths=self.lengths.index_select(0,torch.tensor(sub,device='cuda:0'));lengths=torch.nn.functional.pad(lengths,(0,64-len(sub)),value=1);mask=torch.arange(self.passage.shape[1],device='cuda:0')[None,:]>=lengths[:,None];logits.masked_fill_(mask[:,None,:],-torch.inf);v=logits.max(2).values.sum(1);out.append(v[:len(sub)])
  vals=torch.cat(out).cpu().tolist();return [indices[j] for j in self.env['stable_ranking']([self.ids[j] for j in indices],vals)]
 def execute(self,plan,text,probe=None):
  if plan=='Full-Visual':cand=list(range(len(self.ids)))
  else:
   order,_=probe if probe is not None else self.bm.search(text)
   if plan=='BM25':return [self.ids[i] for i in order[:10]]
   if plan not in ['BM25-R50','BM25-R100','BM25-R200']:raise ValueError(plan)
   cand=order[:int(plan[6:])]
  if not cand:return []
  return [self.ids[i] for i in self.score(self.encode(text),cand)[:10]]
