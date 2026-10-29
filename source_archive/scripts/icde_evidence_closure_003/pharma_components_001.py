import common as C
import engine_001 as E
import numpy as np,sys,time,json
gpu=sys.argv[1];C.start('E2/pharma_components_001');C.STAGE='E2_PHARMA_COST_COMPONENTS';E.idle(gpu)
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call
guard=ResourceGuard(int(gpu),C.A/'resource_guard.json')
try:
 import torch
 eng=E.Engine('pharma_fixedcorpus_index_001',guard);queries=[q for q in C.load(C.ART/'E2/allowed_fitdev_queries.json','ALLOWED_DEV_TEXT_ONLY') if q['role']=='development'];rows=[]
 with torch.inference_mode():
  eng.encode(queries[0]['query'])
  for i,q in enumerate(queries):
   guard.check();bm=[];ps=[];enc=[]
   for rep in range(3):
    t=time.perf_counter();ct=time.process_time();probe=eng.bm.search(q['query']);bm.append((time.process_time()-ct,time.perf_counter()-t));t=time.perf_counter();ct=time.process_time();E.probe_features(eng.bm,q['query'],*probe);ps.append((time.process_time()-ct,time.perf_counter()-t));torch.cuda.synchronize();t=time.perf_counter();ct=time.process_time();encoded=eng.encode(q['query']);torch.cuda.synchronize();enc.append((time.process_time()-ct,time.perf_counter()-t))
   encoded,st=profile_call(lambda:eng.encode(q['query']),C.A/f'encode_{q["query_id"]}.json');rows.append({'query_id':q['query_id'],'component':'visual_query_encoding','GPU_activity_s':st['gpu_active_union_seconds'],'uninstrumented_CPU_mean_s':np.mean(enc,axis=0)[0],'uninstrumented_wall_mean_s':np.mean(enc,axis=0)[1],'query_n':1,'repeats':3,'scope':'separate component call, not a free online cache'})
   for k in [50,100,len(eng.ids)]:
    cand=probe[0][:k] if k<len(eng.ids) else list(range(len(eng.ids)));_,st=profile_call(lambda:eng.score(encoded,cand),C.A/f'score_{q["query_id"]}_{k}.json');rows.append({'query_id':q['query_id'],'component':f'visual_score_{k}','GPU_activity_s':st['gpu_active_union_seconds'],'uninstrumented_CPU_mean_s':None,'uninstrumented_wall_mean_s':None,'scope':'encoding and candidate generation excluded; components not forced to sum'})
   rows.append({'query_id':q['query_id'],'component':'BM25_search','GPU_activity_s':0.,'uninstrumented_CPU_mean_s':np.mean(bm,axis=0)[0],'uninstrumented_wall_mean_s':np.mean(bm,axis=0)[1],'scope':'actual CPU search; GPU zero by CPU-only implementation'})
   rows.append({'query_id':q['query_id'],'component':'Probe_P13_statistics','GPU_activity_s':0.,'uninstrumented_CPU_mean_s':np.mean(ps,axis=0)[0],'uninstrumented_wall_mean_s':np.mean(ps,axis=0)[1],'scope':'reuse paid BM25 scores, CPU-only'})
 C.table(C.R/'pharma_cost_components.csv',rows);C.write(C.A/'component_rows.json',rows)
finally:guard.close()
C.finish('PASS_MEASURED_DEV_ONLY_COST_COMPONENTS','Pharma Dev45的查询视觉编码、R50/R100/FullV打分、BM25和P13 CPU统计已分项实测；无qrels访问。','只补成本归因，不改路由模型/工作点；分开调用的component profiles不强制相加为完整路径成本。')
