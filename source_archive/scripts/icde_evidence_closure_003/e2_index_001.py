import common as C
import engine_001 as E
import sys,time,os,io,json,shutil,resource
gpu=sys.argv[1];C.start('E2/index_build_001');C.STAGE='E2_INDEX_BUILD'
index=C.ROOT/'cache/indexes/pharma_fixedcorpus_index_001';assert not index.exists(),'Existing index immutable';assert shutil.disk_usage(C.ROOT).free>20*2**30
E.idle(gpu)
from mdrouter_v5.gpu_accounting import ResourceGuard,profile_call
guard=ResourceGuard(int(gpu),C.A/'resource_guard.json')
try:
 import pyarrow.parquet as pq
 from PIL import Image
 import torch
 env=E.backend();source=C.ROOT/'data/primary/vidore_v3_pharmaceuticals/262e203d7c59c55947f7042812e0bcc1eee190b1/corpus';shards=sorted(source.glob('*.parquet'));assert len(shards)==2
 expected=['0f51d221a1def60c6d1436a31507e69be42a6a1da4a9ae45ff6e231751466ac5','827305c58164c9e69581953560b1e76349dbc86a5a06df91d16c92023175e468']
 for p,h in zip(shards,expected):assert C.sha(p)==h;C.INPUT[str(p.relative_to(C.ROOT))]={'sha256':h,'role':'OFFICIAL_UNLABELED_CORPUS'}
 index.mkdir();(index/'visual_parts').mkdir();recipe={'time_bj':C.stamp(),'source_revision':source.parent.name,'pages':2313,'visual_backend':'same frozen ColQwen2 FP16 embeddings, FP32 exact MaxSim','batch_size':2,'resize_longest_edge':768*28*28,'dense':'NOT_NEEDED_FOR_FOUR_PLAN_PHARMA_STAGE','code_sha256':C.sha(__file__),'cal_eval_queries_or_labels_read':0};C.write(C.A/'prebuild_recipe.json',recipe)
 t0=time.perf_counter();ct=time.process_time();pages=[r for p in shards for r in pq.read_table(p,columns=['corpus_id','doc_id','markdown','page_number_in_doc']).to_pylist()];pages.sort(key=lambda r:int(r['corpus_id']));assert len(pages)==len({p['corpus_id'] for p in pages})==2313;C.write(index/'pages.json',pages)
 model,processor=env['load_visual']();assert processor.image_processor.size['longest_edge']==768*28*28;done=set();stats=[];tv=time.perf_counter();blockid=0
 with torch.inference_mode():
  for shard in shards:
   pf=pq.ParquetFile(shard)
   for block in pf.iter_batches(batch_size=64,columns=['corpus_id','image']):
    rows=block.to_pylist()
    def encode_block():
     for start in range(0,len(rows),2):
      guard.check();assert shutil.disk_usage(C.ROOT).free>15*2**30;batch=rows[start:start+2];ids=[int(r['corpus_id']) for r in batch];images=[Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB') for r in batch];inputs=processor.process_images(images).to('cuda:0');embs=model(**inputs);valid=inputs['attention_mask'].bool();embs=[e[m].detach().cpu() for e,m in zip(embs,valid)];assert all(torch.isfinite(e).all() for e in embs);p=index/'visual_parts'/f'{ids[0]:06d}.pt';assert not p.exists();torch.save({'ids':ids,'embeddings':embs},p);done.update(ids)
    _,st=profile_call(encode_block,C.A/f'build_block_{blockid:04d}.json');stats.append({'block':blockid,'pages':len(rows),**st});blockid+=1;print(json.dumps({'stage':'E2_INDEX','completed_pages':len(done),'total':2313,'wall_s':time.perf_counter()-tv,'time_bj':C.stamp()}),flush=True)
 assert done=={int(p['corpus_id']) for p in pages};visualtime=time.perf_counter()-tv
 manifest={**recipe,'source_corpus':{str(p.relative_to(C.ROOT)):C.sha(p) for p in shards},'visual_seconds':visualtime,'visual_GPU_activity_s':sum(s['gpu_active_union_seconds'] for s in stats),'total_setup_wall_s':time.perf_counter()-t0,'setup_CPU_s':time.process_time()-ct,'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_host_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_manifest_sha256':C.sha(C.ROOT/'manifests/hr_models_verified.json'),'files':{str(p.relative_to(index)):C.sha(p) for p in index.rglob('*') if p.is_file()}};C.write(index/'index_manifest.json',manifest);C.write(C.A/'index_manifest.json',manifest);C.table(C.R/'build_profile_blocks.csv',stats);print(json.dumps({'status':'PASS_INDEX','pages':len(done),'index_manifest_sha256':C.sha(index/'index_manifest.json'),'time_bj':C.stamp()}),flush=True)
finally:guard.close()
