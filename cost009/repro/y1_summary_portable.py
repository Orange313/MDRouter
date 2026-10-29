import common as C
import collections,json,math,sys
import numpy as np
import frontier_math as F
C.start('Y1/summary_001');ocr=C.load('artifacts/icde_eab_independent_007/T1_preparation/ocr_001/setup_cost.json','REUSED_SHARED_OCR_NOT_NEW_EXECUTION');proto=C.load('artifacts/icde_eab_independent_007/T1/biomedical_frozen_protocol.json');groups=C.load(C.ROOT/proto['query_family_map'],'FROZEN_GROUPS');group={int(x['query_id']):x for x in groups};ids=proto['query_ids'];src=np.array([group[q]['source_group'] for q in ids]);setup=[];measured=[];cold=[];warm=[];amort=[];breaks=[];vectors=[]
textrows=[];textcold=[]
for p in range(3):textrows+=C.load(C.REPORT/f'Y1/BM25_CPU_p{p}_001/requests.csv');textcold.append(C.load(C.ART/f'Y1/BM25_CPU_p{p}_001/completed.json'))
assert len(textrows)==480
GPU_old={}
for x in C.load('reports/icde_eab_independent_007/T2/summary_001/biomedical_main_table.csv'):
    if x['method'] in C.PLANS:GPU_old['ColQwen2',x['method']]=float(x['GPU_active_s'])
for x in C.load('reports/icde_eab_lastmile_008/L1/summary_001/biomedical_colpali_fullv_r50_r100_bm25.csv'):GPU_old['ColPali',x['plan']]=float(x['GPU_active_s'])
for model in ['ColQwen2','ColPali']:
    rr=[];cc=[];first=[];completed=[]
    for p in range(4):
        stem=f'Y1/{model}_p{p}_001';done=C.load(C.ART/stem/'completed.json','NEW_COST_EXECUTION_COMPLETE');assert done['status']=='PASS_COST_ONLY' and done['qrels_reads']==0;completed.append(done)
        guard=C.load(C.ART/stem/'resource_guard.json','ACTUAL_RESOURCE_GUARD');assert guard['valid'] and not guard['foreign_pids'] and not guard['errors']
        cc+=C.load(C.REPORT/stem/'setup_components.csv');ff=C.load(C.REPORT/stem/'first_requests.csv');first.append(ff)
        if p<3:rr+=C.load(C.REPORT/stem/'requests.csv')
        for f in ff:cold.append({'model':model,'process':p,**f,'scope':'only firstFullV is process first; later first-plan requests already followFullV; processcold, OS cache uncontrolled'})
    assert len(rr)==1920
    def comp(name,field='wall_s',ps=[0,1,2]):
        v=[float(x[field]) for x in cc if x['component']==name and int(x['process']) in ps and x.get(field) not in [None,'']];return float(np.mean(v)) if v else None
    def gpu(name):return comp(name,'GPU_s',[3])
    build=comp('full_index_build_and_persist',ps=[0]);buildCPU=comp('full_index_build_and_persist','CPU_s',[0]);blocks=C.load(C.REPORT/f'Y1/{model}_p3_001/index_block_costs.csv');buildGPU=sum(float(x['gpu_active_union_seconds']) for x in blocks if x['kind']=='encoding_block_profiled');assert sum(int(x['pages']) for x in blocks if x['kind']=='encoding_block_profiled')==1016
    for p in [0,3]:
        vv=C.load(C.REPORT/f'Y1/{model}_p{p}_001/index_vector_equivalence.csv');assert len(vv)==1016 and max(float(x['maxabs']) for x in vv)<=1e-3;vectors.append({'model':model,'process':p,'pages':1016,'maxabs':max(float(x['maxabs']) for x in vv),'bitwise_equal_pages':sum(x['bitwise_equal']=='True' for x in vv),'frozen_quality_outputs_replaced':False})
    # Observed operational startup excludes the first request; it includes imports/metadata/audit work.
    reuse_starts=[float(first[p][0]['operational_cold_to_first_s'])-float(first[p][0]['first_request_wall_s']) for p in [1,2]]
    startup_service=float(np.mean(reuse_starts));bmwall=comp('BM25_build');base_visual=startup_service-bmwall
    oldGPU_scope='frozen007/008 instrumented160-query complete paths; not new009 profile12 mean'
    summary={'model':model,'queries':160,'corpus_pages':1016,'unprofiled_processes':3,'new_unprofiled_requests':1920,'profiled_processes':1,'model_load_wall_s':comp('model_load'),'model_load_CPU_s':comp('model_load','CPU_s'),'model_load_GPU_s':gpu('model_load'),'CUDA_init_wall_s':comp('CUDA_context_init'),'CUDA_init_GPU_s':None,'CUDA_init_GPU_scope':'NOT_MEASURED','index_build_unprofiled_wall_s':build,'index_build_unprofiled_CPU_s':buildCPU,'index_build_profiled_GPU_s':buildGPU,'index_build_profiled_wall_s':comp('full_index_build_and_persist',ps=[3]),'index_residency_wall_s':comp('index_residency'),'index_residency_GPU_s':gpu('index_residency'),'BM25_build_wall_s':bmwall,'adapter_overlay_wall_s':comp('adapter_overlay'),'resident_assets_process_startup_s':startup_service,'resident_assets_cold_first_FullV_s':np.mean([float(first[p][0]['operational_cold_to_first_s']) for p in [1,2]]),'fresh_index_operational_first_FullV_s':float(first[0][0]['operational_cold_to_first_s']),'index_disk_bytes':completed[0]['index_bytes'],'peak_GPU_bytes':max(x['peak_GPU_bytes'] for x in completed),'peak_RSS_KiB':max(x['peak_RSS_KiB'] for x in completed),'OCR_shared_wall_s':ocr['wall_s'],'OCR_CPU_core_s':ocr['parent_CPU_s']+ocr['child_CPU_s'],'energy':'NOT_MEASURED','index_write_scope':'serialization/write returns; no fsync durability claim','setup_uncertainty':'index unprofiled n1; profiled index n1 cannot substitute a second unprofiled repeat; process load n3'}
    setup.append(summary);measured+=cc
    bucket=collections.defaultdict(list)
    for x in rr:bucket[int(x['query_id']),int(x['process']),x['plan']].append(x)
    T=np.array([[[float(bucket[q,p,m][0]['E2E_s']) for m in C.PLANS] for p in range(3)] for q in ids]);CPU=np.array([[[float(bucket[q,p,m][0]['CPU_s']) for m in C.PLANS] for p in range(3)] for q in ids]);ix=F.boots(src,2000,C.SEED+1);pi=np.random.default_rng(C.SEED+43).integers(0,3,(2000,3));bt=np.array([T[k][:,p].mean((0,1)) for k,p in zip(ix,pi)])
    for j,m in enumerate(C.PLANS):
        dist=[float(x['E2E_s']) for x in rr if x['plan']==m];lo,hi=np.quantile(bt[:,j],[.025,.975]);saving=100*(1-T[:,:,j].mean()/T[:,:,0].mean());slo,shi=np.quantile(100*(1-bt[:,j]/bt[:,0]),[.025,.975]);warm.append({'model':model,'plan':m,'n_queries':160,'processes':3,'requests':480,'warm_E2E_ms':1000*T[:,:,j].mean(),'source_process95_low_ms':1000*lo,'source_process95_high_ms':1000*hi,'P50_ms':1000*np.median(dist),'P95_ms':1000*np.quantile(dist,.95),'CPU_ms':1000*CPU[:,:,j].mean(),'GPU_active_s':GPU_old[model,m],'GPU_scope':oldGPU_scope,'within_model_warm_saving_pp':saving,'saving_source95_low_pp':slo,'saving_source95_high_pp':shi,'scope':'new009 cost cohort only; does not replace primary007 QP/R100 paired contrast; no cross-model AA'})
        for scenario in ['PLAN_COMPONENT_FROM_SCRATCH','PERSISTENT_ASSETS_COLD_LAUNCH','ALREADY_RESIDENT','FOUR_PLAN_SERVICE_FROM_SCRATCH']:
            if scenario=='ALREADY_RESIDENT':shared=0.;fixed=0.;scope='setup prepaid by assumption; not missing=0'
            elif scenario=='FOUR_PLAN_SERVICE_FROM_SCRATCH':shared=ocr['wall_s'];fixed=build+startup_service;scope='common service hostsall4plans; OCR once, visualindex once, model/startup once; not standalone BM25'
            elif m=='BM25':shared=ocr['wall_s'] if scenario=='PLAN_COMPONENT_FROM_SCRATCH' else 0.;fixed=np.mean([x['process_startup_excluding_first_s'] for x in textcold]);scope='CPU-only measured startup; sharedOCR prepaid only for persistentassets'
            else:shared=ocr['wall_s'] if m!='Full-Visual' and scenario=='PLAN_COMPONENT_FROM_SCRATCH' else 0.;fixed=base_visual+(bmwall if m!='Full-Visual' else 0.)+(build if scenario=='PLAN_COMPONENT_FROM_SCRATCH' else 0.);scope='component-composed standalone scenario from measured phases; FullV requires noOCR/BM25; residual imports/metadata/audits retained; not an independently measured standalone cold total'
            for N in [100,1000,10000]:
                unit=T[:,:,j].mean() if m!='BM25' or scenario=='FOUR_PLAN_SERVICE_FROM_SCRATCH' else np.mean([float(x['E2E_s']) for x in textrows]);total=shared+fixed+N*unit
                visual_needed=m!='BM25' or scenario=='FOUR_PLAN_SERVICE_FROM_SCRATCH';paid=scenario!='ALREADY_RESIDENT';scratch=scenario.endswith('FROM_SCRATCH')
                gpusetup=(gpu('model_load')+gpu('index_residency')+(buildGPU if scratch else 0.)) if visual_needed and paid else 0.
                cpusetup=(ocr['parent_CPU_s']+ocr['child_CPU_s']) if shared else 0.
                if paid:
                    if visual_needed:cpusetup+=comp('model_load','CPU_s')+comp('CUDA_context_init','CPU_s')+comp('index_residency','CPU_s')+(buildCPU if scratch else 0.)+(comp('adapter_overlay','CPU_s') if model=='ColPali' else 0.)
                    if m!='Full-Visual' or scenario=='FOUR_PLAN_SERVICE_FROM_SCRATCH':cpusetup+=comp('BM25_build','CPU_s')
                cpuunit=CPU[:,:,j].mean() if m!='BM25' or scenario=='FOUR_PLAN_SERVICE_FROM_SCRATCH' else np.mean([float(x['CPU_s']) for x in textrows])
                amort.append({'model':model,'plan':m,'scenario':scenario,'N':N,'C_shared_OCR_s':shared,'C_plan_setup_s':fixed,'C_warm_s':unit,'C_total_wall_s':total,'amortized_wall_ms_per_query':1000*total/N,'CPU_warm_core_s_per_query':cpuunit,'CPU_setup_known_partial_core_s':cpusetup,'CPU_total_known_partial_core_s':cpusetup+N*cpuunit,'CPU_setup_limit':'known component core-seconds only; cold import/metadata residualCPU not separately measured','GPU_warm_activity_s_per_query':GPU_old[model,m],'GPU_setup_known_partial_s':gpusetup,'GPU_total_known_partial_s':gpusetup+N*GPU_old[model,m],'GPU_context_setup_s':None if visual_needed and paid else 0.,'GPU_setup_limit':'partial; contextGPU not measured; profile cohorts separate; zero only for explicitly prepaid/noGPU scenario','scope':scope,'uncertainty':'conditional scenario, no independent Q selected; singleton build/oldOCR uncertainty not covered by warm interval','energy':'NOT_MEASURED'})
    for plan in ['BM25-R50','BM25-R100']:
        j=C.PLANS.index(plan);delta=T[:,:,0].mean()-T[:,:,j].mean();n=math.ceil((ocr['wall_s']+bmwall)/delta) if delta>0 else None;breaks.append({'model':model,'plan':plan,'warm_saving_s':delta,'extra_required_OCR_BM25_setup_s':ocr['wall_s']+bmwall,'conditional_break_even_queries':n,'scope':'component scenario with sharedvisualindex; newcohort values, not business threshold or chosen deploymentQ','status':'FINITE_CONDITIONAL' if n else 'NO_POSITIVE_WARM_MARGIN'})
C.table(C.R/'setup_measured.csv',setup);C.table(C.R/'setup_component_raw.csv',measured);C.table(C.R/'cold_first_requests.csv',cold);C.table(C.R/'warm_static_costs_009.csv',warm);C.table(C.R/'amortization_scenarios.csv',amort);C.table(C.R/'break_even_queries.csv',breaks);C.table(C.R/'index_equivalence.csv',vectors);C.table(C.R/'cpu_only_BM25_requests.csv',textrows)
C.prose(C.R/'setup_reuse_matrix.yaml',json.dumps({'scope':'logical required components plus separately observed commonfourplanservice','BM25':['OCR_once','CPU_BM25_startup'],'BM25-R50':['OCR_once','BM25','visual_model','full1016_visualindex_once'],'BM25-R100':['OCR_once','BM25','visual_model','full1016_visualindex_once'],'Full-Visual':['visual_model','full1016_visualindex_once; OCR/BM25 not logically required'],'both_models_joint_use':'OCR+BM25 may be shared once; two standalone modelscenarios must not be blindly summed','accounting':'CPUcoreseconds/GPUactivity/wall separate; no add-overlap wall; profilerCPU/wall is not unprofiled production cost','missing':['contextinitializationGPU','fsync durable persistence timing','network/queue/fullhost energy','independent unprofiled indexbuild repetition']},indent=2)+'\n')
desc='''# Actual measured cost scopes and assumptions

Both frozen retrievers, all160 queries/1016 pages and four staticplans are retained. New009 unprofiled warm paths have3840 GPU-worker requests plus480 CPU-onlyBM25 requests. Each model has3 independent unprofiled model loads,1 unprofiled full index build and a separate fully profiled build/load/residency; profiling is never counted as another unprofiled repeat. All new vectors are compared against old frozen vectors, and no quality table/model/ref is replaced. Model load GPU activity is now measured; CUDA-context GPU remainsNOT_MEASURED. CPU and GPU overlap are not added to wall. All observed resident/first-request/rawtiming/process costs are retained.

Cold means fresh process/model/CUDA/residency with uncontrolled OS page cache. It is not physical storage-cold latency. Guard/idle delays, worker imports/metadata/hash/equivalence audits and profiler overhead are explicit scopes. Only the firstFullV in each worker is first overall; subsequent first-plan records are not independent cold launches. Index serialization is measured at file-write-return; durable fsync/NAS persistence is unmeasured. New independent unprofiled index replication is insufficient (n1/model).

The reported N100/1000/10000 curves are hypothetical accounting scenarios Ctotal=Cshared+Cplan+N*Cwarm. A FullV-only logical service requires noOCR/BM25; rerank/BM25 use the shared407.725s OCR exactly once. Standalone-plan totals are component-composed, not a new end-to-end standalone launch measurement. Separate common-four-plan-service and persistent/resident scenarios expose reuse assumptions. Existing OCR is audited, not rerun. Source/process intervals on newwarm costs are conditional descriptive intervals; they do not cover singleton setup uncertainty. Old007 QP/R100 main paired intervals and old008 ColPali quality remain unchanged. Models are not compared for latency superiority: noABBA/AAgate was registered or passed. Energy and complete business net benefit are not established.
'''
C.prose(C.R/'cold_warm_breakdown.md',desc);C.prose(C.R/'AA_resource_guard.md','# PASS resource guard / NOT_RUN cross-model AA\n\n8GPU worker processes passed3idle checks andResourceGuard; GPU4 used. Noforeign process or unguarded job is accepted. CPU-onlyBM25 uses noCUDA. All request repeats/16profile subgroups are cost observations, not new independent relevance labels. Models sharehardware identity but no cross-model ABBA orA/A equivalence; cross-model superiority is not claimed.\n')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
figdir=C.R/'figures';figdir.mkdir()
for model in ['ColQwen2','ColPali']:
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    for plan in C.PLANS:
        a=[x for x in amort if x['model']==model and x['plan']==plan and x['scenario']=='PLAN_COMPONENT_FROM_SCRATCH'];ax.plot([x['N'] for x in a],[x['amortized_wall_ms_per_query'] for x in a],marker='o',label=plan)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Queries N (conditional scenario)');ax.set_ylabel('Setup-amortized known wall cost (ms/query)');ax.set_title(model+'; OCR counted only where required');ax.legend();fig.tight_layout();fig.savefig(figdir/(model+'_amortization.png'),dpi=180);fig.savefig(figdir/(model+'_amortization.pdf'));plt.close(fig)
facts='; '.join('{} load={:.3f}s/GPU{:.3f}s,index={:.2f}s/CPU{:.2f}s/GPU{:.2f}s, resident-cold first={:.2f}s'.format(x['model'],x['model_load_wall_s'],x['model_load_GPU_s'],x['index_build_unprofiled_wall_s'],x['index_build_unprofiled_CPU_s'],x['index_build_profiled_GPU_s'],x['resident_assets_cold_first_FullV_s']) for x in setup)
C.prose(C.R/'Y1_decision.md','# PASS actual cost collection / conditional amortization\n\n'+facts+'\n\n'+desc)
C.finish('Y1','PASS_ACTUAL_SETUP_COLD_WARM_COSTS__AMORTIZATION_CONDITIONAL',facts,'按真实依赖单次计入OCR；报告N100/1000/10000条件曲线和break-even，不替用户选择Q、不改变主结果；模型间延迟优势无AA证据。')
