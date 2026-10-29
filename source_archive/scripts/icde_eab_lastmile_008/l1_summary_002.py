import common as C
import sys,collections,json
import numpy as np
sys.path.insert(1,str(C.ROOT/'scripts/icde_eab_validation_005'))
import frontier_math as F
C.start('L1/summary_001');protocol=C.load(C.ART/'L1_preparation_001/protocol.json','POST_HOC_EXTENSION_DESIGN')
for n,h in protocol['code_SHA256'].items():assert C.sha(C.ROOT/'scripts'/C.RUN/n)==h,'L1 design code changed'
raw=[];setups=[]
for i in range(3):
    suffix='002' if i==2 else '001'
    completed=C.load(C.ART/f'L1/process_{i}_{suffix}/completed.json','ACTUAL_EXECUTION_COMPLETION');assert completed['queries']==160
    guard=C.load(C.ART/f'L1/process_{i}_{suffix}/resource_guard.json','RESOURCE_GUARD');assert guard['valid'] and not guard['foreign_pids'] and not guard['errors']
    raw+=C.load(C.REPORT/f'L1/process_{i}_{suffix}/requests.csv','ACTUAL_FULL_PATH_REQUESTS');setups.append(C.load(C.ART/f'L1/process_{i}_{suffix}/setup.json','MODEL_BM25_AND_RESIDENCY_SETUP'))
assert len(raw)==5760
correct=C.load(C.REPORT/'L1/process_0_001/correctness.csv','SCORER_AND_CANDIDATE_CORRECTNESS');assert len(correct)==640 and max(float(x['processor_score_max_abs_error']) for x in correct)<1e-3
profiles=C.load(C.REPORT/'L1/process_0_001/GPU_profiles.csv','SEPARATE_GPU_PROFILES');assert len(profiles)==1600
groups=C.load(C.ROOT/protocol['query_family_map'],'SAME_FROZEN_157_FAMILY_27_SOURCE_MAP');group={int(x['query_id']):x for x in groups};ids=protocol['queries'];plans=C.PLANS;by=collections.defaultdict(list)
for x in raw:by[int(x['query_id']),int(x['process']),x['plan']].append(x)
assert all(len(by[q,p,m])==3 for q in ids for p in range(3) for m in plans)
T=np.array([[[np.mean([float(x['E2E_s']) for x in by[q,p,m]])*1000 for m in plans] for p in range(3)] for q in ids]);CPU=np.array([[[np.mean([float(x['CPU_s']) for x in by[q,p,m]])*1000 for m in plans] for p in range(3)] for q in ids])
R=np.array([[float(by[q,0,m][0]['r']) for m in plans] for q in ids]);U=np.array([[float(by[q,0,m][0]['nDCG10']) for m in plans] for q in ids]);RB=np.array([[float(by[q,0,m][0]['r_bar']) for m in plans] for q in ids])
for i,q in enumerate(ids):
    for j,m in enumerate(plans):assert all(float(x['r'])==R[i,j] and float(x['nDCG10'])==U[i,j] for p in range(3) for x in by[q,p,m])
GPU=np.array([[float(next(x['gpu_active_union_seconds'] for x in profiles if int(x['query_id'])==q and x['plan']==m and x['component']=='complete_path')) for m in plans] for q in ids]);intervals={};cirows=[]
for grouping,key in [('source','source_group'),('family','query_family')]:
    labels=np.array([group[q][key] for q in ids]);ix=F.boots(labels,2000,C.SEED+(1 if grouping=='source' else 0));pi=np.random.default_rng(C.SEED+43).integers(0,3,(2000,3));br=np.array([R[k].mean(0) for k in ix]);bu=np.array([U[k].mean(0) for k in ix]);bt=np.array([T[k][:,p].mean((0,1)) for k,p in zip(ix,pi)]);bg=np.array([GPU[k].mean(0) for k in ix])
    for j,m in enumerate(plans):
        for metric,arr in [('r',br[:,j]),('nDCG10',bu[:,j]),('E2E_ms',bt[:,j]),('GPU_active_s',bg[:,j]),('within_model_E2E_saving_pp',100*(1-bt[:,j]/bt[:,0]))]:
            lo,hi=np.quantile(arr,[.025,.975]);intervals[grouping,m,metric]=(lo,hi);cirows.append({'grouping':grouping,'groups':len(set(labels)),'plan':m,'metric':metric,'low95':lo,'high95':hi,'scope':'pointwise paired descriptive; EXPLORATORY_POSTHOC_EXTENDED; timing repeats not new queries'})
main=[];per=[]
for j,m in enumerate(plans):
    rr=[x for x in raw if x['plan']==m];row={'retriever':'ColPali-v1.3','plan':m,'queries':160,'families':157,'sources':27,'nDCG10':U[:,j].mean(),'mean_r':R[:,j].mean(),'harm_rate':np.mean(R[:,j]>1e-10),'r_bar':RB[:,j].mean(),'warm_E2E_ms':T[:,:,j].mean(),'P50_ms':1000*np.median([float(x['E2E_s']) for x in rr]),'P95_ms':1000*np.quantile([float(x['E2E_s']) for x in rr],.95),'CPU_ms':CPU[:,:,j].mean(),'GPU_active_s':GPU[:,j].mean(),'within_model_E2E_saving_pp':100*(1-T[:,:,j].mean()/T[:,:,0].mean()),'evidence':'EXPLORATORY_POSTHOC_EXTENDED','cost_scope':'NEW_ACTUAL_COMPLETE_WARM_PATH; separate GPU instrumentation; own FullV reference; not Serving or full business net benefit','new_certification':'NOT_CERTIFIED'}
    for key in ['r','nDCG10','E2E_ms','GPU_active_s','within_model_E2E_saving_pp']:row[key+'_source95_low'],row[key+'_source95_high']=intervals['source',m,key]
    main.append(row)
    for i,q in enumerate(ids):per.append({'query_id':q,'family':group[q]['query_family'],'source_group':group[q]['source_group'],'plan':m,'nDCG10':U[i,j],'r':R[i,j],'r_bar':RB[i,j],'E2E_ms':T[i,:,j].mean(),'CPU_ms':CPU[i,:,j].mean(),'GPU_active_s':GPU[i,j],'harm':R[i,j]>1e-10,'FalseSafe_diagnostic':m!='Full-Visual' and R[i,j]>.05})
old=C.load('reports/icde_eab_independent_007/T2/summary_001/biomedical_main_table.csv','PRIMARY_MODEL_DIFFERENT_HARDWARE_COHORT');cross=[]
for x in old:
    if x['method'] not in plans:continue
    cross.append({'retriever':x['retriever'],'plan':x['method'],'n':160,'nDCG10':x['nDCG10'],'mean_r':x['mean_r'],'r_low95':x['mean_r_source95_low'],'r_high95':x['mean_r_source95_high'],'E2E_ms':x['E2E_mean_ms'],'GPU_s':x['GPU_active_s'],'within_model_E2E_saving_pp':100*(1-float(x['E2E_mean_ms'])/float(old[0]['E2E_mean_ms'])),'cohort':'007_COLQWEN_PREREGISTERED_FIXED_POLICY_DESCRIPTIVE','own_model_reference':True,'cross_model_hardware_pairing':False})
for x in main:cross.append({'retriever':x['retriever'],'plan':x['plan'],'n':160,'nDCG10':x['nDCG10'],'mean_r':x['mean_r'],'r_low95':x['r_source95_low'],'r_high95':x['r_source95_high'],'E2E_ms':x['warm_E2E_ms'],'GPU_s':x['GPU_active_s'],'within_model_E2E_saving_pp':x['within_model_E2E_saving_pp'],'cohort':'008_COLPALI_EXPLORATORY_POSTHOC_EXTENDED','own_model_reference':True,'cross_model_hardware_pairing':False})
index=C.load('cache/indexes/icde_eab_lastmile_008_biomedical_colpali_001/index_manifest.json','MEASURED_NEW_INDEX_SETUP');ocr=C.load('artifacts/icde_eab_independent_007/T1_preparation/ocr_001/setup_cost.json','SHARED_PREVIOUSLY_MEASURED_OCR');amort=[]
for x in main:
    visual=x['plan']!='BM25';knownwall=(index['setup_wall_s']+np.mean([z['model_load_wall_s'] for z in setups])) if visual else 0
    for Q in [160,1000,10000,100000,1000000]:amort.append({'plan':x['plan'],'Q':Q,'shared_OCR_wall_s':ocr['wall_s'],'shared_OCR_scope':'measured007 reused unchanged, not reexecuted008','index_wall_s':index['setup_wall_s'] if visual else 0,'index_CPU_s':index['setup_CPU_s'] if visual else 0,'measured_index_GPU_s':index['setup_GPU_activity_s'] if visual else 0,'model_load_wall_s':np.mean([z['model_load_wall_s'] for z in setups]) if visual else 0,'model_load_GPU_s':None if visual else 0,'warm_E2E_ms':x['warm_E2E_ms'],'setup_amortized_wall_ms':x['warm_E2E_ms']+1000*(knownwall+ocr['wall_s']+np.mean([z['BM25_setup_wall_s'] for z in setups]))/Q,'GPU_warm_plus_measured_index_s':x['GPU_active_s']+(index['setup_GPU_activity_s']/Q if visual else 0),'scope':'shared visual-index accounting scenario, model-load GPU/energy/network missing; CPU/GPU overlap not added; not full business net benefit'})
# Empty OCR sensitivity uses official relevance only for this already authorized dataset.
import pyarrow.parquet as pq
bio=C.load('artifacts/icde_eab_independent_007/T1/biomedical_frozen_protocol.json');pages=C.load('artifacts/icde_eab_independent_007/T1_preparation/ocr_001/pages.json','UNLABELLED_SAME_OCR');empty={int(x['corpus_id']) for x in pages if not (x.get('markdown') or '').strip()};qp=C.ROOT/bio['official_inputs']['qrels']['path'];assert C.sha(qp)==bio['official_inputs']['qrels']['sha256'];labels=pq.read_table(qp).to_pylist();affected={int(x['query-id']) for x in labels if int(x['corpus-id']) in empty and float(x['score'])>0};sensitivity=[]
C.write(C.A/'public_benchmark_qrels_access.json',{'time_bj':C.stamp(),'path':str(qp.relative_to(C.ROOT)),'sha256':C.sha(qp),'role':'AUTHORIZED_PUBLIC_BIOMEDICAL_EMPTY_OCR_SENSITIVITY','project_sealed_reads':0})
for name,keep in [('relevant_empty_OCR_page',affected),('no_relevant_empty_OCR_page',set(ids)-affected)]:
    ix=[i for i,q in enumerate(ids) if q in keep]
    for j,m in enumerate(plans):sensitivity.append({'group':name,'plan':m,'queries':len(ix),'mean_r':R[ix,j].mean() if ix else None,'nDCG10':U[ix,j].mean() if ix else None,'E2E_ms':T[ix,:,j].mean() if ix else None,'empty_OCR_pages':len(empty),'removed_copyright_image_page_flags':'UNKNOWN_OFFICIAL_RELEASE_HAS_NO_COMPLETE_PAGE_LEVEL_FLAGS','scope':'post-hoc sensitivity, all160 retained, no query or page filtering'})
for n,x in [('biomedical_colpali_fullv_r50_r100_bm25.csv',main),('cross_model_within_domain_comparison.csv',cross),('setup_amortization.csv',amort),('source_family_intervals.csv',cirows),('per_query.csv',per),('empty_OCR_copyright_sensitivity.csv',sensitivity),('gpu_cost_breakdown.csv',profiles)]:C.table(C.R/n,x)
facts='Biomedical ColPali160，5760次真实计时/3进程×3重复，1600GPU分项；'+ '; '.join('{} U={:.4f},r={:.5f}[{:.5f},{:.5f}],E2E={:.2f}ms'.format(x['plan'],x['nDCG10'],x['mean_r'],x['r_source95_low'],x['r_source95_high'],x['warm_E2E_ms']) for x in main)
C.prose(C.R/'L1_decision.md','# PASS execution / EXPLORATORY_POSTHOC_EXTENDED\n\n'+facts+'\n\n本实验发生在ColQwen结果已知之后，非新盲测。相同160问、1016页与OCR/分组/权重固定，无Q/QP移植。各模型使用自己的FullV，模型间硬件cohort不同，不作配对时间优越性检验。评分与candidate containment阈值保持不变；setup/OCR/GPU分项与空OCR敏感性见表，版权缺图页级标签UNKNOWN，不按风险删查询。新K1 NOT_CERTIFIED，封存读取0。\n\n支持论文跨模型领域相关静态前沿表，不证明模型无关部署保证。\n')
C.finish('L1_summary_001','PASS_BIOMEDICAL_COLPALI_EXPLORATORY_POSTHOC_EXTENDED',facts,'报告同数据域双检索器的各自前沿，不把事后补充伪装预注册，不转移旧证书或模型。')
