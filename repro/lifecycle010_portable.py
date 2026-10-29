import common as C
import math, sys, collections
import numpy as np
# frontier_math is supplied by the portable cost runtime
import frontier_math as F
C.start('A0/replay_002')
old=C.ROOT/'reports/icde_eab_submission_gate_009'
# private provenance inventory omitted; public input manifests verify every numeric input
# private provenance inventory omitted; public input manifests verify every numeric input
# private provenance inventory omitted; public input manifests verify every numeric input
setup=C.load(old/'Y1/summary_001/setup_measured.csv')
warm=C.load(old/'Y1/summary_001/warm_static_costs_009.csv')
amort=C.load(old/'Y1/summary_001/amortization_scenarios.csv')
breaks=C.load(old/'Y1/summary_001/break_even_queries.csv')
groups=C.load('reports/icde_eab_independent_007/T1/biomedical_query_family_map_002.csv')
groups={int(x['query_id']):x['source_group'] for x in groups}
out=[]; reuse=[]; checks=[]; cohorts=[]
oldq=C.load('reports/icde_eab_independent_007/T2/summary_001/biomedical_main_table.csv')
oldp=C.load('reports/icde_eab_lastmile_008/L1/summary_001/biomedical_colpali_fullv_r50_r100_bm25.csv')
scenarios=[('OCR_EXCLUSIVE_NEW_INDEX',1.,1.,False),('OCR_SHARED_HALF_NEW_INDEX',.5,1.,False),('OCR_PREPAID_NEW_INDEX',0.,1.,False),('EXISTING_INDEX_OCR_NEW',1.,0.,False),('EXISTING_INDEX_OCR_PREPAID',0.,0.,False),('ALREADY_RESIDENT_ALL_PREPAID',0.,0.,True)]
for s in setup:
    model=s['model']; startup=float(s['resident_assets_process_startup_s']); bm=float(s['BM25_build_wall_s']); base=startup-bm; index=float(s['index_build_unprofiled_wall_s']); ocr=float(s['OCR_shared_wall_s'])
    rr=[]
    for p in range(3):rr+=C.load(old/f'Y1/{model}_p{p}_001/requests.csv','IMMUTABLE_009_WARM_COST_ONLY')
    bucket={(int(x['query_id']),int(x['process']),x['plan']):x for x in rr}
    assert len(bucket)==1920
    ids=sorted({k[0] for k in bucket});src=np.array([groups[q] for q in ids]);plans=['Full-Visual','BM25-R100']
    T=np.array([[[float(bucket[q,p,m]['E2E_s']) for m in plans] for p in range(3)] for q in ids])
    ix=F.boots(src,2000,C.SEED+1);pi=np.random.default_rng(C.SEED+43).integers(0,3,(2000,3));bt=np.array([T[k][:,p].mean((0,1)) for k,p in zip(ix,pi)])
    u=T.mean((0,1));delta=u[0]-u[1];bd=bt[:,0]-bt[:,1]
    for j,m in enumerate(plans):
        row=next(x for x in warm if x['model']==model and x['plan']==m)
        err=abs(u[j]-float(row['warm_E2E_ms'])/1000);assert err<1e-12
        checks.append({'model':model,'check':'raw_request_to009_warm_mean','plan':m,'max_abs_error_s':err,'status':'PASS'})
    original=next(x for x in breaks if x['model']==model and x['plan']=='BM25-R100');assert math.ceil((ocr+bm)/delta)==int(original['conditional_break_even_queries'])
    for scenario,f,charge_index,resident in scenarios:
        sf=0. if resident else base+charge_index*index
        sr=0. if resident else startup+charge_index*index+f*ocr
        extra=sr-sf;nstar=math.ceil(extra/delta) if delta>0 else None
        bns=np.maximum(0.,extra/bd[bd>0])
        lo,hi=np.quantile(bns,[.025,.975])
        reuse.append({'model':model,'scenario':scenario,'OCR_allocation_fraction':f,'visual_index_charged':charge_index,'FullV_setup_s':sf,'R100_setup_s':sr,'incremental_R100_setup_s':extra,'warm_margin_s':delta,'conditional_break_even_N':nstar,'break_even_warm_only_95_low_N':math.ceil(lo),'break_even_warm_only_95_high_N':math.ceil(hi),'scope':'conditional component accounting; setup uncertainty NOT included; fraction0 prepaid, fraction.5 allocation assumption, not free OCR','status':'EXPLORATORY'})
        for N in [100,1000,10000,100000]:
            cf=sf+N*u[0];cr=sr+N*u[1];ratio=cr/cf;bdiff=(sr-sf)+N*(bt[:,1]-bt[:,0]);bratio=(sr+N*bt[:,1])/(sf+N*bt[:,0]);dl,dh=np.quantile(bdiff,[.025,.975]);rl,rh=np.quantile(bratio,[.025,.975])
            row={'model':model,'scenario':scenario,'N':N,'FullV_total_wall_s':cf,'R100_total_wall_s':cr,'R100_over_FullV_ratio':ratio,'R100_minus_FullV_s':cr-cf,'difference_warm_only_95_low_s':dl,'difference_warm_only_95_high_s':dh,'ratio_warm_only_95_low':rl,'ratio_warm_only_95_high':rh,'FullV_setup_s':sf,'R100_setup_s':sr,'OCR_charged_once_s':f*ocr if not resident else 0.,'CPU_GPU_wall_addition':'separate resources; never summed','energy':'NOT_MEASURED','uncertainty':'source/process paired bootstrap 2000; conditional on singleton index/OCR/setup values','status':'EXPLORATORY'};out.append(row)
            if scenario=='OCR_EXCLUSIVE_NEW_INDEX' and N<=10000:
                for m,value in [('Full-Visual',cf),('BM25-R100',cr)]:
                    prior=next(x for x in amort if x['model']==model and x['plan']==m and x['scenario']=='PLAN_COMPONENT_FROM_SCRATCH' and int(x['N'])==N)
                    err=abs(value-float(prior['C_total_wall_s']));assert err<1e-9;checks.append({'model':model,'check':'009_lifecycle_exact_replay','plan':m,'N':N,'max_abs_error_s':err,'status':'PASS'})
    older=oldq if model=='ColQwen2' else oldp
    for j,m in enumerate(plans):
        row=next(x for x in older if x.get('method',x.get('plan'))==m)
        keys=[k for k in row if 'E2E' in k or 'wall' in k]
        key=next((k for k in ['E2E_mean_ms','E2E_ms','warm_E2E_ms','E2E_s','E2E_wall_ms','E2E_wall_s'] if k in row),None)
        if key is None:
            raise ValueError('Unknown old warm field '+str(keys))
        value=float(row[key])/(1000 if key.endswith('_ms') else 1)
        cohorts.append({'model':model,'plan':m,'old_cohort':'007' if model=='ColQwen2' else '008','old_warm_s':value,'new009_warm_s':u[j],'old_over_new_descriptive_ratio':value/u[j],'comparability':'NOT_ESTABLISHED; no randomized cross-cohort A/A; GPU/host/cache/methodmix differences unresolved','allowed_use':'each cohort within its own model/pair; no algorithm speedup or cross-model latency claim','status':'INSUFFICIENT_CROSS_COHORT'})
C.table(C.R/'lifecycle_cost_replay.csv',out);C.table(C.R/'reuse_sensitivity.csv',reuse);C.table(C.R/'cohort_comparison.csv',cohorts);C.table(C.R/'replay_checks.csv',checks)
lines=['# A0 lifecycle and cohort audit','',f'Executed {C.stamp()}; CPU only, no retrieval or new labels.','', 'Ctotal = model/process startup + charged visual index + charged OCR once + optional BM25 startup + N × measured009 warm mean. OCR fraction0 is prepaid/marginal accounting, not measured free work. Shared fraction.5 is an allocation sensitivity; total OCR remains407.725s. Existing index removes build only, not process/model/residency startup. Already-resident removes only explicitly prepaid setup.','', 'Unprofiled warm009 has3 independent processes/model and160 queries/process/plan; 2000 source/process bootstrap replicates are not new labels. Index build and reused OCR each lack independent repeats. Intervals cover warm sampling only. No physical storage-cold/fsync/complete energy/network/fullbusiness ROI claim. CPU core-seconds and GPU active seconds remain separate from wall.','', 'Old007/008 absolute warm means differ markedly from009. No cross-cohort equivalence experiment was conducted. Primary QP quality/paired cost contrast stays in007; lifecycle projection uses009 warm consistently. Oldwarm+newsetup mixing is disallowed for main results. No cross-model latency superiority follows.','', '| Model | exclusive OCR newindex ratio N10k | ratio N100k | conditional crossing | prepaidOCR crossing |','|---|---:|---:|---:|---:|']
for model in ['ColQwen2','ColPali']:
    a=[x for x in out if x['model']==model and x['scenario']=='OCR_EXCLUSIVE_NEW_INDEX'];r=next(x for x in reuse if x['model']==model and x['scenario']=='OCR_EXCLUSIVE_NEW_INDEX');p=next(x for x in reuse if x['model']==model and x['scenario']=='OCR_PREPAID_NEW_INDEX');lines.append(f"|{model}|{a[2]['R100_over_FullV_ratio']:.6f}|{a[3]['R100_over_FullV_ratio']:.6f}|{r['conditional_break_even_N']}|{p['conditional_break_even_N']}|")
lines+=['','Claims: lifecycleTableVIII and conditional costFigure; supported by raw009 request/cohort manifests. No independent calibration, deployment ROI, or cross-model speed claim.']
C.prose(C.R/'cohort_consistency_audit.md','\n'.join(lines)+'\n')
C.prose(C.R/'cost_claim_to_source.md','# Claim-to-source\n\n1.60/1.55: lifecycle_cost_replay.csv OCR_EXCLUSIVE_NEW_INDEX N10000, exact009amortization replay. 77k/88k: reuse_sensitivity.csv same scenario, positive within009 margin. Existing/shared sensitivity: all rows retained, no businessN chosen. New100k is CPU projection, not100k measured independent requests. Raw phase costs/units/model pins remain frozen009. Input hashes: stageinputledger and inventory. Crosscohort speedup: unsupported, explicitly INSUFFICIENT.\n')
C.prose(C.R/'A0_decision.md','\n'.join(lines)+'\n\nPASS arithmetic and provenance; lifecycle comparison conditional; cross-cohort causality INSUFFICIENT; no GPU required.\n')
facts='; '.join(f"{x['model']}:exclusive OCR break-even={x['conditional_break_even_N']},95%warm-only[{x['break_even_warm_only_95_low_N']},{x['break_even_warm_only_95_high_N']}]" for x in reuse if x['scenario']=='OCR_EXCLUSIVE_NEW_INDEX')
C.finish('A0','PASS_CPU_REPLAY__CONDITIONAL_LIFECYCLE',facts,'保留1.60/1.55和77k/88k的独占OCR条件；补齐N100k、共享OCR/已有索引敏感性；新旧计时不支持加速归因，不新增GPU实验。')

