"""Direct paired diagnostics on frozen007 outputs only; no new policy fitting."""
import common as C
import sys,collections,json
import numpy as np
sys.path.insert(1,str(C.ROOT/'scripts/icde_eab_validation_005'))
import frontier_math as F
C.start('L0')
oldA=C.ROOT/'artifacts/icde_eab_independent_007';oldR=C.ROOT/'reports/icde_eab_independent_007'
assert C.sha(oldA/'manifest.json')=='5c7f55e8745b58c1f04f1bb120d253cccfe2889b76c0ec010b262ab943bea22d'
inventory=C.load(oldA/'hash_inventory.json','CLOSED_007_INVENTORY')
def bound(p,role):
    assert C.sha(p)==inventory['files'][str(p.relative_to(C.ROOT))],p
    return C.load(p,role)
proto=bound(oldA/'T1/biomedical_frozen_protocol.json','FROZEN_007_PROTOCOL')
raw=[];quality=None
for p in range(3):
    a=bound(oldR/f'T2/process_{p}_001/quality.csv','ALREADY_EXECUTED_007_QUALITY_NOT_NEW_LABEL_ACCESS')
    if quality is None:quality=a
    else:assert [(x['query_id'],x['method'],x['r'],x['nDCG10']) for x in quality]==[(x['query_id'],x['method'],x['r'],x['nDCG10']) for x in a]
    raw+=bound(oldR/f'T2/process_{p}_001/requests.csv','ACTUAL_FROZEN_COMPLETE_PATH_TIMINGS')
    g=bound(oldA/f'T2/process_{p}_001/resource_guard.json','007_RESOURCE_GUARD');assert g['valid'] and not g['foreign_pids'] and not g['errors']
methods=C.PLANS+['Q_regret','QP_regret','RandomMix','RR_hard','RR_soft'];ids=proto['query_ids'];N=len(ids);assert N==160 and len(raw)==12960
qm={(int(x['query_id']),x['method']):x for x in quality};bucket=collections.defaultdict(list)
for x in raw:bucket[int(x['query_id']),int(x['process']),x['method']].append(x)
assert all(len(bucket[q,p,m])==3 for q in ids for p in range(3) for m in methods)
T=np.array([[[np.mean([float(x['E2E_s']) for x in bucket[q,p,m]])*1000 for m in methods] for p in range(3)] for q in ids])
R=np.array([[float(qm[q,m]['r']) for m in methods] for q in ids]);U=np.array([[float(qm[q,m]['nDCG10']) for m in methods] for q in ids]);qi=methods.index('QP_regret');controls=['BM25-R100','BM25-R50','Q_regret'];B=2000
summary=[];draws=[];per=[];loso=[];severity=[];groups_per={}
for grouping,seed_offset in [('source_group',1),('family',0)]:
    groups=np.array([qm[q,'Full-Visual'][grouping] for q in ids]);assert len(set(groups))==(27 if grouping=='source_group' else 157)
    indices=F.boots(groups,B,int(proto['seed'])+seed_offset);processes=np.random.default_rng(int(proto['seed'])+43).integers(0,3,(B,3))
    br=np.array([R[ix].mean(0) for ix in indices]);bu=np.array([U[ix].mean(0) for ix in indices]);bt=np.array([T[ix][:,pi].mean((0,1)) for ix,pi in zip(indices,processes)])
    for ctrl in controls:
        j=methods.index(ctrl);dr=br[:,qi]-br[:,j];du=bu[:,qi]-bu[:,j];dt=bt[:,qi]-bt[:,j]
        for metric,point,arr,direction in [('positive_regret_delta',R[:,qi].mean()-R[:,j].mean(),dr,'negative_better'),('nDCG10_delta',U[:,qi].mean()-U[:,j].mean(),du,'positive_better'),('complete_E2E_ms_delta',T[:,:,qi].mean()-T[:,:,j].mean(),dt,'negative_better')]:
            lo,hi=np.quantile(arr,[.025,.975]);blo,bhi=np.quantile(arr,[.0125,.9875])
            summary.append({'contrast':'QP_minus_'+ctrl,'control':ctrl,'grouping':grouping,'queries':N,'groups':len(set(groups)),'metric':metric,'estimate':point,'low95':lo,'high95':hi,'low97p5_for_two_endpoint_sensitivity':blo,'high97p5_for_two_endpoint_sensitivity':bhi,'favorable_direction':direction,'bootstrap_draw_fraction_QP_lower_r_and_lower_E2E':float(np.mean((dr<0)&(dt<0))),'bootstrap_repetitions':B,'scope':'POST_HOC_PAIRED_DESCRIPTIVE_POINTWISE;97.5% percentile sensitivity only adjusts the two r/E2E endpoints per comparator, not all analyses; no delta-PAC claim'})
        for b in range(B):draws.append({'grouping':grouping,'draw':b,'control':ctrl,'r_delta':dr[b],'nDCG_delta':du[b],'E2E_ms_delta':dt[b],'scope':'bootstrap draws are not independent queries or frequentist hypothesis-test p-values'})
    groups_per[grouping]=groups
source=groups_per['source_group']
for src in sorted(set(source)):
    ix=np.flatnonzero(source!=src)
    for ctrl in controls:
        j=methods.index(ctrl);dr=R[ix,qi].mean()-R[ix,j].mean();du=U[ix,qi].mean()-U[ix,j].mean();dt=T[ix,:,qi].mean()-T[ix,:,j].mean()
        loso.append({'excluded_source':src,'remaining_queries':len(ix),'control':ctrl,'r_delta':dr,'nDCG10_delta':du,'E2E_ms_delta':dt,'QP_lower_r_and_lower_E2E':bool(dr<0 and dt<0),'scope':'leave-one-source-out sensitivity, not independent replications'})
for i,q in enumerate(ids):
    for j,m in enumerate(methods):
        x=qm[q,m];r=R[i,j];per.append({'query_id':q,'family':x['family'],'source_group':x['source_group'],'method':m,'action':x['action'],'r':r,'nDCG10':U[i,j],'r_bar':float(x['r_bar']),'signed_gap':float(x['signed_gap']),'E2E_ms':T[i,:,j].mean(),'harm':r>1e-10,'FalseSafe_diagnostic_r_gt_0p05_nonreference':x['action']!='Full-Visual' and r>.05,'high_harm_r_ge_0p1':r>=.1,'extreme_harm_r_ge_0p5':r>=.5,'unit':'one existing query; repeated timings not new samples'})
for src in sorted(set(source)):
    ix=np.flatnonzero(source==src)
    for j,m in enumerate(methods):
        rr=R[ix,j];severity.append({'source_group':src,'method':m,'queries':len(ix),'harm_queries':int(np.sum(rr>1e-10)),'FalseSafe_queries':sum(qm[ids[i],m]['action']!='Full-Visual' and R[i,j]>.05 for i in ix),'high_harm_ge_0p1':int(np.sum(rr>=.1)),'extreme_harm_ge_0p5':int(np.sum(rr>=.5)),'mean_r':rr.mean(),'regret_mass':rr.sum(),'share_of_total_method_regret':rr.sum()/R[:,j].sum() if R[:,j].sum() else None})
# Frozen RandomMix action identity is rechecked; it is not reoptimized at a test risk.
import hashlib
for q in ids:
    v=int.from_bytes(hashlib.sha256(f'{proto["seed"]}:Biomedical:{q}:RandomMix'.encode()).digest()[:8],'big')/2**64
    a=C.PLANS[min(int(np.searchsorted(np.cumsum(proto['random_mix_probabilities']),v,side='right')),3)]
    assert a==qm[q,'RandomMix']['action']
budget=[]
for target,label in [(.05,'original_epsilon_0p05'),(float(R[:,qi].mean()),'observed_QP_mean_r')]:
    ev,w=F.envelope(R[:,:4].mean(0),T[:,:,:4].mean((0,1)),target)
    for j,m in enumerate(methods):budget.append({'section':label,'risk_target':target,'method':m,'mean_r':R[:,j].mean(),'nDCG10':U[:,j].mean(),'complete_E2E_ms':T[:,:,j].mean(),'empirical_mean_within_slice':bool(R[:,j].mean()<=target+1e-12),'status':'ACTUAL_FROZEN_EXECUTED_POLICY_NOT_NEW_K1','allocation':None})
    budget.append({'section':label,'risk_target':target,'method':'posthoc_static_convex_envelope','mean_r':float(np.dot(w[0],R[:,:4].mean(0))),'nDCG10':float(np.dot(w[0],U[:,:4].mean(0))),'complete_E2E_ms':ev[0],'empirical_mean_within_slice':True,'status':'RETROSPECTIVE_IDEAL_EXPECTATION_NOT_EXECUTED_NOT_DEPLOYED_NOT_INDEPENDENTLY_VALIDATED','allocation':w[0].tolist()})
main=bound(oldR/'T4/rebuild_004/eab_main_results.csv','BASELINE_COST_FIDELITY_LEDGER');audit=[x for x in main if x['method'].startswith('RR')]
assert len(audit)==10
for m in ['RR_hard','RR_soft']:assert sum(x['method']==m for x in raw)==1440
for name,rows in [('paired_all_comparisons.csv',summary),('paired_qp_vs_r100.csv',[x for x in summary if x['control']=='BM25-R100']),('bootstrap_by_source.csv',[x for x in draws if x['grouping']=='source_group']),('query_family_sensitivity.csv',[x for x in summary if x['grouping']=='family']),('leave_one_source_out.csv',loso),('high_harm_false_safe.csv',per),('false_safe_by_source.csv',severity),('posthoc_envelope_vs_frozen_random.csv',budget),('fidelity_cost_source_audit.csv',audit)]:C.table(C.R/name,rows)
direct=[x for x in summary if x['grouping']=='source_group' and x['control']=='BM25-R100'];r=next(x for x in direct if x['metric']=='positive_regret_delta');t=next(x for x in direct if x['metric']=='complete_E2E_ms_delta');u=next(x for x in direct if x['metric']=='nDCG10_delta');stable=sum(x['QP_lower_r_and_lower_E2E'] for x in loso if x['control']=='BM25-R100')
facts='QP−R100: r={:.6f} 95%[{:.6f},{:.6f}], nDCG={:.6f} [{:.6f},{:.6f}], E2E={:.3f}ms [{:.3f},{:.3f}]; LOSO双均值优势{}/27。'.format(r['estimate'],r['low95'],r['high95'],u['estimate'],u['low95'],u['high95'],t['estimate'],t['low95'],t['high95'],stable)
verdict='POINTWISE_SOURCE_INTERVALS_SUPPORT_BOTH_ENDPOINT_DIRECTIONS_DESCRIPTIVELY' if r['high95']<0 and t['high95']<0 else 'POINT_ESTIMATE_DOMINANCE_NOT_STABLE_IN_BOTH_SOURCE_INTERVALS'
C.prose(C.R/'L0_decision.md','# '+verdict+'\n\n'+facts+'\n\n固定模型与已有结果的事后直接配对诊断；来源组和查询族/进程bootstrap不是生产IID、正式检验或δ-PAC认证。点对点优于R100与不及结果后理想随机包络是不同问题，必须同时保留。False-Safe按非参考动作r>.05定义，不能作为逐查询合同违约；原合同是均值风险。97.5%敏感性仅为每一比较的r/E2E两个端点，不覆盖所有比较。\n\n支持论文RQ2直接比较表；随机基线区别见random_baseline_fidelity。所有160条/27来源保留。\n')
C.prose(C.R/'loss_budget_audit.md','# Original loss contract unchanged\n\nε=.05、δ=.01保持不变。表中每个方法的真实经验均值与两个截面详见posthoc_envelope_vs_frozen_random.csv；Q/QP/R100均>.05。FullV风险0是相对参考定义，不代表给新策略签发K1。新K1 NOT_CERTIFIED；Cal/Eval、旧B-L/test/CS读取0。Bootstrap置信区间不替代原K1证书。不存在把重复12960请求当成独立标签的n扩增。\n')
C.prose(C.R/'random_baseline_fidelity.md','# Frozen execution versus retrospective envelope\n\n1. Frozen RandomMix:007评价前锁定的Pharma概率、seed及SHA256按query分配；本次160个动作全部重新核对一致。原9方法实验的1440次真实路径计时/方法，未新选概率。其风险点不同于QP，不能冒充同风险对照。\n2. posthoc_static_convex_envelope:在已观察到的测试风险/成本上解四静态计划凸混合，只有理想期望，不是已执行部署或独立验证的策略。风险=.05和QP经验风险两个截面全部保留；不得将它改称冻结RandomMix。\n\nBio RR-hard/soft各1440条完整真实请求；旧四Dev的8条RR结果是分项成本组合，没有完整路径尾延迟。两者逐行分级见fidelity_cost_source_audit.csv。本地Q150/Ridge三动作hard/soft target不是官方RetrievalRouter Qwen+LoRA+KL/五动作复现，不简称为官方模型。\n')
C.finish('L0','PASS_DIRECT_PAIRED_DESCRIPTIVE_'+verdict,facts,'论文分别报告确定性R100、实际冻结RandomMix和事后理想包络；不重新选择策略、混合概率或风险参数，不申请新K1。')
