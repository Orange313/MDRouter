"""Requested descriptive information/overhead analysis, explicitly post-hoc."""
import common as C
import numpy as np,sys,collections
sys.path.insert(1,str(C.ROOT/'scripts/icde_eab_validation_005'))
import frontier_math as F
C.start('T4_preparation/probe_diagnostic_001');per=C.load(C.REPORT/'T2/summary_001/biomedical_per_query.csv');raw=[]
for p in range(3):raw+=C.load(C.REPORT/f'T2/process_{p}_001/requests.csv')
ids=sorted({int(r['query_id']) for r in per});methods=['Q_regret','QP_regret'];by={(int(r['query_id']),r['method']):r for r in per};rr=collections.defaultdict(list)
for r in raw:
    if r['method'] in methods:rr[int(r['query_id']),int(r['process']),r['method']].append(r)
Y=np.array([[float(by[q,m]['r']) for m in methods] for q in ids]);U=np.array([[float(by[q,m]['nDCG10']) for m in methods] for q in ids]);T=np.array([[[np.mean([float(r['E2E_s']) for r in rr[q,p,m]]) for m in methods] for p in range(3)] for q in ids]);results=[]
for gi,grouping in enumerate(['family','source_group']):
    g=np.array([by[q,'Q_regret'][grouping] for q in ids]);idx=F.boots(g,2000,C.SEED+gi);proc=np.random.default_rng(C.SEED+99).integers(0,3,(2000,3));dy=np.array([(Y[i,1]-Y[i,0]).mean() for i in idx]);du=np.array([(U[i,1]-U[i,0]).mean() for i in idx]);dt=np.array([1000*(T[i][:,p,1]-T[i][:,p,0]).mean() for i,p in zip(idx,proc)])
    for metric,point,values in [('QP_minus_Q_r',(Y[:,1]-Y[:,0]).mean(),dy),('QP_minus_Q_nDCG',(U[:,1]-U[:,0]).mean(),du),('QP_minus_Q_E2E_ms',1000*(T[:,:,1]-T[:,:,0]).mean(),dt)]:lo,hi=np.quantile(values,[.025,.975]);results.append({'grouping':grouping,'metric':metric,'estimate':point,'low95':lo,'high95':hi,'status':'POST_HOC_PAIRED_DESCRIPTIVE_NOT_CERTIFIED'})
match=C.load(C.REPORT/'T2/summary_001/biomedical_matched_risk_analysis.csv');tax=[]
for m in methods:
    r=next(x for x in match if x['method']==m and x['grouping']=='source_group');a=[x for x in raw if x['method']==m];over=1000*np.mean([sum(float(x[k]) for k in ['Q_wall_s','Probe_extra_wall_s','Probe_statistics_wall_s','planner_wall_s']) for x in a]);probe=1000*np.mean([float(x['Probe_extra_wall_s'])+float(x['Probe_statistics_wall_s']) for x in a]);cost=float(r['actual_policy_E2E_ms']);en=float(r['static_random_envelope_ms']);tax.append({'method':m,'actual_E2E_ms':cost,'retrospective_same_risk_envelope_ms':en,'actual_cost_excess_ms':cost-en,'measured_Q_probe_planner_ms':over,'measured_extra_probe_ms':probe,'excess_even_if_all_measured_information_overhead_free_ms':cost-over-en,'maximum_extra_probe_ms_at_break_even_given_frozen_actions':en-(cost-probe),'scope':'counterfactual accounting holding observed actions fixed; not rerouting or a production causal effect'})
C.table(C.R/'paired_probe_increment.csv',results);C.table(C.R/'probe_break_even.csv',tax);C.prose(C.R/'probe_increment_report.md','# POST_HOC｜探针信息与成本诊断\n\n直接QP−Q比较用同查询族/来源与进程配对重采样，2000次，逐点描述性；不是正式独立假设检验。两个冻结策略风险/成本不同，不能仅凭QP质量更高声称同风险优势。实际额外Probe与规划成本单列；零额外成本会计诊断固定已观察动作，不能视作新部署实验。\n\n'+str(results)+'\n\n'+str(tax)+'\n\n支撑论文F2的信息/成本分解；不支持所有探针无价值、因果机制或新K1。\n')
print(C.clean({'paired':results,'overhead':tax}))
