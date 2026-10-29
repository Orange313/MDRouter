import common as C
import collections,json,sys,shutil
import numpy as np
C.start('T4');rows=[];cross=[];gaps=[]
old=C.load('reports/icde_eab_main_006/X2/rebuild_002/eab_main_table_dev.csv','HISTORICAL_ACTUAL_OR_EXPLICIT_COMPOSED')
for r in old:rows.append({'dataset':r['dataset'],'retriever':r['retriever'],'method':r['action_policy'],'queries':r['N_queries'],'nDCG10':r['nDCG@10'],'mean_r':r['r_pos_mean'],'E2E_ms':r['wall_E2E_mean_ms'],'GPU_active_s':r['GPU_active_s'],'CPU_ms':r['CPU_ms'],'gain_pp':r['risk_matched_cost_gain_pp'],'paired95_low_pp':r['paired_family_CI_low_pp'],'paired95_high_pp':r['paired_family_CI_high_pp'],'cost_scope':r['cost_scope'],'evidence_level':'EXPOSED_DEV_EXPLORATORY','status':r['status'],'baseline_fidelity':r['baseline_fidelity']})
bio=C.load(C.REPORT/'T2/summary_001/biomedical_main_table.csv');match=C.load(C.REPORT/'T2/summary_001/biomedical_matched_risk_analysis.csv')
for r in bio:
    m=next(x for x in match if x['method']==r['method'] and x['grouping']=='source_group');rows.append({'dataset':'Biomedical','retriever':'ColQwen2-v1.0','method':r['method'],'queries':160,'nDCG10':r['nDCG10'],'mean_r':r['mean_r'],'E2E_ms':r['E2E_mean_ms'],'GPU_active_s':r['GPU_active_s'],'CPU_ms':r['CPU_process_ms'],'gain_pp':m['increment_vs_envelope_pp_of_FullV'],'paired95_low_pp':m['paired95_low_pp'],'paired95_high_pp':m['paired95_high_pp'],'cost_scope':r['cost_scope'],'evidence_level':r['evidence_level'],'status':r['status'],'baseline_fidelity':r['baseline_fidelity']})
for r in C.load(C.REPORT/'T3/pilot_summary_001/cross_retriever_main_table.csv'):
    cross.append({'dataset':r['dataset'],'retriever':r['retriever'],'method':r['plan'],'queries':r['queries'],'nDCG10':r['nDCG10'],'mean_r':r['mean_r'],'E2E_ms':r['warm_E2E_ms'],'GPU_active_s':r['complete_path_GPU_s'],'CPU_ms':r['CPU_ms'],'cost_scope':'NEW_ACTUAL_COMPLETE_WARM_PATH','evidence_level':'FROZEN_PILOT_DEV12_PER_DOMAIN','status':r['status'],'baseline_fidelity':'STATIC_OWN_MODEL_EXACT_MAXSIM'})
optional=C.REPORT/'T3/expansion_summary_001/cross_retriever_main_table.csv'
if optional.exists():
    cross=[]
    for r in C.load(optional):cross.append({'dataset':r['dataset'],'retriever':r['retriever'],'method':r['plan'],'queries':r['queries'],'nDCG10':r['nDCG10'],'mean_r':r['mean_r'],'E2E_ms':r['warm_E2E_ms'],'GPU_active_s':r['complete_path_GPU_s'],'CPU_ms':r['CPU_ms'],'cost_scope':'NEW_ACTUAL_COMPLETE_WARM_PATH','evidence_level':'FOUR_DOMAIN_DEV_EXPLORATORY','status':r['status'],'baseline_fidelity':'STATIC_OWN_MODEL_EXACT_MAXSIM'})
for r in old:
    if r['action_policy'] in C.PLANS:cross.append({'dataset':r['dataset'],'retriever':r['retriever'],'method':r['action_policy'],'queries':r['N_queries'],'nDCG10':r['nDCG@10'],'mean_r':r['r_pos_mean'],'E2E_ms':r['wall_E2E_mean_ms'],'GPU_active_s':r['GPU_active_s'],'CPU_ms':r['CPU_ms'],'cost_scope':'HISTORICAL_ACTUAL_COMPLETE_PATH_DIFFERENT_COHORT_FROM_COLPALI','evidence_level':'EXPOSED_DEV_EXPLORATORY','status':r['status'],'baseline_fidelity':'STATIC_OWN_MODEL_EXACT_MAXSIM'})
gaps += [{'dataset':x['dataset'],'evidence':'EXPOSED_DEV','same_catalog':x['catalog'],'risk':x['risk_budget'],'oracle_cost_ms':x['oracle_cost_ms'],'static_random_cost_ms':x['static_random_cost_ms'],'oracle_headroom_pp':x['oracle_headroom_pp'],'status':'GOLD_OFFLINE_NOT_DEPLOYABLE'} for x in C.load('reports/icde_eab_main_006/X2/rebuild_002/oracle_learnability_gap.csv')]
gaps += [{'dataset':'Biomedical','evidence':'NEW_PREREGISTERED_DESCRIPTIVE','same_catalog':x['same_catalog'],'risk':x['risk'],'oracle_cost_ms':x['oracle_cost_ms'],'static_random_cost_ms':x['static_random_cost_ms'],'oracle_headroom_pp':x['oracle_headroom_pp'],'status':x['status']} for x in C.load(C.REPORT/'T2/summary_001/biomedical_oracle_gap.csv')]

for r in rows:r['reference_implementation']='EXACT_VALID_TOKEN_MAXSIM_004' if r['dataset']=='Biomedical' else 'HISTORICAL_FROZEN_LOCAL_005; preserved not replaced'
for r in cross:r['reference_implementation']='EXACT_VALID_TOKEN_MAXSIM_004' if r['retriever'].startswith('ColPali') else 'HISTORICAL_FROZEN_LOCAL_005; own reference'
parity=[]
for p in (C.REPORT/'correctness_supplement').glob('*/primary_scorer_parity.csv'):parity+=C.load(p,'POST_DISCOVERY_DEV_SCORER_PARITY_NO_MODEL_SELECTION')
if parity:C.table(C.R/'primary_reference_parity.csv',parity)
bm={x['method']:x for x in bio};derived=[]
for r in C.load(C.REPORT/'T2/summary_001/biomedical_quality_cost_frontier.csv'):
    w=json.loads(r['allocation']);derived.append({**r,'implied_nDCG10':sum(float(bm[p]['nDCG10'])*v for p,v in zip(C.PLANS,w)),'implied_harm_rate':sum(float(bm[p]['harm_rate'])*v for p,v in zip(C.PLANS,w)),'quality_scope':'expectation of retrospective static random mix, not a measured deployed policy'})
C.table(C.R/'biomedical_quality_cost_frontier.csv',derived)
pq=[]
for x in bio:
    dominators=[s['method'] for s in bio if float(s['E2E_mean_ms'])<=float(x['E2E_mean_ms']) and float(s['nDCG10'])>=float(x['nDCG10']) and (float(s['E2E_mean_ms'])<float(x['E2E_mean_ms']) or float(s['nDCG10'])>float(x['nDCG10']))]
    pq.append({**x,'quality_cost_dominated_by':dominators,'quality_cost_frontier':not dominators,'observed_r_le_0p05':float(x['mean_r'])<=.05,'certification':'NOT_CERTIFIED regardless observed mean'})
C.table(C.R/'biomedical_measured_plan_pareto.csv',pq)

C.table(C.R/'eab_main_results.csv',rows);C.table(C.R/'eab_cross_domain_cross_retriever.csv',cross);C.table(C.R/'eab_oracle_gap_decomposition.csv',gaps)
C.prose(C.R/'eab_baseline_fidelity.md','# 基线与证据边界\n\n旧4域164 Dev：32行历史完整硬件路径，8行RR adapted为组合成本；新Biomedical160：冻结Q/QP、hard/soft适配基线全部新完整路径，统计描述性、预训练重叠UNKNOWN。原RetrievalRouter为LoRA+KL/五动作，与本地Q150/Ridge/三动作不同，明确ADAPTED_NOT_FAITHFUL。Oracle和测试上拟合的随机包络仅离线回顾性。ColPali使用自己的参考和新GPU测量，pilot不是完整四域重做。无新K1，无严格AA通过的Serving结论，无完整业务净收益。\n')
claims=[{'claim':'F1 physical plan quality-cost frontier','evidence':'old164 exposed Dev + Biomedical160 preregistered fixed-policy actual measurements','status':'PASS_MEASURED_DESCRIPTIVE','scope':'warm residency; no full business benefit'}, {'claim':'F2 learned routing stable information gain over strongest static/random','evidence':'paired cluster intervals in main tables; negative results retained','status':'INSUFFICIENT_UNLESS_NEW_TABLE_SUPPORTS','scope':'no new tuning; pretraining UNKNOWN'}, {'claim':'F3 cross-retriever generality','evidence':'frozen ColPali HR12/Industrial12 GPU pilot plus any completed expansion','status':'LIMITED_PILOT_EVIDENCE','scope':'own model reference; no copied learned risk'}, {'claim':'formal risk certification for new policies','evidence':'none; sealedCal/Eval0','status':'NOT_CERTIFIED'}, {'claim':'Serving superior performance','evidence':'prior strict AA not established','status':'INSUFFICIENT'}, {'claim':'original RetrievalRouter reproduction','evidence':'only adapted hard/soft controls','status':'NOT_RUN'}]
if optional.exists():
    claims[2].update({'evidence':'ColPali frozen24 pilot + full164 Dev static execution; primary historical cohort separately labeled','status':'PASS_CROSS_RETRIEVER_STATIC_DEV_ONLY'})
claims[1]['status']='INSUFFICIENT_STABLE_MULTI_DOMAIN_ROUTING_GAIN'
C.table(C.R/'claim_to_evidence_matrix.csv',claims);C.prose(C.R/'eab_claim_to_evidence_matrix.md','# Claim-to-evidence\n\n'+'\n'.join(f'- {x["claim"]}: **{x["status"]}** — {x["evidence"]}.' for x in claims)+'\n')
from figures_002 import draw
draw(C.R/'figures',rows,cross,gaps)
C.finish('T4','PASS_EVIDENCE_TABLES_WITH_EXPLICIT_LIMITS',f'{len(rows)}主表行、{len(cross)}新ColPali行、{len(gaps)}Oracle gap行、3幅脚本生成PDF图；旧Dev/新benchmark、真实/组合成本逐行分列。','新结果进入论文，保留负结果与未完成门槛，不作统一IID或部署保证主张。')
