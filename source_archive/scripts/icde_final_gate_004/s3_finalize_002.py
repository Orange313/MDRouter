import common as C
import numpy as np,json,collections
C.A=C.ART/'S3';C.R=C.REPORT/'S3';C.STAGE='S3';__import__('sys').addaudithook(C.protected)
done=C.load(C.A/'analysis_complete.json');assert done['status']=='PASS_MINIMAL_DIAGNOSTIC';assert all(x['smaller_is_subset'] for x in done['nestedness']);cal=C.load(C.R/'risk_calibration_by_n.csv');comp=C.load(C.R/'subset_composition.csv');gains=C.load(C.R/'paired_learning_gain_matched_risk.csv');curves=C.load(C.R/'matched_risk_learning_curves.csv');summary=[]
for method in ['Q_regret','QP_regret']:
 for n in [20,40,60,81]:
  z=[r for r in cal if r['method']==method and int(r['Fit_n'])==n and float(r['kappa'])==10];bm=[r for r in z if r['head']=='BM25'];r100=[r for r in z if r['head']=='BM25-R100'];s=[r for r in curves if r['method']==method and int(r['Fit_n'])==n and float(r['kappa'])==10][0];summary.append({'method':method,'Fit_n':n,'kappa':10,'mean_r':float(s['mean_Dev_r']),'matched_risk_increment_pp':float(s['matched_risk_increment_pp']),'L2_by_seed':sorted(set(float(r['L2']) for r in z)),'BM25_R2_mean':np.mean([float(r['R2']) for r in bm]),'R100_R2_mean':np.mean([float(r['R2']) for r in r100]),'selected_bias_pred_minus_true':np.mean([float(r['selected_calibration_bias']) for r in r100]),'false_safe_n_mean':np.mean([float(r['false_safe_n']) for r in r100]),'scope':'fixed-kappa10 descriptive example; all7kappa retained'})
C.table(C.R/'attribution_summary_kappa10.csv',summary)
text='# INSUFFICIENT 因果归因；PASS 可复现的最小诊断\n\n'
text+='168组Q/QP既有预测从原Fit子集、原Train-only选择L2与同容量Ridge重新构造，逐值误差<1e-9。12组训练子集/9对相邻规模检查均为嵌套，三个seed配对同一Dev45/44候选族。训练/Dev角色不变，未看Cal/Eval。原E2是四动作学习；本轮实际serving是V/R100/BM25三动作，不能混同。\n\n'
text+='|方法|n|κ10 mean r|同风险静态随机增量pp|BM25 R²|R100 R²|选中风险预测偏差|False-Safe数(3seed均值)|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for r in summary:text+=f'|{r["method"]}|{r["Fit_n"]}|{r["mean_r"]:.5f}|{r["matched_risk_increment_pp"]:.2f}|{r["BM25_R2_mean"]:.3f}|{r["R100_R2_mean"]:.3f}|{r["selected_bias_pred_minus_true"]:.4f}|{r["false_safe_n_mean"]:.2f}|\n'
text+='\n**DATA_SHIFT_OR_LOW_POWER — INSUFFICIENT因果、存在观测线索。** 固定Dev45仅44候选族，query-IID不成立证据未补齐；三seed是训练情景而非三组独立测试。子集文档覆盖、平均relevant页数、BM25/R100非零/大伤害比例及queryID已在subset_composition.csv逐项保留。随着规模变化风险正例组成发生变化；这不能单独证明数据分布漂移是唯一原因。\n\n'
text+='**FEATURE_NOT_INFORMATIVE — 仅支持当前预测器未提取稳定增量。** 在固定容量/训练预算下，各头R²、Spearman、MAE与负预测率见完整risk_calibration_by_n.csv。负R²/宽区间说明当前廉价输入+Ridge不够可靠，不证明任何更大样本下这些特征都无信息。没有因此扩模型。\n\n'
text+='**POLICY_MISMATCH — 关联诊断。** 原选择按c+κ max(pred,0)作argmin，并无对新策略的正式风险校准；预测偏差在决策边界会转为错误安全降级。全7κ曲线与calibration_bins.csv明确区分预测改善和动作变化；κ10负例不推广到所有工作点。κ20的流式策略沿用旧冻结，不在本轮重选。False-Safe贡献及L2变化可解释关联，但缺少独立干预，最终因果为INCONCLUSIVE。\n\n'
text+='**匹配风险的20→81比较（每点配对family+seed情景2000次，非同时/非独立评估）：**\n\n'
for g in gains:
 if float(g['kappa'])==10:text+=f'- {g["method"]}：同风险成本增量变化 {float(g["matched_risk_improvement81_minus20_pp"]):.2f}pp，条件paired95% [{float(g["paired95_matched_improvement_low_pp"]):.2f},{float(g["paired95_matched_improvement_high_pp"]):.2f}]；mean r变化{float(g["risk_change81_minus20"]):.5f}。\n'
text+='\n数据中human/sdg混合的来源与难度分解另见source_difficulty_sensitivity.md和Fit_Dev_source_difficulty.csv；official SDG来源不等于本项目独立人工核验标签。所有κ、n、seed均保留。静态随机包络每个bootstrap重优化是描述性对照；不可当可部署Gold Oracle或独立统计显著性。\n\n**结论—决策—阻塞：** 未显示增加Fit20→81稳定改善同风险真实成本的证据；停止以扩大容量作为后续路线。继续小目录与静态/随机基线的系统性分析。有效独立数据与真正held-out证据缺失；因果归因保持INCONCLUSIVE。\n\n**论文证据：** 支撑Learning/Information Limits中训练规模与校准偏差表及全κ学习图；不支撑独立泛化/已认证新路由/特征本质无用等主张。\n'
C.prose(C.R/'learning_curve_attribution.md',text);C.prose(C.R/'S3_decision.md',text)
facts='PASS 168组既有预测重现、12训练子集完全嵌套；INSUFFICIENT稳定增样收益与因果归因。κ10的20→81同风险成本增量变化Q−28.15pp[−48.22,−5.61]、QP−26.68pp[−46.46,−2.52]（固定Dev条件区间）；全7κ/3seed负结果均保留。'
C.finish('PASS_DIAGNOSTIC_INSUFFICIENT_CAUSAL_ATTRIBUTION',facts,'不增加容量；DATA_SHIFT_OR_LOW_POWER、FEATURE_NOT_INFORMATIVE、POLICY_MISMATCH仅作有限关联诊断，最终因果INCONCLUSIVE；进入S4证据/论文决策。')

