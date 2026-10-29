import common as C
from pathlib import Path
import shutil,json,subprocess,re,os

def _contains_private_marker(text):
    """Reject private endpoint literals and personal home paths; documentation IPs are negative-test sentinels."""
    import ipaddress, re
    for token in re.findall(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])", text):
        try:
            address = ipaddress.ip_address(token)
        except ValueError:
            continue
        if address.is_private and not address.is_loopback:
            return True
    normalized = text.replace(chr(92), '/').replace('//', '/')
    return bool(re.search(r"(?:[A-Za-z]:/Users/|/(?:home|Users)/)[A-Za-z0-9_.-]+", normalized))

C.start('X4');S=Path(__file__).parent
paper=C.R/'paper_source_R4';shutil.copytree(S/'paper_expanded_001',paper)
oldpaper=C.ROOT/'evidence/paper/icde_eab_validation_005/paper_source_005'
for name,h in [('IEEEtran.cls','c972aca108fda004c3514d63658e02816da2e54d9a1451e870b9bd970e003f55'),('IEEEtran.bst','d83aa3c9b47fc120bb3d9d8e476216cb889c738f72ec950beeac3661c45ded24')]:
    assert C.sha(oldpaper/name)==h;shutil.copyfile(oldpaper/name,paper/name)
build=json.loads((S/'paper_receipts/build_R4.json').read_text());assert C.sha(paper/'main_R4.pdf')==build['pdf']['contentHash'];C.write(C.A/'paper_build_R4.json',build)
repo=C.A/'paper_repository_R4';shutil.copytree(paper,repo)
def git(args):return subprocess.check_output(['git','-C',str(repo)]+args,text=True).strip()
git(['init','--quiet']);git(['symbolic-ref','HEAD','refs/heads/codex/eab006-paper-r4']);git(['add','.']);git(['-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','--quiet','-m','Archive inspected Orange Paper R4 and generated evidence'])
bundle=C.A/'paper_R4.bundle';git(['bundle','create',str(bundle),'--all']);C.write(C.A/'paper_commit.json',{'private_paper_commit':git(['rev-parse','HEAD']),'bundle_SHA256':C.sha(bundle),'authoritative_Paper_revision':build['revisionId'],'authoritative_Paper_build':build['build']['id'],'source_attribution':'Orange legacy connector ChatGPT; discovered and stopped, no historical author rewritten'})
release=C.R/'release_candidate_001';release.mkdir();(release/'scripts').mkdir()
old=C.ROOT/'reports/icde_eab_validation_005/A4/reproduction_002/release_candidate'
for p in (old/'scripts').glob('*.py'):
    text=p.read_text();text=text.replace("ROOT=pathlib.Path('/path/to/project')","ROOT=pathlib.Path(os.environ.get('MDROUTER_ROOT','.')).resolve()")
    C.prose(release/'scripts'/p.name,text)
shutil.copytree(old/'legal_inputs',release/'legal_inputs')
for name in ['requirements_recorded.txt','environment_versions.json','LICENSES_AND_DATA.md']:shutil.copyfile(old/name,release/name)
C.prose(release/'README.md','# Numerical replay candidate, not a public release\n\nRun from a new checkout with recorded NumPy/SciPy/Matplotlib environment:\n\n```sh\npython scripts/reproduce_a1_001.py --input-root legal_inputs --output NEW_OUTPUT\n```\n\nCPU replay of archived lawful Dev measurements. It does not rerun GPU hardware, retrain a router, certify risk, or unseal labels. No image corpus/visual weights are redistributed. Seven scientific tables must match the reference hashes. Real hardware reproduction and public anonymous accessibility remain NOT_RUN. Data licensing and author review are required before publication.\n')
env=os.environ.copy();env['PYTHONPATH']=str(C.ROOT/'cache/report_dependencies/prereg_v2_round_001');env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
replay=C.A/'clean_checkout_replay_001'
with (C.ROOT/'artifacts/icde_eab_main_006_logs_001/X4_CPU_replay_001.log').open('x') as log:
    subprocess.run([str(C.ROOT/'.venv/bin/python'),str(release/'scripts/reproduce_a1_001.py'),'--input-root',str(release/'legal_inputs'),'--output',str(replay)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
checks=[]
for name in ['matched_risk_cost_master.csv','static_frontiers.csv','oracle_gap_decomposition.csv','probe_break_even.csv','per_query_routing_diagnostics.csv','false_safe_cases.csv','historical_numeric_verification.csv']:
    actual=replay/'reports/icde_eab_validation_005/A1'/name;ref=C.ROOT/'reports/icde_eab_validation_005/A1'/name
    checks.append({'table':name,'replayed_SHA256':C.sha(actual),'reference_SHA256':C.sha(ref),'status':'PASS' if C.sha(actual)==C.sha(ref) else 'FAIL'})
assert all(x['status']=='PASS' for x in checks);C.table(C.R/'clean_replay_hash_checks.csv',checks)
privacy=[]
for p in release.rglob('*'):
    if p.is_file() and p.suffix in ['.py','.md','.json','.txt','.csv']:
        t=p.read_text(encoding='utf8',errors='replace')
        if _contains_private_marker(t): privacy.append(str(p.relative_to(release)))
C.write(C.A/'release_privacy_scan.json',{'private_path_hits':privacy,'status':'PASS_NO_KNOWN_PRIVATE_SERVER_USER_PATHS' if not privacy else 'FAIL_REQUIRES_SANITIZATION','public_accessibility':'NOT_RUN','full_GPU_clean_reproduction':'NOT_RUN'})
shutil.copyfile(S/'paper_expand_001.py',C.R/'paper_expand_001.py');shutil.copyfile(S/'paper_assemble_001.py',C.R/'paper_assemble_001.py')
C.prose(C.R/'eab_manuscript_diff.md','# PASS｜检查过的Orange Paper R3 → R4\n\n8份TeX文件的实际unified diff见paper_source_R4/manuscript.diff。保留作者占位、EAB标题、现有Introduction/Related Work有效正文与引用；按八节补齐Testbed、Setup、Results、Lessons、Threats。移除已完成的可见红字大纲，原修订完整保留。未恢复待删除GPT联改稿。32行原硬件主表+8行适配表，2张原生矢量图由40行CSV生成。\n\n工作稿10页（含引用），不是提交就绪稿；未将新域/第二模型/新认证或Serving增量写为成功。当前正文的相关工作原稿引用保持，投稿前仍需作者做逐引文事实审阅。\n')
C.prose(C.R/'paper_compilation_receipt.md','# PASS｜权威构建与逐页视觉核验\n\n'+json.dumps({'Paper_revision':build['revisionId'],'build':build['build']['id'],'status':build['build']['status'],'PDF_SHA256':build['pdf']['contentHash'],'pages':10,'visual':'全部10页检查，无裁切/未解析引用，表图可读；末页参考文献与结论','unresolved_source_comments':0},ensure_ascii=False,indent=2)+'\n\n署名异常透明保留：本轮先前Orange06 R55/R56及Paper R4写入经legacy连接绑定ChatGPT。发现后停止该连接，后续使用已核验CodeX token的resource工具；不重写历史作者、不自称真人批准。\n')
C.prose(C.R/'reproduce_from_clean_checkout.md','# PASS 数值CPU重放 / INSUFFICIENT 完整公开复现\n\nrelease_candidate_001/ 为隔离目录。上述唯一命令实际执行；7张科学表逐字节SHA一致，见clean_replay_hash_checks.csv。重绘生成的原生LaTeX图表由paper_expand_001.py与paper_assemble_001.py从CSV写入；尚需整理输入参数为公开CLI。\n\n完整新GPU硬件复现、原版RetrievalRouter、外部域评价和匿名公开URL均NOT_RUN。不能把数值重放当作重新进行GPU测量。记录环境版本，模型和索引仅提供官方获取配方；任何公有发布前保留数据/权重许可证、作者及匿名检查。\n')
C.prose(C.R/'submission_checklist.md','# EAB CONDITIONAL_NOT_READY\n\n- PASS: 当前既有Paper八节、10页英文工作稿、主表与2图、逐页检查、7表CPU重放。\n- INSUFFICIENT: 独立外部评价、第二检索器GPU结果、原版外部基线/适配基线完整路径、Serving可靠性、模型训练重叠审计。\n- NOT_RUN: 匿名公开仓库可访问性、全套硬件clean-checkout、作者署名/ORCID、AI内容披露和合作者最终确认。\n- SEALED: Pharma Cal81/Eval73与旧B-L/test/CS。没有自动揭封或投稿授权。\n- 10/29协议冻结，10/30科学Go/No-Go，11/4主表，11/5英文稿；不以篇幅或更多Dev曲线替代独立证据。\n')
C.prose(C.R/'claim_to_evidence_matrix.md','# 主张—证据矩阵\n\n|主张|证据/正文|等级与状态|\n|---|---|---|\n|F1 静态计划存在不同风险成本点|X2/rebuild_002/40行表；Table II/Fig1；旧E1候选修复|DEV_EXPLORATORY；跨模型PENDING|\n|F2 当前轻量路由未证明稳定同风险增量|X2表/paired CI；Table II/III Fig2|DEV条件；RR成本组合，不等于原版复现|\n|F3 有限标签和独立性限制新K1|封存元数据/zero-loss下界；RQ3|NOT_CERTIFIED；没有新标签证书|\n|F4 服务与总响应需各自AA|旧2720 AA/X3队列分解；RQ4|实测负结果；新Serving比较NOT_RUN|\n|跨视觉检索器|X1训练权重/兼容性审计|GPU pilot/扩展完成前PENDING|\n|独立英文benchmark泛化|X3官方输入与重叠审计|评价NOT_RUN；训练重叠UNKNOWN|\n|完整公开复现|7表CPU重放|局部PASS；公开URL/全硬件NOT_RUN|\n')
C.write(C.A/'prepared.json',{'time_bj':C.stamp(),'paper_pages':10,'paper_PDF_SHA256':build['pdf']['contentHash'],'paper_build':'PASS','visual_QA':'PASS_ALL_10_PAGES','CPU_replay_tables_identical':7,'private_path_hits':privacy,'readiness':'INSUFFICIENT_EAB_SUBMISSION','source_generation':'PASS_NUMERIC_CSV_TO_TEX','new_Gpu_requests':0})
print(json.dumps({'status':'PASS_PAPER_AND_CPU_REPLAY_NOT_SUBMISSION_READY','time_bj':C.stamp(),'privacy_hits':privacy}),flush=True)
