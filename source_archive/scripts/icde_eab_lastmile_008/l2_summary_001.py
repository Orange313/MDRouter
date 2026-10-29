import common as C
import json
C.start('L2/summary_001');S=C.ART/'L2';r=C.load(S/'final_release_receipt_006.json','EXACT_FINAL_COMMIT_CLEAN_REPLAY_RECEIPT');replay=C.load(S/'clean_replay_006/reproduction_receipt.json','30_CSV_BITWISE_REPLAY');gpu=C.load(S/'minimal_GPU_sample_001/minimal_e2e_result.json','ACTUAL_TECHNICAL_GPU_SMOKE');scan=C.load(S/'release_candidate_006/privacy_size_scan.json','DECOMPRESSED_PRIVACY_AND_SIZE_SCAN')
assert len(replay['checks'])==30 and all(x['identical'] for x in replay['checks']) and r['tracked_files']==283 and scan['status']=='PASS' and gpu['label_reads']==0
assert C.sha(S/'MDRouter_release_candidate_006.zip')==r['zip_SHA256']
C.write(C.R/'release_file_inventory.json',r['file_inventory']);C.write(C.R/'clean_checkout_receipt.json',r)
facts=f"最终候选commit {r['commit']}；{r['tracked_files']}文件，{r['working_tree_bytes']}字节，ZIP {r['zip_bytes']}字节；最大文件{r['max_file_bytes']}字节。独立干净checkout/新装CPU依赖实际复算30张CSV全部SHA一致，6幅图（3组PDF/PNG），{r['wall_s']:.2f}s。16页/1固定查询/12计时的独立源码GPU技术样例PASS，标签0；复用既有固定GPU依赖，不声称新装GPU环境。"
body='# PASS reproduction / NOT_RUN public publication\n\n'+facts+'\n\n用户明确要求上传前确认，本轮没有push、创建远程提交或上传附件。目标https://github.com/Orange313/MDRouter.git。不是私有目录存在就已公开；批准后还需检查无需登录的真实访问。ICDE单盲，额外作者匿名镜像不是必需条件。\n\n去身份扫描包含gzip解压内容；未携带权重、页面图像、完整原始qrels、大索引向量、封存Cal/Eval/B-L/test/CS、凭据、私有论文或私有Git历史。大GPU trace数组仅留hash与私有存档；实际分项/完整请求标量、来源分组和guard已包括。软件许可MIT只是待所有者确认的建议；数据/模型权利分列，未擅自授权。\n\n支持论文工件可移植性及实际复现回执，不支撑公开已完成、新认证、严格Serving或业务净收益。\n'
for name in ['public_release_readiness.md','clean_checkout_reproduction.md','license_privacy_audit.md','artifact_availability.md','upload_review.md']:C.prose(C.R/name,body)
C.prose(C.R/'failure_inventory.md','# Preserved failures and changes\n\n候选001依赖元数据查找失败；首次clean replay001因gitignore误排小型索引元数据失败；002修正后15表通过。003含210MB详细GPU trace，保留但不入发布候选；004改为scalar/trace hash；Git旧版不支持init --initial-branch，候选004未提交即失败；005使用兼容初始化方式，30表通过；最终006加入回执、OCR固定版本与获取源码，干净clone30表再次通过。全部旧版本和日志保留，不覆盖，也未因此修改统计方法。\n')
C.finish('L2_summary_001','PASS_CLEAN_CHECKOUT_30_CSV_AND_GPU_SMOKE__PUBLIC_NOT_RUN_PENDING_USER_APPROVAL',facts,'向用户交付精确commit/ZIP/hash/文件清单供最终上传审批；未获审批不push。软件许可待确认。继续完成论文证据审计。')
