import common as C
import json
C.A=C.ART/'X4';C.R=C.REPORT/'X4';x=C.load(C.A/'prepared.json')
assert x['paper_build']=='PASS' and x['CPU_replay_tables_identical']==7
facts='PASS 现有Paper R3检查后扩写至R4，八节10页英文工作稿（含引用）、32行完整路径主表+8行适配表、2原生矢量图；权威build成功且10页逐页核验。7张科学表隔离CPU重放SHA逐字节一致；无新GPU/独立Eval/新认证。PDF SHA256 '+x['paper_PDF_SHA256']+'。'
if x['private_path_hits']:facts+='公开候选隐私扫描发现路径残留，发布BLOCKED；未上传公开仓库。'
C.prose(C.R/'X4_decision.md','# PASS_WRITING_AND_NUMERIC_REPLAY / INSUFFICIENT_EAB_READINESS\n\n'+facts+'\n\n只推进EAB写作，不宣称ready。支持已有开发观察和测量可靠性分析，跨检索器/独立结果待X1/X3。最终作者信息、AI披露、官方许可复核、匿名公开artifact与clean硬件仍未完成。\n')
C.finish('X4','PASS_PAPER_R4_CPU_REPLAY_INSUFFICIENT_SUBMISSION',facts,'保持EAB有条件推进；不揭封、不投稿、不恢复待删除稿。签名连接异常在编译回执披露，后续CodeX正确身份写回。')
