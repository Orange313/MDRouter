"""Publication figures from bound CSV rows; no manually entered measurements."""
def draw(out,rows,cross,gaps):
    import sys
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    out=Path(out);out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    domains=list(dict.fromkeys(r['dataset'] for r in rows));short={'vidore_v3_hr':'HR','vidore_v3_finance_en':'Finance','vidore_v3_industrial':'Industrial','Pharmaceuticals':'Pharma','Biomedical':'Biomedical'}
    fig,aa=plt.subplots(2,3,figsize=(9.2,5.7));axes=aa.ravel();axes[-1].axis('off')
    for ax,d in zip(axes,domains):
        for r in rows:
            if r['dataset']!=d or r['method'] not in ['Full-Visual','BM25-R50','BM25-R100','BM25','Q_regret','QP_regret','RandomMix']:continue
            marker='o' if r['method'] in ['Full-Visual','BM25-R50','BM25-R100','BM25'] else 'x'
            ax.scatter(float(r['E2E_ms']),float(r['nDCG10']),marker=marker,s=25);ax.annotate(r['method'].replace('BM25-','').replace('_regret',''),(float(r['E2E_ms']),float(r['nDCG10'])),fontsize=6,xytext=(3,3),textcoords='offset points')
        ax.set(title=short.get(d,d),xlabel='Complete warm path (ms)',ylabel='Actual mean nDCG@10');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'quality_cost.pdf',bbox_inches='tight');fig.savefig(out/'quality_cost.png',dpi=180);plt.close(fig)
    learned=[r for r in rows if r['method'] in ['Q_regret','QP_regret']];fig,(ax,oracle_ax)=plt.subplots(1,2,figsize=(10.5,4.7),gridspec_kw={'width_ratios':[1.6,1]})
    for i,r in enumerate(learned):
        x=float(r['gain_pp']);lo=float(r['paired95_low_pp']);hi=float(r['paired95_high_pp']);ax.errorbar(x,i,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt='o',capsize=2,color='#1766a3')
    ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(range(len(learned)),[short.get(r['dataset'],r['dataset'])+'/'+r['method'].replace('_regret','') for r in learned]);ax.set_xlabel('Actual-policy gain vs same-risk envelope (pp of FullV)');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    og=[r for r in gaps if abs(float(r['risk'])-.05)<1e-9 and r.get('retriever','ColQwen2')!='ColPali-v1.3'];oracle_ax.barh(range(len(og)),[float(r['oracle_headroom_pp']) for r in og],color='#aaaaaa');oracle_ax.set_yticks(range(len(og)),[short.get(r['dataset'],r['dataset']) for r in og]);oracle_ax.invert_yaxis();oracle_ax.set_xlabel('Offline oracle headroom at r=.05 (pp)');oracle_ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(out/'oracle_vs_learned.pdf',bbox_inches='tight');fig.savefig(out/'oracle_vs_learned.png',dpi=180);plt.close(fig)
    cross_domains=list(dict.fromkeys(r['dataset'] for r in cross));fig,axes=plt.subplots(1,len(cross_domains),figsize=(12,3.8))
    for ax,d in zip(axes,cross_domains):
        data=[r for r in cross if r['dataset']==d]
        for r in data:
            is_colpali=r['retriever'].startswith('ColPali');ax.scatter(float(r['E2E_ms']),float(r['mean_r']),marker='s' if is_colpali else 'o',color='#d46b16' if is_colpali else '#1766a3');ax.annotate(('P/' if is_colpali else 'Q/')+r['method'],(float(r['E2E_ms']),float(r['mean_r'])),fontsize=5,xytext=(3,3),textcoords='offset points')
        ax.set(title=short[d]+' / own model references',xlabel='Complete warm path (ms)',ylabel='Positive regret r');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'cross_domain_retriever.pdf',bbox_inches='tight');fig.savefig(out/'cross_domain_retriever.png',dpi=180);plt.close(fig)
