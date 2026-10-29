"""Cell-only quota allocation with uniform within-cell random assignment."""
from collections import Counter
import hashlib
import json
import math
import random
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from .certification import ucb

REFERENCE='V-Multi'
def assign_quotas(window,quotas,seed):
    rng=random.Random(seed)
    assignment=[REFERENCE]*len(window)
    for cell in sorted({r['cell'] for r in window}):
        positions=[i for i,r in enumerate(window) if r['cell']==cell]
        rng.shuffle(positions)
        cursor=0
        for plan in sorted(p for h,p in quotas if h==cell):
            quota=quotas[cell,plan]
            for pos in positions[cursor:cursor+quota]:assignment[pos]=plan
            cursor+=quota
        if cursor!=len(positions):raise ValueError('Quota does not cover cell')
    return assignment

def certify_cells(samples,plans,delta,method='bounded_kl',min_n=10,family_factor=1):
    if REFERENCE not in plans or len(plans)<2:raise ValueError('Reference and alternatives required')
    if family_factor<1:raise ValueError('Invalid simultaneous family')
    H,K=len(samples),len(plans)-1
    alpha=delta/(max(1,H)*K*family_factor)
    result={}
    for cell,byplan in samples.items():
        result[cell]={REFERENCE:{'UCB':0.0,'n':None,'eligible':True,'reason':'reference risk is zero by definition'}}
        for p in plans:
            if p==REFERENCE:continue
            row=ucb(byplan.get(p,[]),alpha,method)
            row['eligible']=row['n']>=min_n
            row['reason']='developer certificate only' if row['eligible'] else 'empty/sparse -> reference fallback'
            if not row['eligible']:row['UCB']=1.0
            result[cell][p]=row
    return {'cells':result,'H':H,'K':K,'alpha':alpha,'delta':delta,'method':method,'min_n':min_n,
            'family_factor':family_factor,'status':'DEV_OR_SYNTHETIC_ONLY','independent_certification':False}

def allocate(window,certificate,costs,budget,seed,reference=REFERENCE,time_limit=10):
    """Only cell, cell-constant costs/UCB and independent seed affect assignments."""
    if not math.isfinite(budget) or budget<0:raise ValueError('Invalid certified budget')
    if len({r['query_id'] for r in window})!=len(window):raise ValueError('Duplicate IDs')
    started=time.perf_counter()
    m=len(window)
    counts=Counter(r['cell'] for r in window)
    cells=sorted(counts)
    cert=certificate['cells']
    fallback=None
    if any(c not in costs or reference not in costs[c] for c in cells):fallback='missing cell cost profile'
    entries=[]
    if fallback is None:
        for cell in cells:
            for plan,cost in sorted(costs[cell].items()):
                if not math.isfinite(cost) or cost<0:raise ValueError('Invalid cell cost')
                bound=cert.get(cell,{}).get(plan,{})
                eligible=plan==reference or (bound.get('eligible',False) and plan!=reference)
                risk=0.0 if plan==reference else float(bound.get('UCB',1.0))
                if not 0<=risk<=1:raise ValueError('Unbounded certificate')
                entries.append((cell,plan,cost,risk,counts[cell] if eligible else 0))
    def finish(assignment,extra):
        output=[{'query_id':r['query_id'],'cell':r['cell'],'plan_id':p,
                 'UCB':0.0 if p==reference else cert[r['cell']][p]['UCB']} for r,p in zip(window,assignment)]
        total=sum(r['UCB'] for r in output)
        assert total<=m*budget+1e-7
        digest=hashlib.sha256(json.dumps(output,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return {'allocation':output,'allocation_hash':digest,'seed':seed,'m':m,'budget':budget,
                'certified_risk_mean':total/max(1,m),'planning_ms':(time.perf_counter()-started)*1000,
                'independent_certification':False,'status':'DEV_OR_SYNTHETIC_ONLY',**extra}
    if not m or fallback or not any(p!=reference and upper>0 for c,p,cost,u,upper in entries):
        return finish([reference]*m,{'fallback':fallback or 'no eligible nonreference plan','solver':'none',
            'cost_estimate':None if fallback else sum(costs[r['cell']][reference] for r in window)})
    A=np.zeros((len(cells)+1,len(entries)))
    for j,(cell,p,c,u,upper) in enumerate(entries):
        A[cells.index(cell),j]=1
        A[-1,j]=u
    rhs=np.array([counts[c] for c in cells]+[m*budget],float)
    lower=rhs.copy();lower[-1]=-np.inf
    upper=np.array([e[4] for e in entries],float)
    c=np.array([e[2] for e in entries])
    bounds=Bounds(np.zeros(len(entries)),upper)
    result=milp(c=c,integrality=np.ones(len(entries)),bounds=bounds,
        constraints=LinearConstraint(A,lower,rhs),options={'time_limit':time_limit,'mip_rel_gap':0})
    if result.status!=0:
        return finish([reference]*m,{'fallback':f'MILP status {result.status}','solver':'MILP',
            'cost_estimate':sum(costs[r['cell']][reference] for r in window)})
    lp=linprog(c,A_ub=A[-1:],b_ub=rhs[-1:],A_eq=A[:-1],b_eq=rhs[:-1],bounds=list(zip(np.zeros(len(entries)),upper)),method='highs')
    assert lp.success
    integer=np.rint(result.x).astype(int)
    assert np.all(np.abs(integer-result.x)<1e-6)
    quotas={(cell,p):int(integer[j]) for j,(cell,p,*_) in enumerate(entries)}
    rounded=np.floor(lp.x+1e-8).astype(int)
    for cell in cells:
        refj=next(j for j,e in enumerate(entries) if e[0]==cell and e[1]==reference)
        rounded[refj]+=counts[cell]-sum(rounded[j] for j,e in enumerate(entries) if e[0]==cell)
    assert float(A[-1]@rounded)<=rhs[-1]+1e-7
    round_cost=float(c@rounded)
    max_saving=max([0.0]+[costs[cell][reference]-cost for cell,p,cost,u,up in entries if up>0])
    assert lp.fun<=result.fun+1e-7<=round_cost+1e-7
    assert round_cost-lp.fun<=max_saving+1e-6
    assignment=assign_quotas(window,quotas,seed)
    return finish(assignment,{'fallback':None,'solver':'MILP','cost_estimate':float(result.fun),
        'lp_cost':float(lp.fun),'rounded_lp_cost':round_cost,'mip_gap':float(result.mip_gap),
        'one_query_rounding_bound':max_saving,'quotas':{f'{h}|{p}':v for (h,p),v in quotas.items()}})
