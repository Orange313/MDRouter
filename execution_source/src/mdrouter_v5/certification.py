"""Fixed-sample bounded-loss UCBs. No independence is inferred by this module."""
import math
import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta

BOUNDS=('hoeffding','bounded_kl','empirical_bernstein','cp_harm_indicator')
def kl(p,q):
    if q<=0:return 0.0 if p==0 else math.inf
    if q>=1:return 0.0 if p==1 else math.inf
    return (p*math.log(p/q) if p>0 else 0.0)+((1-p)*math.log((1-p)/(1-q)) if p<1 else 0.0)

def ucb_moments(n,mean,variance,alpha,method,harm_count=None):
    if not 0<alpha<1:raise ValueError('Invalid tail allocation')
    if method not in BOUNDS:raise ValueError('Unknown bound')
    if n<1:return 1.0
    if not 0<=mean<=1 or variance<0:raise ValueError('Loss moments must be bounded')
    if method=='hoeffding':return min(1.0,mean+math.sqrt(math.log(1/alpha)/(2*n)))
    if method=='bounded_kl':
        if mean==1:return 1.0
        if mean==0:return -math.expm1(math.log(alpha)/n)
        threshold=math.log(1/alpha)/n
        if kl(mean,1-1e-14)<=threshold:return 1.0
        return float(brentq(lambda q:kl(mean,q)-threshold,mean,1-1e-14,xtol=1e-14))
    if method=='empirical_bernstein':
        if n<2:return 1.0
        log=math.log(2/alpha)
        return min(1.0,mean+math.sqrt(2*variance*log/n)+7*log/(3*(n-1)))
    if harm_count is None or not 0<=harm_count<=n:raise ValueError('CP requires a Bernoulli harm count')
    return 1.0 if harm_count==n else float(beta.ppf(1-alpha,harm_count+1,n-harm_count))

def ucb(samples,alpha,method):
    x=np.array(samples,dtype=float)
    if len(x) and (not np.isfinite(x).all() or x.min()<0 or x.max()>1):
        raise ValueError('Only r or r_bar in [0,1], never signed gap')
    n=len(x)
    mean=float(x.mean()) if n else 0.0
    var=float(x.var(ddof=1)) if n>=2 else 0.0
    value=ucb_moments(n,mean,var,alpha,method,int((x>0).sum()))
    return {'n':n,'sample_mean':mean if n else None,'sample_variance':var if n>=2 else None,
            'alpha':alpha,'method':method,'UCB':value,'width':value-mean if n else None,
            'target_semantics':'mean <= harm probability' if method=='cp_harm_indicator' else 'bounded-loss mean'}

def k1_budget(epsilon,delta):
    if not 0<=epsilon<1 or not 0<delta<1:raise ValueError('Invalid K1 parameters')
    return (epsilon-delta)/(1-delta) if epsilon>delta else None
