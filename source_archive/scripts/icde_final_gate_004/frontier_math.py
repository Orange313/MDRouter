import numpy as np,itertools
def envelope(R,T,target):
 # Vectorized batch, four static plans, all pair mixtures. Target may be risk or cost.
 R=np.atleast_2d(R);T=np.atleast_2d(T);target=np.broadcast_to(np.asarray(target),(len(R),));result=np.where(R<=target[:,None]+1e-12,T,np.inf).min(1);weights=np.zeros_like(R)
 for i in range(len(R)):
  j=int(np.argmin(np.where(R[i]<=target[i]+1e-12,T[i],np.inf)));weights[i,j]=1
 for j,k in itertools.combinations(range(R.shape[1]),2):
  den=R[:,k]-R[:,j];valid=abs(den)>1e-12;w=np.divide(target-R[:,j],den,out=np.zeros(len(R)),where=valid);ok=valid&(w>=0)&(w<=1);cost=(1-w)*T[:,j]+w*T[:,k];improve=ok&(cost<result);result[improve]=cost[improve];weights[improve]=0;weights[improve,j]=1-w[improve];weights[improve,k]=w[improve]
 return result,weights
def boots(groups,B,seed):
 rng=np.random.default_rng(seed);u=np.unique(groups);return [np.concatenate([np.flatnonzero(groups==h) for h in rng.choice(u,len(u),replace=True)]) for _ in range(B)]
