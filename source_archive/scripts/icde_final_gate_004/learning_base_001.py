import numpy as np,hashlib,math,time
L2=[1.,8.,64.];KAPPA=[0.,1.,5.,10.,20.,50.,100.];METHODS=['regret','hard','soft'];SEEDS=[202610270930,202610270931,202610270932]
def folds(g,k,seed):
 u=np.unique(g);k=min(k,len(u));sz={h:int((g==h).sum()) for h in u};order=sorted(u,key=lambda h:(-sz[h],hashlib.sha256((str(seed)+str(h)).encode()).hexdigest()));loads=np.zeros(k,int);a={}
 for h in order:j=int(np.argmin(loads));a[h]=j;loads[j]+=sz[h]
 return [np.array([a[h]==j for h in g]) for j in range(k)]
def fit(X,y,l2):
 cols=np.isfinite(X).any(0);a=X[:,cols];med=np.nanmedian(a,0);mis=~np.isfinite(a);a=np.c_[np.where(mis,med,a),mis.astype(float)];mu=a.mean(0);sd=a.std(0);sd[sd<1e-6]=1;z=(a-mu)/sd;bias=y.mean(0);w=z.T@np.linalg.solve(z@z.T+l2*np.eye(len(z)),y-bias)
 return cols,med,mu,sd,w,bias
def predict(m,X):
 cols,med,mu,sd,w,bias=m;a=X[:,cols];mis=~np.isfinite(a);a=np.c_[np.where(mis,med,a),mis.astype(float)];return (a-mu)/sd@w+bias
def targets(y,c,method,kappa):
 util=c+kappa*y
 if method=='regret':return y
 if method=='hard':return np.eye(y.shape[1])[np.argmin(util,axis=1)]
 z=-util/.1;z-=z.max(1,keepdims=True);z=np.exp(z);return z/z.sum(1,keepdims=True)
def actions(pred,c,method,kappa):return np.argmin(c+kappa*np.maximum(pred,0),axis=1) if method=='regret' else np.argmax(pred,axis=1)
def tune(X,y,c,g,method,kappa,seed):
 choices=[]
 for l in L2:
  pp=np.empty_like(y)
  for val in folds(g,3,seed):
   assert not set(g[val])&set(g[~val]);m=fit(X[~val],targets(y[~val],c[~val],method,kappa),l);pp[val]=predict(m,X[val])
  act=actions(pp,c,method,kappa);utility=np.mean(c[np.arange(len(y)),act]+kappa*y[np.arange(len(y)),act]);choices.append((utility,{64.:0,8.:1,1.:2}[l],l))
 return min(choices)[2]
def subset(train,g,n,seed):
 if n=='all' or int(n)>=len(train):return train
 rng=np.random.default_rng(seed);groups=np.unique(g[train]);rng.shuffle(groups);order=[]
 for h in groups:
  a=train[g[train]==h].copy();rng.shuffle(a);order.extend(a.tolist())
 return np.array(order[:int(n)],int)
def bootstrap_diff(values,g,seed=202610270933,B=2000):
 rng=np.random.default_rng(seed);u=np.unique(g);s=[]
 for _ in range(B):
  ix=np.concatenate([np.flatnonzero(g==x) for x in rng.choice(u,len(u),replace=True)]);s.append(np.mean(values[ix]))
 return np.quantile(s,[.025,.975])
