"""Standalone reproduction of CausalLOB frozen Step16 artificial known-DGP experiment.
This test generates SIMULATED data only, not FI-2010 model results.
Execute: python Code/src/reproduce_synthetic.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SEED=5162026
REPS=40
SYN_TRAIN=800
SYN_TEST=300
SYN_LAG=5
OUT=Path('Results/Tables')
OUT.mkdir(parents=True,exist_ok=True)

def design(x,t,lags=5):
    return np.concatenate([x[t-k] for k in range(1,lags+1)],axis=0)

def fit_target(X,y,columns,alpha=1.):
    A=X[:,columns]; am=A.mean(0); ym=y.mean()
    Ac=A-am
    gram=Ac.T@Ac+alpha*np.eye(len(columns))
    coef=np.linalg.solve(gram,Ac.T@(y-ym))
    intercept=float(ym-am@coef)
    resid=y-(A@coef+intercept)
    variance=float(np.mean(resid**2))
    assert variance>1e-9
    return (np.asarray(columns,int),coef,intercept,variance)

def evaluate_fitted(model,X,y):
    cols,b,c,var=model
    pred=X[:,cols]@b+c
    residual=y-pred
    return {'rmse':float(np.sqrt(np.mean(residual**2))),
            'gaussian_nll':float(np.mean(.5*(np.log(2*np.pi*var)+residual**2/var))),
            'coverage90':float(np.mean(np.abs(residual)<=1.6448536269514722*np.sqrt(var))),
            'interval_width90':float(2*1.6448536269514722*np.sqrt(var))}

def synth_sequence(rng,n,alpha_cross,initial=None):
    a=np.empty((n,6),dtype=np.float64)
    prev=np.zeros(6,dtype=np.float64) if initial is None else np.asarray(initial,dtype=np.float64).copy()
    for i in range(n):
        noise=rng.standard_normal(6)*np.array([.6,.6,.6,.6,.6,.45])
        v=.4*prev+noise
        v[0]=.65*prev[0]+noise[0]
        v[5]=.45*prev[5]+alpha_cross*prev[0]+noise[5]
        a[i]=v; prev=v
    return a

def synthetic_run():
    rng=np.random.default_rng(SEED)
    results=[]
    for rep in range(REPS):
        tr=synth_sequence(rng,SYN_TRAIN+SYN_LAG,.8)
        start=tr[-1].copy()
        streams={'stable_same_DGP':synth_sequence(rng,SYN_TEST,-0.0+0.8,initial=start),
                 'constructed_coefficient_flip':synth_sequence(rng,SYN_TEST,-.8,initial=start)}
        Xtr=np.stack([design(tr,t) for t in range(SYN_LAG,len(tr))])
        ytr=tr[SYN_LAG:,5]
        cols={'AR5':tuple(5+6*(k-1) for k in range(1,6)),
              'KNOWN_TRUE_EDGE':tuple(sorted(set([5+6*(k-1) for k in range(1,6)]+[0]))),
              'WRONG_EDGE':tuple(sorted(set([5+6*(k-1) for k in range(1,6)]+[1]))),
              'DENSE_VAR5':tuple(range(30))}
        fitted={name:fit_target(Xtr,ytr,cc) for name,cc in cols.items()}
        for regime,data in streams.items():
            combined=np.vstack([tr[-SYN_LAG:],data])
            Xt=np.stack([design(combined,t) for t in range(SYN_LAG,len(combined))])
            yt=data[:,5]
            for name,m in fitted.items():
                results.append({'replicate':rep,'regime':regime,'model':name,
                    'n_train':len(ytr),'n_test':len(yt),'missing_target':'x5_t',
                    'cross_edge_true_in_training':'x0_(t-1) -> x5_t',
                    **evaluate_fitted(m,Xt,yt)})
    f=pd.DataFrame(results)
    assert len(f)==REPS*2*4 and f.groupby(['replicate','regime']).size().eq(4).all()
    return f

f=synthetic_run()
archived=OUT/'11_synthetic_40rep_raw.csv'
if archived.exists():
    ref=pd.read_csv(archived)
    cols=['rmse','gaussian_nll','coverage90','interval_width90']
    assert len(f)==len(ref) and (f[['replicate','regime','model']].values==ref[['replicate','regime','model']].values).all()
    np.testing.assert_allclose(f[cols].values,ref[cols].values,rtol=1e-11,atol=1e-11)
    print('PASS: 320 simulated model evaluations exactly match archived Step16 rerun values')

rows=[]
for regime,g in f.groupby('regime',sort=True):
    piv=g.pivot(index='replicate',columns='model',values='rmse')
    for baseline in ['AR5','WRONG_EDGE','DENSE_VAR5']:
        diff=piv[baseline]-piv['KNOWN_TRUE_EDGE']
        n=len(diff); mu=float(diff.mean()); sd=float(diff.std(ddof=1))
        h=float(stats.t.ppf(.975,n-1)*sd/np.sqrt(n))
        rows.append({'regime':regime,'comparator':baseline,'replications':n,
                     'mean_rmse_gain_baseline_minus_true':mu,'sample_sd_of_paired_difference':sd,
                     'ci95_lower_synthetic':mu-h,'ci95_upper_synthetic':mu+h,
                     'paired_t_p_simulation_only':float(stats.ttest_1samp(diff,0).pvalue),
                     'true_model_wins':int((diff>0).sum())})
out=pd.DataFrame(rows)
out.to_csv(OUT/'13_synthetic_paired_statistics.csv',index=False)
print(out.to_string(index=False,float_format=lambda x:f'{x:.9f}'))
