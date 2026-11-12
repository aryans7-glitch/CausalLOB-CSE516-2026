"""Recover original 15-bin confidence reliability diagrams from frozen Step 6 probabilities.

python Code/src/rebuild_reliability.py --zip /tmp/CausalLOB_FROZEN_REPRO_EXPORT.zip
"""
import argparse
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]

def main(archive):
    allrows=[]
    with zipfile.ZipFile(archive) as z:
        for horizon in (10,20,30,50,100):
            with np.load(io.BytesIO(z.read(f'STEP6_FROZEN_DBN/results/03_PREDICTED_CLASS_PROBABILITIES_h{horizon}.npz'))) as d:
                p=d['p_down_neutral_up']
                y=d['label_ordered']
            conf=p.max(axis=1)
            correct=p.argmax(axis=1)==y
            bins=np.minimum((conf*15).astype(int),14)
            ECE=0.
            for b in range(15):
                sel=bins==b
                n=int(sel.sum())
                c=float(conf[sel].mean()) if n else np.nan
                acc=float(correct[sel].mean()) if n else np.nan
                if n: ECE += (n / len(y)) * abs(c-acc)
                allrows.append(dict(horizon_raw_messages=horizon,confidence_bin=b+1,bin_lower=b/15,bin_upper=(b+1)/15,n=n,mean_confidence=c,empirical_accuracy=acc))
            print(f'h={horizon}: n={len(y)}, recomputed ECE={ECE:.12f}')
    data=pd.DataFrame(allrows)
    out=ROOT/'Results/Tables/18_reliability_bins_from_original_predictions.csv'
    data.to_csv(out,index=False)
    fig,axs=plt.subplots(1,2,figsize=(8.1,3.05),sharex=True,sharey=True)
    for ax,h,lab in zip(axs,(10,30),('Short horizon: h=10','Intermediate horizon: h=30')):
        x=data[(data.horizon_raw_messages==h)&(data.n>0)].copy()
        ax.plot([0,1],[0,1],ls='--',lw=1,alpha=.7,color='0.55',label='Ideal calibration')
        ax.plot(x.mean_confidence,x.empirical_accuracy,linewidth=1.7,marker='o',markersize=3,color='#245577',label='DBN confidence bins')
        ax.set_title(lab,fontsize=10)
        ax.set_xlim(.3,1.0);ax.set_ylim(.3,1.0)
        ax.grid(alpha=.18)
        ax.set_xlabel('Mean predicted confidence',fontsize=9)
        ax.tick_params(labelsize=8)
    axs[0].set_ylabel('Observed accuracy in bin',fontsize=9)
    axs[0].legend(loc='lower right',fontsize=7)
    fig.tight_layout()
    path=ROOT/'Results/Figures/11_reliability_original_frozen_probabilities.png'
    fig.savefig(path,dpi=240,bbox_inches='tight')
    plt.close(fig)
    print('WROTE:',out,'and',path)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--zip',type=Path,default=Path('/tmp/CausalLOB_FROZEN_REPRO_EXPORT.zip'))
    main(ap.parse_args().zip)
