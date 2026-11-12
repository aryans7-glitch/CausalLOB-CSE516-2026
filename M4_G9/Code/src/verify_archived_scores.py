"""Audit CSV against actual printed outputs in the supplied frozen notebook.
Does not load / re-evaluate Google Drive model checkpoints.
Run from M4_G9/: python Code/src/verify_archived_scores.py
"""
import json,re
from pathlib import Path
import pandas as pd
import numpy as np
nb=json.loads(Path('Code/src/PGM_Project_Frozen.ipynb').read_text())

def source_output(i):
    out=[]
    for cellout in nb['cells'][i].get('outputs',[]):
        val=cellout.get('text',cellout.get('data',{}).get('text/plain',''))
        out.append(''.join(val) if isinstance(val,list) else str(val))
    return '\n'.join(out)

# Step13 was already executed in original Google Drive environment.
src=source_output(10)
pat=re.compile(r'^\s*(10|20|30|50|100)\s+(DEEPLOB|DBN_COMPLETE|PRIOR_D1_D7)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)',re.M)
matches=pat.findall(src)
assert len(matches)==15,f'Expected 15 rows from Step13 printed table, found {len(matches)}'
model_rows={(int(h),name):dict(n=int(n),balanced_accuracy=float(ba),logloss=float(ll),brier=float(br)) for h,name,n,ba,ll,br in matches}
table=pd.read_csv('Results/Tables/14_matched_forecast_full_probabilistic_kpis.csv')
for row in table.itertuples():
    h=row.horizon_raw_messages
    for name,prefix in [('DEEPLOB','deeplob'),('DBN_COMPLETE','dbn'),('PRIOR_D1_D7','training_prior')]:
        srcrow=model_rows[(h,name)]
        assert srcrow['n']==row.n_matched
        for out_col,in_col in [('logloss','logloss'),('brier','brier')]:
            np.testing.assert_allclose(float(getattr(row,f'{prefix}_{out_col}')),srcrow[in_col],atol=5e-7,rtol=0)
        if name!='PRIOR_D1_D7':
            np.testing.assert_allclose(float(getattr(row,f'{prefix}_balanced_accuracy')),srcrow['balanced_accuracy'],atol=5e-7,rtol=0)
print('PASS: all five horizons, matched row counts, 15 log-loss scores, 15 Brier scores and 10 balanced-accuracy values agree with original executed Step13 output')
