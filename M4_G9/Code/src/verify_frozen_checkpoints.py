"""Verify provenance and score replay from the ORIGINAL frozen CausalLOB export.

From M4_G9/: python Code/src/verify_frozen_checkpoints.py --zip Code/checkpoints/CausalLOB_FROZEN_REPRO_EXPORT.zip

The export supplies original model weights and saved predictions. This script checks
archive bytes and recomputes recorded outcome scores; it does not refit LPCMCI,
Gaussian transitions or DeepLOB network weights.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def readnpz(z, path):
    with np.load(io.BytesIO(z.read(path)), allow_pickle=False) as x:
        return {k: x[k] for k in x.files}


def compare(actual, expected, name, tol=5e-7):
    np.testing.assert_allclose(float(actual), float(expected), atol=tol, rtol=0,
                               err_msg=f'{name}: expected {expected}, got {actual}')


def validate(archive):
    verified = {}
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('FROZEN_CHECKPOINT_MANIFEST.json'))
        entries = manifest['files']
        filenames = set(z.namelist())
        assert len(entries) == 244, 'Frozen-export count changed; re-audit the source'
        for path, metadata in entries.items():
            assert path in filenames, f'Export entry missing: {path}'
            data = z.read(path)
            assert len(data) == metadata['size_bytes'], path + ': byte length mismatch'
            assert sha(data) == metadata['sha256'], path + ': digest mismatch'
        for required in manifest['required_artifacts_verified']:
            assert required in filenames, f'Required original artifact missing: {required}'
        print(f'PASS: {len(entries)} original exported files, {len(manifest["required_artifacts_verified"])} required checkpoint paths, no digest mismatch')

        mask = readnpz(z, 'STEP4_VALIDATION_FROZEN/checkpoints/PRIMARY_FROZEN_PARENT_MASK.npz')
        frozen = readnpz(z, 'STEP6_FROZEN_DBN/checkpoints/01_FROZEN_STATE_SPACE.npz')
        assert np.array_equal(mask['parent_mask'], frozen['frozen_parent_mask'])
        assert int(mask['parent_mask'].sum()) == 39, 'Expected 39 lagged parent slots'
        assert str(mask['mask_hash'].item()) == 'eb38b17c1431c0a98d46ed1c792953c932c58572799b64aee79165dfc58439da'
        assert frozen['B_sparse'].shape == (6, 30)
        assert frozen['v_sparse'].shape == (6,)
        assert np.all(frozen['v_sparse'] > 0)
        assert np.all(frozen['v_ar'] > 0)
        print('PASS: original Step 4 mask agrees with Step 6 frozen 39-slot linear-Gaussian state-space checkpoint')

        transition = pd.read_csv(io.BytesIO(z.read('STEP8_GRAPH_SPECIFIC_READONLY/results/01_transition_all_models.csv')))
        out_table = pd.read_csv(ROOT / 'Results/Tables/15_transition_kpis.csv')
        refnames = {'AR5':'FROZEN_AR5', 'Sparse-39':'FROZEN_CAUSALLOB', 'Dense_VAR5':'DENSE_VAR5'}
        available = sorted(transition.model.unique().tolist())
        for item in out_table.itertuples():
            if item.model == 'Median_100_matched_random':
                r = pd.read_csv(io.BytesIO(z.read('STEP8_GRAPH_SPECIFIC_READONLY/results/04_graph_null_transition_summary.csv')))
                pooled = r[r.day.astype(str).eq('POOLED')].set_index('metric')
                for metric, col in [('transition_rmse','state_rmse'),('transition_joint_nll','joint_nll'),('return_rmse','return_rmse'),('return_nll','return_nll')]:
                    compare(pooled.loc[metric,'null_median'],getattr(item,col),f'random median {col}',tol=8e-7)
                continue
            src = refnames[item.model]
            assert src in available, f'Transition baseline {src} not in archive: {available}'
            row = transition[(transition.model.eq(src)) & (transition.day.astype(str).eq('POOLED'))].iloc[0]
            for src_col,col in [('transition_rmse','state_rmse'),('transition_joint_nll','joint_nll'),('return_rmse','return_rmse'),('return_nll','return_nll')]:
                compare(row[src_col],getattr(item,col),f'{item.model} {col}',tol=8e-7)
        print('PASS: every final transition KPI equals archived Step 8 results (six-digit rounding tolerance)')

        outcome = pd.read_csv(io.BytesIO(z.read('STEP6_FROZEN_DBN/results/04_OUTCOME_METRICS_D8_D10.csv')))
        matched = pd.read_csv(io.BytesIO(z.read('STEP12_EVIDENCE_AND_DEEPLOB/results/11_deeplob_matched_comparison.csv')))
        final = pd.read_csv(ROOT / 'Results/Tables/14_matched_forecast_full_probabilistic_kpis.csv')
        final_rows=[]
        for item in final.itertuples():
            h = int(item.horizon_raw_messages)
            d=readnpz(z,f'STEP6_FROZEN_DBN/results/03_PREDICTED_CLASS_PROBABILITIES_h{h}.npz')
            y=d['label_ordered'].astype(int)
            p=d['p_down_neutral_up']
            prior=d['prior_down_neutral_up']
            assert p.shape == (len(y),3), (h,p.shape)
            assert np.all((y>=0) & (y<3))
            assert np.isfinite(p).all() and np.max(np.abs(p.sum(axis=1)-1)) < 1e-9
            ll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean())
            br=float(((p-np.eye(3)[y])**2).sum(axis=1).mean())
            ll_prior=float(-np.log(np.clip(prior[y],1e-15,1)).mean())
            br_prior=float(((prior[None,:]-np.eye(3)[y])**2).sum(axis=1).mean())
            obs=outcome[(outcome.horizon_raw_messages==h) & (outcome.day.astype(str)=='POOLED')]
            a=obs[obs.model=='ORDERED_PROBIT_SPARSE_DBN_HEAD'].iloc[0]
            b=obs[obs.model=='PRIOR_ONLY'].iloc[0]
            for val, expected, name in [(ll,a.log_loss,'LL'),(br,a.brier_multiclass,'Brier'),(ll_prior,b.log_loss,'prior LL'),(br_prior,b.brier_multiclass,'prior Brier')]:
                compare(val,expected,f'Step6 h{h} {name}',tol=2e-10)
            for source_model, prefix in [('DEEPLOB_SAME_COHORT','deeplob'),('FROZEN_DBN_HEAD_SAME_COHORT','dbn'),('TRAIN_PRIOR_SAME_COHORT','training_prior')]:
                c=matched[(matched.horizon==h)&(matched.day.astype(str)=='POOLED')&(matched.model==source_model)]
                assert len(c)==1, (h,source_model)
                c=c.iloc[0]
                assert int(c.n)==int(item.n_matched)
                compare(c.log_loss,getattr(item,prefix+'_logloss'),f'matched {h} {prefix} LL')
                compare(c.brier_multiclass,getattr(item,prefix+'_brier'),f'matched {h} {prefix} Brier')
                if prefix != 'training_prior':
                    compare(c.balanced_accuracy,getattr(item,prefix+'_balanced_accuracy'),f'matched {h} {prefix} BA')
            final_rows.append(dict(horizon=h,n_full=len(y),replayed_full_DBN_logloss=ll,replayed_full_DBN_brier=br,replayed_full_prior_logloss=ll_prior,replayed_full_prior_brier=br_prior,matched_rows_from_original=item.n_matched))
        print('PASS: all 5 horizon Step 6 prediction arrays recomputed against original outcome CSV; matched DeepLOB/DBN/prior rows agree with original Step 12 comparison')

        # Check full 12-row consecutive-gap array against original Step 9 CSV.
        gap=pd.read_csv(io.BytesIO(z.read('STEP9_GAP_STRESS_READONLY/results/03_gap_contrasts.csv')))
        target=pd.read_csv(ROOT/'Results/Tables/06_consecutive_missingness.csv')
        for rec in target.itertuples():
            sc={'Return + OFI':'MISS_RETURN_AND_OFI','History only':'HISTORY_ONLY'}[rec.scenario]
            # Contrast defined as prior - model in Step 9 outcome comparison.
            rows=gap[(gap.scenario==sc)&(gap.gap==rec.gap_representation_steps)&(gap.horizon==rec.horizon_events)&(gap.day.astype(str)=='POOLED')]
            print_flag=rows.comparison.unique()
            selector=rows[rows.comparison.astype(str).str.contains('PRIOR',case=False)]
            assert len(selector)==1, (sc,rec.gap_representation_steps,rec.horizon_events,print_flag)
            compare(selector.iloc[0].outcome_logloss_gain,rec.prior_minus_DBN_logloss,f'gap {sc} {rec.gap_representation_steps} h{rec.horizon_events}')
        print('PASS: every reported consecutive-gap contrast agrees with archived Step 9 results')

        outfile=ROOT/'Results/Tables/17_original_checkpoint_score_replay.csv'
        pd.DataFrame(final_rows).to_csv(outfile,index=False)
        print('WROTE:',outfile)
        return len(entries)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--zip',type=Path,default=Path('/tmp/CausalLOB_FROZEN_REPRO_EXPORT.zip'))
    args=ap.parse_args()
    validate(args.zip)
