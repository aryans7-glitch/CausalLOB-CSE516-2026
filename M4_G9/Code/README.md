# Code, model and checkpoint reproduction

All commands below run from the **M4_G9 root** (parent of Code/).

## 1. Local verification and score replay

```bash
python -m pip install -r Code/requirements.txt
python Code/src/verify_archive.py Data/fi2010_canonical_raw40_labels.npz
python Code/src/restore_original_checkpoints.py
python Code/src/verify_frozen_checkpoints.py
python Code/src/rebuild_reliability.py
python Code/src/verify_archived_scores.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python Code/src/reproduce_synthetic.py
```

The 11 checkpoint parts in `Code/checkpoints/` reconstruct the **original** 201,392,753-byte export, SHA256:

`f8ca4d6e25ef48fd1265af349cb354e0291f063b541b4c58b9009b5563889f22`

The original source manifest within that ZIP includes hashes and lengths for 244 original files. The `verify_frozen_checkpoints.py` script confirms every hash and required path, verifies the Step4/Step6 graph-mask parity, recomputes Step6 log-loss/Brier from saved row probabilities across five horizons, and checks archived transition, matched-forecast, and gap-score tables against the displayed result CSVs. Its score replay does not retrain the models.

`rebuild_reliability.py` uses the original saved Step6 probability arrays to regenerate 15-bin confidence reliability tables and a two-horizon illustration. Archived and independently recomputed confidence ECE agree on all five horizons. This is an analysis of retrospective predictions, not a newly trained model.

`reproduce_synthetic.py` is the original Step16 known-mechanism experiment independently re-implemented with seed 5162026, 40 independent replication blocks, six-dimensional Gaussian DGP, train/test targets 800/300 per replication/regime, 5 lags, Ridge alpha 1, four comparator models and two test mechanisms. It validates 320 model-by-replication-by-regime scores and writes paired statistics and *synthetic-only* 95% confidence intervals.

## 2. Restore Google Drive checkpoint paths

In a Colab session after mounting Drive, use:

```bash
python Code/src/restore_original_checkpoints.py \
  --output /tmp/CausalLOB_FROZEN_REPRO_EXPORT.zip \
  --extract-root /content/drive/MyDrive/CausalLOB
```

This reconstructs checkpoint, result, source and dataset-stage files without changing the original exported bytes. A preexisting file with a different SHA-256 causes an error rather than a silent overwrite. Any restorer error must be investigated instead of bypassing checkpoint guards.

The restored stage paths include `STEP2_REBUILD/checkpoints/04_temporal_primary.npz`, `STEP4_VALIDATION_FROZEN/checkpoints/PRIMARY_FROZEN_PARENT_MASK.npz`, `STEP6_FROZEN_DBN/checkpoints/01_FROZEN_STATE_SPACE.npz`, five Step6 ordered-probit heads and prediction NPZ files, and five DeepLOB model weight/result directories.

## 3. Optional full clean notebook execution

Open `Code/src/PGM_Project_Frozen.ipynb` in the original Colab/Drive environment, mount the same CausalLOB path, and execute existing cells in their written order. Some source cells contain absolute Drive paths and historical setup assumptions; an ordinary desktop Jupyter runtime is **not** plug-and-play for these notebook cells. The historical runtime was not locked for bitwise reruns. Do not rename the canonical user NPZ to the specialized Step2 state checkpoint or disable checksum guards.

Original essential Python dependencies include `tigramite==5.2.10.1`, NumPy, pandas, SciPy, scikit-learn, matplotlib, PyArrow, KaggleHub and PyTorch. The local audit dependencies are deliberately smaller; see `Code/requirements.txt`. Specific DeepLOB settings in frozen code: seed 5162026; windows 100 x 40; batch size 256; max epochs 15; patience 3; learning rate 0.001; weight decay 0.00001; gradient clipping 5. The inference/missingness experiments retain their original fixed parameters. Differences in PyTorch/BLAS/CPU/GPU versions may prevent bitwise matching on full retraining.

**Scope of verification:** Archived checkpoint integrity, direct probabilistic-score replay, table/parity checks and synthetic mechanism rerun were completed. A fresh end-to-end LPCMCI/DeepLOB **refit** was not run, and no new market holdout was evaluated.
