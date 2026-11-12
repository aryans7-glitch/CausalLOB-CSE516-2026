# CausalLOB — Dataset

Dataset: FI-2010 NoAuction Decimal Precision

The project uses 394,337 limit-order-book observations spanning 10 trading days, with 40 raw LOB columns and five prediction-horizon labels.

The canonical dataset is stored as `fi2010_canonical_raw40_labels.npz`.

Preprocessing reconstructs six market-state variables and identifies 50 day-by-security sequences to prevent temporal leakage.

See `Code/src/PGM_Project_Frozen.ipynb` for the original preprocessing and experimental pipeline.

Run the verification scripts documented in `Code/README.md` to validate the dataset and frozen experimental artifacts.
