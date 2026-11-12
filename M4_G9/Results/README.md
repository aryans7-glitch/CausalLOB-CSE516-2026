# Archived evidence tables and figures

Figures 01–07 and tables 01–10/11 (MC) come from the prior M3 results packaging and are based on executed original notebook outputs. They have been carried forward without refitting the market model. Figures 08–10 are a newly drawn **diagram of the existing frozen graph** and plots of the separately replayed 40-seed synthetic experiment. The graph figure is illustrative and contains no new edges.

`11_synthetic_40rep_raw.csv` and `12_synthetic_40rep_summary.csv` are the earlier independent Step16 rerun's saved synthetic output. The executable `Code/src/reproduce_synthetic.py` recomputes the raw 320 rows with exact-score tolerance and writes `13_synthetic_paired_statistics.csv`. `14_matched_forecast_full_probabilistic_kpis.csv` is derived from Step13 and individually checked against original notebook text with `verify_archived_scores.py`. `15_transition_kpis.csv` and `16_full_observation_ece_unmatched_cohort.csv` represent the distinct Step8 transition and Step6 ECE cohorts.

**Caution:** Step6 ECE comes from a different full-observation cohort than the strictly matched DeepLOB comparison. Do not merge the two cohorts. D8–D10 are historically exposed. The 95% intervals and p-values in `13_synthetic_paired_statistics.csv` describe *only* simulated known-mechanism replications.

## Original frozen checkpoint audit extension

- `Tables/17_original_checkpoint_score_replay.csv`: independently recalculated full-observation Step6 five-horizon log-loss and Brier from original saved probabilities, alongside the number of original matched DeepLOB comparison rows.
- `Tables/18_reliability_bins_from_original_predictions.csv`: all 15 confidence bins at each of five forecast horizons, recovered from original saved Step6 prediction arrays.
- `Figures/11_reliability_original_frozen_probabilities.png`: example h=10 and h=30 reliability curves based on the preceding row-level tables, not invented values.
- `Code/checkpoints/`: original 244-file frozen export reconstructed from eleven 18-MiB parts; its complete per-file SHA-256 manifest is inside the restored ZIP.

These replays verify the original frozen results without claiming any new market-model training or future data.
