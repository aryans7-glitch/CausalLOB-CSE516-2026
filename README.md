# CausalLOB — Milestone 3: Results & Analysis

**Course:** CSE 516 — Probabilistic Graphical Models  
**Project:** CausalLOB: Temporal Causal Discovery and Dynamic Bayesian Inference for Limit Order Book Price Formation  
**Group:** G9  
**Project Mode:** Applied AI  
**Domain:** Financial Market Microstructure  
**Dataset:** FI-2010 NoAuction Decimal Precision  
**Milestone:** M3 — Experimental Results, Evaluation and Analysis

**Repository:** https://github.com/aryans7-glitch/CausalLOB-CSE516-2026

## 1. Project Overview

CausalLOB investigates whether a Dynamic Bayesian Network (DBN) can represent meaningful lagged dependencies between limit-order-book states and support interpretable probabilistic inference.

The project focuses on three core Probabilistic Graphical Model components:

- **Representation:** A six-variable temporal Bayesian graphical model describing liquidity, imbalance, order flow and midpoint returns.
- **Learning:** LPCMCI-based temporal structure discovery, followed by regularized parameter estimation.
- **Inference:** Exact Gaussian posterior inference under incomplete observations, alongside probabilistic forecasting and approximate-inference comparisons.

Because FI-2010 is observational, discovered dependencies are treated as candidate temporal relationships rather than experimentally verified causal effects.

## 2. Continuity with Previous Milestones

### Milestone 1 — Proposal and Scoping

Established the research problem, FI-2010 data preparation, six-variable state representation, temporal structure-learning plan, SOTA references, baselines and evaluation KPIs.

### Milestone 2 — Technical Foundation

Implemented the CausalLOB pipeline, including:

- Canonical reconstruction of 394,337 representations.
- Fifty boundary-separated day-by-security sequences.
- LPCMCI-based temporal structure discovery.
- Frozen Sparse-39 DBN with 30 own-history and 9 cross-variable parent slots.
- Ridge-based linear-Gaussian parameter learning.
- Five ordered-probit probabilistic outcome heads.
- Exact inference and preliminary AR(5) baseline comparisons.

### Milestone 3 — Results and Analysis

Evaluates the frozen M2 model through baseline comparisons, probability-quality KPIs, robustness analysis, ablations, statistical summaries, inference benchmarking, error analysis and revised SOTA positioning.

The original research question, dataset, primary state variables and frozen graph are preserved.

## 3. Milestone 3 Objectives

This milestone investigates:

1. Under which conditions does Sparse-39 outperform an equal-history AR(5) baseline?
2. How does the sparse graphical representation compare with dense VAR(5) and predictive benchmarks?
3. How well calibrated are the predicted class probabilities?
4. How robust is inference under missing observations and consecutive observation gaps?
5. What is the relationship between approximate-inference error and computational cost?
6. Which assumptions and model components limit performance?

The objective is to understand when graphical modeling is useful, not to demonstrate superiority in every experiment.

## 4. Repository Structure

```text
M3_G9/
├── README.md
├── Report/
│   ├── M3_G9_Report.pdf
│   └── M3_G9_Report.tex
├── Code/
│   ├── README.md
│   ├── requirements.txt
│   └── src/
│       ├── PGM_Project_Original.ipynb
│       ├── M3_G9_Results_Reanalysis.ipynb
│       └── verify_m3_tables.py
├── Data/
│   └── README.md
└── Results/
    ├── README.md
    ├── EVIDENCE_STATUS.md
    ├── Source_provenance.sha256
    ├── Figures/
    └── Tables/
```

## 5. Experimental Evaluation

The report covers the following categories:

| Category | Evaluation |
|---|---|
| Representation | Frozen graph structure, sparsity and dependency interpretation |
| Baselines | AR(5), dense VAR(5), random-graph diagnostic and predictive comparisons |
| Learning | Frozen structure and fitted transition parameters |
| Inference | Posterior reconstruction, missing-state inference and numerical approximation |
| Probability quality | Log-loss, Brier score and calibration |
| Robustness | Missing-state patterns and consecutive observation gaps |
| Ablation | Own-history-only, sparse graphical structure and alternative model assumptions |
| Statistical analysis | Descriptive day-level mean ± standard deviation and diagnostic uncertainty |
| SOTA | Comparison with LPCMCI research and predictive limit-order-book methods |

## 6. Principal Findings

The experimental evidence shows mixed performance:

- Sparse-39 provides modest improvements over AR(5) for selected reconstruction and inference tasks.
- Dense VAR(5) achieves slightly better pooled transition RMSE than Sparse-39.
- DeepLOB-style predictive benchmarks achieve lower matched log-loss across all five tested horizons.
- The training class-prior baseline outperforms the DBN in all 12 evaluated consecutive-observation-gap settings.
- Approximate Gaussian inference exhibits decreasing numerical error as Monte Carlo sample count increases, with additional computational cost.

These findings identify both the benefits and limitations of the graphical representation.

## 7. Statistical Methodology and Evidence Limitations

The primary evaluation follows the original chronological setup:

- **Days 1–5:** Structure discovery.
- **Days 6–7:** Candidate graph validation.
- **Days 1–7:** Model parameter fitting.
- **Days 8–10:** Retrospective evaluation.

Trading days are treated as the outer descriptive evaluation blocks. The 50 separated sequences and overlapping observation windows are not treated as statistically independent experiments.

Day-level mean ± standard deviation is reported where supported. The available three-day retrospective evaluation does not justify strong confirmatory significance claims.

The report distinguishes:

- Previously recorded FI-2010 empirical results.
- Descriptive statistical summaries.
- Synthetic Gaussian numerical experiments.

The supplied M3 re-analysis notebook reproduces archived tables, plots and numerical consistency checks. It does not claim a fresh end-to-end FI-2010 rerun because the original Google Drive checkpoints and raw dataset files were unavailable in the submission-building environment.

## 8. Running the Optional Code

Use Python 3.10 or newer.

From the `M3_G9` directory, install dependencies:

```bash
python -m pip install -r Code/requirements.txt
```

Validate the packaged results:

```bash
python Code/src/verify_m3_tables.py
```

Open the re-analysis notebook:

```bash
jupyter notebook Code/src/M3_G9_Results_Reanalysis.ipynb
```

The original notebook requires its separately documented FI-2010 data preparation and Google Drive checkpoints. See `Code/README.md` and `Data/README.md` for additional details.

## 9. State-of-the-Art Positioning

The primary methodological reference remains:

**Gerhardus, A. and Runge, J. (2020).** *High-Recall Causal Discovery for Autocorrelated Time Series with Latent Confounders.* NeurIPS.

LPCMCI motivates the structure-discovery component of CausalLOB.

The project extends this methodological foundation into a domain-specific Dynamic Bayesian Network for probabilistic inference and evaluation.

Predictive comparisons, including DeepLOB-style models, show that the interpretability and probabilistic structure of CausalLOB do not automatically translate into superior forecasting performance.

The project therefore positions itself as an experimental study of interpretable temporal probabilistic modeling, rather than a state-of-the-art trading predictor or proof of market causality.

## 10. Limitations and Future Work

Current limitations include:

- Observational data and possible latent confounding.
- Only ten trading days and anonymous securities.
- Retrospective rather than researcher-blind held-out evaluation.
- Sensitivity to missing-observation patterns.
- Simplifying Gaussian and conditional-independence assumptions.
- Lack of complete ground-truth causal structure.

Future work includes external dataset replication, stronger calibration analysis, additional genuinely unseen evaluation periods, more extensive uncertainty quantification and improved handling of missing observations.

**CSE 516 — Probabilistic Graphical Models | CausalLOB | Group G9 | Milestone 3**
