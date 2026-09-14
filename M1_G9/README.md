# CausalLOB — CSE 516 Milestone 1

## Project
CausalLOB: Temporal Causal Discovery and Dynamic Bayesian Inference for Limit Order Book Price Formation

## Track
Applied AI

## Domain
Financial Market Microstructure / Limit Order Books

## Principal Dataset
FI-2010 NoAuction Decimal Precision

## Research Question
Which lagged limit-order-book states are stable probabilistic antecedents of short-horizon midpoint price formation, and can a Dynamic Bayesian Network represent those dependencies for calibrated probabilistic inference?

## PGM Mapping

### Representation
A temporal graphical model over:
- log relative spread
- log total L1-L5 depth
- L1 imbalance
- L2-L5 imbalance
- depth-normalized quote-change OFI proxy
- one-step midpoint return

Rolling volatility is used as a regime/sensitivity variable.

### Learning
- LPCMCI for temporal structure learning
- PCMCI+ as a causally sufficient comparison
- regularized/Bayesian parameter learning for the DBN

### Inference
- posterior probabilities of Up / Stationary / Down
- MAP inference
- Variable Elimination
- approximate inference under partial evidence

## M1 Contents
- Report/
- Video/
- Code/
- Data/
- Results/

## Reproducibility
The Colab notebook writes versioned checkpoints to Google Drive so preprocessing does not need to be repeated after runtime disconnects.
