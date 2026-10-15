# Data

The project uses the FI-2010 Limit Order Book dataset,
NoAuction, Decimal Precision representation.

The full dataset is not stored directly in this GitHub repository because of size.

The M2 code documents the download/preparation procedure and reconstructs
the leakage-safe modelling variables used in CausalLOB.

Expected canonical observations: 394,337.

Important preprocessing rule:
temporal lags must never cross day/security sequence boundaries.
