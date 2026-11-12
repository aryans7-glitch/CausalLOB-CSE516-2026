# Evidence provenance — final frozen-checkpoint audit

| Finding | Source and verification | Qualification |
|---|---|---|
| Canonical 394,337 rows, 40 raw variables, 5 labels, 50 segment starts | Supplied canonical NPZ independently SHA-256/shape/day verified | Does not independently regenerate publisher message-level labels |
| Frozen six-node, 39-slot graph and Gaussian parameters | Original Step4 parent-mask NPZ matches original Step6 frozen state-space NPZ byte-for-byte in graph membership; 244 source export digests passed | Does not establish identified market causal edges |
| AR(5), Sparse-39, dense VAR(5), 100 randomized-graph comparisons | Pooled summary KPIs cross-checked against original Step8 score CSV | Same retrospective market data; graph alternatives are not independent new experiments |
| Full-evidence five-horizon ordered-probit log-loss and Brier | Recalculated directly from every original saved Step6 probability NPZ; matches original Step6 score CSV | Original outputs, not freshly fitted predictions |
| Matched DeepLOB vs DBN vs class prior, all five horizons | Original Step12 comparison CSV validated against report's complete matched KPI table and executed Step13 notebook printout | DeepLOB weights supplied and hash-verified, but neural forward inference/retraining not rerun |
| DBN confidence ECE and reliability bins | Recomputed 15-bin reliability tables from original Step6 prediction arrays, all five ECE values equal source results | Step6 full-evidence cohort, not DeepLOB matched cohort |
| Six missing-state masks and 12 sequential gap contrasts | Original Step7/Step9 results preserved; every displayed gap score cross-checked with Step9 | Missingness imposed artificially, not historical exchange outages |
| 40-replication same-mechanism/flip simulation | Independently rerun from Step16 mechanism; all 320 scores matched source; genuine synthetic paired CI/p computed | Does not prove a FI-2010 causal or statistical effect |
| Separate 50-replication Gaussian MC numerical fixture | Archived M3 notebook results | Not a remeasured inference latency for the actual fitted market model |

**Original checkpoint export:** 244 hashed entries; 26 explicitly required files; one fixed frozen mask; five ordered-probit heads; five DeepLOB weight sets. The entire 201,392,753-byte original ZIP is restored from eleven GitHub-friendly chunks. All original archive file hashes passed the manifest verification.

**Not claimed:** fresh LPCMCI discovery, new model fitting, DeepLOB forward rerun, independent reconstruction of original message-level outcome labeling, genuine post-development market holdout, market-side confirmatory p-values or 95% confidence intervals, trading profitability or external validation. Days 8–10 were previously examined. The simulation's 95% intervals and tests apply only to its artificial mechanism.
