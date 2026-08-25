# Baseline validation results

## Run summary

The random-legal, most-common-legal, and tile-efficiency baselines were evaluated on a random-shard sample of the processed 2018 dataset. The most-common-legal baseline was fitted on the full 2017 dataset (training corpus), not on 2018, so there is no fit/eval leakage in this run.

```bash
PYTHONPATH=src .venv/bin/python -m mahjong_mind.modelling.baselines.baseline_predictions \
  data/processed/2018 --frequency-fit-dataset data/processed/2017 \
  --sample-shard-count 60 --max-decisions-per-shard 5000 --seed 0 --tile-efficiency
```

- Frequency-fit dataset: `data/processed/2017`, dataset SHA-256 `ab65113473f948f7736f25fbc00d1bf9af8ac2c91d594297c77da3a88ee4b273`
- Evaluation dataset: `data/processed/2018`, dataset SHA-256 `bb6ed4ae2f6d54eef5e38e146153d8944d67418ac9867149b44b978df41ab1f3`
- Shard sampling: 60 shards randomly selected (seed 0) out of the 891 shards in the full 2018 dataset, each capped at 5,000 decisions, for exactly 300,000 evaluated decisions.

## Results

| Baseline | Top-1 accuracy | Top-3 accuracy | MRR | Cross-entropy |
| --- | ---: | ---: | ---: | ---: |
| Random legal | 14.5% | 33.0% | 0.323 | 2.227 |
| Most-common legal | 34.4% | 63.2% | 0.532 | 2.006 |
| Tile efficiency | **50.8%** | **82.2%** | **0.680** | **1.602** |

The tile-efficiency baseline again performed best on every metric, improving Top-1 accuracy by 16.4 percentage points and Top-3 accuracy by 19.0 percentage points over the most-common-legal baseline.

## Comparison with earlier results

These numbers are close to both the 2009 smoke test (Top-1 48.8% / Top-3 78.9% / MRR 0.658 for tile efficiency) and the earlier 3-shard/300,000-decision run on the same 2018 dataset (Top-1 50.7% / Top-3 82.0% / MRR 0.679), which is a reasonable sanity check that discard behaviour is stable both across sample years and across different random samples of 2018 itself.

## Methodology notes

- Sampling was done at shard granularity (`sample_shard_paths`, seed 0), rather than by truncating to the first N rows, to avoid biasing the sample toward whichever matches were written earliest into the dataset.
- This run improves on an earlier attempt that covered the same total decision count (300,000) using only 3 full shards. Reading 3 full shards is effectively cluster sampling with 3 clusters: because shards are built from matches processed in roughly chronological order, each shard is a contiguous block of ~194 matches, and decisions within a shard are correlated (shared time window, overlapping player pool) rather than independent. Standard-error formulas that assume independent observations understate the true uncertainty under that design.
- This run instead spreads the same 300,000-decision budget across 60 shards (about 7% of the 891 available), capping each shard's contribution to 5,000 decisions. With many more, much thinner clusters, no single contiguous block of matches can dominate the sample, which substantially reduces the design effect from intra-cluster correlation compared to the 3-shard version, at effectively the same runtime (60 file opens adds negligible overhead versus the per-decision cost of the tile-efficiency baseline).
- 300,000 decisions across 60 independent shards is treated as sufficient to serve as this baseline validation benchmark for the rest of the project.
