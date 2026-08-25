# Baseline smoke results

## Run summary

The random-legal, most-common-legal, and tile-efficiency baselines were evaluated on the first 1,000 decisions in the processed 2009 development dataset. The run completed successfully in 8.01 seconds.

```bash
PYTHONPATH=src .venv/bin/python -m mahjong_mind.modelling.baselines.baseline_predictions \
  data/processed/2009 --limit 1000 --seed 0 --tile-efficiency
```

Dataset SHA-256: `6b864d432f46839ae1d4e752fba8ad9422873d35b4391b47c4e65482cb813cd0`

## Results

| Baseline | Top-1 accuracy | Top-3 accuracy | MRR | Cross-entropy |
| --- | ---: | ---: | ---: | ---: |
| Random legal | 17.0% | 36.8% | 0.346 | 2.208 |
| Most-common legal | 34.1% | 59.9% | 0.523 | 2.008 |
| Tile efficiency | **48.8%** | **78.9%** | **0.658** | **1.647** |

The tile-efficiency baseline performed best on every reported metric. It improved Top-1 accuracy by 14.7 percentage points and Top-3 accuracy by 19.0 percentage points over the most-common-legal baseline on this sample.

## Limitations

These are development smoke results intended to verify the dataset-to-evaluation pipeline, not final validation results. The sample contains the first 1,000 decisions rather than a random or held-out selection, and the most-common baseline was fitted on the same 2009 dataset, causing leakage.
