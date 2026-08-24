# MLP baseline results

## Run summary

The dense MLP discard baseline was trained on 2017 and evaluated on set A, the same 300,000-decision random-shard sample of 2018 used for the non-learned baselines in `docs/baseline-validation-results.md`. The checkpoint evaluated here was never scored against set A during training or checkpoint selection; this is the first time it was measured on that data.

### Training configuration

```bash
PYTHONPATH=src .venv/bin/python -m mahjong_mind.modelling.mlp_baseline \
  data/processed/2017 --sample-shard-count 100 --max-decisions-per-shard 5000 \
  --epochs 5 --batch-size 256 --seed 0 \
  --checkpoint-dir data/checkpoints/mlp_baseline \
  --validation-dataset-directory data/processed/2018 \
  --validation-sample-shard-count 20 --validation-max-decisions-per-shard 1000 \
  --validation-seed 1 --exclude-sample-shard-count 60 --exclude-sample-seed 0
```

- Architecture: two hidden layers of 256 units (ReLU, dropout 0.1), input the shared `dense-observation-v1` encoding (928 features), output 37 discard logits, illegal actions masked to `-1e9` before softmax.
- Features were standardized (zero mean, unit variance) using statistics computed once from the training sample.
- Training data: 500,000 decisions from 2017 (100 shards, seed 0, capped at 5,000/shard).
- Optimizer: AdamW, learning rate 1e-3, batch size 256, 5 epochs.
- Early stopping: each epoch's checkpoint was scored against set B, a disjoint 20,000-decision sample of 2018 (20 shards, seed 1), explicitly excluding the 60 shards used for set A. Epoch 5 had the best validation Top-1 accuracy (56.74%) and was selected.

## Results on set A (300,000 decisions, 60 shards, seed 0)

| Model | Top-1 accuracy | Top-3 accuracy | MRR | Cross-entropy |
| --- | ---: | ---: | ---: | ---: |
| Random legal | 14.5% | 33.0% | 0.323 | 2.227 |
| Most-common legal | 34.4% | 63.2% | 0.532 | 2.006 |
| Tile efficiency | 50.8% | **82.2%** | 0.680 | 1.602 |
| **MLP (epoch 5)** | **56.5%** | 81.5% | **0.709** | **1.324** |

The MLP improves Top-1 accuracy by 5.7 percentage points over tile-efficiency, the strongest non-learned baseline, and clearly improves MRR and cross-entropy as well. Tile-efficiency remains marginally ahead on Top-3 accuracy (82.2% vs 81.5%) — the MLP is more often exactly right and better calibrated, but very slightly less likely to have the true discard somewhere in its top three.

## Consistency check against the validation slice

Epoch 5's score on set B during training (Top-1 56.74%, Top-3 81.22%, MRR 0.7097, cross-entropy 1.3209) is close to its score on set A above (Top-1 56.50%, Top-3 81.46%, MRR 0.7091, cross-entropy 1.3242). Since set A never influenced which epoch was selected, this closeness is a reasonable sanity check that the selection wasn't overfit to a lucky validation sample.

## Methodology notes

- Set A and set B are guaranteed disjoint: `sample_shard_paths` excludes set A's 60 shards before sampling set B's 20, so no decision was scored by both.
- This evaluation used the exact same set-A shard sample (`sample_shard_paths(shard_count=60, seed=0)`) as the non-learned baselines, so the comparison table above is a genuine apples-to-apples measurement, not a re-sampled approximation.
- Training loss decreased every epoch (1.479 → 1.329 → 1.277 → 1.243 → 1.215) and had not clearly plateaued by epoch 5, so more epochs or more training data could plausibly improve results further; that was out of scope for this run.
