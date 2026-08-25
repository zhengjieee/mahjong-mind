# Transformer model results

## Run summary

The Transformer discard policy was trained on 2017 and evaluated on set A, the same 300,000-decision random-shard sample of 2018 used for the non-learned baselines (`docs/baseline-validation-results.md`) and the MLP (`docs/mlp-baseline-results.md`). The checkpoint evaluated here was never scored against set A during training or checkpoint selection; this is the first time it was measured on that data.

### Training configuration

Training ran in two stages: 5 epochs first, then 5 more resumed from the `epoch-5` checkpoint after the first stage's validation trend suggested more headroom (no sign of plateauing at epoch 5). Resuming reused the first stage's weights and normalization statistics rather than redoing the already-completed epochs.

```bash
# Stage 1: epochs 1-5
PYTHONPATH=src .venv/bin/python -m mahjong_mind.modelling.transformer_model \
  data/processed/2017 --sample-shard-count 100 --max-decisions-per-shard 5000 \
  --epochs 5 --batch-size 256 --seed 0 \
  --checkpoint-dir data/checkpoints/transformer_model \
  --validation-dataset-directory data/processed/2018 \
  --validation-sample-shard-count 20 --validation-max-decisions-per-shard 1000 \
  --validation-seed 1 --exclude-sample-shard-count 60 --exclude-sample-seed 0

# Stage 2: epochs 6-10, resumed from epoch-5
PYTHONPATH=src .venv/bin/python -m mahjong_mind.modelling.transformer_model \
  data/processed/2017 --sample-shard-count 100 --max-decisions-per-shard 5000 \
  --epochs 5 --batch-size 256 --seed 0 \
  --checkpoint-dir data/checkpoints/transformer_model \
  --validation-dataset-directory data/processed/2018 \
  --validation-sample-shard-count 20 --validation-max-decisions-per-shard 1000 \
  --validation-seed 1 --exclude-sample-shard-count 60 --exclude-sample-seed 0 \
  --initial-checkpoint data/checkpoints/transformer_model/epoch-5.pt \
  --starting-epoch 6
```

- Architecture: 3-layer Transformer encoder, `d_model=128`, 4 attention heads, feed-forward hidden size 256, dropout 0.1. Input is a tokenized sequence (own hand, dora markers, opponents' unidentified concealed tiles, every discard with tsumogiri/riichi/called flags, every meld tile) plus a separate 37-dim context vector (scores, winds, riichi states, etc.) prepended as one extra token, whose encoded output is used as the pooled representation for the final head — the same role a BERT-style `[CLS]` token plays.
- Context features were standardized (zero mean, unit variance) the same way the MLP's dense features were, using statistics computed once from the training sample and reused across both training stages.
- Training data: the exact same 500,000 decisions from 2017 used for the MLP (100 shards, seed 0, capped at 5,000/shard), read once per epoch across all 10 epochs, so any difference in results reflects architecture and training length, not data budget.
- Optimizer: AdamW, learning rate 1e-3, batch size 256. Stage 1 (5 epochs) took 1 hour 25 minutes; stage 2 (5 more epochs, resumed) took 1 hour 39 minutes — training is substantially slower than the MLP's ~10 minutes for the same data budget, since self-attention costs more per decision than the MLP's plain matrix multiplies, and padding variable-length sequences to a batch's longest sequence adds further overhead.
- Early stopping: each epoch's checkpoint was scored against the exact same set B used for the MLP (20,000 decisions, 20 shards, seed 1, excluding set A's 60 shards). Validation Top-1 accuracy across all 10 epochs: 52.0%, 55.3%, 58.3%, 59.5%, 60.4%, 60.8%, 61.2%, 61.8%, 61.9%, 62.5%. Gains were large early (epochs 1-5: +8.4 points) and clearly diminishing later (epochs 6-10: +2.1 points), though still positive every single epoch — epoch 10 was selected as best.
- Checkpoints are saved locally under `data/checkpoints/transformer_model/` (gitignored, not committed).

## Results on set A (300,000 decisions, 60 shards, seed 0)

| Model | Top-1 accuracy | Top-3 accuracy | MRR | Cross-entropy |
| --- | ---: | ---: | ---: | ---: |
| Random legal | 14.5% | 33.0% | 0.323 | 2.227 |
| Most-common legal | 34.4% | 63.2% | 0.532 | 2.006 |
| Tile efficiency | 50.8% | 82.2% | 0.680 | 1.602 |
| MLP (epoch 5) | 56.5% | 81.5% | 0.709 | 1.324 |
| **Transformer (epoch 10)** | **62.8%** | **87.4%** | **0.762** | **1.097** |

The Transformer beats every model on every metric. It improves Top-1 accuracy by 6.3 percentage points over the MLP and 12.0 points over tile-efficiency, the strongest non-learned baseline.

## Consistency check against the validation slice

Epoch 10's score on set B during training (Top-1 62.52%, Top-3 86.96%, MRR 0.7591, cross-entropy 1.1045) is close to its score on set A above (Top-1 62.84%, Top-3 87.40%, MRR 0.7623, cross-entropy 1.0973). As with the MLP and the earlier 5-epoch checkpoint, this closeness is a reasonable sanity check that checkpoint selection was not overfit to a lucky validation sample.

## Methodology notes

- Set A and set B are the identical samples used for the MLP's evaluation, guaranteed disjoint by `sample_shard_paths`'s `exclude` parameter, so this comparison table is a genuine apples-to-apples measurement across all five models.
- Training was extended from 5 to 10 epochs specifically because the 5-epoch run showed no sign of plateauing (unlike the MLP, which had flattened by epoch 2-3) — this was a decision made from direct evidence in that run, not a speculative hyperparameter search. By epoch 10, gains had clearly slowed (diminishing but still positive), which is a reasonable point to stop without further investigation.
- `train_transformer` gained an `initial_checkpoint`/`starting_epoch` resume capability specifically to make this extension possible without repeating the first 5 epochs' ~1.5 hours of already-completed work. The optimizer itself restarts fresh on resume (its momentum/variance state isn't saved) — a deliberate simplification that doesn't appear to have caused any issue here, given the loss and validation trends continued smoothly across the resume point.
- Evaluating the checkpoint on set A took about 3 minutes — much faster than training, since it's a single forward pass with no backward pass or optimizer step.
