# Frozen final test results (2019)

## Run summary

This is the one and only evaluation the final selected model will ever receive against 2019, the project's frozen final test set. 2019 was reserved from the start and never touched during model development, training, hyperparameter decisions, or checkpoint selection for any model — the baselines, the MLP, or the Transformer.

The final model is the Transformer, epoch 10 (`docs/transformer-model-results.md`), which had already beaten every non-learned baseline and the MLP on set A (2018).

### Evaluation configuration

```bash
# Reproduces the sample used for this evaluation
shard_paths = sample_shard_paths(Path("data/processed/2019"), shard_count=60, seed=0)
evaluate_transformer_checkpoint(
    Path("data/checkpoints/transformer_model/epoch-10.pt"),
    Path("data/processed/2019"),
    shard_paths=shard_paths,
    max_decisions_per_shard=5000,
)
```

- Sample: 300,000 decisions from 2019 (60 shards, seed 0, capped at 5,000/shard) — the same shard-level random sampling design used for set A on 2018, applied here to 2019 for the same reason: avoiding the cluster-sampling bias a handful of full shards would introduce.
- Model: `data/checkpoints/transformer_model/epoch-10.pt`, unchanged from the checkpoint evaluated in `docs/transformer-model-results.md`.
- Legal-action rate: 100% by construction. Illegal actions are masked to near-zero probability before ranking (`mask_illegal_logits`), so the model cannot output an illegal action regardless of training quality — this isn't something that needed separate measurement.

## Results

| Sample | Top-1 | Top-3 | MRR | Cross-entropy |
| --- | ---: | ---: | ---: | ---: |
| Set A (2018, prior comparison) | 62.84% | 87.40% | 0.7623 | 1.0973 |
| **2019 (frozen final test)** | **62.58%** | **87.29%** | **0.7605** | **1.1027** |

The frozen test-set result is close to the set-A result: 0.26 percentage points lower on Top-1, 0.11 points lower on Top-3, essentially unchanged MRR and cross-entropy. This is a small, unremarkable gap consistent with ordinary sampling variation between two different years' data, not a sign of overfitting to 2018-specific patterns. It is the strongest generalization evidence available in this project, since 2019 never influenced any training or selection decision for this or any other model.

## Limitations

- Raw per-decision predictions were not preserved during this evaluation, so segmented analyses or reliability plots cannot be reproduced from this run without a second pass over 2019. Adding that capability now and rerunning would mean evaluating the frozen test set more than once, defeating its purpose — this is accepted as a deliberate scope tradeoff rather than retroactively fixed.
- As with set A, this is a 300,000-decision random-shard sample (60 of 2019's 880 shards), not the full ~88M-decision corpus, for the same statistical and practicality reasons documented in `docs/baseline-validation-results.md`.
- This evaluation used the exact same sampling seed (0) and shard count (60) as set A, but applied to a different dataset (2019 instead of 2018), so the specific shards selected are not the same files — only the sampling *method* is identical.
