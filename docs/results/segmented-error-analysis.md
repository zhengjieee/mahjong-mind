# Segmented error analysis (Transformer, epoch 10)

## Run summary

The final model (Transformer, epoch 10) was evaluated on set A (the same 300,000-decision sample of 2018 used for every other final comparison) with metrics broken down by five segment dimensions, in addition to the aggregate numbers already reported in `docs/results/transformer-model-results.md`. This reuses set A rather than the frozen 2019 test set, since only the one-time headline evaluation on 2019 is off-limits for repeated use — further analysis on set A carries no such restriction.

Every field used to build the segments (turn/draws-remaining, melds, dealer, riichi state, legal-action count) already exists in the Parquet decision schema, so this required no new data collection — only grouping decisions that were already being scored.

```bash
# Reproduces the sample
shard_paths = sample_shard_paths(Path("data/processed/2018"), shard_count=60, seed=0)
evaluate_transformer_checkpoint_segmented(
    Path("data/checkpoints/transformer_model/epoch-10.pt"),
    Path("data/processed/2018"),
    shard_paths=shard_paths,
    max_decisions_per_shard=5000,
)
```

## Results by segment

| Dimension | Bucket | Decisions | Top-1 | Top-3 | MRR | Cross-entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Overall | — | 300,000 | 62.8% | 87.4% | 0.762 | 1.097 |
| Game phase | early (draws_remaining ≥ 47) | 147,180 | 61.9% | 89.4% | 0.764 | 1.090 |
| Game phase | mid (24–46) | 109,158 | 63.2% | 85.3% | 0.758 | 1.127 |
| Game phase | late (≤ 23) | 43,662 | 65.0% | 85.9% | 0.769 | 1.048 |
| Hand openness | closed | 239,036 | 61.7% | 86.7% | 0.754 | 1.133 |
| Hand openness | open | 60,964 | 67.3% | 90.1% | 0.795 | 0.955 |
| Dealer status | dealer | 77,515 | 65.7% | 88.5% | 0.781 | 1.017 |
| Dealer status | non-dealer | 222,485 | 61.9% | 87.0% | 0.756 | 1.125 |
| Opponent riichi | no riichi | 247,752 | 64.4% | 88.4% | 0.774 | 1.054 |
| Opponent riichi | **riichi present** | 52,248 | **55.2%** | 82.7% | 0.707 | 1.304 |
| Candidate-set size | 10+ | 221,521 | 59.6% | 86.3% | 0.742 | 1.193 |
| Candidate-set size | 7–9 | 50,273 | 61.9% | 86.4% | 0.755 | 1.123 |
| Candidate-set size | 4–6 | 11,155 | 74.8% | 93.4% | 0.846 | 0.724 |
| Candidate-set size | 1–3 | 17,051 | 99.2% | 100.0% | 0.996 | 0.020 |

## Key finding: opponent riichi pressure is the clearest genuine weak point

Top-1 accuracy drops 9.2 percentage points (64.4% → 55.2%) when any opponent has an accepted riichi, the largest gap of any segment that isn't a mechanical artifact of the decision being easier or harder by construction (see caveat below). This is a meaningful, specific weakness: once an opponent declares riichi, the correct discard often depends on defensive tile-safety reasoning (which tiles are more likely to deal into their hand) rather than pure hand-building efficiency — a consideration the model was never given as an explicit input feature, only implicitly through the riichi-state tokens/flags it would have to learn the significance of purely by imitating human discards.

## Caveat: the candidate-set-size result is misleading at face value

The "1–3" bucket's 99.2% Top-1 accuracy is **not** evidence of strong model performance — it's largely a data-distribution artifact. When there is only 1 legal action (a common case immediately after declaring riichi, when only the newly drawn tile can be discarded), Top-1 accuracy is 100% for *any* predictor by construction, since there's nothing to get wrong. This bucket mixes those trivial forced-discard decisions with genuinely small (but non-trivial) 2–3-candidate decisions, inflating the average. The same effect likely contributes, to a lesser extent, to the "late" game-phase and "open" hand-openness buckets scoring better than their counterparts, since both correlate with fewer live options (later in a hand, or after committing to an open hand's shape, there are simply fewer sensible discards to choose between).

## Manual review: five wrong predictions under opponent riichi pressure

Five decisions where opponent riichi was present and the model's top choice didn't match the actual discard, pulled from set A:

1. **`...c0ef9756:681`** — Hand includes `2s 2s 3s`, `4p 4p`, `5pr`. Human discarded the isolated `4m`; model ranked `3s`/`2s` (breaking the 2s-2s-3s shape) above it, with `4m` third at 19.4% probability. A plausible, non-egregious disagreement — the model's top pick isn't unreasonable, just not the human's choice.
2. **`...c0ef9756:683`** — Human tsumogiri'd `F` (an isolated honor tile just drawn) — a standard safe, low-commitment discard, especially reasonable under riichi pressure. Model preferred `6m` (25.6% vs `F`'s 21.6%). This looks like a case where the model favored hand-efficiency over the safer, more defensively-standard play.
3. **`...c28f86e7:158`** — The model's top three picks (`6p`, `P`, `9p`) all draw from the hand's pin-suit shapes; the human instead cleared the one isolated souzu tile (`8s`), which only ranked 5th at 4.9% probability. The largest miss of the five — the model appears to have misjudged which shape to preserve.
4. **`...c28f86e7:259`** — Hand holds a `C C` (red dragon) pair. Human discarded one `C`, breaking the pair — plausibly to shed a tile that becomes progressively more dangerous to hold against a riichi the longer it's kept. Model ranked it third (20.2%) behind `1s` and `3m`.
5. **`...c28f86e7:265`** — A near three-way tie: `1p` (32.8%), `3m` (32.1%, the human's actual choice), `2m` (29.7%). Not a meaningful error — the model's probabilities barely separate the top options.

Reading across these: three of the five (2, 3, 4) are plausibly explained by the model underweighting defensive/safety reasoning relative to hand-building efficiency once a riichi is live, consistent with the segment-level finding above. One (5) isn't really an "error" at all — the model's top choice and the human's choice were nearly tied. This is a small, qualitative sample and shouldn't be read as proof of a specific mechanism, only as a plausible, consistent story alongside the aggregate numbers.

## Methodology notes

- Segment thresholds: game phase splits on `draws_remaining` (early ≥ 47, mid 24–46, late ≤ 23 — roughly thirds of a hand's live wall); candidate-set size buckets the legal-action count (1–3, 4–6, 7–9, 10+). Both are simple, documented choices, not tuned or validated against any ground truth — reasonable starting buckets for exploratory analysis, adjustable later if needed.
- This analysis required a new module, `segmented_evaluation.py`, since the existing training/evaluation dataset (`TransformerDiscardDataset`) discards the raw row after tokenizing and never carried segment-relevant fields forward. The new module reuses the existing tokenizer, collate function, masking, and prediction logic unchanged — the only new code is deriving segment labels from fields already in the schema and tracking one accumulator per (dimension, bucket).
- This is exploratory analysis, not a model change. No decision was made based on these results that affects the already-frozen 2019 test result in `docs/results/frozen-test-results.md`.
