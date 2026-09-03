# Model lifecycle

How a checkpoint becomes the served model, and how to take it back out.

## What is served, and how to tell

The service loads one checkpoint at startup, named by `MAHJONG_MIND_CHECKPOINT`
and defaulting to `data/checkpoints/transformer_model/epoch-10.pt`. That file is
tracked in git — every other checkpoint is not — so an image builds from the
repository alone and a deploy is reproducible from one commit.

The served model identifies itself as `<file stem>@<first 8 of its sha256>`,
derived at load time rather than written down. `GET /health` and every
`/recommend` response report it. A hardcoded string would keep naming the old
model after the file changed underneath it; a content hash cannot.

```
$ curl -s https://mahjong-mind-908350631195.asia-southeast1.run.app/health
{"status":"ok","model_version":"epoch-10@bdfc2200"}
```

## Promotion criterion

A checkpoint replaces the served model only if all of the following hold. This
is a deliberately small bar for a single-maintainer project; it exists so the
decision is written down rather than remembered.

1. **Ranking quality.** Top-1 on set A at least as good as the incumbent's
   62.84%, measured once, after selection is finished. Set A is the fixed
   300,000-decision sample from 2018 (60 shards, seed 0, capped at 5,000 per
   shard) used for final comparison only.
2. **Model selection kept separate.** Early stopping and any other choice
   between candidates uses a set-B-style sample that excludes set A's shards.
   A checkpoint chosen using set A is not eligible, because its set A score
   would then report the luck of that sample rather than its quality.
3. **No defensive regression.** Top-1 with an opponent in riichi no worse than
   the incumbent's 55.2%. This is the model's known weak segment, and an
   overall gain that comes out of this bucket is not an improvement worth
   shipping.
4. **Legal actions only.** Illegal actions masked before ranking, and every
   recommended tile in the player's hand. Guaranteed by construction, and
   covered by the `/recommend` tests.
5. **Green pipeline.** Tests, ruff and mypy pass, and the deployed revision
   answers `/health`.

The 2019 frozen test set is **not** part of this. It was evaluated exactly once,
in Week 7, and its value came entirely from being touched once. Do not evaluate
any future checkpoint against it.

## Deploying a different model

Replace `data/checkpoints/transformer_model/epoch-10.pt`, or repoint
`MAHJONG_MIND_CHECKPOINT`, and push to `main`. CI builds an image tagged with
the commit sha, deploys it, and fails the workflow if the new revision does not
become healthy. Confirm the swap took effect by reading `model_version` from
`/health` — it changes with the file.

## Rolling back

Every deployed image is tagged with the commit that produced it, so rollback is
a redeploy of an earlier tag:

```bash
gcloud run deploy mahjong-mind \
  --image=asia-southeast1-docker.pkg.dev/project-d62b6639-d6d0-430b-98b/mahjong-mind/service:<earlier-sha> \
  --region=asia-southeast1
```

Roughly thirty seconds. The registry keeps the three most recent tagged images,
so there are always two earlier versions to return to; anything older is
rebuilt from git instead, which takes a few minutes.

Cloud Run also lists earlier *revisions*, which can outlive the images they
reference. A revision whose image has been cleaned up will fail on cold start,
so prefer redeploying a tag you have confirmed is still in the registry.
