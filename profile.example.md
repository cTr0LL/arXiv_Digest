# Research profile

<!--
Copy this file to profile.md and replace everything below with your own work.
profile.md is gitignored, so what you write there stays on your machine.

This file is the prompt, twice over: the prefilter embeds each line of it to
choose candidates, and the model reads all of it to score them. Write it the
way you would brief a new labmate.

Four things that matter more than they look:

1. Name methods, benchmarks and datasets explicitly. The prefilter matches
   semantically, so listing both "domain adaptation" and "distribution shift"
   costs nothing and widens recall.
2. Fill in "What I do not need". Anything under a heading matching that phrase
   is *subtracted* from a paper's score rather than added, which is what stops
   the model inflating every result to a 4. Rename that heading and the meaning
   flips.
3. Every line is embedded on its own and a paper only has to match one of
   them, so keep lines specific. A vague line matches everything weakly, which
   is worse than useless.
4. If your work sits at the intersection of two fields, say so in a single
   sentence rather than splitting it across two bullets. Separate lines make
   the filter return the union -- every paper in either field -- rather than
   the overlap you actually care about.

HTML comments like this one are stripped before embedding, and so are headings,
so instructions in here cost you nothing.

The profile below belongs to an invented researcher. It is filled in properly
so you can see the level of detail that works. Delete it.
-->

## What I work on

Crop-type mapping from Sentinel-2 satellite image time series, specifically
models that stay accurate when transferred to a region or growing season they
were not trained on. The hard part is that spectral signatures for the same
crop shift with latitude, soil and weather, so a model fitted in one country
degrades badly in the next.

## Methods I track

- Temporal encoders for satellite image series: U-TAE, temporal attention, LSTM and transformer baselines
- Self-supervised pretraining on unlabelled imagery: SatMAE, masked autoencoders, contrastive approaches
- Geospatial foundation models and how well they transfer without fine-tuning
- Label-efficient learning: few-shot adaptation, active learning, weak supervision from crop statistics
- Unsupervised domain adaptation across regions and across years

## Benchmarks and data I care about

PASTIS, BigEarthNet, EuroSAT and Sen1Floods11. Sentinel-1 radar fusion when it
is used to see through cloud cover rather than as an extra channel for its own
sake. Results reported across more than one country or season, since
single-region numbers say nothing about transfer.

## What I do not need

- Object detection in aerial or drone imagery, which is a different problem
- Super-resolution and image enhancement work with no downstream task
- Pure remote sensing applications with no machine learning contribution
- Neural architecture search, quantization and inference efficiency
- Surveys, position papers and dataset-only releases

## Context

I am reproducing a temporal transformer baseline on PASTIS at the moment, so
papers that report per-region breakdowns or release training code are worth
more to me than ones reporting a single aggregate number.
