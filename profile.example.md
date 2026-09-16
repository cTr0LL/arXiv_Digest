# Research profile

<!--
Copy this file to profile.md and edit it. profile.md is gitignored, so what
you write there stays on your machine.

This file is the prompt, twice over: the prefilter embeds each line of it to
choose candidates, and the model reads all of it to score them. Write it the
way you would brief a new labmate.

Three things that matter more than they look:

1. Name methods, benchmarks and datasets explicitly. The prefilter matches
   semantically, so listing both "replay" and "rehearsal" costs nothing and
   widens recall.
2. Fill in "What I do not need". Negative examples prevent most false
   positives and are what stop the model inflating every score to a 4.
3. Every line is embedded on its own and a paper only has to match one of
   them. So keep lines specific -- a vague line matches everything weakly,
   which is worse than useless.

HTML comments like this one are stripped before embedding, so instructions in
here cost you nothing. Headings are stripped too.

The content below is a worked example. Replace it with yours.
-->

## What I work on

Continual learning for medical image classification, specifically replay-based
methods under domain shift, where the label set stays fixed but the scanner,
site or acquisition protocol changes between tasks.

## Methods I track

- Replay and rehearsal: experience replay, generative replay, buffer selection, herding
- Regularization-based continual learning: EWC, synaptic intelligence, MAS
- Prompt- and adapter-based continual learning: L2P, DualPrompt, CODA-Prompt, LoRA
- Forgetting metrics, stability-plasticity tradeoffs, task-order sensitivity

## Benchmarks and data I care about

Domain-incremental and class-incremental splits of medical imaging datasets such
as CheXpert, MIMIC-CXR, ISIC and Camelyon17. Split-CIFAR and Split-ImageNet only
when the method itself is the contribution rather than the application.

## What I do not need

- Continual learning for language models with no vision component
- Federated learning unless continual learning is the main axis of the work
- Neural architecture search, quantization, inference efficiency
- Surveys, position papers and benchmark-only contributions

## Context

I am implementing a replay baseline right now, so papers reporting buffer size
ablations or concrete implementation detail are worth more to me than
high-level surveys.
