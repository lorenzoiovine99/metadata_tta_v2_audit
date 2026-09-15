# Metadata TTA — Current Pipeline, Reproduction and Status

## 1. Scope

This document describes the current implementation and experimental protocol for the **Metadata Test-Time Adaptation (Metadata TTA)** method used in this repository.

The current implementation has been validated on:

* FMoW
* Yearbook
* HuffPost

The objective of the current phase was to obtain a clean Metadata TTA baseline before evaluating more complex adaptation methods such as:

* TENT
* Temporal Gradient
* Consensus

The previous joint-training formulation of the double-head model has been replaced by a formulation designed to preserve the original main-task source model exactly.

---

# 2. Core idea

At test time, the metadata associated with the current sample is assumed to be available **before the main-task prediction**.

For a test sample \(x_t\), with metadata target \(m_t\), the protocol is:

1. restore the online adapter to the source-model state;
2. compute the auxiliary metadata loss for the current sample;
3. perform one gradient update of the online adapter using the metadata loss;
4. predict the main label for the same sample;
5. discard the adapted adapter state;
6. repeat independently for the next sample.

Therefore, the current Metadata TTA implementation is:

**episodic + pre-prediction + sample-wise**.

There is no accumulation of adaptation across test samples.

---

# 3. Model architecture

The model contains:

* shared representation / trunk;
* online adapter;
* main classification head;
* auxiliary metadata head.

The important distinction from the old implementation is how the double-head model is constructed.

## Old formulation

Previously, the double-head model was trained jointly with an objective of the form:

$$
L = L_{\text{main}} + \lambda L_{\text{aux}}
$$

This meant that the auxiliary task could modify the representation used by the main classifier.

As a consequence:

```text
single_head_frozen != double_head_frozen
```

and comparisons between Metadata TTA and the single-head baseline were confounded by a different source model.

## Current formulation

The new pipeline is:

```text
train single source model
        |
        v
copy the complete main path exactly
        |
        v
create double-head model
        |
        v
train ONLY aux_head on source metadata
        |
        v
Metadata TTA
```

The single source model is therefore the reference main predictor.

The double-head model starts from an exact copy of it.

Only the newly initialized auxiliary head is trained after the copy.

---

# 4. Single → Double exact initialization

The function:

```python
initialize_double_from_single(...)
```

copies the complete compatible state of the trained single-head model into the double-head model.

The only parameters expected to be absent from the single model are:

```text
aux_head.weight
aux_head.bias
```

A sanity test was added to verify that the main predictions are exactly preserved.

Expected output:

```text
FINAL_AUX_ONLY_MAIN_SANITY |
max_abs_logit_diff=0.000000000000e+00 |
predictions_equal=True
```

This sanity check has passed on:

* FMoW
* Yearbook
* HuffPost

Therefore:

```text
single_head_frozen == double_head_frozen
```

before Metadata adaptation.

This is a key property of the current implementation.

---

# 5. Auxiliary-head-only source training

After copying the single model into the double model, the auxiliary metadata classifier is trained using:

```python
train_aux_head_only(...)
```

During this stage:

```text
shared representation: frozen
online adapter:         frozen
main head:              frozen
auxiliary head:         trainable
```

The auxiliary head learns to predict metadata from the representation produced by the source model.

After training, its parameters are frozen again.

The auxiliary head is not intended to improve the source main-task classifier directly.

Its role is to provide a differentiable metadata objective during TTA.

---

# 6. Why a frozen auxiliary head can adapt the model

During Metadata TTA, the auxiliary head itself is frozen.

However, the auxiliary loss has the computational path:

```text
input
  |
  v
online adapter
  |
  v
features
  |
  v
frozen aux head
  |
  v
metadata loss
```

The gradient therefore propagates through the frozen auxiliary classifier and reaches the online adapter.

Only the online adapter is updated.

The auxiliary head acts as a fixed differentiable metadata probe.

---

# 7. Metadata TTA update

For every test sample:

```text
reset adapter to source
        |
        v
current sample x_t
        |
        v
known metadata m_t
        |
        v
auxiliary CE loss
        |
        v
one gradient step on online_adapter
        |
        v
main prediction y_t
        |
        v
discard adapted state
```

The important configuration is:

```yaml
metadata:
  batch_size: 1
  steps: 1
  episodic_pre_prediction: true
```

The current experiments also use:

```yaml
regularization: 0.0
gradient_clip: 0.1
normalized_aux_loss_threshold: 0.0
```

The learning rate is tuned independently for each dataset using pseudo-OOD source years.

---

# 8. Fast episodic reset

The original episodic implementation used a complete method reset for every test sample.

This was unnecessarily expensive because it could:

* reload the whole model;
* clone state;
* rebuild the optimizer;
* repeat this tens of thousands of times.

The Metadata method now implements:

```python
reset_online_adapter_to_source()
```

This restores only the parameters that can actually change during Metadata TTA.

It also:

```python
optimizer.zero_grad(set_to_none=True)
optimizer.state.clear()
```

so Adam does not preserve momentum/state between episodic samples.

This preserves the intended sample-wise episodic protocol while avoiding full model reconstruction.

---

# 9. Source temporal training

The configuration:

```yaml
training:
  temporal:
    enabled: true
```

must not be confused with:

```yaml
methods:
  temporal_gradient: false
```

These are different mechanisms.

`training.temporal` belongs to the supervised ID/source training protocol.

`methods.temporal_gradient` is a test-time adaptation method evaluated on OOD data.

For the canonical FMoW / Yearbook / HuffPost runs, supervised temporal source training remains enabled.

The current common setup uses:

```yaml
training:
  base:
    epochs: 50

    early_stopping:
      enabled: true
      patience: 5
      min_delta: 0.0001

  temporal:
    enabled: true
    epochs: 50
    reset_optimizer_each_year: false

    early_stopping:
      enabled: true
      patience: 5
      min_delta: 0.0001
```

Dataset-specific hyperparameters such as learning rate, weight decay and hidden dimension are tuned separately.

---

# 10. Hyperparameter tuning protocol

The tuning procedure has two relevant stages for the current Metadata implementation.

## Stage 1 — Single-head source model

The single-head supervised model is tuned on the full source period.

Example:

```bash
python scripts/tune.py \
  --config configs/tuning/yearbook_tuning.yaml \
  --stage baseline
```

The old double-head joint-training stage is currently disabled:

```yaml
double_head:
  enabled: false
```

This is intentional.

The Metadata double model is now generated directly from the best single model.

---

# 11. TTA pseudo-OOD tuning

Metadata hyperparameters are selected without using the real OOD test period.

The final portion of the source period is used as pseudo-OOD.

The earlier source years form the pseudo-source period.

Conceptually:

```text
true source period
-------------------------------------------------
pseudo-source                 pseudo-OOD
```

The model is trained using pseudo-source years and Metadata TTA is evaluated on the held-out later source years.

Command:

```bash
python scripts/tune.py \
  --config configs/tuning/<dataset>_tuning.yaml \
  --stage tta
```

The resulting trials are stored under:

```text
results/tuning/<dataset>/tta/metadata/trials.csv
```

---

# 12. Dataset-specific tuning splits

## FMoW

True source:

```text
2002–2012
```

Metadata tuning split:

```text
pseudo-source: 2002–2008
pseudo-OOD:    2009–2012
```

True OOD:

```text
2013–2017
```

Selected Metadata learning rate:

```text
3e-4
```

---

## Yearbook

True source:

```text
1930–1970
```

Metadata tuning split:

```text
pseudo-source: 1930–1966
pseudo-OOD:    1967–1970
```

True OOD:

```text
1971–2013
```

Selected Metadata learning rate:

```text
3e-6
```

---

## HuffPost

True source:

```text
2012–2015
```

Because only four source years are available, the last source year is used as pseudo-OOD:

```text
pseudo-source: 2012–2014
pseudo-OOD:    2015
```

True OOD:

```text
2016–2018
```

Selected Metadata learning rate:

```text
3e-6
```

---

# 13. Baseline tuning results

## Yearbook

Best single-head trial:

```text
learning_rate      = 3e-4
weight_decay       = 1e-4
ifdropout            = 0.1
shared_hidden_dim  = 512
batch_size         = 32
```

Mean source validation accuracy:

```text
0.98661
```

Yearbook is therefore already close to the performance ceiling on much of the source period.

---

## HuffPost

Best single-head trial:

```text
learning_rate      = 3e-4
weight_decay       = 0.0
dropout            = 0.1
shared_hidden_dim  = 512
batch_size         = 32
```

Mean source validation accuracy:

```text
0.74353
```

---

# 14. Metadata tuning results

## FMoW

Pseudo-OOD 2009–2012:

```text
LR = 3e-4
mean delta ≈ +1.66 pp
4 / 4 years improved
```

This was the strongest Metadata result among the tested datasets.

---

## Yearbook

Pseudo-OOD 1967–1970:

```text
LR = 3e-6
mean delta ≈ +0.15 pp
worst delta = 0
1 / 4 years improved
```

`3e-6` and `1e-5` produced the same prediction-level result.

The smaller value was selected as a conservative tie-break.

The very high frozen accuracy creates a substantial ceiling effect.

---

## HuffPost

Pseudo-OOD 2015:

```text
1e-6  -> unchanged
3e-6  -> +0.038 pp
1e-5  -> +0.038 pp
3e-5  -> unchanged
1e-4  -> -0.076 pp
3e-4  -> -0.457 pp
```

This shows a clear dependence on adaptation magnitude.

Larger Metadata learning rates reproduce the degradation seen in older experiments.

The conservative value:

```text
3e-6
```

was selected.

---

# 15. Final OOD results

## FMoW

Final years:

```text
2013–2017
```

Metadata versus frozen source:

```text
2013: +1.009 pp
2014: -0.133 pp
2015: +1.374 pp
2016: +0.553 pp
2017: +0.728 pp
```

Summary:

```text
4 / 5 years improved
simple mean ≈ +0.71 pp
weighted mean ≈ +0.70 pp
≈ +177 correct predictions
```

FMoW currently provides the clearest evidence that Metadata TTA can improve the main classifier.

### Important methodological note

The FMoW 2013–2017 period had already been inspected during earlier iterations of the method and motivated architectural changes.

Therefore these results should be treated as exploratory/post-hoc OOD evidence rather than a completely untouched blind final test.

---

## Yearbook

Final OOD:

```text
1971–2013
```

Total samples:

```text
4142
```

Canonical run:

```text
frozen correct   = 3626
Metadata correct = 3627
```

Weighted accuracy improvement:

```text
≈ +0.024 percentage points
```

Prediction-level changes occurred in only five years:

```text
1975: -1 correct
1985: +1 correct
1991: -1 correct
1998: +1 correct
2007: +1 correct
```

Summary:

```text
3 years improved
2 years worsened
38 years unchanged
net = +1 correct prediction
```

Interpretation:

Metadata is essentially neutral/stable on Yearbook.

It does not provide a strong improvement, but it also does not systematically damage the already strong source model.

---

## HuffPost

Final OOD:

```text
2016–2018
```

Total samples:

```text
4881
```

Results:

```text
2016:
frozen   = 0.6639658849
Metadata = 0.6639658849

2017:
frozen   = 0.6393629124
Metadata = 0.6393629124

2018:
frozen   = 0.6606683805
Metadata = 0.6606683805
```

Summary:

```text
0 years improved
0 years worsened
3 years unchanged
weighted delta = 0
net correct predictions = 0
```

This is important relative to the old Metadata experiments.

The previous implementation could degrade HuffPost.

With the new conservative episodic formulation, the degradation disappears.

Metadata is exactly neutral at the prediction level for the selected learning rate.

---

# 16. Current overall conclusion

The current Metadata implementation produces three qualitatively different but coherent outcomes.

```text
FMoW:
clear positive effect

Yearbook:
almost completely neutral,
very small positive net effect

HuffPost:
exactly neutral
```

The method should therefore not be described as universally improving accuracy.

A more accurate current conclusion is:

> Metadata TTA can improve performance when the metadata gradient provides a useful adaptation direction, while the source-preserving and episodic formulation greatly reduces the risk of degrading the original classifier.

An equally important result is that the new implementation removes the source-model confound.

For all three datasets:

```text
single_head_frozen == double_head_frozen
```

before adaptation.

---

# 17. Running a complete experiment

The main evaluation entry point is:

```bash
python scripts/run.py \
  --config configs/<dataset>_eval_stream_tas.yaml
```

Examples:

```bash
python scripts/run.py \
  --config configs/fmow_eval_stream_tas.yaml
```

```bash
python scripts/run.py \
  --config configs/yearbook_eval_stream_tas.yaml
```

```bash
python scripts/run.py \
  --config configs/huffpost_eval_stream_tas.yaml
```

A useful way to retain the complete terminal output is:

```bash
python scripts/run.py \
  --config configs/huffpost_eval_stream_tas.yaml \
  2>&1 | tee huffpost_final_metadata.log
```

---

# 18. Required sanity check

After every Metadata experiment, verify:

```bash
grep "FINAL_AUX_ONLY_MAIN_SANITY" <log_file>
```

Expected:

```text
FINAL_AUX_ONLY_MAIN_SANITY |
max_abs_logit_diff=0.000000000000e+00 |
predictions_equal=True
```

Any failure of this condition must be investigated before interpreting Metadata results.

---

# 19. Result files

Final OOD metrics are stored under:

```text
results/<dataset>/eval_stream_tas/<run_id>/metrics/ood.csv
```

For example:

```text
results/yearbook/eval_stream_tas/
results/huffpost/eval_stream_tas/
results/fmow/eval_stream_tas/
```

The OOD CSV contains:

```text
year
method
accuracy
n_samples
```

The important comparisons for the current Metadata stage are:

```text
single_head_frozen
double_head_frozen
metadata
```

The first two must match exactly.

---

# 20. Relevant implementation files

The main modified files for the current Metadata pipeline are:

```text
src/metadata_tta/training/supervised.py
src/metadata_tta/training/__init__.py

src/metadata_tta/tta/metadata.py

src/metadata_tta/evaluation/evaluator.py

src/metadata_tta/tuning/baseline_tuner.py
src/metadata_tta/tuning/tta_tuner.py

src/metadata_tta/experiment/eval_stream_tas_runner.py
```

Relevant configurations include:

```text
configs/fmow_eval_stream_tas.yaml
configs/yearbook_eval_stream_tas.yaml
configs/huffpost_eval_stream_tas.yaml

configs/tuning/fmow_tuning.yaml
configs/tuning/yearbook_tuning.yaml
configs/tuning/huffpost_tuning.yaml
```

---

# 21. Important implementation changes already completed

The following issues have already been addressed.

## Reproducibility

Random seeds are reset after model construction during tuning so that architecture creation does not change the training RNG trajectory.

This fixed the earlier mismatch between:

```text
single
```

and:

```text
double with lambda = 0
```

---

## Source-preserving double model

Implemented exact single-to-double transfer.

---

## Auxiliary-only training

Implemented source training in which only the metadata auxiliary classifier is optimized.

---

## Episodic pre-prediction evaluation

Implemented Metadata adaptation before the current sample's main-task prediction.

---

## Fast sample-wise reset

Implemented adapter-only reset with optimizer state clearing.

---

## TTA tuner dependency

The TTA tuner no longer requires a separately tuned old joint double-head model.

It derives the Metadata double model directly from the best single-head configuration.

---

## Main-path sanity checks

Added checks both during tuning and final evaluation to guarantee equality between single and frozen double main predictions.

---

# 22. What is intentionally NOT being tuned

At this stage, some choices are kept fixed across datasets to avoid unnecessarily expanding the search space.

These include the common training protocol:

```text
maximum epochs = 50
early stopping = enabled
patience = 5
min_delta = 1e-4
```

and the Metadata structure:

```text
batch size = 1
steps = 1
regularization = 0
gradient clipping = 0.1
normalized auxiliary threshold = 0
episodic pre-prediction = true
```

The primary Metadata hyperparameter currently tuned per dataset is the learning rate.

---

# 23. What remains to be done

Metadata TTA is now considered sufficiently implemented and validated to serve as the basic metadata adaptation baseline.

The next experimental phases should be performed independently and in order.

## 23.1 TENT

Implement / validate TENT using exactly the same source models and OOD protocol.

TENT should be evaluated independently of Metadata first.

This provides a metadata-free TTA reference.

---

## 23.2 Temporal Gradient

Only after Metadata and TENT are stable should the Temporal Gradient method be activated.

The method-specific parameters should be tuned using the same pseudo-OOD discipline.

The real OOD period must not be used for hyperparameter selection.

---

## 23.3 Consensus

Consensus should be evaluated only after the simpler gradient-based method is understood.

Again:

```text
pseudo-OOD -> select hyperparameters
true OOD   -> final evaluation only
```

## Future extension — cumulative Metadata TTA and adaptive reset

The current Metadata TTA implementation is intentionally **fully episodic**.

For every test sample:

```text
restore source adapter
        |
        v
metadata update on x_t
        |
        v
main prediction
        |
        v
discard adapted state
```

Therefore, no information acquired through Metadata TTA is currently accumulated across samples.

This design was chosen first because it provides a conservative and easy-to-interpret Metadata baseline. It also strongly limits catastrophic drift, which was observed in earlier continual Metadata experiments.

However, this is not necessarily the final form of the method.

A natural extension is **cumulative Metadata TTA**, in which the adapted online adapter is retained across consecutive samples:

```text
source adapter
     |
     v
update on x_t
     |
     v
predict y_t
     |
     v
keep adapted adapter
     |
     v
update on x_{t+1}
     |
     v
...
```

Earlier experiments showed that naive continual accumulation can drift substantially. Therefore, cumulative adaptation should probably not be used without an explicit reset mechanism.

### Adaptive reset

A future version could use a detector that decides when the accumulated adapted state is no longer trustworthy and should be reset to the source model.

Possible reset signals include:

* sudden increase in auxiliary metadata loss;
* moving-average auxiliary loss degradation;
* abnormal gradient norm;
* abrupt change in gradient direction;
* excessive parameter distance from the source adapter;
* sudden increase in prediction entropy;
* strong change in metadata distribution;
* disagreement between current and historical adaptation directions;
* explicit temporal change-point detection.

The resulting method would be a hybrid between fully episodic and fully continual adaptation:

```text
adapt
  |
  v
accumulate
  |
  v
accumulate
  |
  v
detector triggers
  |
  v
RESET TO SOURCE
  |
  v
adapt again
```

This creates three conceptually distinct Metadata TTA variants:

```text
1. Episodic Metadata TTA
   reset after every sample

2. Continual Metadata TTA
   never reset

3. Adaptive-reset Metadata TTA
   accumulate updates until a drift/change detector triggers a reset
```

The current repository has validated only the first variant.

The second variant was explored experimentally and showed significant drift, so it should not currently be considered a stable baseline.

The third variant is a promising future direction and should be evaluated only after the current episodic Metadata baseline and the other TTA baselines are fully established.

An important future research question is therefore not only:

> Should metadata adaptation be accumulated?

but also:

> For how long should adaptation be accumulated before returning to the source state?

This reset decision can itself become part of the TTA algorithm.

# 24. Recommended order from this point

```text
1. Freeze current Metadata implementation
2. Clean remaining debug / legacy code
3. Validate TENT on FMoW
4. Validate TENT on Yearbook
5. Validate TENT on HuffPost
6. Summarize Metadata vs TENT
7. Validate Temporal Gradient
8. Validate Consensus
9. Run final controlled comparisons
10. Produce aggregate tables and figures
```

---

# 25. Remaining code cleanup

Before considering the repository final, the following cleanup tasks should be reviewed.

## Temporary debug prints

`tta_tuner.py` may still contain temporary messages such as:

```text
DEBUG_METADATA
before_method_construction
after_method_construction
before_evaluation
after_evaluation
```

These should be removed once no longer needed.

## Legacy comments

Some comments still refer to continuous Metadata streams.

The current Metadata implementation is episodic per sample, so these comments should be updated.

## Legacy `metadata.episodic`

The important switch for the current implementation is:

```yaml
episodic_pre_prediction: true
```

The older:

```yaml
episodic: false
```

configuration field should eventually be reviewed or documented to avoid ambiguity.

## Old double-head tuning code

The repository still contains code paths for the original joint double-head training because they may be useful for reference experiments.

They should not be confused with the current Metadata source construction.

---

# 26. Experimental discipline going forward

For every new TTA method:

1. tune the source baseline using source data only;
2. choose a temporal pseudo-OOD split inside the source period;
3. tune TTA parameters only using that pseudo-OOD split;
4. freeze all method hyperparameters;
5. run the true OOD period;
6. never retune based on final OOD performance;
7. report frozen source and adapted results together;
8. retain prediction-equivalence sanity checks when applicable.

This is especially important because earlier exploratory FMoW OOD results have already influenced development decisions.

Future comparisons should keep a clear distinction between:

```text
development / exploratory evidence
```

and:

```text
untouched final evaluation
```

---

# 27. Current status

As of the current repository state:

```text
Metadata source formulation:       DONE
Auxiliary-only training:           DONE
Episodic pre-prediction TTA:       DONE
Fast episodic reset:               DONE
Single/double equality sanity:     DONE

FMoW Metadata validation:          DONE
Yearbook Metadata validation:      DONE
HuffPost Metadata validation:      DONE

TENT validation:                   TODO
Temporal Gradient validation:      TODO
Consensus validation:              TODO
Final multi-method comparison:     TODO
```

Metadata TTA can now be treated as the stable first TTA baseline for the next phase of the project.

## Future extension — cumulative Metadata TTA and adaptive reset

The current Metadata TTA implementation is intentionally **fully episodic**.

For every test sample:

```text
restore source adapter
        |
        v
metadata update on x_t
        |
        v
main prediction
        |
        v
discard adapted state
```

Therefore, no information acquired through Metadata TTA is currently accumulated across samples.

This design was chosen first because it provides a conservative and easy-to-interpret Metadata baseline. It also strongly limits catastrophic drift, which was observed in earlier continual Metadata experiments.

However, this is not necessarily the final form of the method.

A natural extension is **cumulative Metadata TTA**, in which the adapted online adapter is retained across consecutive samples:

```text
source adapter
     |
     v
update on x_t
     |
     v
predict y_t
     |
     v
keep adapted adapter
     |
     v
update on x_{t+1}
     |
     v
...
```

Earlier experiments showed that naive continual accumulation can drift substantially. Therefore, cumulative adaptation should probably not be used without an explicit reset mechanism.

### Adaptive reset

A future version could use a detector that decides when the accumulated adapted state is no longer trustworthy and should be reset to the source model.

Possible reset signals include:

* sudden increase in auxiliary metadata loss;
* moving-average auxiliary loss degradation;
* abnormal gradient norm;
* abrupt change in gradient direction;
* excessive parameter distance from the source adapter;
* sudden increase in prediction entropy;
* strong change in metadata distribution;
* disagreement between current and historical adaptation directions;
* explicit temporal change-point detection.

The resulting method would be a hybrid between fully episodic and fully continual adaptation:

```text
adapt
  |
  v
accumulate
  |
  v
accumulate
  |
  v
detector triggers
  |
  v
RESET TO SOURCE
  |
  v
adapt again
```

This creates three conceptually distinct Metadata TTA variants:

```text
1. Episodic Metadata TTA
   reset after every sample

2. Continual Metadata TTA
   never reset

3. Adaptive-reset Metadata TTA
   accumulate updates until a drift/change detector triggers a reset
```

The current repository has validated only the first variant.

The second variant was explored experimentally and showed significant drift, so it should not currently be considered a stable baseline.

The third variant is a promising future direction and should be evaluated only after the current episodic Metadata baseline and the other TTA baselines are fully established.

An important future research question is therefore not only:

> Should metadata adaptation be accumulated?

but also:

> For how long should adaptation be accumulated before returning to the source state?

This reset decision can itself become part of the TTA algorithm.
