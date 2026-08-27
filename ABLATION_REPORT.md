# Chemprop's Size Leak

**Ablation report — 27 Aug 2026 · 18 runs · 5 endpoints · 10-fold Butina CV**

A forked message-passing block never beat stock chemprop on any endpoint. Its one
apparent win came from the aggregation function quietly encoding molecule size — a
hypothesis that then predicted the sign of the effect on three more datasets before
they were run.

| | |
|---|---|
| Architecture variants tested | 9 |
| Variants that beat baseline | 0 |
| Extra parameters carried | 270,600 (+85%) |
| Advance predictions held | 3 / 3 |

---

## 1. What was actually different

The modded model started as an experiment in giving chemprop's message-passing
recurrence a gated feed-forward block, plus AdamW in place of the hardcoded Adam. By
the time it was measured properly it differed from stock in four ways at once, which
is why the first comparison was uninterpretable and the rest of this work is an
ablation.

| Component | Baseline | Modded (final) |
|---|---|---|
| Message passing | `BondMessagePassing` | `ModdedBondMessagePassing` |
| Residual block | none | gated `ResidualFFN`, shared |
| Aggregation | `MeanAggregation` | `NormAggregation` |
| Optimizer | Adam | AdamW, `wd = 0` |
| Depth | 3 | 3 |
| Parameters | 318,301 | 588,901 |

One of those rows is already inert. **AdamW at `weight_decay = 0` is mathematically
identical to Adam**, and every configuration that reached parity ran at zero decay — so
the optimizer change, which was the fork's original motivation, has no effect in the
model as it now stands.

### Weight decay, tested directly

Decay was A/B'd on its own: `0.01` against `0.0`, nothing else changed.

| metric | wd = 0.01 | wd = 0.0 | paired diff | p |
|---|---|---|---|---|
| MAE | 0.4061 | **0.3989** | +0.0072 | 0.282 |
| RMSE | 0.5460 | **0.5368** | +0.0092 | 0.202 |
| R² | 0.2281 | **0.2540** | −0.0259 | 0.203 |

All three favour *no* decay; none significant. Neutral at best, mildly harmful at face
value.

---

## 2. Nine variants, none of them better

**HLM_CLINT · 3,087 compounds · 1,561 Butina clusters**

Every configuration was cross-validated on identical Butina splits at
`random_state=42`, making all comparisons paired. The search covered capacity, depth,
initialisation, aggregation, optimizer and training budget.

| config | params | MAE | RMSE | R² | vs baseline |
|---|---|---|---|---|---|
| **baseline, 60 epochs** | **318 K** | **0.3769** | **0.4980** | **0.3576** | — |
| baseline, 40 epochs | 318 K | 0.3799 | 0.4978 | 0.3586 | tie, p=0.35 |
| v5 — shared block, 100 ep | 589 K | 0.3804 | 0.4972 | 0.3584 | tie, p=0.70 |
| v3 — zero-init + dropout + Norm | 589 K | 0.3815 | 0.4979 | 0.3577 | tie |
| A — random init | 589 K | 0.3836 | 0.5019 | 0.3467 | tie, p=0.35 |
| v4 — per-level blocks | 1.13 M | 0.3837 | 0.5036 | 0.3435 | tie |
| B — depth 6 | 589 K | 0.3843 | 0.5103 | 0.3262 | worse on RMSE, sign p=0.022 |
| v1 — depth 6, Sum, per-level | 3.6 M | 0.3967 | 0.5232 | 0.2907 | worse, p=0.003 |
| v2 — no dropout, Mean, wd=0 | 589 K | 0.3989 | 0.5368 | 0.2540 | worse, p=0.006 |
| v2 — no dropout, Mean, wd=0.01 | 589 K | 0.4061 | 0.5460 | 0.2281 | worse, p=0.005 |

Sorted by MAE. Nothing lands above the baseline.

### Two hypotheses that died on contact

**Overparameterisation.** v1 carried 11× the baseline's weights, so cutting it to 1.85×
should have closed the gap. It didn't — v2 was marginally worse.

**An idle block.** The `ResidualFFN` zero-initialises its output projection, so it
starts as an exact identity; the natural guess is that it stays there. A forward-hook
probe measured the actual residual contribution on held-out molecules:

> `‖f(x) − x‖ / ‖x‖` = **2.34×** for the shared block (per-level variants: 4.48× and
> 2.32×). It rewrites the hidden state at more than twice the magnitude of its own
> input, and converts none of that into accuracy. **Capacity was never the constraint.**

The probe also caught a real bug in the per-level variant: `__init__` allocated `depth`
blocks while the forward loop only indexed `depth − 1` of them, leaving the last block
never called and 270,600 parameters permanently dead. That variant's score was measured
against two working blocks, not three.

---

## 3. The aggregation is not normalising anything

Repairing v2 up to parity took three simultaneous changes: zero-initialising the output
projection, plumbing dropout into the block, and swapping `MeanAggregation` for
`NormAggregation`. Arm A later showed the initialisation was not load-bearing. That
leaves the aggregation, and its implementation is the whole story
(`chemprop/nn/agg.py`):

```python
# h = (1/|V|) sum h_v  -- denominator is this molecule's atom count
class MeanAggregation(Aggregation):
    ...  reduce="mean"

# h = (1/c) sum h_v  -- denominator is a constant, for every molecule
class NormAggregation(SumAggregation):
    def __init__(self, dim=0, *args, norm: float = 100.0, **kwargs):
        ...
    def forward(self, H, batch):
        return super().forward(H, batch) / self.norm
```

`MeanAggregation` divides by `|V|`, so doubling a molecule's atoms doubles both the sum
and the divisor — the embedding's magnitude is invariant to size. `NormAggregation`
divides by a fixed `100.0` regardless: a 40-atom molecule sums forty vectors and
divides by the same constant as a 20-atom one. **The norm of the representation handed
to the readout therefore grows roughly linearly with atom count.**

Despite the name this is *not* a size normalisation. It is `SumAggregation` with a
cosmetic rescale that keeps activations in range. The size signal is not removed, only
shrunk uniformly — which makes it a free feature wherever molecular size happens to
predict the target.

| endpoint | n | atoms (mean ± sd) | corr(size, target) |
|---|---|---|---|
| MDR1_MDCK_ER | 2,642 | 23.4 ± 5.3 | **+0.493** |
| HLM_CLINT | 3,087 | 23.1 ± 5.3 | +0.320 |
| RLM_CLINT | 3,054 | 23.0 ± 5.4 | +0.224 |
| PK_AUC | 187 | 26.3 ± 4.6 | +0.060 |
| SOLUBILITY | 2,173 | 23.1 ± 5.2 | **−0.213** |

The atom-count distributions are near-identical across endpoints. What varies
enormously is how strongly size predicts the target — from +0.49 on MDR1, where size
alone explains roughly a quarter of the variance, to −0.21 on solubility, where the
same signal points the wrong way.

---

## 4. Three forecasts, made in advance

That mechanism makes a falsifiable claim: the modded model's advantage should track
`corr(size, target)` and change sign with it. Each prediction below was written down
before its run started.

| endpoint | corr | predicted | observed RMSE diff | p | folds won | verdict |
|---|---|---|---|---|---|---|
| MDR1_MDCK_ER | +0.49 | modded wins | −0.0255 | 0.070 | 7/10 | held |
| HLM_CLINT | +0.32 | tie | −0.0008 | 0.943 | 5/10 | held |
| SOLUBILITY | −0.21 | modded loses | +0.0330 | 0.062 | 2/10 | held |
| PK_AUC | +0.06 | tie | −0.0297 | 0.695 | 5/10 | uninformative |

Monotone in the correlation across the three informative endpoints, with the effect
crossing zero right about where the correlation does.

**The solubility result is the load-bearing one.** No account in which the
`ResidualFFN` is genuinely learning chemistry predicts that the same architecture would
get *worse* on a new endpoint — but the size-leak account predicts exactly that,
including which metrics move (RMSE and R² shift while MAE barely does, because the
effect lands on the large-error compounds).

PK_AUC is excluded from the pattern: both arms scored below a mean-predictor there, so
its position carries no information about the architecture. See below.

### Full endpoint results

| endpoint | arm | MAE | RMSE | R² |
|---|---|---|---|---|
| MDR1_MDCK_ER | baseline | 0.3870 | 0.5244 | 0.4161 |
| MDR1_MDCK_ER | modded | **0.3821** | **0.4989** | **0.4689** |
| HLM_CLINT | baseline | **0.3769** | 0.4980 | 0.3576 |
| HLM_CLINT | modded | 0.3804 | **0.4972** | **0.3584** |
| SOLUBILITY | baseline | **0.3735** | **0.5601** | **0.3002** |
| SOLUBILITY | modded | 0.3896 | 0.5931 | 0.2258 |
| PK_AUC | baseline | 1.0045 | 1.2366 | −0.0308 |
| PK_AUC | modded | **0.9840** | **1.2069** | −0.0351 |

---

## 5. The endpoint that broke both models

**PK_AUC · 187 compounds · Butina cutoff 0.4**

The in-house PK set needed work before it could be used at all: SMILES live in a `mol`
column, and raw AUC spans 0 to 110,845 with skew 3.70, so the target became
`log10(AUC + 1)`. The `+1` offset exists to absorb nine true zeros. It is a congeneric
series, so Butina at the standard 0.65 put a third of the set in one cluster and
produced folds ranging from 2 to 62 compounds; the cutoff was loosened to 0.4 to get
workable splits.

Both models then finished with **negative R²** — worse than predicting the training
mean. A random forest on Morgan fingerprints, given the identical folds, does not have
that problem:

| model | split | pooled R² | MAE |
|---|---|---|---|
| Random forest, Morgan FP | random | **+0.417** | 0.717 |
| Random forest, Morgan FP | Butina scaffold | **+0.176** | 0.927 |
| Predict global mean | — | 0.000 | 1.086 |
| chemprop, modded | Butina scaffold | −0.035 | 0.984 |
| chemprop, baseline | Butina scaffold | −0.031 | 1.005 |

A random forest extracts real signal under the same scaffold split where both graph
networks land below the mean-predictor. This is not a hard-dataset problem — it is a
**small-data problem specific to chemprop here**, fitting 318 K–589 K parameters to
roughly 168 training compounds per fold. The baseline did not even converge cleanly,
hitting its epoch ceiling in 3 of 10 folds.

PK_AUC is therefore unusable as an architecture benchmark; any A/B run on it measures
noise. If the endpoint itself matters, fingerprints with a tree ensemble is the working
approach today, and pretraining the GNN on the larger ADME endpoints before fine-tuning
is the obvious thing to try next.

One data caveat: nine compounds have AUC exactly 0. If those are below-detection rather
than true zeros they are censored observations and should not be plain regression
targets at all.

---

## 6. What to do with the fork

- **The `ResidualFFN` has never earned its parameters.** Across nine variants and four
  usable endpoints, no configuration beat baseline for a reason attributable to the
  block. It costs +85% parameters and demonstrably rewrites the hidden state without
  converting that into accuracy.
- **The AdamW change is currently a no-op.** At `weight_decay = 0` it is Adam. Decay was
  directly tested and was neutral-to-slightly-harmful, so switching it back on is not
  the fix.
- **Aggregation is the one real lever found.** `NormAggregation` supplies molecule size
  as a feature. That is worth a measurable amount where size predicts the target and
  costs a measurable amount where it doesn't — and it needs zero extra parameters.
- **Choose aggregation per endpoint.** Computing `corr(size, target)` on the training
  split takes seconds and correctly called the sign of the effect three times out of
  three. It is a cheaper and more reliable knob than anything in the fork.

### Open: the missing control

The decisive experiment has not been run: **stock chemprop with `NormAggregation` and
no `ResidualFFN`**, on MDR1. If it reproduces the modded model's gain, the entire fork
reduces to a one-line aggregation swap and 270,600 parameters can be deleted. It needs
a small change to `models/chemprop_ref.py` to make aggregation selectable, then roughly
15 minutes of GPU.

---

## 7. How much of this to believe

Two limits apply to everything above and neither is small.

**Every comparison is single-seed.** Splits are shared across arms so the pairing is
valid, but model initialisation and data order are fixed at one value. Fold-to-fold
spread on these endpoints is comparable to the effects being measured, so a seed sweep
could move individual results.

**No correction for multiple comparisons.** Roughly fifteen paired tests were run. An
uncorrected `p = 0.06` in that context is not a finding on its own. What carries weight
here is the pattern — three sign predictions stated before the runs and confirmed
after — not any single p-value.

The size-leak mechanism is well supported: it is visible directly in the aggregation
source, it explains the v3 repair, and it predicted three outcomes in advance. The
individual endpoint effect sizes are suggestive and should be re-measured across seeds
before anything is built on them.

---

## Reproduction

All runs used 10-fold Butina cross-validation at `random_state=42`, batch size 32,
early-stopping patience 10, on identical splits per endpoint. Butina cutoff 0.65 for
ADME endpoints, 0.4 for PK_AUC. Results and per-fold metrics are under `results/`, one
directory per run.

```bash
# baseline
uv run python cli.py evaluate --endpoint HLM_CLINT --model CHEMPROP \
  --train-config.max-epochs 60

# modded, best-known config (v5)
uv run python cli.py evaluate --endpoint HLM_CLINT --model CHEMPROP_MODDED \
  --train-config.max-epochs 60 --train-config.weight-decay 0.0
```

Depth and block initialisation are configuration knobs rather than source edits:

- `--train-config.mp-depth INT` — message-passing depth (default 3)
- `--train-config.ffn-zero-init` / `--train-config.no-ffn-zero-init` — whether
  `ResidualFFN.down_proj` starts at zero (default on)

Both are defined in `config.py` and plumbed through `models/chemprop_modded_ref/`.
