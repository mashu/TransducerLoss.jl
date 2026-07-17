# Choosing a loss

Pick a variant from the table below before reading API details. Every entry
point is a **loss function** on logits your model already computed.

## Decision table

| Variant | Entry point | Logit inputs (shapes) | When to use | Why it exists |
|---------|-------------|----------------------|-------------|---------------|
| **RNN-T** | [`rnnt_loss_batched`](@ref) | `logits` `(V, T, U+1, B)` | Default transducer training; Graves-style blank/label lattice | Baseline end-to-end ASR objective (Graves 2012) |
| **TDT** | [`tdt_loss_batched`](@ref) | `logits` `(V, T, U+1, B)` + `dur_logits` `(D, T, U+1, B)` + `durations` (`AbstractVector` or `NTuple` of ints) | Frame-skipping decode, Parakeet-TDT-style models, multi-blank / variable stride | Generalizes RNN-T with a duration head (Xu et al., ICML 2023) |
| **Monotonic RNN-T** | [`monotonic_rnnt_loss_batched`](@ref) | `logits` `(V, T, U+1, B)` | At most one label per frame; RNA / diagonal-only alignments | Restricts lattice to diagonal moves (TDT with `durations = [1]`) |
| **HAT** | [`hat_loss_batched`](@ref) | `label_logits` `(V, T, U+1, B)` + `blank_logits` `(T, U+1, B)` — **blank-free** `V` | LM fusion with internal-LM subtraction; production shallow-fusion setups | Factorized blank enables principled LM combination (Variani et al., ICASSP 2020) |
| **Pruned RNN-T** | [`pruned_rnnt_loss_batched`](@ref) + [`pruning_bounds`](@ref) | `pruned_logits` `(V, T, S, B)` + `band_offsets` `(T, B)`; bounds helper takes `am` `(V, T, B)` + `lm` `(V, U+1, B)` | Long utterances where full `O(V·T·U·B)` joint does not fit in memory | Banded lattice matches vanilla RNN-T when band is wide enough (Kuang et al., Interspeech 2022) |
| **FastEmit** | `fastemit_lambda` on RNN-T / TDT | Same as parent variant | Reduce emission latency at decode time | Gradient-only regularization; loss scalar unchanged (Yu et al., arXiv:2010.11148) |
| **Minimum-latency** | `latency_lambda` on RNN-T only | Same as RNN-T | Penalize late label emissions during training | Adds exact expected emission frame to the objective (Shinohara & Watanabe, Interspeech 2022) |

### Tensor shape legend

| Symbol | Meaning |
|--------|---------|
| `V` | Vocabulary size (includes blank for RNN-T / TDT / pruned; **excludes** blank for HAT labels) |
| `T` | Max encoder frames (padded axis) |
| `U` | Max target length in the batch |
| `B` | Batch size |
| `D` | Number of duration classes (`length(durations)`) |
| `S` | Band width (`size(pruned_logits, 3)`); lattice cell `u = band_offsets[t, b] + s` |

**TDT extras:** `durations` must be ascending, unique, non-negative, and include
a value `≥ 1`. Blank transitions use durations `≥ 1`; tokens may use `d = 0`.
`sigma` (default `0`) is NeMo's logit under-normalization (≈ `0.05` recommended).

**Pruned extras:** `band_offsets` must be monotone per sample with
`band_offsets[1, b] == 0`. [`pruning_bounds`](@ref) estimates offsets from
trivial-joint posteriors (k2-inspired, not bit-compatible with k2's
gradient-based bounds).

## Typical use-cases

**Start with RNN-T** when building a standard encoder–predictor–joint transducer
and memory is not a bottleneck. One logit tensor, well-understood training
dynamics.

**Pick TDT** when the model has separate token and duration heads and you want
frame skipping at decode (e.g. NVIDIA Parakeet-TDT). Use `durations =
[0, 1, 2, 3, 4]` and `sigma ≈ 0.05` for NeMo-compatible training. Subsumes
multi-blank transducer variants.

**Pick Monotonic RNN-T** when alignments must stay on the diagonal (one token
per frame maximum). No duration head in your model — pass only token logits.

**Pick HAT** when you need **internal-LM subtraction** for principled fusion
with an external language model. Your joint network outputs separate label and
blank logits over a blank-free label vocabulary.

**Pick Pruned RNN-T** when utterances are long and the full `(V, T, U+1, B)`
joint tensor is too large. Use [`pruning_bounds`](@ref) to get `band_offsets`,
gather logits into `(V, T, S, B)`, then call [`pruned_rnnt_loss_batched`](@ref).
With `S = U + 1` and zero offsets the loss equals vanilla RNN-T exactly.

**Add `fastemit_lambda`** (RNN-T or TDT) when decode latency is too high —
scales label-emission gradients by `(1 + λ)` without changing the reported NLL.
Typical values are small (e.g. `0.004`).

**Add `latency_lambda`** (RNN-T only) when you want the loss itself to penalize
late emissions via the alignment posterior's expected frame index.

## Out of scope

| Idea | Status |
|------|--------|
| Multi-blank transducer | Subsumed by TDT duration head |
| Bayes-risk / delay-penalty (k2-style) | Not implemented |
| Alignment-restricted RNN-T (external alignments) | Not implemented |

## Memory

Vanilla RNN-T, TDT, and HAT materialize `O(T · U · B)` lattice state and consume
`O(V · T · U · B)` joint logits. Pruned RNN-T bounds the joint to
`O(V · T · S · B)` with `S ≪ U`. Shorter training windows or smaller batches
help when memory is tight.

## Automatic differentiation

Gradients are analytic `rrule`s computed inside the forward-backward — Zygote
never traces the lattice kernels. See [Examples](@ref) for a Zygote snippet.
