# TransducerLoss.jl

The **transducer-loss family** for Julia — the companion package to
[CTCLoss.jl](https://github.com/mashu/CTCLoss.jl), sharing its conventions
(blank = **last** class by default, loss = **batch mean**, analytic gradients
w.r.t. **raw logits** delivered through ChainRulesCore `rrule`s, device-agnostic
kernels via [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl) —
CPU and CUDA run the same code).

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/mashu/TransducerLoss.jl")
```

From a local checkout:

```julia
Pkg.develop(path = "/path/to/TransducerLoss.jl")
```

## Quick start

```julia
using TransducerLoss

# Vanilla RNN-T (Graves 2012): logits (V, T, U+1, B)
loss = rnnt_loss_batched(logits, targets, input_lengths)
loss = rnnt_loss_batched(logits, targets, input_lengths, blank)

# TDT — Token-and-Duration Transducer (Xu et al., ICML 2023):
# token head (V, T, U+1, B) + duration head (D, T, U+1, B)
loss = tdt_loss_batched(logits, dur_logits, targets, input_lengths,
                        [0, 1, 2, 3, 4]; blank, sigma = 0.05)

# Monotonic RNN-T (diagonal-only alignments): TDT with durations = [1]
loss = monotonic_rnnt_loss_batched(logits, targets, input_lengths)

# HAT — Hybrid Autoregressive Transducer (Variani et al., ICASSP 2020):
# label_logits (V, T, U+1, B), blank_logits (T, U+1, B) — blank-free V
loss = hat_loss_batched(label_logits, blank_logits, targets, input_lengths)

# Pruned RNN-T — banded lattice (Kuang et al., Interspeech 2022):
offsets = pruning_bounds(am, lm, targets, input_lengths; band_width = 4)
loss = pruned_rnnt_loss_batched(pruned_logits, offsets, targets, input_lengths)

# Regularization kwargs on RNN-T / TDT
loss = rnnt_loss_batched(logits, targets, input_lengths; fastemit_lambda = 0.004)
loss = rnnt_loss_batched(logits, targets, input_lengths; latency_lambda = 0.01)
```

### Tensor shapes

- **Logits**: `(V, T, U+1, B)` — raw joint-network outputs before softmax.
  `T` bounds encoder frames; `U+1` bounds prediction states (`⟨start⟩` plus up
  to `U` labels).
- **Targets**: `Vector{Vector{Int}}` — label indices (no blank).
- **Input lengths**: `Vector{Int}` — valid encoder frames per sample.
- **Blank** defaults to `size(logits, 1)` (last class), matching CTCLoss.jl.

For TDT, `dur_logits` has shape `(D, T, U+1, B)` with one class per entry of
`durations` (ascending, e.g. `[0, 1, 2, 3, 4]`). The `sigma` parameter is
NeMo's logit under-normalization constant (≈ 0.05 recommended; `0` disables).

For HAT, `label_logits` is `(V, T, U+1, B)` over a blank-free vocabulary and
`blank_logits` is `(T, U+1, B)`.

For pruned RNN-T, `pruned_logits` is `(V, T, S, B)` where `S` is the band
width; `band_offsets` is `(T, B)` with `u = band_offsets[t, b] + s`.

### Automatic differentiation

Gradients are supplied through ChainRulesCore `rrule`s computed inside the
forward-backward pass, so [Zygote](https://github.com/FluxML/Zygote.jl) never
traces the lattice kernels:

```julia
using TransducerLoss, Zygote

grad = Zygote.gradient(
    l -> rnnt_loss_batched(l, targets, input_lengths),
    logits,
)[1]

gtok, gdur = Zygote.gradient(
    (l, d) -> tdt_loss_batched(l, d, targets, input_lengths, durations),
    logits, dur_logits,
)
```

## Why one package

All variants operate on the same `(t frames consumed, u labels emitted)`
lattice and share emission gather, α initialization, NLL read-out, label
packing, numerics, and test oracles in `src/core/`. The `src/rnnt/`,
`src/tdt/`, `src/hat/`, and `src/pruned/` subtrees depend **only on
core/**, never on each other — so extracting one later is a file move;
bundling avoids duplicated kernels or a micro-package reaching into another
package's unexported internals. NeMo, k2, and torchaudio ship their
transducer variants together for the same reason.

```
src/
  TransducerLoss.jl   module root
  core/
    utils.jl          logaddexp
    labels.jl         pack_transducer_targets
    kernels.jl        emission gather, α init, NLL read-out (loss-agnostic)
  rnnt/
    kernels.jl        RNN-T diagonal + gradient kernels
    loss.jl           driver, API, rrules
  tdt/
    kernels.jl        TDT diagonal + gradient kernels
    loss.jl           driver, API, rrule
  hat/
    kernels.jl        HAT gather + factorized gradient
    loss.jl           driver, API, rrule
  pruned/
    kernels.jl        banded-lattice column kernels + trivial joint
    loss.jl           banded loss, API, pruning_bounds
test/
  reference.jl        pure-Julia references + brute-force oracles
  test_rnnt.jl, test_tdt.jl, test_hat.jl, test_pruned.jl
```

## Algorithms

Forward-backwards run as **diagonal wavefronts** (RNN-T, TDT, HAT) or
**column sweeps** (pruned): lattice cells on one anti-diagonal or frame
column are independent given their neighbours — one kernel launch per step,
parallel over `(u, batch)` or `batch`.

**RNN-T**: blank advances time by one, label `y_{u+1}` advances the label
axis in place; exit via a final blank from `(T, U+1)`.

**TDT** (NeMo-compatible semantics): two independently normalized heads;
every transition also draws a duration `d` from an ascending set — blank
requires `d ≥ 1` (no self-loop), tokens may take `d = 0`; the exit blank
lands exactly on frame `T + 1`; all transition scores are under-normalized
by a constant `sigma` (paper recommends ≈ 0.05, `0` disables).

**HAT**: blank is factorized as a per-cell Bernoulli `σ(blank_logits)`;
labels share `(1 − σ) · softmax(label_logits)` over a blank-free vocabulary,
enabling principled internal-LM subtraction for LM fusion.

**Pruned RNN-T**: the joint is evaluated only on `S` band cells per frame;
with a full-width band the loss equals vanilla RNN-T exactly.
[`pruning_bounds`](@ref) estimates the band from trivial-joint label
posteriors (k2-inspired, not bit-compatible with k2's gradient-based bounds).

**FastEmit** ([Yu et al., arXiv:2010.11148](https://arxiv.org/abs/2010.11148)):
optional `fastemit_lambda` on both [`rnnt_loss_batched`](@ref) and
[`tdt_loss_batched`](@ref). Scales label-emission gradients by `(1 + λ)`;
the loss scalar stays the standard NLL (gradient-only regularization).

**Minimum-latency** (Shinohara & Watanabe, Interspeech 2022): optional
`latency_lambda` on [`rnnt_loss_batched`](@ref). Augments the loss with the
exact expected mean label-emission frame via a first-moment forward-backward.

## Transducer family coverage

| Variant | Entry point |
|---------|-------------|
| **RNN-T** (Graves 2012) | [`rnnt_loss_batched`](@ref) |
| **TDT** (Xu et al., ICML 2023) | [`tdt_loss_batched`](@ref) |
| **Monotonic RNN-T / RNA** | [`monotonic_rnnt_loss_batched`](@ref) — TDT with `durations = [1]` |
| **FastEmit** regularization | `fastemit_lambda` kwarg on RNN-T and TDT |
| **Minimum-latency** | `latency_lambda` kwarg on RNN-T |
| **HAT** (Variani et al. 2020) | [`hat_loss_batched`](@ref) |
| **Pruned RNN-T** (Kuang et al. 2022) | [`pruned_rnnt_loss_batched`](@ref) + [`pruning_bounds`](@ref) |
| **Multi-blank transducer** | Subsumed by TDT (duration head generalizes frame skipping) |
| **Bayes-risk / delay-penalty** | Out of scope (k2-style risk objectives) |
| **Alignment-restricted (Ar-RNN-T)** | Out of scope (requires external alignments) |

## Verification

Every recurrence and gradient formula was validated **before transcription**
against brute-force enumeration of all lattice paths and central finite
differences (max error ≈ 1e-10 in double precision), including empty targets,
`T = 1`, zero-length inputs, padded prediction axes, duration sets without 0,
and `sigma ≠ 0`. FastEmit (`λ > 0`) is checked by decomposing gradients into
the `λ = 0` base plus label-posterior scaling. Minimum-latency is
finite-difference verified. The test suite repeats every check in pure Julia,
plus Zygote integration and heterogeneous-batch consistency.

## Memory

Vanilla RNN-T, TDT, and HAT materialize `O(T · U · B)` lattice tensors and
consume `O(V · T · U · B)` joint logits — the classic transducer cost.
[`pruned_rnnt_loss_batched`](@ref) bounds the joint to `O(V · T · S · B)`
with `S ≪ U` for long utterances. Train on shorter windows or smaller
batches when memory is tight.

## Dependencies

- [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
- [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl)
- [NNlib](https://github.com/FluxML/NNlib.jl) (for `logsoftmax`)

## License

MIT.
