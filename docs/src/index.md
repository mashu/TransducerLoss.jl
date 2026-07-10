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
loss = rnnt_loss_batched(logits, targets, input_lengths)            # blank = V
loss = rnnt_loss_batched(logits, targets, input_lengths, blank)

# TDT — Token-and-Duration Transducer (Xu et al., ICML 2023):
# token head (V, T, U+1, B) + duration head (D, T, U+1, B)
loss = tdt_loss_batched(logits, dur_logits, targets, input_lengths,
                        [0, 1, 2, 3, 4]; blank, sigma = 0.05)

# Monotonic RNN-T (diagonal-only alignments): TDT with durations = [1]
loss = monotonic_rnnt_loss_batched(logits, targets, input_lengths)

# FastEmit latency regularization (torchaudio / NeMo / k2 compatible)
loss = rnnt_loss_batched(logits, targets, input_lengths; fastemit_lambda = 0.004)
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

## Why one package for two losses

Both losses operate on the same `(t frames consumed, u labels emitted)`
lattice, and TDT with `durations = [0, 1]` has an alignment space that
strictly contains RNN-T's. They share the emission gather, α initialization,
NLL read-out, label packing, numerics, and test oracles — all of which live
in `src/core/`. The `src/rnnt/` and `src/tdt/` subtrees depend **only on
core/**, never on each other, so extracting one into its own package later is
a file move; bundling today avoids either duplicated kernels or a
micro-package reaching into another package's unexported internals. NeMo,
k2, and torchaudio ship their transducer variants together for the same
reason.

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
test/
  reference.jl        pure-Julia references + brute-force oracles
  test_rnnt.jl, test_tdt.jl
```

## Algorithms

Forward-backwards run as **diagonal wavefronts**: all lattice cells with
`t + u = d` are independent given neighbouring diagonals — one kernel launch
per diagonal, parallel over `(u, batch)`.

**RNN-T**: blank advances time by one, label `y_{u+1}` advances the label
axis in place; exit via a final blank from `(T, U+1)`.

**TDT** (NeMo-compatible semantics): two independently normalized heads;
every transition also draws a duration `d` from an ascending set — blank
requires `d ≥ 1` (no self-loop), tokens may take `d = 0`; the exit blank
lands exactly on frame `T + 1`; all transition scores are under-normalized
by a constant `sigma` (paper recommends ≈ 0.05, `0` disables). At decode
time the duration head's prediction advances the encoder index — the
frame-skipping behind the 2–3× throughput lead of TDT models on the open
ASR leaderboards.

**FastEmit** ([Yu et al., arXiv:2010.11148](https://arxiv.org/abs/2010.11148)):
optional `fastemit_lambda` on both [`rnnt_loss_batched`](@ref) and
[`tdt_loss_batched`](@ref). Scales label-emission gradients by `(1 + λ)`;
the loss scalar stays the standard NLL (gradient-only regularization, as in
torchaudio / NeMo / k2). Verified analytically against label-emission
posteriors; `λ = 0` recovers the base transducer.

## Transducer family coverage

| Variant | Status in this package |
|---------|------------------------|
| **RNN-T** (Graves 2012) | [`rnnt_loss_batched`](@ref) |
| **TDT** (Xu et al., ICML 2023) | [`tdt_loss_batched`](@ref) |
| **Monotonic RNN-T** | [`monotonic_rnnt_loss_batched`](@ref) — TDT with `durations = [1]` |
| **FastEmit** regularization | `fastemit_lambda` kwarg on RNN-T and TDT |
| **Multi-blank transducer** | Subsumed by TDT (duration head generalizes frame skipping) |
| **RNA / diagonal-only** | Same lattice as Monotonic RNN-T (`durations = [1]`) |
| **Pruned RNN-T** (k2) | Planned — banded lattice, same objective, ~10× memory |
| **HAT** (blank factorization + ILM) | Planned — separate blank stream for shallow fusion |
| **Bayes-risk / delay-penalty** | Out of scope for v0.1 (k2-style risk objectives) |
| **Alignment-restricted (Ar-RNN-T)** | Out of scope (requires external alignments) |

Monotonic RNN-T is not a separate kernel tree: every token transition must
advance time by exactly one frame, so the alignment lattice is a subset of
vanilla RNN-T's. Setting TDT `durations = [1]` enforces this — blank and
label moves are both diagonal, with a trivial single-class duration head.

## Verification

Every recurrence and gradient formula was validated **before transcription**
against brute-force enumeration of all lattice paths and central finite
differences (max error ≈ 1e-10 in double precision), including empty targets,
`T = 1`, zero-length inputs, padded prediction axes, duration sets without 0,
and `sigma ≠ 0`. FastEmit (`λ > 0`) is checked by decomposing gradients into
the `λ = 0` base plus label-posterior scaling. The test suite repeats every
check in pure Julia, plus Zygote integration (both heads for TDT) and
heterogeneous-batch consistency.

## Memory

Both losses materialize `O(T · U · B)` lattice tensors and consume
`O(V · T · U · B)` joint logits — the classic transducer cost. Train on
shorter windows or smaller batches for long inputs. The natural next
optimizations: a gather-based fused joint, and the **pruned transducer**
(k2), which bounds the lattice to a thin band for ~10× memory reduction at
equal accuracy.

## Dependencies

- [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl)
- [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl)
- [NNlib](https://github.com/FluxML/NNlib.jl) (for `logsoftmax`)

## License

MIT.
