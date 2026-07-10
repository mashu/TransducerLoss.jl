"""
    TransducerLoss

The **transducer-loss family** for Julia — GPU-capable via
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl),
with analytic gradients delivered through
[ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) `rrule`s so
Zygote never traces the lattice kernels. The companion package to
[CTCLoss.jl](https://github.com/mashu/CTCLoss.jl), sharing its conventions:
blank defaults to the **last** class, the loss is the **batch mean**, and
gradients w.r.t. **raw logits** are computed inside the forward-backward.

Two losses, one lattice core:

- [`rnnt_loss_batched`](@ref) — vanilla RNN-T (Graves 2012,
  arXiv:1211.3711): blank advances time by one, labels advance the label
  axis in place.
- [`tdt_loss_batched`](@ref) — Token-and-Duration Transducer (Xu et al.,
  ICML 2023, arXiv:2304.06795): a second duration head lets every emission
  advance time by a learned stride, enabling frame-skipping decode; with
  `durations = [0, 1]` its alignment space strictly contains RNN-T's.
- [`monotonic_rnnt_loss_batched`](@ref) — Monotonic RNN-T as TDT with
  `durations = [1]` (diagonal-only alignments).
- **FastEmit** — `fastemit_lambda` on RNN-T and TDT (Yu et al.,
  arXiv:2010.11148); gradient-only latency regularization.
- **Minimum-latency** — `latency_lambda` on RNN-T (Shinohara & Watanabe,
  Interspeech 2022): augments the loss with the exact expected label
  emission frame via a first-moment (expectation-semiring) forward-backward.
- [`hat_loss_batched`](@ref) — Hybrid Autoregressive Transducer (Variani et
  al. 2020): Bernoulli blank factorization enabling internal-LM subtraction;
  reuses the shared lattice kernels with its own emissions and gradient.
- [`pruned_rnnt_loss_batched`](@ref) + [`pruning_bounds`](@ref) — banded
  lattice (Kuang et al. 2022): the joint lives on `O(V·T·S·B)` instead of
  `O(V·T·U·B)`; with a full-width band it equals vanilla RNN-T exactly.

They live in one package deliberately: both operate on the same
`(t frames, u labels)` lattice and share the emission gather, α
initialization, NLL read-out, label packing, numerics, and test oracles
(`src/core/`). The `src/rnnt/` and `src/tdt/` subtrees depend only on
`core/`, never on each other — so splitting one out later is trivial, but
bundling avoids duplicated kernels or a micro-package depending on another
package's internals (the same reason NeMo, k2, and torchaudio ship their
transducer variants together).

Both forward-backwards run as diagonal wavefronts: all lattice cells with
`t + u = d` are independent given neighbouring diagonals, so the host loops
over diagonals and each kernel launch fills one anti-diagonal for the whole
batch. Every recurrence and gradient formula was validated against
brute-force enumeration of all lattice paths and central finite differences
(max error ~1e-10) before transcription; the test suite repeats both checks
in pure Julia.
"""
module TransducerLoss

using KernelAbstractions
using ChainRulesCore: ChainRulesCore, NoTangent
using NNlib: logsoftmax

export pack_transducer_targets,
       rnnt_forward_backward,
       rnnt_loss_batched,
       tdt_forward_backward,
       tdt_loss_batched,
       monotonic_rnnt_loss_batched,
       hat_forward_backward,
       hat_loss_batched,
       pruned_forward_backward,
       pruned_rnnt_loss_batched,
       pruning_bounds

include("core/utils.jl")     # logaddexp
include("core/labels.jl")    # pack_transducer_targets
include("core/kernels.jl")   # gather, α init, NLL read-out (loss-agnostic)
include("rnnt/kernels.jl")   # RNN-T diagonal/gradient kernels
include("rnnt/loss.jl")      # RNN-T driver + API + rrules
include("hat/kernels.jl")    # HAT gather + factorized gradient
include("hat/loss.jl")       # HAT driver + API + rrule
include("pruned/kernels.jl") # banded-lattice column kernels + trivial joint
include("pruned/loss.jl")    # banded loss, API, pruning_bounds
include("tdt/kernels.jl")    # TDT diagonal/gradient kernels
include("tdt/loss.jl")       # TDT driver + API + rrule

end
