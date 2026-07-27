# TransducerLoss.jl

[![CI](https://github.com/mashu/TransducerLoss.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mashu/TransducerLoss.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mashu/TransducerLoss.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mashu/TransducerLoss.jl)
[![Documentation](https://img.shields.io/badge/docs-blue.svg)](https://mashu.github.io/TransducerLoss.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Transducer-loss family** for Julia — GPU-capable via KernelAbstractions,
differentiable via ChainRulesCore (Zygote-compatible). Companion to
[CTCLoss.jl](https://github.com/mashu/CTCLoss.jl): blank defaults to the **last**
class, loss is the **batch mean**, gradients w.r.t. **raw logits** are computed
inside one forward-backward per loss.

```julia
using Pkg; Pkg.add(url = "https://github.com/mashu/TransducerLoss.jl")
using TransducerLoss, Zygote

loss = rnnt_loss(logits, targets, input_lengths)
loss = tdt_loss(logits, dur_logits, targets, input_lengths,
                        [0, 1, 2, 3, 4]; sigma = 0.05)

grad = Zygote.gradient(l -> rnnt_loss(l, targets, input_lengths), logits)[1]
```

## The loss family

| Variant | Entry point | Notes |
|---|---|---|
| RNN-T (Graves 2012) | `rnnt_loss` | baseline transducer |
| TDT (Xu et al., ICML 2023) | `tdt_loss` | duration head, frame-skipping decode |
| Monotonic RNN-T / RNA | `monotonic_rnnt_loss` | TDT special case `durations = [1]` |
| FastEmit (Yu et al. 2021) | `fastemit_lambda` on RNN-T/TDT | gradient-level emission regularization |
| Minimum-latency (Shinohara & Watanabe 2022) | `latency_lambda` on RNN-T | exact expected emission frame via a first-moment forward-backward; FD-verified |
| HAT (Variani et al. 2020) | `hat_loss` | Bernoulli blank factorization → internal-LM subtraction for principled LM fusion |
| Banded / pruned RNN-T (Kuang et al. 2022) | `pruned_rnnt_loss` + `pruning_bounds` | `O(V·T·S·B)` joint; full-width band equals vanilla RNN-T; bounds are k2-inspired (posterior-centred, not bit-compatible) |
| Banded / pruned TDT | `pruned_tdt_loss` + `tdt_pruning_bounds` | Same banding for TDT (incl. FastEmit); bounds from the **TDT** lattice (occupancy-centred), not RNN-T's |
| Multi-blank (Xu et al., ICASSP 2023) | subsumed | superseded by TDT's duration head |

Gradients are analytic and verified in `test/` against brute-force path enumeration
and central finite differences.

## Related

- [CTCLoss.jl](https://github.com/mashu/CTCLoss.jl) — companion package with matching conventions
