# TransducerLoss.jl

[![CI](https://github.com/mashu/TransducerLoss.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/mashu/TransducerLoss.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/mashu/TransducerLoss.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/mashu/TransducerLoss.jl)
[![Documentation](https://img.shields.io/badge/docs-blue.svg)](https://mashu.github.io/TransducerLoss.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Batched **RNN-T** and **Token-and-Duration Transducer (TDT)** losses for Julia, with GPU support via [KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl). Differentiable via [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl) (Zygote-compatible). Companion to [CTCLoss.jl](https://github.com/mashu/CTCLoss.jl).

```julia
using TransducerLoss

loss = rnnt_loss_batched(logits, targets, input_lengths)
loss = tdt_loss_batched(logits, dur_logits, targets, input_lengths,
                        [0, 1, 2, 3, 4]; sigma = 0.05)
```
