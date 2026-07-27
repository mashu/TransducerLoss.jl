# TransducerLoss.jl

Batched **transducer-loss** functions for Julia — GPU-capable via
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl),
differentiable via [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl)
(Zygote-compatible). Companion to
[CTCLoss.jl](https://github.com/mashu/CTCLoss.jl): blank defaults to the
**last** class, loss is the **batch mean**, gradients w.r.t. **raw logits**
are computed inside one forward-backward per call.

## Losses, not model heads

This package exports **training objectives** — functions that take logits your
model already produced and return a scalar loss (plus analytic gradients via
`rrule`s). It does **not** define encoder, predictor, or joint-network layers.

In a typical ASR model, the joint network outputs one or more **logit
tensors**. You pass those tensors here:

| Your model produces | You call |
|---------------------|----------|
| One joint logit tensor `(V, T, U+1, B)` | [`rnnt_loss`](@ref) |
| Token head + duration head | [`tdt_loss`](@ref) |
| Label head + blank head | [`hat_loss`](@ref) |
| Banded joint `(V, T, S, B)` + offsets | [`pruned_rnnt_loss`](@ref) |
| Banded token + duration heads + offsets | [`pruned_tdt_loss`](@ref) |

TDT and HAT correspond to architectures with **multiple output heads**, but
TransducerLoss only implements the **loss computation** on their logits — not
the heads themselves.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/mashu/TransducerLoss.jl")
```

From a local checkout:

```julia
Pkg.develop(path = "/path/to/TransducerLoss.jl")
```

## Where to go next

| Page | Purpose |
|------|---------|
| [Choosing a loss](@ref) | Decision table — variant, shapes, when to use, why it exists |
| [Examples](@ref) | One minimal snippet per exported loss |
| [API](@ref) | Full docstrings and internal forward-backward entry points |

## Shared conventions

All variants share these inputs (unless noted otherwise):

- **`targets`**: `Vector{Vector{Int}}` — label indices per utterance (no blank).
- **`input_lengths`**: `Vector{Int}` — valid encoder frames per sample (`≤ T`).
- **`blank`**: defaults to `size(logits, 1)` (last class), matching CTCLoss.jl.

Lattice axes: **`T`** bounds encoder frames; **`U+1`** bounds prediction states
(`⟨start⟩` plus up to **`U`** target labels). Logits are **raw** (softmax /
sigmoid applied inside the loss).

## License

MIT.
