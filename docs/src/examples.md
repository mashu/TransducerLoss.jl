# Examples

Minimal calls for every exported loss. Shapes follow [Choosing a loss](@ref).
Replace `logits`, `targets`, and `input_lengths` with your model outputs.

## RNN-T

```julia
using TransducerLoss

# logits: (V, T, U+1, B) — raw joint-network output
loss = rnnt_loss(logits, targets, input_lengths)
loss = rnnt_loss(logits, targets, input_lengths; blank = V)
```

## TDT

```julia
# logits (V, T, U+1, B), dur_logits (D, T, U+1, B)
durations = [0, 1, 2, 3, 4]
loss = tdt_loss(logits, dur_logits, targets, input_lengths, durations;
                        sigma = 0.05)
```

## Monotonic RNN-T

```julia
# logits (V, T, U+1, B) — no duration head needed
loss = monotonic_rnnt_loss(logits, targets, input_lengths)
```

## HAT

```julia
# label_logits (V, T, U+1, B) blank-free; blank_logits (T, U+1, B)
loss = hat_loss(label_logits, blank_logits, targets, input_lengths)
```

## Pruned RNN-T

```julia
# Step 1: estimate band from encoder (am) and predictor (lm) logits
offsets = pruning_bounds(am, lm, targets, input_lengths; band_width = 4)

# Step 2: loss on banded joint logits (V, T, S, B)
loss = pruned_rnnt_loss(pruned_logits, offsets, targets, input_lengths)
```

[`pruning_bounds`](@ref) alone — useful when you only need offsets before
gathering pruned logits in your model code:

```julia
offsets = pruning_bounds(am, lm, targets, input_lengths; band_width = 4)
# offsets: (T, B) Int matrix; cell u = offsets[t, b] + s for s in 1:S
```

## Pruned TDT

```julia
# Bounds from the TDT lattice (occupancy), not RNN-T's
offsets = tdt_pruning_bounds(am, lm, am_dur, lm_dur, targets, input_lengths;
                             band_width = 4, durations)

# Banded token (V, T, S, B) + duration (D, T, S, B) logits
loss = pruned_tdt_loss(pruned_logits, pruned_dur_logits, offsets,
                               targets, input_lengths, durations;
                               sigma = 0.05)
```

## Regularization

```julia
# FastEmit — gradient-only; works on RNN-T and TDT
loss = rnnt_loss(logits, targets, input_lengths; fastemit_lambda = 0.004)
loss = tdt_loss(logits, dur_logits, targets, input_lengths, durations;
                        fastemit_lambda = 0.004)

# Minimum-latency — RNN-T only; augments the loss scalar
loss = rnnt_loss(logits, targets, input_lengths; latency_lambda = 0.01)
```

## Zygote gradients

```julia
using TransducerLoss, Zygote

grad = Zygote.gradient(
    l -> rnnt_loss(l, targets, input_lengths),
    logits,
)[1]

gtok, gdur = Zygote.gradient(
    (l, d) -> tdt_loss(l, d, targets, input_lengths, durations),
    logits, dur_logits,
)
```
