# API

```@docs
TransducerLoss
```

## Loss functions

```@docs
rnnt_loss
tdt_loss
monotonic_rnnt_loss
hat_loss
pruned_rnnt_loss
pruned_tdt_loss
```

## Helpers

```@docs
pack_transducer_targets
pruning_bounds
tdt_pruning_bounds
```

## Internal (advanced)

```@docs
rnnt_forward_backward
tdt_forward_backward
hat_forward_backward
pruned_forward_backward
TransducerLoss.logaddexp
```

## Summary

| Function | Description |
|----------|-------------|
| `rnnt_loss` | RNN-T / vanilla transducer loss (Graves 2012); `(V,T,U+1,B)` or `(V,T,U+1)` |
| `tdt_loss` | Token-and-Duration Transducer loss (Xu et al., ICML 2023); 4D or 3D |
| `monotonic_rnnt_loss` | Monotonic RNN-T via TDT with `durations = [1]` |
| `hat_loss` | Hybrid Autoregressive Transducer with Bernoulli blank factorization |
| `pruned_rnnt_loss` | RNN-T on a banded lattice; full-width band equals vanilla RNN-T |
| `pruned_tdt_loss` | TDT on a banded lattice; full-width band equals vanilla TDT |
| `pruning_bounds` | Estimate band offsets from trivial-joint label posteriors |
| `tdt_pruning_bounds` | Estimate band offsets on the TDT lattice (occupancy-centred) |
| `pack_transducer_targets` | Pack ragged label sequences into a padded matrix |
| `rnnt_forward_backward` | Internal RNN-T forward-backward; returns `(loss, grad)` |
| `tdt_forward_backward` | Internal TDT forward-backward; returns `(loss, grad_tok, grad_dur)` |
| `hat_forward_backward` | Internal HAT forward-backward; returns `(loss, grad_labels, grad_blank)` |
| `pruned_forward_backward` | Internal banded forward-backward; returns `(loss, grad)` |
