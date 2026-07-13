# API

```@docs
TransducerLoss
```

## Loss functions

```@docs
rnnt_loss_batched
tdt_loss_batched
monotonic_rnnt_loss_batched
hat_loss_batched
pruned_rnnt_loss_batched
```

## Helpers

```@docs
pack_transducer_targets
pruning_bounds
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
| `rnnt_loss_batched` | Batched RNN-T / vanilla transducer loss (Graves 2012) |
| `tdt_loss_batched` | Batched Token-and-Duration Transducer loss (Xu et al., ICML 2023) |
| `monotonic_rnnt_loss_batched` | Monotonic RNN-T via TDT with `durations = [1]` |
| `hat_loss_batched` | Hybrid Autoregressive Transducer with Bernoulli blank factorization |
| `pruned_rnnt_loss_batched` | RNN-T on a banded lattice; full-width band equals vanilla RNN-T |
| `pruning_bounds` | Estimate band offsets from trivial-joint label posteriors |
| `pack_transducer_targets` | Pack ragged label sequences into a padded matrix |
| `rnnt_forward_backward` | Internal RNN-T forward-backward; returns `(loss, grad)` |
| `tdt_forward_backward` | Internal TDT forward-backward; returns `(loss, grad_tok, grad_dur)` |
| `hat_forward_backward` | Internal HAT forward-backward; returns `(loss, grad_labels, grad_blank)` |
| `pruned_forward_backward` | Internal banded forward-backward; returns `(loss, grad)` |
