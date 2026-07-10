# API

```@docs
TransducerLoss
```

## Loss functions

```@docs
rnnt_loss_batched
tdt_loss_batched
```

## Helpers

```@docs
pack_transducer_targets
```

## Internal (advanced)

```@docs
rnnt_forward_backward
tdt_forward_backward
TransducerLoss.logaddexp
```

## Summary

| Function | Description |
|----------|-------------|
| `rnnt_loss_batched` | Batched RNN-T / vanilla transducer loss (Graves 2012) |
| `tdt_loss_batched` | Batched Token-and-Duration Transducer loss (Xu et al., ICML 2023) |
| `pack_transducer_targets` | Pack ragged label sequences into a padded matrix |
| `rnnt_forward_backward` | Internal RNN-T forward-backward; returns `(loss, grad)` |
| `tdt_forward_backward` | Internal TDT forward-backward; returns `(loss, grad_tok, grad_dur)` |
