# TDT (Token-and-Duration Transducer) loss: driver and public API.

"""
    tdt_forward_backward(logits, dur_logits, labels, target_lengths,
                         input_lengths, durations, blank, sigma)

Internal: run the TDT forward-backward on whatever device holds `logits` and
return `(loss, grad_tok, grad_dur)` — the batch-mean NLL and the gradients
w.r.t. both raw-logit tensors (already scaled by `1/B`).
"""
function tdt_forward_backward(logits::AbstractArray{T,4},
                              dur_logits::AbstractArray{T,4},
                              labels::Matrix{Int32},
                              target_lengths::Vector{Int32},
                              input_lengths::Vector{Int},
                              durations::Vector{Int}, blank::Int,
                              sigma::Real, fastemit_lambda::Real) where {T}
    V, Tmax, U1max, B = size(logits)
    size(dur_logits)[2:4] == (Tmax, U1max, B) || throw(ArgumentError(
        "dur_logits must share (T, U+1, B) with logits"))
    size(dur_logits, 1) == length(durations) || throw(ArgumentError(
        "dur_logits has $(size(dur_logits, 1)) duration classes but " *
        "$(length(durations)) durations were given"))
    1 <= blank <= V || throw(ArgumentError("blank $blank outside 1:$V"))
    issorted(durations) && allunique(durations) && all(>=(0), durations) ||
        throw(ArgumentError("durations must be ascending, unique, non-negative"))
    any(>=(1), durations) ||
        throw(ArgumentError("durations must include a value ≥ 1"))
    maximum(target_lengths; init = Int32(0)) + 1 <= U1max ||
        throw(ArgumentError("logits provide $U1max prediction states but " *
            "targets need $(maximum(target_lengths) + 1)"))
    backend = KernelAbstractions.get_backend(logits)
    σ = T(sigma)
    λ = T(fastemit_lambda)
    fastemit_lambda >= 0 || throw(ArgumentError("fastemit_lambda must be ≥ 0"))

    clamped = Int32.(clamp.(input_lengths, 0, Tmax))
    lab_d = copyto!(similar(logits, Int32, size(labels)...), labels)
    Ul_d  = copyto!(similar(logits, Int32, B), target_lengths)
    Tl_d  = copyto!(similar(logits, Int32, B), clamped)
    dur_d = copyto!(similar(logits, Int32, length(durations)),
                    Int32.(durations))

    lp = logsoftmax(logits; dims = 1)
    ld = logsoftmax(dur_logits; dims = 1)

    em_b = similar(logits, T, Tmax, U1max, B)
    em_l = similar(logits, T, Tmax, U1max, B)
    gather_emissions_kernel!(backend)(em_b, em_l, lp, lab_d, Ul_d, Tl_d,
                                     Int32(blank); ndrange = (Tmax, U1max, B))

    α = fill!(similar(logits, T, Tmax, U1max, B), T(-Inf))
    lattice_fwd_init_kernel!(backend)(α, Tl_d; ndrange = B)
    for d in Int32(3):Int32(Tmax + U1max)
        tdt_fwd_diag_kernel!(backend)(α, em_b, em_l, ld, dur_d, Ul_d, Tl_d,
                                      σ, d; ndrange = (U1max, B))
    end

    # The backward recursion embeds the exit terms, so the diagonal loop
    # writes every cell — no separate init.
    β = fill!(similar(logits, T, Tmax, U1max, B), T(-Inf))
    for d in Int32(Tmax + U1max):-Int32(1):Int32(2)
        tdt_bwd_diag_kernel!(backend)(β, em_b, em_l, ld, dur_d, Ul_d, Tl_d,
                                      σ, d; ndrange = (U1max, B))
    end

    nll = similar(logits, T, B)
    lattice_nll_kernel!(backend)(nll, β, Tl_d; ndrange = B)

    gtok = similar(logits)
    gdur = similar(dur_logits)
    tdt_grad_kernel!(backend)(gtok, gdur, α, β, em_b, em_l, lp, ld, lab_d,
                              dur_d, Ul_d, Tl_d, nll, σ, Int32(blank), λ;
                              ndrange = (Tmax, U1max, B))

    KernelAbstractions.synchronize(backend)

    sum(nll) / T(B), gtok ./ T(B), gdur ./ T(B)
end

"""
    tdt_loss_batched(logits, dur_logits, targets, input_lengths, durations;
                     blank = size(logits, 1), sigma = 0, fastemit_lambda = 0)

Batched **Token-and-Duration Transducer** loss (Xu et al., ICML 2023,
arXiv:2304.06795) — the frame-skipping generalization of RNN-T behind
NVIDIA's Parakeet-TDT models. Mean over the batch; blank defaults to the
last token class (the CTCLoss.jl / TransducerLoss.jl convention).

- `logits`: `(V, T, U+1, B)` raw token-head outputs.
- `dur_logits`: `(D, T, U+1, B)` raw duration-head outputs, one class per
  entry of `durations`.
- `durations`: ascending duration set, e.g. `[0, 1, 2, 3, 4]`. Blank
  transitions use durations ≥ 1; tokens may use 0 (multiple tokens per
  frame, as in RNN-T). The exit blank lands exactly on frame `T + 1`.
- `sigma`: NeMo's logit under-normalization constant (≈ 0.05 recommended;
  0 disables).
- `fastemit_lambda`: FastEmit regularization; scales label-emission
  gradients by `(1 + λ)` on both token and duration heads. Loss scalar
  unchanged; see [`rnnt_loss_batched`](@ref).

With `durations = [0, 1]` the alignment space is a strict superset of
vanilla RNN-T's (tokens may also advance time directly). With
`durations = [1]` every emission advances time by one frame — equivalent
to **Monotonic RNN-T** (see [`monotonic_rnnt_loss_batched`](@ref)).
Gradients for **both** logit tensors arrive through a ChainRulesCore
`rrule` computed inside the same forward-backward pass.
"""
function tdt_loss_batched(logits::AbstractArray{T,4},
                          dur_logits::AbstractArray{T,4},
                          targets::Vector{Vector{Int}},
                          input_lengths::Vector{Int},
                          durations::Vector{Int};
                          blank::Int = size(logits, 1),
                          sigma::Real = 0,
                          fastemit_lambda::Real = 0) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, _, _ = tdt_forward_backward(logits, dur_logits, labels,
                                      target_lengths, input_lengths,
                                      durations, blank, sigma, fastemit_lambda)
    loss
end

"""
    monotonic_rnnt_loss_batched(logits, targets, input_lengths;
                                blank = size(logits, 1), sigma = 0,
                                fastemit_lambda = 0)

Batched **Monotonic RNN-T** loss — at most one label per frame, no vertical
token transitions. Implemented as TDT with `durations = [1]` and a trivial
duration head (`dur_logits` all zeros ⇒ softmax probability 1). Equivalent
to restricting the lattice to diagonal label moves plus blank advances.
"""
function monotonic_rnnt_loss_batched(logits::AbstractArray{T,4},
                                     targets::Vector{Vector{Int}},
                                     input_lengths::Vector{Int};
                                     blank::Int = size(logits, 1),
                                     sigma::Real = 0,
                                     fastemit_lambda::Real = 0) where {T}
    dur_logits = zeros(T, 1, size(logits, 2), size(logits, 3), size(logits, 4))
    tdt_loss_batched(logits, dur_logits, targets, input_lengths, [1];
                     blank, sigma, fastemit_lambda)
end

function ChainRulesCore.rrule(::typeof(tdt_loss_batched),
                              logits::AbstractArray{T,4},
                              dur_logits::AbstractArray{T,4},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int},
                              durations::Vector{Int};
                              blank::Int = size(logits, 1),
                              sigma::Real = 0,
                              fastemit_lambda::Real = 0) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, gtok, gdur = tdt_forward_backward(logits, dur_logits, labels,
                                            target_lengths, input_lengths,
                                            durations, blank, sigma,
                                            fastemit_lambda)
    tdt_pullback(Δ) = (NoTangent(), Δ * gtok, Δ * gdur, NoTangent(),
                       NoTangent(), NoTangent(), NoTangent())
    loss, tdt_pullback
end

function ChainRulesCore.rrule(::typeof(monotonic_rnnt_loss_batched),
                              logits::AbstractArray{T,4},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int};
                              blank::Int = size(logits, 1),
                              sigma::Real = 0,
                              fastemit_lambda::Real = 0) where {T}
    dur_logits = zeros(T, 1, size(logits, 2), size(logits, 3), size(logits, 4))
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, gtok, _ = tdt_forward_backward(logits, dur_logits, labels,
                                         target_lengths, input_lengths,
                                         [1], blank, sigma, fastemit_lambda)
    mono_pullback(Δ) = (NoTangent(), Δ * gtok, NoTangent(), NoTangent(),
                        NoTangent(), NoTangent(), NoTangent())
    loss, mono_pullback
end
