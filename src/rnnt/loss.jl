# Vanilla RNN-T: batched forward-backward driver and public API.

"""
    rnnt_forward_backward(logits, labels, target_lengths, input_lengths,
                          blank, fastemit_lambda)

Internal: run the transducer forward-backward on whatever device holds
`logits` and return `(loss, grad)` where `loss` is the batch-mean NLL and
`grad` is `∂loss/∂logits` (same shape as `logits`, already scaled by `1/B`).
Use `pack_transducer_targets` to build `labels` and `target_lengths`.
"""
function rnnt_forward_backward(logits::AbstractArray{T,4},
                               labels::Matrix{Int32},
                               target_lengths::Vector{Int32},
                               input_lengths::Vector{Int},
                               blank::Int, fastemit_lambda::Real,
                               latency_lambda::Real = 0) where {T}
    V, Tmax, U1max, B = size(logits)
    1 <= blank <= V || throw(ArgumentError("blank $blank outside 1:$V"))
    maximum(target_lengths; init = Int32(0)) + 1 <= U1max ||
        throw(ArgumentError("logits provide $U1max prediction states but " *
            "targets need $(maximum(target_lengths) + 1)"))
    backend = KernelAbstractions.get_backend(logits)
    λ = T(fastemit_lambda)
    fastemit_lambda >= 0 || throw(ArgumentError("fastemit_lambda must be ≥ 0"))
    λlat = T(latency_lambda)
    latency_lambda >= 0 || throw(ArgumentError("latency_lambda must be ≥ 0"))

    lab_d, Ul_d, Tl_d = device_inputs(logits, labels, target_lengths,
                                      input_lengths, Tmax)

    log_probs = logsoftmax(logits; dims = 1)

    em_b = similar(logits, T, Tmax, U1max, B)
    em_l = similar(logits, T, Tmax, U1max, B)
    gather_emissions_kernel!(backend)(em_b, em_l, log_probs, lab_d, Ul_d, Tl_d,
                                      Int32(blank); ndrange = (Tmax, U1max, B))

    α = fill!(similar(logits, T, Tmax, U1max, B), T(-Inf))
    lattice_fwd_init_kernel!(backend)(α, Tl_d; ndrange = B)
    for d in Int32(3):Int32(Tmax + U1max)          # cell (1,1) is diagonal 2
        lattice_fwd_diag_kernel!(backend)(α, em_b, em_l, Ul_d, Tl_d, d;
                                       ndrange = (U1max, B))
    end

    β = fill!(similar(logits, T, Tmax, U1max, B), T(-Inf))
    lattice_bwd_init_kernel!(backend)(β, em_b, Ul_d, Tl_d; ndrange = B)
    for d in Int32(Tmax + U1max):-Int32(1):Int32(2)
        lattice_bwd_diag_kernel!(backend)(β, em_b, em_l, Ul_d, Tl_d, d;
                                       ndrange = (U1max, B))
    end

    nll = similar(logits, T, B)
    lattice_nll_kernel!(backend)(nll, β, Tl_d; ndrange = B)

    # Minimum-latency term (Shinohara & Watanabe, Interspeech 2022): first
    # latency moments Z̃ via the expectation semiring; E_b = Z̃_b / Z_b is the
    # expected sum of label emission frames, normalized per sample by U_b.
    if λlat > zero(T)
        m_α = fill!(similar(α), T(-Inf))
        for d in Int32(3):Int32(Tmax + U1max)
            latency_moment_fwd_kernel!(backend)(m_α, α, em_b, em_l, Ul_d,
                                                Tl_d, d; ndrange = (U1max, B))
        end
        m_β = fill!(similar(β), T(-Inf))
        for d in Int32(Tmax + U1max):-Int32(1):Int32(2)
            latency_moment_bwd_kernel!(backend)(m_β, β, em_b, em_l, Ul_d,
                                                Tl_d, d; ndrange = (U1max, B))
        end
        E = vec(ifelse.(isfinite.(nll),
                        exp.(view(m_β, 1, 1, :) .+ nll), zero(T)))
        total = ifelse.(isfinite.(nll),
                        nll .+ λlat .* E ./ max.(T.(Ul_d), one(T)), nll)
    else
        m_α, m_β = α, β                       # placeholders; never read
        E, total = nll, nll
    end

    grad = similar(logits)
    rnnt_grad_kernel!(backend)(grad, α, β, em_b, em_l, log_probs, lab_d,
                               Ul_d, Tl_d, m_α, m_β, nll, E, Int32(blank),
                               λ, λlat; ndrange = (Tmax, U1max, B))

    KernelAbstractions.synchronize(backend)

    sum(total) / T(B), grad ./ T(B)
end


"""
    rnnt_loss_batched(logits, targets, input_lengths [; blank, fastemit_lambda])
    rnnt_loss_batched(logits, targets, input_lengths, blank [; fastemit_lambda])

Batched RNN-T / transducer loss (Graves 2012), mean over batch.

- `logits`: `(V, T, U+1, B)` raw joint-network outputs (softmax applied
  internally), where `T` bounds encoder frames and `U+1` bounds prediction
  states (`⟨start⟩` plus up to `U` labels).
- `targets`: `Vector{Vector{Int}}` label sequences (no blank).
- `input_lengths`: valid encoder frames per sample.
- `blank` defaults to `size(logits, 1)` (last class), matching CTCLoss.jl.
- `latency_lambda`: minimum-latency training (Shinohara & Watanabe,
  Interspeech 2022) — augments the loss with `λ · E[Σᵤ tᵤ] / U`, the
  expected mean label-emission frame under the alignment posterior, computed
  exactly with a first-moment (expectation-semiring) forward-backward.
  A genuine objective: finite-difference verified. `0` disables.
- `fastemit_lambda`: FastEmit regularization (Yu et al., arXiv:2010.11148).
  Scales label-emission gradients by `(1 + λ)`; `0` disables. The reported
  loss scalar is unchanged (standard NLL); only gradients are modified,
  matching torchaudio / NeMo / k2.

Gradients are supplied through a ChainRulesCore `rrule` computed inside the
same forward-backward pass, so Zygote never traces the kernels. Memory note:
the joint tensor is `O(V·T·U·B)` — for long inputs train on shorter windows
or smaller batches.
"""
function rnnt_loss_batched(logits::AbstractArray{T,4},
                           targets::Vector{Vector{Int}},
                           input_lengths::Vector{Int};
                           blank::Int = size(logits, 1),
                           fastemit_lambda::Real = 0,
                           latency_lambda::Real = 0) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, _ = rnnt_forward_backward(logits, labels, target_lengths,
                                    input_lengths, blank, fastemit_lambda,
                                    latency_lambda)
    loss
end

function rnnt_loss_batched(logits::AbstractArray{T,4},
                           targets::Vector{Vector{Int}},
                           input_lengths::Vector{Int},
                           blank::Int;
                           fastemit_lambda::Real = 0) where {T}
    rnnt_loss_batched(logits, targets, input_lengths; blank,
                      fastemit_lambda, latency_lambda)
end

function ChainRulesCore.rrule(::typeof(rnnt_loss_batched),
                              logits::AbstractArray{T,4},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int};
                              blank::Int = size(logits, 1),
                              fastemit_lambda::Real = 0,
                              latency_lambda::Real = 0) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, grad = rnnt_forward_backward(logits, labels, target_lengths,
                                       input_lengths, blank, fastemit_lambda,
                                       latency_lambda)
    rnnt_pullback(Δ) = (NoTangent(), Δ * grad, NoTangent(), NoTangent(),
                        NoTangent())
    loss, rnnt_pullback
end

function ChainRulesCore.rrule(::typeof(rnnt_loss_batched),
                              logits::AbstractArray{T,4},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int},
                              blank::Int;
                              fastemit_lambda::Real = 0) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, grad = rnnt_forward_backward(logits, labels, target_lengths,
                                       input_lengths, blank, fastemit_lambda)
    rnnt_pullback(Δ) = (NoTangent(), Δ * grad, NoTangent(), NoTangent(),
                        NoTangent())
    loss, rnnt_pullback
end
