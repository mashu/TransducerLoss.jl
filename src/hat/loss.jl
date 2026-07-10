# HAT driver and public API.

"""
    hat_forward_backward(label_logits, blank_logits, labels, target_lengths,
                         input_lengths)

Internal: `(loss, grad_labels, grad_blank)` — batch-mean HAT NLL and
gradients w.r.t. both raw-logit tensors (scaled by `1/B`).
"""
function hat_forward_backward(label_logits::AbstractArray{T,4},
                              blank_logits::AbstractArray{T,3},
                              labels::Matrix{Int32},
                              target_lengths::Vector{Int32},
                              input_lengths::Vector{Int}) where {T}
    V, Tmax, U1max, B = size(label_logits)
    size(blank_logits) == (Tmax, U1max, B) || throw(ArgumentError(
        "blank_logits must be (T, U+1, B) matching label_logits"))
    maximum(target_lengths; init = Int32(0)) + 1 <= U1max ||
        throw(ArgumentError("label_logits provide $U1max prediction states " *
            "but targets need $(maximum(target_lengths) + 1)"))
    backend = KernelAbstractions.get_backend(label_logits)
    lab_d, Ul_d, Tl_d = device_inputs(label_logits, labels, target_lengths,
                                      input_lengths, Tmax)

    lp = logsoftmax(label_logits; dims = 1)
    em_b = similar(label_logits, T, Tmax, U1max, B)
    em_l = similar(label_logits, T, Tmax, U1max, B)
    hat_gather_kernel!(backend)(em_b, em_l, blank_logits, lp, lab_d, Ul_d,
                                Tl_d; ndrange = (Tmax, U1max, B))

    α = fill!(similar(em_b), T(-Inf))
    lattice_fwd_init_kernel!(backend)(α, Tl_d; ndrange = B)
    for d in Int32(3):Int32(Tmax + U1max)
        lattice_fwd_diag_kernel!(backend)(α, em_b, em_l, Ul_d, Tl_d, d;
                                          ndrange = (U1max, B))
    end
    β = fill!(similar(em_b), T(-Inf))
    lattice_bwd_init_kernel!(backend)(β, em_b, Ul_d, Tl_d; ndrange = B)
    for d in Int32(Tmax + U1max):-Int32(1):Int32(2)
        lattice_bwd_diag_kernel!(backend)(β, em_b, em_l, Ul_d, Tl_d, d;
                                          ndrange = (U1max, B))
    end
    nll = similar(label_logits, T, B)
    lattice_nll_kernel!(backend)(nll, β, Tl_d; ndrange = B)

    gy = similar(label_logits)
    gb = similar(blank_logits)
    hat_grad_kernel!(backend)(gb, gy, α, β, em_b, em_l, blank_logits, lp,
                              lab_d, Ul_d, Tl_d, nll;
                              ndrange = (Tmax, U1max, B))
    KernelAbstractions.synchronize(backend)
    sum(nll) / T(B), gy ./ T(B), gb ./ T(B)
end

"""
    hat_loss_batched(label_logits, blank_logits, targets, input_lengths)

**Hybrid Autoregressive Transducer** loss (Variani et al., ICASSP 2020,
arXiv:2003.07705). Blank is factorized out as a per-cell Bernoulli:
`P(blank) = σ(blank_logits[t,u,b])`, and labels share
`(1 − σ) · softmax(label_logits)` over a **blank-free** vocabulary — so
`label_logits` is `(V, T, U+1, B)` with all `V` classes real labels, and
`blank_logits` is `(T, U+1, B)`.

The payoff is principled language-model fusion: because the label
distribution is a proper conditional LM given "not blank", the model's
internal LM can be estimated (run the prediction network with the encoder
contribution zeroed) and subtracted before adding an external LM — the
production-grade alternative to plain shallow fusion. Gradients for both
tensors arrive via one ChainRulesCore `rrule`.
"""
function hat_loss_batched(label_logits::AbstractArray{T,4},
                          blank_logits::AbstractArray{T,3},
                          targets::Vector{Vector{Int}},
                          input_lengths::Vector{Int}) where {T}
    labels, target_lengths = pack_transducer_targets(targets, 0)  # no blank id
    loss, _, _ = hat_forward_backward(label_logits, blank_logits, labels,
                                      target_lengths, input_lengths)
    loss
end

function ChainRulesCore.rrule(::typeof(hat_loss_batched),
                              label_logits::AbstractArray{T,4},
                              blank_logits::AbstractArray{T,3},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int}) where {T}
    labels, target_lengths = pack_transducer_targets(targets, 0)
    loss, gy, gb = hat_forward_backward(label_logits, blank_logits, labels,
                                        target_lengths, input_lengths)
    hat_pullback(Δ) = (NoTangent(), Δ * gy, Δ * gb, NoTangent(), NoTangent())
    loss, hat_pullback
end
