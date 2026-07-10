# Vanilla RNN-T: batched forward-backward driver and public API.

"""
    rnnt_forward_backward(logits, labels, target_lengths, input_lengths, blank)

Internal: run the transducer forward-backward on whatever device holds
`logits` and return `(loss, grad)` where `loss` is the batch-mean NLL and
`grad` is `∂loss/∂logits` (same shape as `logits`, already scaled by `1/B`).
Use `pack_transducer_targets` to build `labels` and `target_lengths`.
"""
function rnnt_forward_backward(logits::AbstractArray{T,4},
                               labels::Matrix{Int32},
                               target_lengths::Vector{Int32},
                               input_lengths::Vector{Int},
                               blank::Int) where {T}
    V, Tmax, U1max, B = size(logits)
    1 <= blank <= V || throw(ArgumentError("blank $blank outside 1:$V"))
    maximum(target_lengths; init = Int32(0)) + 1 <= U1max ||
        throw(ArgumentError("logits provide $U1max prediction states but " *
            "targets need $(maximum(target_lengths) + 1)"))
    backend = KernelAbstractions.get_backend(logits)

    clamped = Int32.(clamp.(input_lengths, 0, Tmax))
    lab_d = copyto!(similar(logits, Int32, size(labels)...), labels)
    Ul_d  = copyto!(similar(logits, Int32, B), target_lengths)
    Tl_d  = copyto!(similar(logits, Int32, B), clamped)

    log_probs = logsoftmax(logits; dims = 1)

    em_b = similar(logits, T, Tmax, U1max, B)
    em_l = similar(logits, T, Tmax, U1max, B)
    gather_emissions_kernel!(backend)(em_b, em_l, log_probs, lab_d, Ul_d, Tl_d,
                                      Int32(blank); ndrange = (Tmax, U1max, B))

    α = fill!(similar(logits, T, Tmax, U1max, B), T(-Inf))
    lattice_fwd_init_kernel!(backend)(α, Tl_d; ndrange = B)
    for d in Int32(3):Int32(Tmax + U1max)          # cell (1,1) is diagonal 2
        rnnt_fwd_diag_kernel!(backend)(α, em_b, em_l, Ul_d, Tl_d, d;
                                       ndrange = (U1max, B))
    end

    β = fill!(similar(logits, T, Tmax, U1max, B), T(-Inf))
    rnnt_bwd_init_kernel!(backend)(β, em_b, Ul_d, Tl_d; ndrange = B)
    for d in Int32(Tmax + U1max):-Int32(1):Int32(2)
        rnnt_bwd_diag_kernel!(backend)(β, em_b, em_l, Ul_d, Tl_d, d;
                                       ndrange = (U1max, B))
    end

    nll = similar(logits, T, B)
    lattice_nll_kernel!(backend)(nll, β, Tl_d; ndrange = B)

    grad = similar(logits, T, Tmax, U1max, B)
    rnnt_grad_kernel!(backend)(grad, α, β, em_b, em_l, log_probs, lab_d,
                               Ul_d, Tl_d, nll, Int32(blank);
                               ndrange = (Tmax, U1max, B))

    KernelAbstractions.synchronize(backend)

    sum(nll) / T(B), grad ./ T(B)
end


"""
    rnnt_loss_batched(logits, targets, input_lengths [; blank])
    rnnt_loss_batched(logits, targets, input_lengths, blank)

Batched RNN-T / transducer loss (Graves 2012), mean over batch.

- `logits`: `(V, T, U+1, B)` raw joint-network outputs (softmax applied
  internally), where `T` bounds encoder frames and `U+1` bounds prediction
  states (`⟨start⟩` plus up to `U` labels).
- `targets`: `Vector{Vector{Int}}` label sequences (no blank).
- `input_lengths`: valid encoder frames per sample.
- `blank` defaults to `size(logits, 1)` (last class), matching CTCLoss.jl.

Gradients are supplied through a ChainRulesCore `rrule` computed inside the
same forward-backward pass, so Zygote never traces the kernels. Memory note:
the joint tensor is `O(V·T·U·B)` — for long inputs train on shorter windows
or smaller batches.
"""
function rnnt_loss_batched(logits::AbstractArray{T,4},
                           targets::Vector{Vector{Int}},
                           input_lengths::Vector{Int};
                           blank::Int = size(logits, 1)) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, _ = rnnt_forward_backward(logits, labels, target_lengths,
                                    input_lengths, blank)
    loss
end

function rnnt_loss_batched(logits::AbstractArray{T,4},
                           targets::Vector{Vector{Int}},
                           input_lengths::Vector{Int},
                           blank::Int) where {T}
    rnnt_loss_batched(logits, targets, input_lengths; blank)
end

function ChainRulesCore.rrule(::typeof(rnnt_loss_batched),
                              logits::AbstractArray{T,4},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int};
                              blank::Int = size(logits, 1)) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, grad = rnnt_forward_backward(logits, labels, target_lengths,
                                       input_lengths, blank)
    rnnt_pullback(Δ) = (NoTangent(), Δ * grad, NoTangent(), NoTangent())
    loss, rnnt_pullback
end

function ChainRulesCore.rrule(::typeof(rnnt_loss_batched),
                              logits::AbstractArray{T,4},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int},
                              blank::Int) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, grad = rnnt_forward_backward(logits, labels, target_lengths,
                                       input_lengths, blank)
    rnnt_pullback(Δ) = (NoTangent(), Δ * grad, NoTangent(), NoTangent(),
                        NoTangent())
    loss, rnnt_pullback
end
