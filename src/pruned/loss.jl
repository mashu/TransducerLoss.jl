# Banded-lattice loss driver, public API, and the pruning-bounds helper.

"""
    pruned_forward_backward(logits, band_offsets, labels, target_lengths,
                            input_lengths, blank)

Internal: `(loss, grad)` for the banded lattice. `logits` is `(V, T, S, B)`
— the joint evaluated only on band cells — and `band_offsets` maps band slot
`s` at frame `t` to lattice cell `u = band_offsets[t, b] + s`.
"""
function pruned_forward_backward(logits::AbstractArray{T,4},
                                 band_offsets::Matrix{Int32},
                                 labels::Matrix{Int32},
                                 target_lengths::Vector{Int32},
                                 input_lengths::Vector{Int},
                                 blank::Int) where {T}
    V, Tmax, S, B = size(logits)
    1 <= blank <= V || throw(ArgumentError("blank $blank outside 1:$V"))
    size(band_offsets) == (Tmax, B) || throw(ArgumentError(
        "band_offsets must be (T, B) = ($Tmax, $B)"))
    all(>=(0), band_offsets) || throw(ArgumentError("offsets must be ≥ 0"))
    for b in 1:B
        issorted(view(band_offsets, :, b)) || throw(ArgumentError(
            "band offsets must be monotone non-decreasing (sample $b)"))
        band_offsets[1, b] == 0 || throw(ArgumentError(
            "band must contain the start cell: band_offsets[1, $b] == 0"))
    end
    backend = KernelAbstractions.get_backend(logits)
    lab_d, Ul_d, Tl_d = device_inputs(logits, labels, target_lengths,
                                      input_lengths, Tmax)
    off_d = copyto!(similar(logits, Int32, Tmax, B), band_offsets)

    lp = logsoftmax(logits; dims = 1)
    α = fill!(similar(logits, T, Tmax, S, B), T(-Inf))
    for t in Int32(1):Int32(Tmax)
        pruned_fwd_col_kernel!(backend)(α, lp, off_d, lab_d, Ul_d, Tl_d,
                                        Int32(blank), t; ndrange = B)
    end
    β = fill!(similar(α), T(-Inf))
    for t in Int32(Tmax):-Int32(1):Int32(1)
        pruned_bwd_col_kernel!(backend)(β, lp, off_d, lab_d, Ul_d, Tl_d,
                                        Int32(blank), t; ndrange = B)
    end
    nll = similar(logits, T, B)
    lattice_nll_kernel!(backend)(nll, β, Tl_d; ndrange = B)   # β[1,1] = start

    grad = similar(logits)
    pruned_grad_kernel!(backend)(grad, α, β, lp, off_d, lab_d, Ul_d, Tl_d,
                                 nll, Int32(blank); ndrange = (Tmax, S, B))
    KernelAbstractions.synchronize(backend)
    sum(nll) / T(B), grad ./ T(B)
end

"""
    pruned_rnnt_loss_batched(logits, band_offsets, targets, input_lengths;
                             blank = size(logits, 1))

RNN-T loss on a **banded lattice** — the memory-critical core of the pruned
transducer (Kuang et al., Interspeech 2022, arXiv:2206.13236). The joint is
evaluated only on `S = size(logits, 3)` cells per frame, at lattice
positions `u = band_offsets[t, b] + s`; alignments are confined to the band,
so the joint tensor shrinks from `O(V·T·U·B)` to `O(V·T·S·B)`.

Offsets must be monotone non-decreasing per sample with
`band_offsets[1, b] == 0`; obtain them from [`pruning_bounds`](@ref) (or any
alignment source). With `S = U + 1` and zero offsets this is *exactly* the
full RNN-T loss — the test suite asserts that equality.
"""
function pruned_rnnt_loss_batched(logits::AbstractArray{T,4},
                                  band_offsets::AbstractMatrix{<:Integer},
                                  targets::Vector{Vector{Int}},
                                  input_lengths::Vector{Int};
                                  blank::Int = size(logits, 1)) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, _ = pruned_forward_backward(logits, Int32.(Matrix(band_offsets)),
                                      labels, target_lengths, input_lengths,
                                      blank)
    loss
end

function ChainRulesCore.rrule(::typeof(pruned_rnnt_loss_batched),
                              logits::AbstractArray{T,4},
                              band_offsets::AbstractMatrix{<:Integer},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int};
                              blank::Int = size(logits, 1)) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, grad = pruned_forward_backward(logits, Int32.(Matrix(band_offsets)),
                                         labels, target_lengths,
                                         input_lengths, blank)
    pruned_pullback(Δ) = (NoTangent(), Δ * grad, NoTangent(), NoTangent(),
                          NoTangent())
    loss, pruned_pullback
end

"""
    pruning_bounds(am, lm, targets, input_lengths; band_width,
                   blank = size(am, 1)) -> Matrix{Int32}

k2-inspired band estimation from the **trivial joint** `am ⊕ lm` (encoder
logits `(V, T, B)`, prediction logits `(V, U+1, B)`), which never
materializes a `(V, T, U+1, B)` tensor: per-cell normalizers are reduced
inside a kernel, the shared lattice forward-backward runs on the resulting
`(T, U+1, B)` emissions, and the band is centred on the label-posterior mass
per frame (monotone, start-anchored, exit-reachable).

Simplification stated honestly: k2 derives bounds from the *gradients* of a
smoothed trivial-joint loss; centring on posterior mass is a simpler proxy
that keeps the same guarantees (monotone, feasible) but is not
bit-compatible with k2. Throws if `band_width` is too small to sweep from
the start cell to the exit within `T` frames.
"""
function pruning_bounds(am::AbstractArray{T,3}, lm::AbstractArray{T,3},
                        targets::Vector{Vector{Int}},
                        input_lengths::Vector{Int};
                        band_width::Int,
                        blank::Int = size(am, 1)) where {T}
    V, Tmax, B = size(am)
    size(lm, 1) == V && size(lm, 3) == B || throw(ArgumentError(
        "lm must be (V, U+1, B) matching am"))
    U1max = size(lm, 2)
    band_width >= 2 || throw(ArgumentError("band_width must be ≥ 2"))
    labels, target_lengths = pack_transducer_targets(targets, blank)
    maximum(target_lengths; init = Int32(0)) + 1 <= U1max ||
        throw(ArgumentError("lm provides $U1max prediction states but " *
            "targets need $(maximum(target_lengths) + 1)"))
    backend = KernelAbstractions.get_backend(am)
    lab_d, Ul_d, Tl_d = device_inputs(am, labels, target_lengths,
                                      input_lengths, Tmax)

    em_b = similar(am, T, Tmax, U1max, B)
    em_l = similar(am, T, Tmax, U1max, B)
    trivial_joint_kernel!(backend)(em_b, em_l, am, lm, lab_d, Ul_d, Tl_d,
                                   Int32(blank); ndrange = (Tmax, U1max, B))
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
    KernelAbstractions.synchronize(backend)

    # Small (T, B) result: finish on CPU.
    A, Bt, El = Array(α), Array(β), Array(em_l)
    off = zeros(Int32, Tmax, B)
    for b in 1:B
        Tb = clamp(input_lengths[b], 0, Tmax)
        Ub = Int(target_lengths[b])
        Tb < 1 && continue
        logZ = Bt[1, 1, b]
        prev = 0.0
        for t in 1:Tb
            num = den = 0.0
            for u in 1:Ub
                p = exp(A[t, u, b] + El[t, u, b] + Bt[t, u + 1, b] - logZ)
                num += u * p
                den += p
            end
            center = den > 0 ? num / den : prev
            prev = center
            cand = clamp(round(Int, center - band_width / 2), 0,
                         max(0, Ub + 1 - band_width))
            off[t, b] = max(t > 1 ? off[t - 1, b] : Int32(0), Int32(cand))
        end
        for t in 2:Tb                      # cap forward jumps at w−1 so a
            off[t, b] = clamp(off[t, b],   # path can climb within the band
                              off[t - 1, b],
                              off[t - 1, b] + Int32(band_width - 1))
        end
        # exit must be reachable: raise the tail with a feasible ramp
        required = max(0, Ub + 1 - band_width)
        off[Tb, b] = max(off[Tb, b], Int32(required))
        for t in (Tb - 1):-1:1
            off[t, b] = max(off[t, b], off[t + 1, b] - Int32(band_width - 1))
        end
        off[1, b] == 0 || throw(ArgumentError(
            "band_width = $band_width too small to reach $(Ub) labels in " *
            "$(Tb) frames (sample $b)"))
        for t in 2:Tb                                    # re-assert monotone
            off[t, b] = max(off[t, b], off[t - 1, b])
        end
    end
    off
end
