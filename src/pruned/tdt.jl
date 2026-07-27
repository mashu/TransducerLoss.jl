# Banded-lattice TDT: driver, public API, and duration-aware bound estimation.

"""
    pruned_tdt_forward_backward(logits, dur_logits, band_offsets, labels,
                                target_lengths, input_lengths, durations,
                                blank, sigma, fastemit_lambda)

Internal: `(loss, grad_tok, grad_dur)` for the banded TDT lattice. Both logit
tensors are `(·, T, S, B)` — the joint evaluated only on band cells — and
`band_offsets` maps band slot `s` at frame `t` to lattice cell
`u = band_offsets[t, b] + s`.
"""
function pruned_tdt_forward_backward(logits::AbstractArray{T,4},
                                     dur_logits::AbstractArray{T,4},
                                     band_offsets::Matrix{Int32},
                                     labels::Matrix{Int32},
                                     target_lengths::Vector{Int32},
                                     input_lengths::Vector{Int},
                                     durations::DurationSpec, blank::Int,
                                     sigma::Real, fastemit_lambda::Real) where {T}
    V, Tmax, S, B = size(logits)
    size(dur_logits)[2:4] == (Tmax, S, B) || throw(ArgumentError(
        "dur_logits must share (T, S, B) with logits"))
    size(dur_logits, 1) == length(durations) || throw(ArgumentError(
        "dur_logits has $(size(dur_logits, 1)) duration classes but " *
        "$(length(durations)) durations were given"))
    1 <= blank <= V || throw(ArgumentError("blank $blank outside 1:$V"))
    issorted(durations) && allunique(durations) && all(>=(0), durations) ||
        throw(ArgumentError("durations must be ascending, unique, non-negative"))
    any(>=(1), durations) ||
        throw(ArgumentError("durations must include a value ≥ 1"))
    fastemit_lambda >= 0 || throw(ArgumentError("fastemit_lambda must be ≥ 0"))
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
    σ = T(sigma)
    λ = T(fastemit_lambda)
    lab_d, Ul_d, Tl_d = device_inputs(logits, labels, target_lengths,
                                      input_lengths, Tmax)
    off_d = copyto!(similar(logits, Int32, Tmax, B), band_offsets)
    dur_d = copyto!(similar(logits, Int32, length(durations)),
                    collect(Int32, durations))

    lp = logsoftmax(logits; dims = 1)
    ld = logsoftmax(dur_logits; dims = 1)

    # Columns, not diagonals: within a column the dd = 0 token chain is
    # serial, and every cross-column read (t − dd) is already complete.
    α = fill!(similar(logits, T, Tmax, S, B), T(-Inf))
    for t in Int32(1):Int32(Tmax)
        pruned_tdt_fwd_col_kernel!(backend)(α, lp, ld, off_d, lab_d, dur_d,
                                            Ul_d, Tl_d, σ, Int32(blank), t;
                                            ndrange = B)
    end
    β = fill!(similar(α), T(-Inf))
    for t in Int32(Tmax):-Int32(1):Int32(1)
        pruned_tdt_bwd_col_kernel!(backend)(β, lp, ld, off_d, lab_d, dur_d,
                                            Ul_d, Tl_d, σ, Int32(blank), t;
                                            ndrange = B)
    end
    nll = similar(logits, T, B)
    lattice_nll_kernel!(backend)(nll, β, Tl_d; ndrange = B)   # β[1,1] = start

    gtok = similar(logits)
    gdur = similar(dur_logits)
    pruned_tdt_grad_kernel!(backend)(gtok, gdur, α, β, lp, ld, off_d, lab_d,
                                     dur_d, Ul_d, Tl_d, nll, σ, Int32(blank),
                                     λ; ndrange = (Tmax, S, B))
    KernelAbstractions.synchronize(backend)
    sum(nll) / T(B), gtok ./ T(B), gdur ./ T(B)
end

"""
    pruned_tdt_loss_batched(logits, dur_logits, band_offsets, targets,
                            input_lengths, durations; blank, sigma,
                            fastemit_lambda)

**Token-and-Duration Transducer loss on a banded lattice** — pruning (Kuang
et al., Interspeech 2022, arXiv:2206.13236) applied to TDT (Xu et al., ICML
2023). The joint is evaluated only on `S = size(logits, 3)` cells per frame,
at lattice positions `u = band_offsets[t, b] + s`, so both logit tensors
shrink from `O(T·U·B)` to `O(T·S·B)`.

This is the memory lever for transducer training. Everything else — joint
width, batch size, utterance length — is a constant factor on a tensor that
is quadratic in the lattice; banding attacks the lattice itself. At a band of
5 against 160 label positions it is a ~32× cut.

- `logits`: `(V, T, S, B)` raw token-head outputs on band cells.
- `dur_logits`: `(D, T, S, B)` raw duration-head outputs on the same cells.
- `band_offsets`: `(T, B)`, monotone non-decreasing per sample with
  `band_offsets[1, b] == 0`. Get them from [`tdt_pruning_bounds`](@ref).
- `durations`, `blank`, `sigma`, `fastemit_lambda`: as
  [`tdt_loss_batched`](@ref). FastEmit works here — it is a reweighting of
  the token-departure gradient, which is a per-cell quantity, so banding
  does not interfere with it.

With `S = U + 1` and zero offsets this is **exactly** [`tdt_loss_batched`](@ref);
the test suite asserts that equality on both the loss and both gradients.

Alignments that leave the band are dropped, so the returned value is an
upper bound on the true NLL. A band chosen by [`tdt_pruning_bounds`](@ref)
keeps essentially all of the mass; a hand-picked one may not.
"""
function pruned_tdt_loss_batched(logits::AbstractArray{T,4},
                                 dur_logits::AbstractArray{T,4},
                                 band_offsets::AbstractMatrix{<:Integer},
                                 targets::Vector{Vector{Int}},
                                 input_lengths::Vector{Int},
                                 durations::DurationSpec;
                                 blank::Int = size(logits, 1),
                                 sigma::Real = 0,
                                 fastemit_lambda::Real = 0) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, _, _ = pruned_tdt_forward_backward(logits, dur_logits,
                                             Int32.(Matrix(band_offsets)),
                                             labels, target_lengths,
                                             input_lengths, durations, blank,
                                             sigma, fastemit_lambda)
    loss
end

function ChainRulesCore.rrule(::typeof(pruned_tdt_loss_batched),
                              logits::AbstractArray{T,4},
                              dur_logits::AbstractArray{T,4},
                              band_offsets::AbstractMatrix{<:Integer},
                              targets::Vector{Vector{Int}},
                              input_lengths::Vector{Int},
                              durations::DurationSpec;
                              blank::Int = size(logits, 1),
                              sigma::Real = 0,
                              fastemit_lambda::Real = 0) where {T}
    labels, target_lengths = pack_transducer_targets(targets, blank)
    loss, gtok, gdur = pruned_tdt_forward_backward(logits, dur_logits,
                                                   Int32.(Matrix(band_offsets)),
                                                   labels, target_lengths,
                                                   input_lengths, durations,
                                                   blank, sigma,
                                                   fastemit_lambda)
    pruned_tdt_pullback(Δ) = (NoTangent(), Δ * gtok, Δ * gdur, NoTangent(),
                              NoTangent(), NoTangent(), NoTangent())
    loss, pruned_tdt_pullback
end

"""
    tdt_pruning_bounds(am, lm, am_dur, lm_dur, targets, input_lengths;
                       band_width, durations, blank = size(am, 1), sigma = 0)
        -> Matrix{Int32}

Duration-aware band estimation from the **trivial joint**: token logits
`am (V, T, B)` ⊕ `lm (V, U+1, B)` and duration logits `am_dur (D, T, B)` ⊕
`lm_dur (D, U+1, B)`. The `(V, T, U+1, B)` token tensor never materializes —
per-cell normalizers are reduced inside a kernel — and the duration tensor
that *does* materialize is `(D, T, U+1, B)`, smaller by `V/D`.

Uses the TDT lattice, not RNN-T's. That matters: a duration-`d` blank skips
`d` frames without advancing `u`, so TDT occupancy sits lower and flatter in
`u` than RNN-T's at the same frame. Bounds estimated with the wrong
transition model push real alignment mass outside the band, and the loss
silently rises rather than failing.

Centres each frame's band on **cell occupancy** `α + β − logZ` rather than on
RNN-T `pruning_bounds`' label-emission posterior. Occupancy is the
probability of passing through `(t, u)` at all, which is the quantity the
band is supposed to contain, and it needs no duration-weighted sum over exit
transitions. Offsets are then made monotone, capped at `band_width − 1`
growth per frame so a path can still climb inside the band, and ramped so
the exit stays reachable.

Throws if `band_width` is too small to sweep from the start cell to the exit
within `T` frames.
"""
function tdt_pruning_bounds(am::AbstractArray{T,3}, lm::AbstractArray{T,3},
                            am_dur::AbstractArray{T,3},
                            lm_dur::AbstractArray{T,3},
                            targets::Vector{Vector{Int}},
                            input_lengths::Vector{Int};
                            band_width::Int, durations::DurationSpec,
                            blank::Int = size(am, 1), sigma::Real = 0) where {T}
    V, Tmax, B = size(am)
    size(lm, 1) == V && size(lm, 3) == B || throw(ArgumentError(
        "lm must be (V, U+1, B) matching am"))
    U1max = size(lm, 2)
    D = length(durations)
    size(am_dur) == (D, Tmax, B) || throw(ArgumentError(
        "am_dur must be (D, T, B) = ($D, $Tmax, $B)"))
    size(lm_dur) == (D, U1max, B) || throw(ArgumentError(
        "lm_dur must be (D, U+1, B) = ($D, $U1max, $B)"))
    band_width >= 2 || throw(ArgumentError("band_width must be ≥ 2"))
    issorted(durations) && allunique(durations) && all(>=(0), durations) ||
        throw(ArgumentError("durations must be ascending, unique, non-negative"))
    labels, target_lengths = pack_transducer_targets(targets, blank)
    maximum(target_lengths; init = Int32(0)) + 1 <= U1max ||
        throw(ArgumentError("lm provides $U1max prediction states but " *
            "targets need $(maximum(target_lengths) + 1)"))
    backend = KernelAbstractions.get_backend(am)
    σ = T(sigma)
    lab_d, Ul_d, Tl_d = device_inputs(am, labels, target_lengths,
                                      input_lengths, Tmax)
    dur_d = copyto!(similar(am, Int32, D), collect(Int32, durations))

    em_b = similar(am, T, Tmax, U1max, B)
    em_l = similar(am, T, Tmax, U1max, B)
    trivial_joint_kernel!(backend)(em_b, em_l, am, lm, lab_d, Ul_d, Tl_d,
                                   Int32(blank); ndrange = (Tmax, U1max, B))
    ldt = similar(am, T, D, Tmax, U1max, B)
    trivial_duration_kernel!(backend)(ldt, am_dur, lm_dur, Ul_d, Tl_d;
                                      ndrange = (Tmax, U1max, B))

    α = fill!(similar(em_b), T(-Inf))
    lattice_fwd_init_kernel!(backend)(α, Tl_d; ndrange = B)
    for d in Int32(3):Int32(Tmax + U1max)
        tdt_fwd_diag_kernel!(backend)(α, em_b, em_l, ldt, dur_d, Ul_d, Tl_d,
                                      σ, d; ndrange = (U1max, B))
    end
    # TDT's backward embeds the exit terms — no separate init kernel.
    β = fill!(similar(em_b), T(-Inf))
    for d in Int32(Tmax + U1max):-Int32(1):Int32(2)
        tdt_bwd_diag_kernel!(backend)(β, em_b, em_l, ldt, dur_d, Ul_d, Tl_d,
                                      σ, d; ndrange = (U1max, B))
    end
    KernelAbstractions.synchronize(backend)

    # Small (T, B) result: finish on CPU.
    A, Bt = Array(α), Array(β)
    centres = zeros(Float64, Tmax, B)
    for b in 1:B
        Tb = clamp(input_lengths[b], 0, Tmax)
        Ub = Int(target_lengths[b])
        Tb < 1 && continue
        logZ = Bt[1, 1, b]
        prev = 0.0
        for t in 1:Tb
            num = den = 0.0
            for u in 1:(Ub + 1)
                p = exp(A[t, u, b] + Bt[t, u, b] - logZ)
                num += u * p
                den += p
            end
            prev = den > 0 ? num / den : prev
            centres[t, b] = prev
        end
    end
    band_offsets_from_centres(centres, target_lengths, input_lengths,
                              band_width, Tmax)
end
