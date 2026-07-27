# Banded-lattice TDT kernels — the pruned transducer (Kuang et al. 2022,
# arXiv:2206.13236) applied to Token-and-Duration (Xu et al. 2023).
#
# Same band contract as `pruned/kernels.jl`: cells live at
# `u = off[t, b] + s` for `s ∈ 1:S`, offsets monotone non-decreasing with
# `off[1, b] == 0`, so both logit tensors are `(·, T, S, B)` instead of
# `(·, T, U+1, B)`.
#
# What differs from banded RNN-T is the *reach*. RNN-T's blank moves
# `(t, u) → (t+1, u)`, so a column depends only on its immediate predecessor.
# TDT emissions carry a duration, so a column depends on **every** column
# `t − dd` for `dd ∈ durations`. The band slot shifts by a different amount
# for each of those: arriving at `u` from column `tp` reads slot
# `u − off[tp, b]`, not `s`. Cells whose source slot falls outside `1:S` are
# outside the band and contribute `-Inf` — that is the pruning approximation,
# and it is why the bounds should be estimated with a *duration-aware*
# forward-backward (see `tdt_pruning_bounds`) rather than RNN-T's.
#
# Within a column the `dd == 0` token transition `(t, u−1) → (t, u)` is a
# serial chain, exactly as in banded RNN-T: the forward walks `s` ascending
# and the backward descending so the same-column neighbour is already written.

@kernel function pruned_tdt_fwd_col_kernel!(α::AbstractArray{T,3},
        @Const(lp::AbstractArray{T,4}), @Const(ld::AbstractArray{T,4}),
        @Const(off::AbstractMatrix{Int32}), @Const(lab::AbstractMatrix{Int32}),
        @Const(dur::AbstractVector{Int32}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32}), σ::T, blank::Int32,
        t::Int32) where {T}
    b = @index(Global)
    @inbounds if t <= Tl[b]
        S = Int32(size(α, 2))
        D = Int32(length(dur))
        for s in Int32(1):S
            u = off[t, b] + s
            u <= Ul[b] + Int32(1) || continue
            if t == Int32(1) && u == Int32(1)
                α[t, s, b] = T(0)
                continue
            end
            v = T(-Inf)
            for i in Int32(1):D
                dd = dur[i]
                tp = t - dd
                tp < Int32(1) && break          # durations ascend: none left fit
                if dd >= Int32(1)               # blank arrival from (tp, u)
                    sp = u - off[tp, b]
                    if Int32(1) <= sp <= S
                        v = logaddexp(v, α[tp, sp, b] + lp[blank, tp, sp, b] +
                                         ld[i, tp, sp, b] - σ)
                    end
                end
                if u > Int32(1)                 # token arrival from (tp, u−1)
                    sq = u - Int32(1) - off[tp, b]
                    if Int32(1) <= sq <= S
                        v = logaddexp(v, α[tp, sq, b] +
                                         lp[lab[u - Int32(1), b], tp, sq, b] +
                                         ld[i, tp, sq, b] - σ)
                    end
                end
            end
            α[t, s, b] = v
        end
    end
end

@kernel function pruned_tdt_bwd_col_kernel!(β::AbstractArray{T,3},
        @Const(lp::AbstractArray{T,4}), @Const(ld::AbstractArray{T,4}),
        @Const(off::AbstractMatrix{Int32}), @Const(lab::AbstractMatrix{Int32}),
        @Const(dur::AbstractVector{Int32}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32}), σ::T, blank::Int32,
        t::Int32) where {T}
    b = @index(Global)
    @inbounds if t <= Tl[b]
        S = Int32(size(β, 2))
        D = Int32(length(dur))
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        for s in S:-Int32(1):Int32(1)
            u = off[t, b] + s
            u <= U1 || continue
            v = T(-Inf)
            for i in Int32(1):D
                dd = dur[i]
                tn = t + dd
                if dd >= Int32(1)               # blank departure
                    if tn <= Tb
                        sn = u - off[tn, b]
                        if Int32(1) <= sn <= S
                            v = logaddexp(v, lp[blank, t, s, b] +
                                             ld[i, t, s, b] - σ + β[tn, sn, b])
                        end
                    elseif tn == Tb + Int32(1) && u == U1   # exit lands on T+1
                        v = logaddexp(v, lp[blank, t, s, b] +
                                         ld[i, t, s, b] - σ)
                    end
                end
                if u < U1 && tn <= Tb           # token departure (dd = 0 allowed)
                    sn = u + Int32(1) - off[tn, b]
                    if Int32(1) <= sn <= S
                        v = logaddexp(v, lp[lab[u, b], t, s, b] +
                                         ld[i, t, s, b] - σ + β[tn, sn, b])
                    end
                end
            end
            β[t, s, b] = v
        end
    end
end

@kernel function pruned_tdt_grad_kernel!(gtok::AbstractArray{T,4},
        gdur::AbstractArray{T,4}, @Const(α::AbstractArray{T,3}),
        @Const(β::AbstractArray{T,3}), @Const(lp::AbstractArray{T,4}),
        @Const(ld::AbstractArray{T,4}), @Const(off::AbstractMatrix{Int32}),
        @Const(lab::AbstractMatrix{Int32}), @Const(dur::AbstractVector{Int32}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        @Const(nll::AbstractVector{T}), σ::T, blank::Int32,
        fastemit::T) where {T}
    t, s, b = @index(Global, NTuple)
    @inbounds begin
        V = size(gtok, 1)
        D = Int32(size(gdur, 1))
        S = Int32(size(gtok, 3))
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        u = off[t, b] + Int32(s)
        nll_b = nll[b]
        if t > Tb || u > U1 || (isinf(nll_b) && nll_b > T(0))
            for k in Int32(1):Int32(V)
                gtok[k, t, s, b] = T(0)
            end
            for i in Int32(1):D
                gdur[i, t, s, b] = T(0)
            end
        else
            occ = exp(α[t, s, b] + β[t, s, b] + nll_b)
            for k in Int32(1):Int32(V)
                gtok[k, t, s, b] = exp(lp[k, t, s, b]) * occ
            end
            for i in Int32(1):D
                gdur[i, t, s, b] = exp(ld[i, t, s, b]) * occ
            end
            for i in Int32(1):D
                dd = dur[i]
                tn = t + dd
                if dd >= Int32(1)
                    βn = T(-Inf)
                    if tn <= Tb
                        sn = u - off[tn, b]
                        Int32(1) <= sn <= S && (βn = β[tn, sn, b])
                    elseif tn == Tb + Int32(1) && u == U1
                        βn = T(0)
                    end
                    post = exp(α[t, s, b] + lp[blank, t, s, b] +
                               ld[i, t, s, b] - σ + βn + nll_b)
                    gtok[blank, t, s, b] -= post
                    gdur[i, t, s, b] -= post
                end
                if u < U1 && tn <= Tb
                    sn = u + Int32(1) - off[tn, b]
                    if Int32(1) <= sn <= S
                        post = exp(α[t, s, b] + lp[lab[u, b], t, s, b] +
                                   ld[i, t, s, b] - σ + β[tn, sn, b] + nll_b)
                        scale = T(1) + fastemit
                        gtok[lab[u, b], t, s, b] -= scale * post
                        gdur[i, t, s, b] -= scale * post
                    end
                end
            end
        end
    end
end

# Trivial *duration* joint for bound estimation: logsoftmax over D of
# `am_dur ⊕ lm_dur`, materialized as (D, T, U+1, B). Unlike the token joint —
# which reduces to two scalars per cell and never materializes — the TDT
# recursion needs every duration class at every cell. D is the duration-set
# size (single digits to low tens), so this is orders of magnitude smaller
# than the (V, T, U+1, B) tensor pruning exists to avoid.
@kernel function trivial_duration_kernel!(ldt::AbstractArray{T,4},
        @Const(am_dur::AbstractArray{T,3}), @Const(lm_dur::AbstractArray{T,3}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32})) where {T}
    t, u, b = @index(Global, NTuple)
    @inbounds begin
        D = Int32(size(ldt, 1))
        if t <= Tl[b] && u <= Ul[b] + Int32(1)
            m = T(-Inf)
            for i in Int32(1):D
                x = am_dur[i, t, b] + lm_dur[i, u, b]
                x > m && (m = x)
            end
            z = zero(T)
            for i in Int32(1):D
                z += exp(am_dur[i, t, b] + lm_dur[i, u, b] - m)
            end
            z = m + log(z)
            for i in Int32(1):D
                ldt[i, t, u, b] = am_dur[i, t, b] + lm_dur[i, u, b] - z
            end
        else
            for i in Int32(1):D
                ldt[i, t, u, b] = T(-Inf)
            end
        end
    end
end
