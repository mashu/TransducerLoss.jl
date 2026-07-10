# Banded-lattice (pruned transducer) kernels — the memory-critical core of
# Kuang et al. 2022 (arXiv:2206.13236). The joint is evaluated only on cells
# u ∈ (off[t,b], off[t,b] + S], with monotone non-decreasing offsets, so the
# joint tensor is (V, T, S, B) instead of (V, T, U+1, B).
#
# Band columns have a same-t label chain, so — unlike the full lattice — the
# host loops over t (like CTC) and each work-item walks its column's S cells
# serially; S is the small band width, so the serial walk is cheap.

@kernel function pruned_fwd_col_kernel!(α::AbstractArray{T,3},
        @Const(lp::AbstractArray{T,4}), @Const(off::AbstractMatrix{Int32}),
        @Const(lab::AbstractMatrix{Int32}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32}), blank::Int32, t::Int32) where {T}
    b = @index(Global)
    @inbounds if t <= Tl[b]
        S = Int32(size(α, 2))
        for s in Int32(1):S
            u = off[t, b] + s
            u <= Ul[b] + Int32(1) || continue
            if t == Int32(1) && u == Int32(1)
                α[t, s, b] = T(0)
                continue
            end
            v = T(-Inf)
            if t > Int32(1)
                sp = u - off[t - Int32(1), b]        # blank from (t−1, u)
                if Int32(1) <= sp <= S
                    v = α[t - Int32(1), sp, b] +
                        lp[blank, t - Int32(1), sp, b]
                end
            end
            if u > Int32(1) && s > Int32(1)          # label from (t, u−1)
                v = logaddexp(v, α[t, s - Int32(1), b] +
                    lp[lab[u - Int32(1), b], t, s - Int32(1), b])
            end
            α[t, s, b] = v
        end
    end
end

@kernel function pruned_bwd_col_kernel!(β::AbstractArray{T,3},
        @Const(lp::AbstractArray{T,4}), @Const(off::AbstractMatrix{Int32}),
        @Const(lab::AbstractMatrix{Int32}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32}), blank::Int32, t::Int32) where {T}
    b = @index(Global)
    @inbounds if t <= Tl[b]
        S = Int32(size(β, 2))
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        for s in S:-Int32(1):Int32(1)
            u = off[t, b] + s
            u <= U1 || continue
            if t == Tb && u == U1                     # exit blank
                β[t, s, b] = lp[blank, t, s, b]
                continue
            end
            v = T(-Inf)
            if t < Tb
                sn = u - off[t + Int32(1), b]
                if Int32(1) <= sn <= S
                    v = lp[blank, t, s, b] + β[t + Int32(1), sn, b]
                end
            end
            if u <= Ul[b] && s < S                    # label to (t, u+1)
                v = logaddexp(v, lp[lab[u, b], t, s, b] +
                                 β[t, s + Int32(1), b])
            end
            β[t, s, b] = v
        end
    end
end

@kernel function pruned_grad_kernel!(grad::AbstractArray{T,4},
        @Const(α::AbstractArray{T,3}), @Const(β::AbstractArray{T,3}),
        @Const(lp::AbstractArray{T,4}), @Const(off::AbstractMatrix{Int32}),
        @Const(lab::AbstractMatrix{Int32}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32}), @Const(nll::AbstractVector{T}),
        blank::Int32) where {T}
    t, s, b = @index(Global, NTuple)
    @inbounds begin
        V = size(grad, 1)
        S = Int32(size(grad, 3))
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        u = off[t, b] + Int32(s)
        nll_b = nll[b]
        if t > Tb || u > U1 || (isinf(nll_b) && nll_b > T(0))
            for k in Int32(1):Int32(V)
                grad[k, t, s, b] = T(0)
            end
        else
            occ = exp(α[t, s, b] + β[t, s, b] + nll_b)
            for k in Int32(1):Int32(V)
                grad[k, t, s, b] = exp(lp[k, t, s, b]) * occ
            end
            βn = T(-Inf)
            if t < Tb
                sn = u - off[t + Int32(1), b]
                Int32(1) <= sn <= S && (βn = β[t + Int32(1), sn, b])
            elseif u == U1
                βn = T(0)
            end
            grad[blank, t, s, b] -= exp(α[t, s, b] + lp[blank, t, s, b] +
                                        βn + nll_b)
            if u <= Ul[b] && Int32(s) < S
                grad[lab[u, b], t, s, b] -= exp(α[t, s, b] +
                    lp[lab[u, b], t, s, b] + β[t, Int32(s) + Int32(1), b] +
                    nll_b)
            end
        end
    end
end

# Trivial-joint emissions for pruning-bound estimation: per-cell logsumexp
# over V of am ⊕ lm, without ever materializing a (V, T, U+1, B) tensor.
@kernel function trivial_joint_kernel!(em_b::AbstractArray{T,3},
        em_l::AbstractArray{T,3}, @Const(am::AbstractArray{T,3}),
        @Const(lm::AbstractArray{T,3}), @Const(lab::AbstractMatrix{Int32}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        blank::Int32) where {T}
    t, u, b = @index(Global, NTuple)
    @inbounds if t <= Tl[b] && u <= Ul[b] + Int32(1)
        V = size(am, 1)
        m = T(-Inf)
        for k in Int32(1):Int32(V)
            x = am[k, t, b] + lm[k, u, b]
            x > m && (m = x)
        end
        z = zero(T)
        for k in Int32(1):Int32(V)
            z += exp(am[k, t, b] + lm[k, u, b] - m)
        end
        z = m + log(z)
        em_b[t, u, b] = am[blank, t, b] + lm[blank, u, b] - z
        em_l[t, u, b] = u <= Ul[b] ?
            am[lab[u, b], t, b] + lm[lab[u, b], u, b] - z : T(-Inf)
    else
        em_b[t, u, b] = T(-Inf)
        em_l[t, u, b] = T(-Inf)
    end
end
