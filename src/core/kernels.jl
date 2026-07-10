# Kernels shared by every transducer-family loss in this package.
#
# All lattices here use the same cell convention — t ∈ 1:Tb frames consumed,
# u ∈ 1:Ub+1 prediction states (u−1 labels emitted) — so emission gathering,
# α initialization, and the final NLL read-out are loss-independent.

@kernel function gather_emissions_kernel!(em_b::AbstractArray{T,3},
        em_l::AbstractArray{T,3}, @Const(lp::AbstractArray{T,4}),
        @Const(lab::AbstractMatrix{Int32}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32}), blank::Int32) where {T}
    t, u, b = @index(Global, NTuple)
    @inbounds if t <= Tl[b] && u <= Ul[b] + Int32(1)
        em_b[t, u, b] = lp[blank, t, u, b]
        em_l[t, u, b] = u <= Ul[b] ? lp[lab[u, b], t, u, b] : T(-Inf)
    else
        em_b[t, u, b] = T(-Inf)
        em_l[t, u, b] = T(-Inf)
    end
end

@kernel function lattice_fwd_init_kernel!(α::AbstractArray{T,3},
        @Const(Tl::AbstractVector{Int32})) where {T}
    b = @index(Global)
    @inbounds Tl[b] >= Int32(1) && (α[1, 1, b] = T(0))
end

@kernel function lattice_nll_kernel!(nll::AbstractVector{T},
        @Const(β::AbstractArray{T,3}), @Const(Tl::AbstractVector{Int32})) where {T}
    b = @index(Global)
    @inbounds nll[b] = Tl[b] >= Int32(1) ? -β[1, 1, b] : T(Inf)
end

# Single-step lattice recursions (blank: t→t+1; label: u→u+1). These are
# emission-agnostic — they consume only (em_b, em_l) — so vanilla RNN-T and
# HAT share them; only emission gathering and gradients differ per loss.

@kernel function lattice_fwd_diag_kernel!(α::AbstractArray{T,3},
        @Const(em_b::AbstractArray{T,3}), @Const(em_l::AbstractArray{T,3}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        d::Int32) where {T}
    u, b = @index(Global, NTuple)
    @inbounds begin
        t = d - u
        if u <= Ul[b] + Int32(1) && Int32(1) <= t <= Tl[b] &&
           !(t == Int32(1) && u == Int32(1))
            a = t > Int32(1) ? α[t - Int32(1), u, b] + em_b[t - Int32(1), u, b] :
                               T(-Inf)
            c = u > Int32(1) ? α[t, u - Int32(1), b] + em_l[t, u - Int32(1), b] :
                               T(-Inf)
            α[t, u, b] = logaddexp(a, c)
        end
    end
end

@kernel function lattice_bwd_init_kernel!(β::AbstractArray{T,3},
        @Const(em_b::AbstractArray{T,3}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32})) where {T}
    b = @index(Global)
    @inbounds begin
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        Tb >= Int32(1) && (β[Tb, U1, b] = em_b[Tb, U1, b])   # exit blank
    end
end

@kernel function lattice_bwd_diag_kernel!(β::AbstractArray{T,3},
        @Const(em_b::AbstractArray{T,3}), @Const(em_l::AbstractArray{T,3}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        d::Int32) where {T}
    u, b = @index(Global, NTuple)
    @inbounds begin
        t = d - u
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        if u <= U1 && Int32(1) <= t <= Tb && !(t == Tb && u == U1)
            a = t < Tb ? em_b[t, u, b] + β[t + Int32(1), u, b] : T(-Inf)
            c = u < U1 ? em_l[t, u, b] + β[t, u + Int32(1), b] : T(-Inf)
            β[t, u, b] = logaddexp(a, c)
        end
    end
end
