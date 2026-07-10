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
