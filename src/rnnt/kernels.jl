# KernelAbstractions kernels specific to the vanilla RNN-T lattice
# (Graves 2012): blank advances time by exactly one, labels advance the
# label axis in place. Shared gather/init/NLL kernels live in core/.

@kernel function rnnt_fwd_diag_kernel!(α::AbstractArray{T,3},
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

@kernel function rnnt_bwd_init_kernel!(β::AbstractArray{T,3},
        @Const(em_b::AbstractArray{T,3}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32})) where {T}
    b = @index(Global)
    @inbounds begin
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        Tb >= Int32(1) && (β[Tb, U1, b] = em_b[Tb, U1, b])   # exit blank
    end
end

@kernel function rnnt_bwd_diag_kernel!(β::AbstractArray{T,3},
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

@kernel function rnnt_grad_kernel!(grad::AbstractArray{T,4},
        @Const(α::AbstractArray{T,3}), @Const(β::AbstractArray{T,3}),
        @Const(em_b::AbstractArray{T,3}), @Const(em_l::AbstractArray{T,3}),
        @Const(lp::AbstractArray{T,4}), @Const(lab::AbstractMatrix{Int32}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        @Const(nll::AbstractVector{T}), blank::Int32, fastemit::T) where {T}
    t, u, b = @index(Global, NTuple)
    @inbounds begin
        V = size(grad, 1)
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        nll_b = nll[b]
        if t > Tb || u > U1 || (isinf(nll_b) && nll_b > T(0))
            # padding cell, or impossible sample: no probability mass to move
            for k in Int32(1):Int32(V)
                grad[k, t, u, b] = T(0)
            end
        else
            # occupancy term: p(k) · exp(α + β − logZ);  logZ = −nll
            occ = exp(α[t, u, b] + β[t, u, b] + nll_b)
            for k in Int32(1):Int32(V)
                grad[k, t, u, b] = exp(lp[k, t, u, b]) * occ
            end
            # subtract the posterior of each realizable transition
            βnext = t < Tb ? β[t + Int32(1), u, b] :
                             (u == U1 ? T(0) : T(-Inf))
            grad[blank, t, u, b] -= exp(α[t, u, b] + em_b[t, u, b] + βnext + nll_b)
            if u < U1
                k = lab[u, b]
                post = exp(α[t, u, b] + em_l[t, u, b] +
                           β[t, u + Int32(1), b] + nll_b)
                grad[k, t, u, b] -= (T(1) + fastemit) * post
            end
        end
    end
end
