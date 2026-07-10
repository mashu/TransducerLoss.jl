# HAT (Hybrid Autoregressive Transducer, Variani et al. 2020,
# arXiv:2003.07705) kernels. Blank is a per-cell Bernoulli σ(b); labels get
# log(1 − σ(b)) + logsoftmax over a blank-free vocabulary. The lattice
# recursions are the shared core kernels — only emission gathering and the
# factorized gradient live here.

@kernel function hat_gather_kernel!(em_b::AbstractArray{T,3},
        em_l::AbstractArray{T,3}, @Const(blogit::AbstractArray{T,3}),
        @Const(lp::AbstractArray{T,4}), @Const(lab::AbstractMatrix{Int32}),
        @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32})) where {T}
    t, u, b = @index(Global, NTuple)
    @inbounds if t <= Tl[b] && u <= Ul[b] + Int32(1)
        x = blogit[t, u, b]
        em_b[t, u, b] = -log1p(exp(-x))                       # log σ(x)
        em_l[t, u, b] = u <= Ul[b] ?
            -log1p(exp(x)) + lp[lab[u, b], t, u, b] : T(-Inf) # log(1−σ) + lp
    else
        em_b[t, u, b] = T(-Inf)
        em_l[t, u, b] = T(-Inf)
    end
end

@kernel function hat_grad_kernel!(gb::AbstractArray{T,3},
        gy::AbstractArray{T,4}, @Const(α::AbstractArray{T,3}),
        @Const(β::AbstractArray{T,3}), @Const(em_b::AbstractArray{T,3}),
        @Const(em_l::AbstractArray{T,3}), @Const(blogit::AbstractArray{T,3}),
        @Const(lp::AbstractArray{T,4}), @Const(lab::AbstractMatrix{Int32}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        @Const(nll::AbstractVector{T})) where {T}
    t, u, b = @index(Global, NTuple)
    @inbounds begin
        V = size(gy, 1)
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        nll_b = nll[b]
        if t > Tb || u > U1 || (isinf(nll_b) && nll_b > T(0))
            gb[t, u, b] = T(0)
            for k in Int32(1):Int32(V)
                gy[k, t, u, b] = T(0)
            end
        else
            βn = t < Tb ? β[t + Int32(1), u, b] :
                          (u == U1 ? T(0) : T(-Inf))
            Pb = exp(α[t, u, b] + em_b[t, u, b] + βn + nll_b)
            Pl = u <= Ul[b] ?
                 exp(α[t, u, b] + em_l[t, u, b] + β[t, u + Int32(1), b] +
                     nll_b) : T(0)
            s = inv(one(T) + exp(-blogit[t, u, b]))           # σ(b)
            gb[t, u, b] = -Pb * (one(T) - s) + Pl * s
            for k in Int32(1):Int32(V)
                gy[k, t, u, b] = Pl * exp(lp[k, t, u, b])
            end
            u <= Ul[b] && (gy[lab[u, b], t, u, b] -= Pl)
        end
    end
end
