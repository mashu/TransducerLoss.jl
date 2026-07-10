# Vanilla RNN-T gradient + minimum-latency moment kernels. The lattice
# recursions live in core/ (shared with HAT); only what is specific to the
# softmax-joint gradient and the latency expectation lives here.




@kernel function rnnt_grad_kernel!(grad::AbstractArray{T,4},
        @Const(α::AbstractArray{T,3}), @Const(β::AbstractArray{T,3}),
        @Const(em_b::AbstractArray{T,3}), @Const(em_l::AbstractArray{T,3}),
        @Const(lp::AbstractArray{T,4}), @Const(lab::AbstractMatrix{Int32}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        @Const(m_α::AbstractArray{T,3}), @Const(m_β::AbstractArray{T,3}),
        @Const(nll::AbstractVector{T}), @Const(E::AbstractVector{T}),
        blank::Int32, fastemit::T, λlat::T) where {T}
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
            if λlat > zero(T)
                # minimum-latency term: ∂E/∂s(τ) = (∂Z̃/∂s)/Z − E·post(τ),
                # with Z̃ the first latency moment (label from (t,u) emits
                # at frame t; the exit blank carries no moment).
                lamU = λlat / T(max(Ul[b], Int32(1)))
                βn  = t < Tb ? β[t + Int32(1), u, b] :
                               (u == U1 ? T(0) : T(-Inf))
                mβn = t < Tb ? m_β[t + Int32(1), u, b] : T(-Inf)
                Pb  = exp(α[t, u, b] + em_b[t, u, b] + βn + nll_b)
                eb  = lamU * (exp(m_α[t, u, b] + em_b[t, u, b] + βn + nll_b) +
                              exp(α[t, u, b] + em_b[t, u, b] + mβn + nll_b) -
                              E[b] * Pb)
                el = zero(T)
                if u <= Ul[b]
                    Pl = exp(α[t, u, b] + em_l[t, u, b] +
                             β[t, u + Int32(1), b] + nll_b)
                    el = lamU * (exp(m_α[t, u, b] + em_l[t, u, b] +
                                     β[t, u + Int32(1), b] + nll_b) +
                                 exp(α[t, u, b] + em_l[t, u, b] +
                                     m_β[t, u + Int32(1), b] + nll_b) +
                                 T(t) * Pl - E[b] * Pl)
                end
                for k in Int32(1):Int32(V)
                    grad[k, t, u, b] -= exp(lp[k, t, u, b]) * (eb + el)
                end
                grad[blank, t, u, b] += eb
                u <= Ul[b] && (grad[lab[u, b], t, u, b] += el)
            end
        end
    end
end

# First latency moment, forward: Z̃-prefix mass (log space). No init needed —
# the empty prefix has latency 0, so m_α(1,1) stays −Inf.
@kernel function latency_moment_fwd_kernel!(m_α::AbstractArray{T,3},
        @Const(α::AbstractArray{T,3}), @Const(em_b::AbstractArray{T,3}),
        @Const(em_l::AbstractArray{T,3}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32}), d::Int32) where {T}
    u, b = @index(Global, NTuple)
    @inbounds begin
        t = d - u
        if u <= Ul[b] + Int32(1) && Int32(1) <= t <= Tl[b] &&
           !(t == Int32(1) && u == Int32(1))
            a = t > Int32(1) ? m_α[t - Int32(1), u, b] + em_b[t - Int32(1), u, b] :
                               T(-Inf)
            v = a
            if u > Int32(1)
                c = m_α[t, u - Int32(1), b] + em_l[t, u - Int32(1), b]
                w = α[t, u - Int32(1), b] + em_l[t, u - Int32(1), b] + log(T(t))
                v = logaddexp(logaddexp(v, c), w)
            end
            m_α[t, u, b] = v
        end
    end
end

# First latency moment, backward. The exit cell keeps −Inf (no suffix moment).
@kernel function latency_moment_bwd_kernel!(m_β::AbstractArray{T,3},
        @Const(β::AbstractArray{T,3}), @Const(em_b::AbstractArray{T,3}),
        @Const(em_l::AbstractArray{T,3}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32}), d::Int32) where {T}
    u, b = @index(Global, NTuple)
    @inbounds begin
        t = d - u
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        if u <= U1 && Int32(1) <= t <= Tb && !(t == Tb && u == U1)
            v = t < Tb ? em_b[t, u, b] + m_β[t + Int32(1), u, b] : T(-Inf)
            if u < U1
                v = logaddexp(v, em_l[t, u, b] + m_β[t, u + Int32(1), b])
                v = logaddexp(v, em_l[t, u, b] + β[t, u + Int32(1), b] +
                                 log(T(t)))
            end
            m_β[t, u, b] = v
        end
    end
end
