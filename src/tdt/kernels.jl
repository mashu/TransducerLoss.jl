# KernelAbstractions kernels for the TDT (Token-and-Duration Transducer)
# forward-backward (Xu et al., ICML 2023, arXiv:2304.06795), following the
# lattice semantics of NVIDIA NeMo's reference kernels:
#
#   * two independently normalized heads per cell: token logprobs (blank =
#     last by our convention) and duration logprobs over an ascending set;
#   * blank transitions require duration ≥ 1 (no self-loop), token
#     transitions allow duration 0 (multiple tokens per frame, like RNN-T);
#   * the exit blank from (t, U+1) must land exactly on frame T+1;
#   * every transition score is under-normalized by a constant σ
#     (NeMo's "logit under-normalization"; 0 disables it).
#
# The backward recursion embeds the exit terms, so — unlike the vanilla
# transducer — no separate β-init kernel is needed: every cell is written by
# the diagonal loop itself. Shared gather/init/NLL kernels live in core/.

@kernel function tdt_fwd_diag_kernel!(α::AbstractArray{T,3},
        @Const(em_b::AbstractArray{T,3}), @Const(em_l::AbstractArray{T,3}),
        @Const(ld::AbstractArray{T,4}), @Const(dur::AbstractVector{Int32}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        σ::T, d::Int32) where {T}
    u, b = @index(Global, NTuple)
    @inbounds begin
        t = d - u
        if u <= Ul[b] + Int32(1) && Int32(1) <= t <= Tl[b] &&
           !(t == Int32(1) && u == Int32(1))
            v = T(-Inf)
            for i in Int32(1):Int32(length(dur))
                dd = dur[i]
                tp = t - dd
                tp < Int32(1) && break          # durations ascend: none left fit
                if dd >= Int32(1)               # blank arrival from (t−dd, u)
                    v = logaddexp(v, α[tp, u, b] + em_b[tp, u, b] +
                                     ld[i, tp, u, b] - σ)
                end
                if u > Int32(1)                 # token arrival from (t−dd, u−1)
                    v = logaddexp(v, α[tp, u - Int32(1), b] +
                                     em_l[tp, u - Int32(1), b] +
                                     ld[i, tp, u - Int32(1), b] - σ)
                end
            end
            α[t, u, b] = v
        end
    end
end

@kernel function tdt_bwd_diag_kernel!(β::AbstractArray{T,3},
        @Const(em_b::AbstractArray{T,3}), @Const(em_l::AbstractArray{T,3}),
        @Const(ld::AbstractArray{T,4}), @Const(dur::AbstractVector{Int32}),
        @Const(Ul::AbstractVector{Int32}), @Const(Tl::AbstractVector{Int32}),
        σ::T, d::Int32) where {T}
    u, b = @index(Global, NTuple)
    @inbounds begin
        t = d - u
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        if u <= U1 && Int32(1) <= t <= Tb
            v = T(-Inf)
            for i in Int32(1):Int32(length(dur))
                dd = dur[i]
                tn = t + dd
                if dd >= Int32(1)               # blank departure
                    if tn <= Tb
                        v = logaddexp(v, em_b[t, u, b] + ld[i, t, u, b] - σ +
                                         β[tn, u, b])
                    elseif tn == Tb + Int32(1) && u == U1
                        v = logaddexp(v, em_b[t, u, b] + ld[i, t, u, b] - σ)
                    end
                end
                if u < U1 && tn <= Tb           # token departure (dd = 0 allowed)
                    v = logaddexp(v, em_l[t, u, b] + ld[i, t, u, b] - σ +
                                     β[tn, u + Int32(1), b])
                end
            end
            β[t, u, b] = v
        end
    end
end

@kernel function tdt_grad_kernel!(gtok::AbstractArray{T,4},
        gdur::AbstractArray{T,4}, @Const(α::AbstractArray{T,3}),
        @Const(β::AbstractArray{T,3}), @Const(em_b::AbstractArray{T,3}),
        @Const(em_l::AbstractArray{T,3}), @Const(lp::AbstractArray{T,4}),
        @Const(ld::AbstractArray{T,4}), @Const(lab::AbstractMatrix{Int32}),
        @Const(dur::AbstractVector{Int32}), @Const(Ul::AbstractVector{Int32}),
        @Const(Tl::AbstractVector{Int32}), @Const(nll::AbstractVector{T}),
        σ::T, blank::Int32) where {T}
    t, u, b = @index(Global, NTuple)
    @inbounds begin
        V = size(gtok, 1)
        D = size(gdur, 1)
        Tb = Tl[b]
        U1 = Ul[b] + Int32(1)
        nll_b = nll[b]
        if t > Tb || u > U1 || (isinf(nll_b) && nll_b > T(0))
            for k in Int32(1):Int32(V)
                gtok[k, t, u, b] = T(0)
            end
            for i in Int32(1):Int32(D)
                gdur[i, t, u, b] = T(0)
            end
        else
            occ = exp(α[t, u, b] + β[t, u, b] + nll_b)
            for k in Int32(1):Int32(V)
                gtok[k, t, u, b] = exp(lp[k, t, u, b]) * occ
            end
            for i in Int32(1):Int32(D)
                gdur[i, t, u, b] = exp(ld[i, t, u, b]) * occ
            end
            for i in Int32(1):Int32(D)
                dd = dur[i]
                tn = t + dd
                if dd >= Int32(1)
                    βn = tn <= Tb ? β[tn, u, b] :
                         (tn == Tb + Int32(1) && u == U1 ? T(0) : T(-Inf))
                    post = exp(α[t, u, b] + em_b[t, u, b] + ld[i, t, u, b] -
                               σ + βn + nll_b)
                    gtok[blank, t, u, b] -= post
                    gdur[i, t, u, b] -= post
                end
                if u < U1 && tn <= Tb
                    post = exp(α[t, u, b] + em_l[t, u, b] + ld[i, t, u, b] -
                               σ + β[tn, u + Int32(1), b] + nll_b)
                    gtok[lab[u, b], t, u, b] -= post
                    gdur[i, t, u, b] -= post
                end
            end
        end
    end
end
