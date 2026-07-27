# Shared test oracles: pure-Julia reference implementations and brute-force
# lattice enumeration for every loss variant. Independent of the kernel code.

# ── Pure-Julia reference (independent loop implementation) ───────────────────

logsoftmax_ref(x; dims) = x .- log.(sum(exp.(x .- maximum(x; dims)); dims)) .-
                          maximum(x; dims)

"Naive single-sample transducer NLL, straight from the Graves 2012 lattice."
function ref_rnnt_nll(logits::Array{Float64,3}, target::Vector{Int}, blank::Int)
    V, T, U1 = size(logits)
    U = length(target)
    lp = logsoftmax_ref(logits; dims = 1)
    α = fill(-Inf, T, U + 1)
    α[1, 1] = 0.0
    for d in 3:(T + U + 1), u in 1:(U + 1)
        t = d - u
        1 <= t <= T || continue
        (t == 1 && u == 1) && continue
        a = t > 1 ? α[t - 1, u] + lp[blank, t - 1, u] : -Inf
        b = u > 1 ? α[t, u - 1] + lp[target[u - 1], t, u - 1] : -Inf
        α[t, u] = logaddexp(a, b)
    end
    -(α[T, U + 1] + lp[blank, T, U + 1])
end

"Brute force: enumerate every monotonic lattice path."
function brute_rnnt_nll(logits::Array{Float64,3}, target::Vector{Int}, blank::Int)
    V, T, _ = size(logits)
    U = length(target)
    lp = logsoftmax_ref(logits; dims = 1)
    total = Ref(-Inf)
    function walk(t, u, acc)
        if t == T && u == U + 1
            total[] = logaddexp(total[], acc + lp[blank, t, u])
            return
        end
        t < T && walk(t + 1, u, acc + lp[blank, t, u])
        u <= U && walk(t, u + 1, acc + lp[target[u], t, u])
    end
    walk(1, 1, 0.0)
    -total[]
end

single(logits, target, blank) =
    rnnt_loss(logits, target, size(logits, 2), blank)

"Brute force TDT: enumerate every (token, duration) lattice path."
function brute_tdt_nll(tok::Array{Float64,3}, dur::Array{Float64,3},
                       target::Vector{Int}, blank::Int,
                       durations::Vector{Int}, sigma::Float64)
    V, T, _ = size(tok)
    U = length(target)
    lp = logsoftmax_ref(tok; dims = 1)
    ld = logsoftmax_ref(dur; dims = 1)
    total = Ref(-Inf)
    function walk(t, u, acc)
        for (i, dd) in enumerate(durations)
            if dd >= 1
                s = acc + lp[blank, t, u] + ld[i, t, u] - sigma
                if t + dd <= T
                    walk(t + dd, u, s)
                elseif t + dd == T + 1 && u == U + 1
                    total[] = logaddexp(total[], s)
                end
            end
            if u <= U && t + dd <= T
                walk(t + dd, u + 1,
                     acc + lp[target[u], t, u] + ld[i, t, u] - sigma)
            end
        end
    end
    walk(1, 1, 0.0)
    -total[]
end

tdt_single(tok, dur, target, blank, durations; sigma = 0.0) =
    tdt_loss(tok, dur, target, durations; blank, sigma)

"Label-emission posteriors P(emit label at (t,u) | x) for FastEmit verification."
function ref_rnnt_label_posts(logits::Array{Float64,3}, target::Vector{Int},
                              blank::Int)
    V, T, U1 = size(logits)
    U = length(target)
    lp = logsoftmax_ref(logits; dims = 1)
    α = fill(-Inf, T, U + 1)
    α[1, 1] = 0.0
    for d in 3:(T + U + 1), u in 1:(U + 1)
        t = d - u
        1 <= t <= T || continue
        (t == 1 && u == 1) && continue
        a = t > 1 ? α[t - 1, u] + lp[blank, t - 1, u] : -Inf
        b = u > 1 ? α[t, u - 1] + lp[target[u - 1], t, u - 1] : -Inf
        α[t, u] = logaddexp(a, b)
    end
    β = fill(-Inf, T, U + 1)
    β[T, U + 1] = lp[blank, T, U + 1]
    for d in (T + U):-1:2, u in 1:(U + 1)
        t = d - u
        1 <= t <= T || continue
        (t == T && u == U + 1) && continue
        a = t < T ? lp[blank, t, u] + β[t + 1, u] : -Inf
        b = u < U + 1 ? lp[target[u], t, u] + β[t, u + 1] : -Inf
        β[t, u] = logaddexp(a, b)
    end
    nll = -β[1, 1]
    posts = Tuple{Int,Int,Int,Float64}[]
    for u in 1:U, t in 1:T
        em_l = lp[target[u], t, u]
        post = exp(α[t, u] + em_l + β[t, u + 1] + nll)
        push!(posts, (t, u, target[u], post))
    end
    posts
end

"Label-emission posteriors for TDT (token index k, duration class i at (t,u))."
function ref_tdt_label_posts(tok::Array{Float64,3}, dur::Array{Float64,3},
                             target::Vector{Int}, blank::Int,
                             durations::Vector{Int}, sigma::Float64)
    V, T, _ = size(tok)
    U = length(target)
    lp = logsoftmax_ref(tok; dims = 1)
    ld = logsoftmax_ref(dur; dims = 1)
    α = fill(-Inf, T, U + 1)
    α[1, 1] = 0.0
    for d in 3:(T + U + 1), u in 1:(U + 1)
        t = d - u
        1 <= t <= T || continue
        (t == 1 && u == 1) && continue
        v = -Inf
        for (i, dd) in enumerate(durations)
            tp = t - dd
            tp < 1 && break
            if dd >= 1
                v = logaddexp(v, α[tp, u] + lp[blank, tp, u] + ld[i, tp, u] - sigma)
            end
            if u > 1
                v = logaddexp(v, α[tp, u - 1] + lp[target[u - 1], tp, u - 1] +
                                 ld[i, tp, u - 1] - sigma)
            end
        end
        α[t, u] = v
    end
    β = fill(-Inf, T, U + 1)
    for d in (T + U + 1):-1:2, u in 1:(U + 1)
        t = d - u
        1 <= t <= T || continue
        v = -Inf
        for (i, dd) in enumerate(durations)
            tn = t + dd
            if dd >= 1
                if tn <= T
                    v = logaddexp(v, lp[blank, t, u] + ld[i, t, u] - sigma + β[tn, u])
                elseif tn == T + 1 && u == U + 1
                    v = logaddexp(v, lp[blank, t, u] + ld[i, t, u] - sigma)
                end
            end
            if u <= U && tn <= T
                v = logaddexp(v, lp[target[u], t, u] + ld[i, t, u] - sigma +
                                 β[tn, u + 1])
            end
        end
        β[t, u] = v
    end
    nll = -β[1, 1]
    posts = Tuple{Int,Int,Int,Int,Float64}[]
    for u in 1:U, t in 1:T
        for (i, dd) in enumerate(durations)
            tn = t + dd
            tn <= T || continue
            em_l = lp[target[u], t, u]
            post = exp(α[t, u] + em_l + ld[i, t, u] - sigma + β[tn, u + 1] + nll)
            push!(posts, (t, u, target[u], i, post))
        end
    end
    posts
end

"HAT reference NLL: per-cell Bernoulli blank + blank-free label softmax."
function ref_hat_nll(blogit::Array{Float64,2}, ylogit::Array{Float64,3},
                     target::Vector{Int})
    T, U1 = size(blogit)
    U = length(target)
    lp = logsoftmax_ref(ylogit; dims = 1)
    em_b = -log1p.(exp.(-blogit))
    em_l = fill(-Inf, T, U1)
    for u in 1:U
        em_l[:, u] .= .-log1p.(exp.(blogit[:, u])) .+ lp[target[u], :, u]
    end
    α = fill(-Inf, T, U + 1)
    α[1, 1] = 0.0
    for d in 3:(T + U + 1), u in 1:(U + 1)
        t = d - u
        1 <= t <= T || continue
        a = t > 1 ? α[t - 1, u] + em_b[t - 1, u] : -Inf
        c = u > 1 ? α[t, u - 1] + em_l[t, u - 1] : -Inf
        α[t, u] = logaddexp(a, c)
    end
    -(α[T, U + 1] + em_b[T, U + 1])
end

hat_emissions(blogit::Array{Float64,2}, ylogit::Array{Float64,3},
              target::Vector{Int}) = begin
    T, U1 = size(blogit)
    U = length(target)
    lp = logsoftmax_ref(ylogit; dims = 1)
    em_b = -log1p.(exp.(-blogit))
    em_l = fill(-Inf, T, U1)
    for u in 1:U
        em_l[:, u] .= .-log1p.(exp.(blogit[:, u])) .+ lp[target[u], :, u]
    end
    em_b, em_l
end

"Brute force HAT: enumerate every monotonic lattice path."
function brute_hat_nll(blogit::Array{Float64,2}, ylogit::Array{Float64,3},
                       target::Vector{Int})
    T, _ = size(blogit)
    U = length(target)
    em_b, em_l = hat_emissions(blogit, ylogit, target)
    total = Ref(-Inf)
    function walk(t, u, acc)
        if t == T && u == U + 1
            total[] = logaddexp(total[], acc + em_b[t, u])
            return
        end
        t < T && walk(t + 1, u, acc + em_b[t, u])
        u <= U && walk(t, u + 1, acc + em_l[t, u])
    end
    walk(1, 1, 0.0)
    -total[]
end

band_slot(off_t::Int, u::Int, S::Int) =
    (s = u - off_t; 1 <= s <= S ? s : 0)

"Brute force pruned RNN-T: enumerate paths confined to the band."
function brute_pruned_rnnt_nll(logits::Array{Float64,3},
                               off::AbstractVector{<:Integer},
                               target::Vector{Int}, blank::Int)
    V, T, S = size(logits)
    U = length(target)
    lp = logsoftmax_ref(logits; dims = 1)
    band_slot(Int(off[1]), 1, S) > 0 || return Inf
    total = Ref(-Inf)
    function walk(t, u, acc)
        s = band_slot(Int(off[t]), u, S)
        s == 0 && return
        if t == T && u == U + 1
            total[] = logaddexp(total[], acc + lp[blank, t, s])
            return
        end
        t < T && band_slot(Int(off[t + 1]), u, S) > 0 &&
            walk(t + 1, u, acc + lp[blank, t, s])
        u <= U && band_slot(Int(off[t]), u + 1, S) > 0 &&
            walk(t, u + 1, acc + lp[target[u], t, s])
    end
    walk(1, 1, 0.0)
    -total[]
end

"Brute force minimum-latency RNN-T objective: NLL + λ·E[Σᵤ tᵤ]/U."
function brute_rnnt_latency_nll(logits::Array{Float64,3}, target::Vector{Int},
                                blank::Int, λ::Real)
    V, T, _ = size(logits)
    U = length(target)
    lp = logsoftmax_ref(logits; dims = 1)
    paths = Tuple{Float64, Float64}[]
    function walk(t, u, acc, frames)
        if t == T && u == U + 1
            push!(paths, (acc + lp[blank, t, u], frames))
            return
        end
        t < T && walk(t + 1, u, acc + lp[blank, t, u], frames)
        u <= U && walk(t, u + 1, acc + lp[target[u], t, u], frames + t)
    end
    walk(1, 1, 0.0, 0.0)
    isempty(paths) && return Inf
    m = maximum(first.(paths))
    Z = log(sum(exp(p[1] - m) for p in paths)) + m
    nll = -Z
    U == 0 && return nll
    E = sum(exp(p[1] - Z) * p[2] for p in paths) / U
    nll + λ * E
end

"Brute force monotonic RNN-T via TDT durations = [1]."
function brute_monotonic_rnnt_nll(tok::Array{Float64,3}, target::Vector{Int},
                                  blank::Int)
  dur = zeros(Float64, 1, size(tok, 2), size(tok, 3))
  brute_tdt_nll(tok, dur, target, blank, [1], 0.0)
end

pruned_single(logits, off, target, blank) =
    pruned_rnnt_loss(logits, off, target; blank)

"""Brute-force banded TDT NLL: enumerate every alignment that stays in band.

Mirrors `brute_tdt_nll` with a `band_slot` guard on every visited cell — the
independent oracle for `pruned_tdt_loss`. Alignments leaving the band
are simply not enumerated, which is exactly what the banded lattice does.
"""
function brute_pruned_tdt_nll(tok::Array{Float64,3}, dur::Array{Float64,3},
                              off::AbstractVector{<:Integer},
                              target::Vector{Int}, blank::Int,
                              durations::Vector{Int}, sigma::Float64)
    V, T, S = size(tok)
    U = length(target)
    lp = logsoftmax_ref(tok; dims = 1)
    ld = logsoftmax_ref(dur; dims = 1)
    band_slot(Int(off[1]), 1, S) > 0 || return Inf
    total = Ref(-Inf)
    function walk(t, u, acc)
        s = band_slot(Int(off[t]), u, S)
        s == 0 && return
        for (i, dd) in enumerate(durations)
            if dd >= 1
                score = acc + lp[blank, t, s] + ld[i, t, s] - sigma
                if t + dd <= T
                    band_slot(Int(off[t + dd]), u, S) > 0 &&
                        walk(t + dd, u, score)
                elseif t + dd == T + 1 && u == U + 1
                    total[] = logaddexp(total[], score)
                end
            end
            if u <= U && t + dd <= T &&
               band_slot(Int(off[t + dd]), u + 1, S) > 0
                walk(t + dd, u + 1,
                     acc + lp[target[u], t, s] + ld[i, t, s] - sigma)
            end
        end
    end
    walk(1, 1, 0.0)
    -total[]
end

pruned_tdt_single(tok, dur, off, target, blank, durations; sigma = 0.0) =
    pruned_tdt_loss(tok, dur, off, target, durations; blank, sigma)
