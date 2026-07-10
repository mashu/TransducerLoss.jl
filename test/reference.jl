# Shared test oracles: pure-Julia reference implementations and brute-force
# lattice enumeration for both losses. Independent of the kernel code.

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
    rnnt_loss_batched(reshape(logits, size(logits)..., 1), [target],
                      [size(logits, 2)], blank)

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
    tdt_loss_batched(reshape(tok, size(tok)..., 1),
                     reshape(dur, size(dur)..., 1), [target],
                     [size(tok, 2)], durations; blank, sigma)

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
