@testset "RNN-T" begin
    @testset "loss vs brute force & reference" begin
        for (V, T, target) in ((4, 4, [1, 2]), (4, 5, [2, 2, 2]),
                               (4, 3, Int[]), (4, 1, [1, 3]),
                               (3, 3, [1, 2, 1]), (5, 5, [1, 4, 2, 4]))
            blank = V
            for _ in 1:3
                logits = randn(rng, V, T, length(target) + 1) .* 2
                l = single(logits, target, blank)
                @test l ≈ brute_rnnt_nll(logits, target, blank) atol = 1e-8
                @test l ≈ ref_rnnt_nll(logits, target, blank) atol = 1e-8
            end
        end
    end
    @testset "padded prediction axis (U1 > U + 1)" begin
        logits = randn(rng, 4, 4, 5)                    # U = 1, two pad slots
        l = single(logits, [2], 4)
        @test l ≈ brute_rnnt_nll(logits[:, :, 1:2], [2], 4) atol = 1e-8
    end
    @testset "gradient vs finite differences" begin
        V, T, target, blank = 3, 3, [1, 2], 3
        logits = randn(rng, V, T, length(target) + 1)
        labels, tlens = pack_transducer_targets([target], blank)
        l4 = reshape(logits, size(logits)..., 1)
        _, grad = rnnt_forward_backward(l4, labels, tlens, [T], blank, 0)
        ε = 1e-6
        for i in eachindex(l4)
            lp = copy(l4); lp[i] += ε
            lm = copy(l4); lm[i] -= ε
            fd = (rnnt_loss(lp, [target], [T], blank) -
                  rnnt_loss(lm, [target], [T], blank)) / 2ε
            @test grad[i] ≈ fd atol = 1e-5
        end
    end
    @testset "Zygote integration (rrule)" begin
        logits = randn(rng, Float32, 4, 5, 3, 2)
        targets = [[1, 2], [3]]
        lens = [5, 4]
        g = Zygote.gradient(l -> rnnt_loss(l, targets, lens), logits)[1]
        labels, tlens = pack_transducer_targets(targets, 4)
        _, grad = rnnt_forward_backward(logits, labels, tlens, lens, 4, 0)
        @test g ≈ grad
        # keyword-blank form differentiates too
        g2 = Zygote.gradient(
            l -> rnnt_loss(l, targets, lens; blank = 4), logits)[1]
        @test g2 ≈ grad
    end
    @testset "heterogeneous batch = mean of singles" begin
        V, blank = 4, 4
        targets = [[1, 2, 3], [2], Int[]]
        lens = [5, 3, 4]
        U1 = maximum(length, targets) + 1
        logits = randn(rng, V, maximum(lens), U1, 3)
        batched = rnnt_loss(logits, targets, lens, blank)
        singles = [single(Array(logits[:, 1:lens[b], 1:length(targets[b]) + 1, b]),
                          targets[b], blank) for b in 1:3]
        @test batched ≈ sum(singles) / 3 atol = 1e-8
    end
    @testset "argument validation" begin
        logits = randn(rng, 4, 3, 2, 1)
        @test_throws ArgumentError rnnt_loss(logits, [[4]], [3], 4)     # blank in target
        @test_throws ArgumentError rnnt_loss(logits, [[1, 2, 3]], [3], 4)  # U+1 > U1max
        # zero-length input ⇒ no valid path ⇒ Inf loss, finite gradients
        l0 = rnnt_loss(logits, [[1]], [0], 4)
        @test isinf(l0)
        g0 = Zygote.gradient(l -> rnnt_loss(l, [[1]], [0], 4), logits)[1]
        @test all(iszero, g0)
    end
    @testset "FastEmit: loss unchanged, gradient scaling" begin
        V, T, target, blank = 4, 4, [1, 2], 4
        logits = randn(rng, V, T, length(target) + 1, 1)
        labels, tlens = pack_transducer_targets([target], blank)
        λ = 0.25
        l0, g0 = rnnt_forward_backward(logits, labels, tlens, [T], blank, 0)
        lλ, gλ = rnnt_forward_backward(logits, labels, tlens, [T], blank, λ)
        @test l0 ≈ lλ
        @test l0 ≈ brute_rnnt_nll(logits[:, :, :, 1], target, blank) atol = 1e-8
        posts = ref_rnnt_label_posts(logits[:, :, :, 1], target, blank)
        for (t, u, k, post) in posts
        end
        # non-label cells are identical
        mask = falses(size(logits))
        for (t, u, k, _) in posts
            mask[k, t, u, 1] = true
        end
        @test gλ[.!mask] ≈ g0[.!mask]
        # λ = 0 matches public API default
        @test rnnt_loss(logits, [target], [T]; fastemit_lambda = λ) ≈ l0
        @test rnnt_loss(logits, [target], [T]; fastemit_lambda = λ) ≈
              brute_rnnt_nll(logits[:, :, :, 1], target, blank) atol = 1e-8
        g_z = Zygote.gradient(
            l -> rnnt_loss(l, [target], [T]; fastemit_lambda = λ),
            logits)[1]
        @test g_z ≈ gλ
    end
    @testset "minimum-latency training" begin
        V, T, target, blank = 4, 5, [1, 2], 4
        logits = randn(rng, V, T, 3, 1)
        λ = 0.3
        @test rnnt_loss(logits, [target], [T]; latency_lambda = 0) ==
              rnnt_loss(logits, [target], [T])
        # loss = nll + λ·E/U, E from the label-posterior reference oracle
        posts = ref_rnnt_label_posts(logits[:, :, :, 1], target, blank)
        E_ref = sum(t * post for (t, u, k, post) in posts)
        nll_ref = ref_rnnt_nll(logits[:, :, :, 1], target, blank)
        l = rnnt_loss(logits, [target], [T]; latency_lambda = λ)
        @test l ≈ brute_rnnt_latency_nll(logits[:, :, :, 1], target, blank, λ) atol = 1e-8
        @test l ≈ nll_ref + λ * E_ref / length(target) atol = 1e-8
        labels, tlens = pack_transducer_targets([target], blank)
        _, grad = rnnt_forward_backward(logits, labels, tlens, [T], blank,
                                        0, λ)
        ε = 1e-6
        for i in eachindex(logits)
            p = copy(logits); p[i] += ε
            m = copy(logits); m[i] -= ε
            fdv = (rnnt_loss(p, [target], [T]; latency_lambda = λ) -
                   rnnt_loss(m, [target], [T]; latency_lambda = λ)) / 2ε
            @test grad[i] ≈ fdv atol = 1e-5
        end
        g = Zygote.gradient(
            l -> rnnt_loss(l, [target], [T]; latency_lambda = λ),
            logits)[1]
        @test g ≈ grad
        # empty target: latency term vanishes exactly
        @test rnnt_loss(logits, [Int[]], [T]; latency_lambda = λ) ==
              rnnt_loss(logits, [Int[]], [T])
        @test_throws ArgumentError rnnt_loss(
            logits, [target], [T]; latency_lambda = -1)
    end
    @testset "positional blank API" begin
        V, T, target, blank = 4, 5, [1, 2], 4
        logits = randn(rng, V, T, 3, 1)
        λ = 0.2
        l_kw = rnnt_loss(logits, [target], [T]; blank, latency_lambda = λ)
        l_pos = rnnt_loss(logits, [target], [T], blank; latency_lambda = λ)
        @test l_pos ≈ l_kw
        g_pos = Zygote.gradient(
            l -> rnnt_loss(l, [target], [T], blank; latency_lambda = λ),
            logits)[1]
        g_kw = Zygote.gradient(
            l -> rnnt_loss(l, [target], [T]; blank, latency_lambda = λ),
            logits)[1]
        @test g_pos ≈ g_kw
        l_fe = rnnt_loss(logits, [target], [T], blank;
                                 fastemit_lambda = 0.1, latency_lambda = λ)
        @test isfinite(l_fe)
    end
end
