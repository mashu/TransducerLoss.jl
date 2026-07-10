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
        _, grad = rnnt_forward_backward(l4, labels, tlens, [T], blank)
        ε = 1e-6
        for i in eachindex(l4)
            lp = copy(l4); lp[i] += ε
            lm = copy(l4); lm[i] -= ε
            fd = (rnnt_loss_batched(lp, [target], [T], blank) -
                  rnnt_loss_batched(lm, [target], [T], blank)) / 2ε
            @test grad[i] ≈ fd atol = 1e-5
        end
    end
    @testset "Zygote integration (rrule)" begin
        logits = randn(rng, Float32, 4, 5, 3, 2)
        targets = [[1, 2], [3]]
        lens = [5, 4]
        g = Zygote.gradient(l -> rnnt_loss_batched(l, targets, lens), logits)[1]
        labels, tlens = pack_transducer_targets(targets, 4)
        _, grad = rnnt_forward_backward(logits, labels, tlens, lens, 4)
        @test g ≈ grad
        # keyword-blank form differentiates too
        g2 = Zygote.gradient(
            l -> rnnt_loss_batched(l, targets, lens; blank = 4), logits)[1]
        @test g2 ≈ grad
    end
    @testset "heterogeneous batch = mean of singles" begin
        V, blank = 4, 4
        targets = [[1, 2, 3], [2], Int[]]
        lens = [5, 3, 4]
        U1 = maximum(length, targets) + 1
        logits = randn(rng, V, maximum(lens), U1, 3)
        batched = rnnt_loss_batched(logits, targets, lens, blank)
        singles = [single(Array(logits[:, 1:lens[b], 1:length(targets[b]) + 1, b]),
                          targets[b], blank) for b in 1:3]
        @test batched ≈ sum(singles) / 3 atol = 1e-8
    end
    @testset "argument validation" begin
        logits = randn(rng, 4, 3, 2, 1)
        @test_throws ArgumentError rnnt_loss_batched(logits, [[4]], [3], 4)     # blank in target
        @test_throws ArgumentError rnnt_loss_batched(logits, [[1, 2, 3]], [3], 4)  # U+1 > U1max
        # zero-length input ⇒ no valid path ⇒ Inf loss, finite gradients
        l0 = rnnt_loss_batched(logits, [[1]], [0], 4)
        @test isinf(l0)
        g0 = Zygote.gradient(l -> rnnt_loss_batched(l, [[1]], [0], 4), logits)[1]
        @test all(iszero, g0)
    end
end
