@testset "pruned / banded lattice" begin
    @testset "full-width band equals vanilla RNN-T" begin
        V, T = 4, 5
        targets = [[1, 2], [3]]
        lens = [T, T - 1]
        U1 = maximum(length, targets) + 1
        logits = randn(rng, V, T, U1, 2)
        off = zeros(Int, T, 2)
        @test pruned_rnnt_loss_batched(logits, off, targets, lens) ≈
              rnnt_loss_batched(logits, targets, lens) atol = 1e-10
        for (target, Tlen) in zip(targets, lens)
            lb = logits[:, 1:Tlen, 1:length(target) + 1, 1]
            off1 = zeros(Int, Tlen)
            @test pruned_single(lb, off1, target, V) ≈
                  single(lb, target, V) atol = 1e-10
            @test pruned_single(lb, off1, target, V) ≈
                  brute_pruned_rnnt_nll(lb, off1, target, V) atol = 1e-8
            @test brute_pruned_rnnt_nll(lb, off1, target, V) ≈
                  brute_rnnt_nll(lb, target, V) atol = 1e-8
        end
        gp = Zygote.gradient(
            l -> pruned_rnnt_loss_batched(l, off, targets, lens), logits)[1]
        gf = Zygote.gradient(
            l -> rnnt_loss_batched(l, targets, lens), logits)[1]
        @test gp ≈ gf atol = 1e-10
    end
    @testset "shifted band: loss vs brute force" begin
        for (T, target, offv) in (
                (5, [1, 2], [0, 0, 1, 1, 1]),
                (4, [2, 1], [0, 0, 1, 1]),
                (3, Int[], [0, 0, 0]))
            S = 2
            blank = 4
            lb = randn(rng, blank, T, S) .* 2
            l = pruned_single(lb, offv, target, blank)
            @test l ≈ brute_pruned_rnnt_nll(lb, offv, target, blank) atol = 1e-8
        end
    end
    @testset "shifted band: gradients vs finite differences" begin
        V, T, S, target, blank = 4, 5, 2, [1, 2], 4
        offm = reshape([0, 0, 1, 1, 1], :, 1)
        lb = randn(rng, V, T, S, 1)
        labels, tlens = pack_transducer_targets([target], blank)
        _, grad = pruned_forward_backward(lb, Int32.(Matrix(offm)), labels,
                                          tlens, [T], blank)
        ε = 1e-6
        for i in eachindex(lb)
            p = copy(lb); p[i] += ε
            m = copy(lb); m[i] -= ε
            fdv = (pruned_rnnt_loss_batched(p, offm, [target], [T]) -
                   pruned_rnnt_loss_batched(m, offm, [target], [T])) / 2ε
            @test grad[i] ≈ fdv atol = 1e-5
        end
        g = Zygote.gradient(
            l -> pruned_rnnt_loss_batched(l, offm, [target], [T]), lb)[1]
        @test g ≈ grad
    end
    @testset "offset validation & pruning_bounds" begin
        lb = randn(rng, 4, 4, 2, 1)
        @test_throws ArgumentError pruned_rnnt_loss_batched(
            lb, reshape([1, 1, 1, 1], :, 1), [[1]], [4])   # start not in band
        @test_throws ArgumentError pruned_rnnt_loss_batched(
            lb, reshape([0, 1, 0, 1], :, 1), [[1]], [4])   # not monotone
        am = randn(rng, 4, 6, 1)
        lm = randn(rng, 4, 4, 1)
        off = pruning_bounds(am, lm, [[1, 2]], [6]; band_width = 4)
        @test size(off) == (6, 1)
        @test off[1, 1] == 0 && issorted(off[:, 1])
        @test all(diff(off[:, 1]) .<= 3)                      # slope ≤ w − 1
        @test off[6, 1] + 4 >= 2 + 1                          # exit reachable
        joint = randn(rng, 4, 6, 4, 1)
        l = pruned_rnnt_loss_batched(joint, off, [[1, 2]], [6])
        @test isfinite(l)
        @test l ≈ brute_pruned_rnnt_nll(joint[:, :, :, 1], Vector(off[:, 1]),
                                        [1, 2], 4) atol = 1e-8
    end
    @testset "heterogeneous batch = mean of singles" begin
        V, blank = 4, 4
        targets = [[1, 2], [3]]
        lens = [5, 4]
        U1 = maximum(length, targets) + 1
        S = U1
        logits = randn(rng, V, maximum(lens), S, 2)
        off = zeros(Int, maximum(lens), 2)
        batched = pruned_rnnt_loss_batched(logits, off, targets, lens; blank)
        singles = [pruned_rnnt_loss_batched(
            logits[:, 1:lens[b], :, b:b], off[1:lens[b], b:b],
            [targets[b]], [lens[b]]; blank) for b in 1:2]
        @test batched ≈ sum(singles) / 2 atol = 1e-8
    end
    @testset "pruning_bounds validation" begin
        am = randn(rng, 4, 4, 1)
        lm = randn(rng, 4, 3, 1)
        @test_throws ArgumentError pruning_bounds(am, lm, [[1, 2, 3]], [4];
                                                  band_width = 1)
        @test_throws ArgumentError pruning_bounds(am, lm, [[1, 2, 3, 4, 5]], [4];
                                                  band_width = 2)
    end
end
