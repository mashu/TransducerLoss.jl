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
        gp = Zygote.gradient(
            l -> pruned_rnnt_loss_batched(l, off, targets, lens), logits)[1]
        gf = Zygote.gradient(
            l -> rnnt_loss_batched(l, targets, lens), logits)[1]
        @test gp ≈ gf atol = 1e-10
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
        off = pruning_bounds(am, lm, [[1, 2, 3]], [6]; band_width = 2)
        @test size(off) == (6, 1)
        @test off[1, 1] == 0 && issorted(off[:, 1])
        @test all(diff(off[:, 1]) .<= 1)                   # slope ≤ w − 1
        @test off[6, 1] >= 3 + 1 - 2                       # exit reachable
        joint = randn(rng, 4, 6, 2, 1)
        @test isfinite(pruned_rnnt_loss_batched(joint, off, [[1, 2, 3]], [6]))
    end
end
