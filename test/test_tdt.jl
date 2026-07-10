@testset "TDT" begin
    @testset "TDT: loss vs brute force" begin
        for (V, T, target, durs, sigma) in (
                (4, 5, [1, 2], [0, 1, 2], 0.0),
                (4, 5, [1, 2], [1, 2], 0.0),          # no zero duration
                (4, 6, [3, 1], [0, 1, 2, 3, 4], 0.0),
                (4, 4, Int[], [0, 1, 2], 0.0),
                (4, 1, [2], [0, 1], 0.0),
                (4, 5, [1, 2, 3], [0, 1, 2], 0.05))   # under-normalization
            blank = V
            for _ in 1:3
                tok = randn(rng, V, T, length(target) + 1) .* 2
                dur = randn(rng, length(durs), T, length(target) + 1) .* 2
                l = tdt_single(tok, dur, target, blank, durs; sigma)
                @test l ≈ brute_tdt_nll(tok, dur, target, blank, durs, sigma) atol = 1e-8
            end
        end
    end
    @testset "TDT: gradients vs finite differences" begin
        V, T, target, durs = 3, 4, [1, 2], [0, 1, 2]
        blank, sigma = 3, 0.05
        tok = randn(rng, V, T, 3)
        dur = randn(rng, length(durs), T, 3)
        t4 = reshape(tok, size(tok)..., 1)
        d4 = reshape(dur, size(dur)..., 1)
        labels, tlens = pack_transducer_targets([target], blank)
        _, gtok, gdur = tdt_forward_backward(t4, d4, labels, tlens, [T],
                                             durs, blank, sigma)
        ε = 1e-6
        for i in eachindex(t4)
            p = copy(t4); p[i] += ε
            m = copy(t4); m[i] -= ε
            fd = (tdt_loss_batched(p, d4, [target], [T], durs; blank, sigma) -
                  tdt_loss_batched(m, d4, [target], [T], durs; blank, sigma)) / 2ε
            @test gtok[i] ≈ fd atol = 1e-5
        end
        for i in eachindex(d4)
            p = copy(d4); p[i] += ε
            m = copy(d4); m[i] -= ε
            fd = (tdt_loss_batched(t4, p, [target], [T], durs; blank, sigma) -
                  tdt_loss_batched(t4, m, [target], [T], durs; blank, sigma)) / 2ε
            @test gdur[i] ≈ fd atol = 1e-5
        end
    end
    @testset "TDT: Zygote + batching + validation" begin
        durs = [0, 1, 2]
        tok = randn(rng, Float32, 4, 5, 3, 2)
        dur = randn(rng, Float32, 3, 5, 3, 2)
        targets = [[1, 2], [3]]
        lens = [5, 4]
        gt, gd = Zygote.gradient(
            (a, b) -> tdt_loss_batched(a, b, targets, lens, durs;
                                       blank = 4, sigma = 0.05),
            tok, dur)
        labels, tlens = pack_transducer_targets(targets, 4)
        _, rt, rd = tdt_forward_backward(tok, dur, labels, tlens, lens,
                                         durs, 4, 0.05)
        @test gt ≈ rt
        @test gd ≈ rd

        # heterogeneous batch equals mean of singles
        singles = [tdt_single(Float64.(tok[:, 1:lens[b], 1:length(targets[b]) + 1, b]),
                              Float64.(dur[:, 1:lens[b], 1:length(targets[b]) + 1, b]),
                              targets[b], 4, durs; sigma = 0.05) for b in 1:2]
        batched = tdt_loss_batched(tok, dur, targets, lens, durs;
                                   blank = 4, sigma = 0.05)
        @test batched ≈ sum(singles) / 2 atol = 1e-4   # Float32 tensors

        @test_throws ArgumentError tdt_loss_batched(tok, dur, targets, lens,
                                                    [2, 1]; blank = 4)      # unsorted
        @test_throws ArgumentError tdt_loss_batched(tok, dur, targets, lens,
                                                    [0]; blank = 4)          # no d ≥ 1
        @test_throws ArgumentError tdt_loss_batched(tok, dur[:, 1:3, :, :],
                                                    targets, lens, durs;
                                                    blank = 4)               # (T,U+1,B) mismatch
    end
end
