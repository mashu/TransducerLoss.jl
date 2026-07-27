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

@testset "pruned TDT" begin
    durs = [0, 1, 2]

    @testset "full-width band equals full TDT (loss and both gradients)" begin
        V, T = 4, 5
        targets = [[1, 2], [3]]
        lens = [T, T - 1]
        U1 = maximum(length, targets) + 1
        tok = randn(rng, V, T, U1, 2)
        dur = randn(rng, length(durs), T, U1, 2)
        off = zeros(Int, T, 2)
        for σ in (0.0, 0.05), λ in (0.0, 0.1)
            @test pruned_tdt_loss_batched(tok, dur, off, targets, lens, durs;
                                          sigma = σ, fastemit_lambda = λ) ≈
                  tdt_loss_batched(tok, dur, targets, lens, durs;
                                   sigma = σ, fastemit_lambda = λ) atol = 1e-10
            gp = Zygote.gradient((a, b) -> pruned_tdt_loss_batched(
                a, b, off, targets, lens, durs; sigma = σ, fastemit_lambda = λ),
                tok, dur)
            gf = Zygote.gradient((a, b) -> tdt_loss_batched(
                a, b, targets, lens, durs; sigma = σ, fastemit_lambda = λ),
                tok, dur)
            @test gp[1] ≈ gf[1] atol = 1e-10
            @test gp[2] ≈ gf[2] atol = 1e-10
        end
    end

    @testset "shifted band: loss vs brute force" begin
        for (T, target, offv) in ((5, [1, 2], [0, 0, 1, 1, 1]),
                                  (4, [2, 1], [0, 0, 1, 1]),
                                  (3, Int[], [0, 0, 0]))
            S, blank = 2, 4
            tok = randn(rng, blank, T, S) .* 2
            dur = randn(rng, length(durs), T, S)
            for σ in (0.0, 0.05)
                @test pruned_tdt_single(tok, dur, offv, target, blank, durs;
                                        sigma = σ) ≈
                      brute_pruned_tdt_nll(tok, dur, offv, target, blank,
                                           durs, σ) atol = 1e-8
            end
        end
    end

    @testset "shifted band: gradients vs finite differences" begin
        V, T, S, target, blank = 4, 5, 2, [1, 2], 4
        offv = [0, 0, 1, 1, 1]
        tok = randn(rng, V, T, S)
        dur = randn(rng, length(durs), T, S)
        gt, gd = Zygote.gradient(
            (a, b) -> pruned_tdt_single(a, b, offv, target, blank, durs),
            tok, dur)
        ε = 1e-6
        for idx in (CartesianIndex(1, 1, 1), CartesianIndex(blank, 3, 2),
                    CartesianIndex(2, 5, 1))
            p = copy(tok); p[idx] += ε
            m = copy(tok); m[idx] -= ε
            fd = (pruned_tdt_single(p, dur, offv, target, blank, durs) -
                  pruned_tdt_single(m, dur, offv, target, blank, durs)) / 2ε
            @test gt[idx] ≈ fd atol = 1e-5
        end
        for idx in (CartesianIndex(1, 2, 1), CartesianIndex(3, 4, 2))
            p = copy(dur); p[idx] += ε
            m = copy(dur); m[idx] -= ε
            fd = (pruned_tdt_single(tok, p, offv, target, blank, durs) -
                  pruned_tdt_single(tok, m, offv, target, blank, durs)) / 2ε
            @test gd[idx] ≈ fd atol = 1e-5
        end
    end

    @testset "durations = [1] reproduces banded monotonic RNN-T" begin
        V, T, S, target, blank = 4, 5, 2, [1, 2], 4
        offv = [0, 0, 1, 1, 1]
        tok = randn(rng, V, T, S)
        dur = zeros(length([1]), T, S)
        @test pruned_tdt_single(tok, dur, offv, target, blank, [1]) ≈
              brute_pruned_tdt_nll(tok, dur, offv, target, blank, [1], 0.0) atol = 1e-8
    end

    @testset "banding never lowers the loss" begin
        # Dropping alignments can only remove probability mass — compare the
        # full joint against the same scores restricted to band cells.
        V, T, U1, blank = 4, 6, 3, 4
        target = [1, 2]
        tok = randn(rng, V, T, U1)
        dur = randn(rng, length(durs), T, U1)
        full = tdt_single(tok, dur, target, blank, durs)
        offv, S = [0, 0, 0, 1, 1, 1], 2
        tok_b = Array{Float64}(undef, V, T, S)
        dur_b = Array{Float64}(undef, length(durs), T, S)
        for t in 1:T, s in 1:S
            tok_b[:, t, s] .= tok[:, t, offv[t] + s]
            dur_b[:, t, s] .= dur[:, t, offv[t] + s]
        end
        narrow = pruned_tdt_single(tok_b, dur_b, offv, target, blank, durs)
        @test narrow >= full - 1e-9
    end

    @testset "tdt_pruning_bounds yields a feasible band" begin
        V, T, B = 5, 8, 2
        targets = [[1, 2, 3], [2]]
        lens = [T, T - 2]
        U1 = maximum(length, targets) + 1
        D = length(durs)
        am = randn(rng, V, T, B)
        lm = randn(rng, V, U1, B)
        am_d = randn(rng, D, T, B)
        lm_d = randn(rng, D, U1, B)
        for w in (2, 3, U1)
            off = tdt_pruning_bounds(am, lm, am_d, lm_d, targets, lens;
                                     band_width = w, durations = durs)
            @test size(off) == (T, B)
            for b in 1:B
                Tb, Ub = lens[b], length(targets[b])
                @test off[1, b] == 0
                @test issorted(view(off, 1:Tb, b))
                @test all(off[t, b] - off[t - 1, b] <= w - 1 for t in 2:Tb)
                @test off[Tb, b] + w >= Ub + 1        # exit reachable
            end
        end
        # A band this wide must reproduce the full TDT exactly.
        off = tdt_pruning_bounds(am, lm, am_d, lm_d, targets, lens;
                                 band_width = U1, durations = durs)
        @test all(iszero, off)
    end
end
