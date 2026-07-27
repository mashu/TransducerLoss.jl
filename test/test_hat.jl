@testset "HAT" begin
    @testset "loss vs brute force & reference" begin
        for (T, V, target) in ((4, 3, [1, 2]), (3, 4, Int[]), (5, 3, [2, 1, 2]),
                                 (4, 4, [1, 3]), (2, 3, [2]))
            for _ in 1:2
                blog = randn(rng, T, length(target) + 1) .* 2
                ylog = randn(rng, V, T, length(target) + 1) .* 2
                l = hat_loss(reshape(ylog, size(ylog)..., 1),
                                     reshape(blog, size(blog)..., 1),
                                     [target], [T])
                @test l ≈ brute_hat_nll(blog, ylog, target) atol = 1e-8
                @test l ≈ ref_hat_nll(blog, ylog, target) atol = 1e-8
            end
        end
    end
    @testset "gradients vs finite differences + Zygote" begin
        T, V, target = 4, 3, [1, 2]
        blog = randn(rng, T, 3, 1)
        ylog = randn(rng, V, T, 3, 1)
        labels, tlens = pack_transducer_targets([target], 0)
        _, gy, gb = hat_forward_backward(ylog, blog, labels, tlens, [T])
        ε = 1e-6
        for i in eachindex(ylog)
            p = copy(ylog); p[i] += ε
            m = copy(ylog); m[i] -= ε
            fdv = (hat_loss(p, blog, [target], [T]) -
                   hat_loss(m, blog, [target], [T])) / 2ε
            @test gy[i] ≈ fdv atol = 1e-5
        end
        for i in eachindex(blog)
            p = copy(blog); p[i] += ε
            m = copy(blog); m[i] -= ε
            fdv = (hat_loss(ylog, p, [target], [T]) -
                   hat_loss(ylog, m, [target], [T])) / 2ε
            @test gb[i] ≈ fdv atol = 1e-5
        end
        g1, g2 = Zygote.gradient(
            (a, b) -> hat_loss(a, b, [target], [T]), ylog, blog)
        @test g1 ≈ gy
        @test g2 ≈ gb
    end
    @testset "heterogeneous batch = mean of singles" begin
        targets = [[1, 2], [3], Int[]]
        lens = [4, 3, 5]
        U1 = maximum(length, targets) + 1
        V = 3
        ylog = randn(rng, V, maximum(lens), U1, 3)
        blog = randn(rng, maximum(lens), U1, 3)
        batched = hat_loss(ylog, blog, targets, lens)
        singles = [hat_loss(
            ylog[:, 1:lens[b], 1:length(targets[b]) + 1, b:b],
            blog[1:lens[b], 1:length(targets[b]) + 1, b:b],
            [targets[b]], [lens[b]]) for b in 1:3]
        @test batched ≈ sum(singles) / 3 atol = 1e-8
    end
    @testset "argument validation" begin
        ylog = randn(rng, 3, 4, 3, 1)
        blog = randn(rng, 4, 3, 1)
        @test_throws ArgumentError hat_loss(ylog, blog[:, 1:2, :],
                                                    [[1, 2]], [4])
        @test_throws ArgumentError pack_transducer_targets([[0]], 0)
    end
end
