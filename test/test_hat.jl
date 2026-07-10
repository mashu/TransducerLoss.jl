@testset "HAT" begin
    @testset "loss vs reference" begin
        for (T, V, target) in ((4, 3, [1, 2]), (3, 4, Int[]), (5, 3, [2, 1, 2]))
            blog = randn(rng, T, length(target) + 1, 1)
            ylog = randn(rng, V, T, length(target) + 1, 1) .* 2
            l = hat_loss_batched(ylog, blog, [target], [T])
            @test l ≈ ref_hat_nll(blog[:, :, 1], ylog[:, :, :, 1], target) atol = 1e-8
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
            fdv = (hat_loss_batched(p, blog, [target], [T]) -
                   hat_loss_batched(m, blog, [target], [T])) / 2ε
            @test gy[i] ≈ fdv atol = 1e-5
        end
        for i in eachindex(blog)
            p = copy(blog); p[i] += ε
            m = copy(blog); m[i] -= ε
            fdv = (hat_loss_batched(ylog, p, [target], [T]) -
                   hat_loss_batched(ylog, m, [target], [T])) / 2ε
            @test gb[i] ≈ fdv atol = 1e-5
        end
        g1, g2 = Zygote.gradient(
            (a, b) -> hat_loss_batched(a, b, [target], [T]), ylog, blog)
        @test g1 ≈ gy
        @test g2 ≈ gb
    end
end
