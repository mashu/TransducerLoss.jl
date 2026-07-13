@testset "pack_transducer_targets" begin
    labels, lens = pack_transducer_targets([[1, 2], [3], Int[]], 4)
    @test labels == [1 3 0; 2 0 0]
    @test lens == Int32[2, 1, 0]

    @test_throws ArgumentError pack_transducer_targets([[4, 1]], 4)
    @test_throws ArgumentError pack_transducer_targets([[0]], 4)
end
