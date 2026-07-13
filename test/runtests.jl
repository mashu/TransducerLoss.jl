using Test
using Random
using Zygote
using TransducerLoss
using TransducerLoss: logaddexp

const rng = MersenneTwister(0)

include("reference.jl")

@testset "TransducerLoss" begin
    include("test_labels.jl")
    include("test_rnnt.jl")
    include("test_tdt.jl")
    include("test_hat.jl")
    include("test_pruned.jl")
end
