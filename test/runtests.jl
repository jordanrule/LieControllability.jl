using Test
using LieControllability
using ForwardDiff
using Random

@testset "LieControllability core" begin
    f_id(x, args...) = x
    h_sq(x, args...) = sum(x .^ 2)

    ld = L_d(h_sq, f_id)
    @test isapprox(ld([1.0, 2.0, 3.0]), 28.0; atol=1e-8)

    f(x, args...) = [x[2], 0.0, 0.0]
    g(x, args...) = [0.0, x[1], 0.0]
    cf, cb = L_bracket(f, g)
    @test isapprox(cf([1.0, 2.0, 3.0]) .- cb([1.0, 2.0, 3.0]), [1.0, -2.0, 0.0]; atol=1e-8)

    @test h1([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0] .* [0.0, 0.0, 1.0]
    @test Xi_1(collect(1.0:10.0), nothing) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]

    cs = control_system()
    @test size(cs.G) == (10, 10)
    @test size(cs.L) == (10, 10)
    @test disease_control(cs; rng=MersenneTwister(1)) == true
    @test disease_measure(cs; rng=MersenneTwister(1)) == true
    @test full_control(cs; rng=MersenneTwister(1)) == true
end


