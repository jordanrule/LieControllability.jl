using Test
using LieControllability
using ForwardDiff
using Random
using LinearAlgebra

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

    xs1, ys1, U1, V1, W1 = slice_field(f8; n=25, backend=:loop)
    xs2, ys2, U2, V2, W2 = slice_field_dense(f8; n=25, backend=:tullio)
    @test xs1 == xs2
    @test ys1 == ys2
    @test isapprox(U1, U2; atol=1e-10)
    @test isapprox(V1, V2; atol=1e-10)
    @test isapprox(W1, W2; atol=1e-10)

    s = dbs_scenario(:high_gain; n_elements=10, seed=2)
    cs_s = control_system(s)
    sim = simulate_trajectory(cs_s, zeros(cs_s.n_elements); tspan=(0.0, 2.0), dt=0.05, method=:rk4)
    @test length(sim.t) == size(sim.x, 2)
    @test size(sim.x, 1) == cs_s.n_elements
    @test length(sim.u) == length(sim.t)

    f0(x, args...) = zeros(3)
    gcar_1(x, args...) = [1.0, 0.0, -x[2]]
    gcar_2(x, args...) = [0.0, 1.0, x[1]]
    M = controllability_distribution(f0, [gcar_1, gcar_2], [0.1, -0.3, 0.5]; max_depth=2)
    @test size(M, 1) == 3
    @test rank(M; atol=1e-8) == 3

    chow_ok = chow_rank_test(
        f0,
        [gcar_1, gcar_2];
        n_state=3,
        n_samples=12,
        sample_bounds=(-1.0, 1.0),
        max_depth=2,
        include_drift=false,
        rank_tol=1e-8,
        rng=MersenneTwister(7),
    )
    @test chow_ok.is_controllable
    @test chow_ok.min_rank == 3
    @test chow_ok.full_rank_fraction == 1.0

    gline(x, args...) = [1.0, 0.0, 0.0]
    chow_bad = chow_rank_test(
        f0,
        [gline];
        n_state=3,
        n_samples=12,
        sample_bounds=(-1.0, 1.0),
        max_depth=2,
        include_drift=false,
        rank_tol=1e-8,
        rng=MersenneTwister(7),
    )
    @test !chow_bad.is_controllable
    @test chow_bad.min_rank == 1

    chow_seed_1 = chow_rank_test(f0, [gcar_1, gcar_2]; n_state=3, n_samples=6, rng=MersenneTwister(11))
    chow_seed_2 = chow_rank_test(f0, [gcar_1, gcar_2]; n_state=3, n_samples=6, rng=MersenneTwister(11))
    @test chow_seed_1.ranks == chow_seed_2.ranks
    @test chow_seed_1.sample_points == chow_seed_2.sample_points
end


