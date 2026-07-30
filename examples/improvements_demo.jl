#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LieControllability
using Plots
using Statistics

gr()

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

function benchmark_dense_eval(; n=181, repeats=4)
    loop_times = Float64[]
    tullio_times = Float64[]
    for _ in 1:repeats
        push!(loop_times, @elapsed slice_field(f8; n=n, z=0.0, backend=:loop))
        push!(tullio_times, @elapsed slice_field_dense(f8; n=n, z=0.0, backend=:tullio))
    end
    return median(loop_times), median(tullio_times)
end

function run_scenario(name::Symbol)
    scenario = dbs_scenario(name)
    cs = control_system(scenario)
    x0 = zeros(cs.n_elements)
    x0[scenario.control_node] = 0.5
    sim = simulate_trajectory(cs, x0; tspan=scenario.tspan, dt=scenario.dt, method=:rk4)
    y = [cs.h(sim.x[:, k], cs.P) for k in eachindex(sim.t)]
    return sim.t, y
end

loop_time, tullio_time = benchmark_dense_eval()
speedup = loop_time / max(tullio_time, eps())

scenarios = [:baseline, :early_drive, :high_gain]
traj = Dict{Symbol, Tuple{Vector{Float64}, Vector{Float64}}}()
for s in scenarios
    traj[s] = run_scenario(s)
end

bench_plot = bar(
    ["loop", "tullio"],
    [loop_time, tullio_time],
    color=[:slategray, :darkorange],
    legend=false,
    ylabel="seconds (median)",
    title="Dense slice evaluation (n = 181)\nSpeedup = $(round(speedup, digits=2))x",
)

traj_plot = plot(
    title="Parameterized DBS scenarios: readout trajectory h(x)",
    xlabel="time",
    ylabel="h_single(x)",
    legend=:bottomright,
)
for s in scenarios
    ts, ys = traj[s]
    plot!(traj_plot, ts, ys, label=String(s), linewidth=2)
end

final_plot = plot(bench_plot, traj_plot; layout=(2, 1), size=(900, 900))
outpath = joinpath(figdir, "improvements_dense_and_scenarios.png")
savefig(final_plot, outpath)

println("Dense-eval benchmark median loop:   ", round(loop_time, digits=4), " s")
println("Dense-eval benchmark median tullio: ", round(tullio_time, digits=4), " s")
println("Speedup: ", round(speedup, digits=2), "x")
println("Saved -> ", outpath)

