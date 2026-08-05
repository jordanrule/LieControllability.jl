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
    # Warm up compilation for both backends before timing.
    slice_field(f8; n=n, z=0.0, backend=:loop)
    slice_field_dense(f8; n=n, z=0.0, backend=:tullio)

    loop_times = [@elapsed slice_field(f8; n=n, z=0.0, backend=:loop) for _ in 1:repeats]
    tullio_times = [@elapsed slice_field_dense(f8; n=n, z=0.0, backend=:tullio) for _ in 1:repeats]
    return (
        n=n,
        loop_median=median(loop_times),
        loop_mean=mean(loop_times),
        loop_std=std(loop_times),
        tullio_median=median(tullio_times),
        tullio_mean=mean(tullio_times),
        tullio_std=std(tullio_times),
    )
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

grid_sizes = [121, 181, 241, 301]
bench_results = [benchmark_dense_eval(; n=n, repeats=6) for n in grid_sizes]

loop_medians = [r.loop_median for r in bench_results]
tullio_medians = [r.tullio_median for r in bench_results]
speedups = loop_medians ./ max.(tullio_medians, eps())

scenarios = [:baseline, :early_drive, :high_gain]
traj = Dict{Symbol, Tuple{Vector{Float64}, Vector{Float64}}}()
for s in scenarios
    traj[s] = run_scenario(s)
end

bench_plot = bar(
    grid_sizes,
    [loop_medians tullio_medians],
    label=["loop median" "tullio median"],
    color=[:slategray :darkorange],
    xlabel="grid size n",
    ylabel="seconds",
    title="Dense slice evaluation: warm-up + 4-grid sweep (6 repeats)",
)

plot!(
    bench_plot,
    grid_sizes,
    loop_medians,
    linewidth=2,
    color=:black,
    label="",
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

for (i, n) in enumerate(grid_sizes)
    r = bench_results[i]
    println(
        "n=$(n): loop median=$(round(loop_medians[i], digits=4)) s, ",
        "loop mean+/-std=$(round(r.loop_mean, digits=4))+/-$(round(r.loop_std, digits=4)) s, ",
        "tullio median=$(round(tullio_medians[i], digits=4)) s, ",
        "tullio mean+/-std=$(round(r.tullio_mean, digits=4))+/-$(round(r.tullio_std, digits=4)) s, ",
        "speedup=$(round(speedups[i], digits=2))x",
    )
end
println("Median speedup across sweep: ", round(median(speedups), digits=2), "x")
println("Saved -> ", outpath)

