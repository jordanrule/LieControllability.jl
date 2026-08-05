#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LieControllability
using CairoMakie
using Random
using Statistics

set_theme!(Theme(fontsize=16))

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

function benchmark_dense_eval(; n::Int, repeats::Int=6)
    # Warm up both backends before timing.
    slice_field(f8; n=n, z=0.0, backend=:loop)
    slice_field_dense(f8; n=n, z=0.0, backend=:tullio)

    loop_times = [@elapsed slice_field(f8; n=n, z=0.0, backend=:loop) for _ in 1:repeats]
    tullio_times = [@elapsed slice_field_dense(f8; n=n, z=0.0, backend=:tullio) for _ in 1:repeats]

    return (
        loop_median=median(loop_times),
        tullio_median=median(tullio_times),
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

# Panel A: Lie derivative map with drift direction overlay.
h_scalar(x, args...) = 2 * x[1] + 3 * x[3]
lie_f8 = L_d(h_scalar, f8)
xs, ys, lie_values = slice_scalar_dense(lie_f8; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=241, backend=:tullio)
xs_q, ys_q, U, V, _ = slice_field_dense(f8; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=23, backend=:tullio)

qpoints = Point2f[]
qdirs = Vec2f[]
for (j, y) in enumerate(ys_q)
    for (i, x) in enumerate(xs_q)
        u = U[j, i]
        v = V[j, i]
        nrm = sqrt(u^2 + v^2)
        scale = nrm > 0 ? 0.16 / nrm : 0.0
        push!(qpoints, Point2f(x, y))
        push!(qdirs, Vec2f(scale * u, scale * v))
    end
end

# Panel B: Chow rank sampling comparison on two built-in systems.
chow_f1 = chow_rank_test(
    f1,
    [g1];
    n_state=3,
    n_samples=240,
    sample_bounds=(-1.5, 1.5),
    max_depth=3,
    include_drift=true,
    rank_tol=1e-8,
    rng=MersenneTwister(2026),
)

chow_f4 = chow_rank_test(
    f4,
    [g1];
    n_state=3,
    n_samples=240,
    sample_bounds=(-1.5, 1.5),
    max_depth=3,
    include_drift=true,
    rank_tol=1e-8,
    rng=MersenneTwister(2027),
)

rank_levels = 1:3
rank_counts_f1 = [count(==(k), chow_f1.ranks) for k in rank_levels]
rank_counts_f4 = [count(==(k), chow_f4.ranks) for k in rank_levels]

# Panel C: Benchmark sweep.
grid_sizes = [121, 181, 241, 301]
bench_results = [benchmark_dense_eval(; n=n, repeats=6) for n in grid_sizes]
loop_medians = [r.loop_median for r in bench_results]
tullio_medians = [r.tullio_median for r in bench_results]
speedups = loop_medians ./ max.(tullio_medians, eps())

# Panel D: Scenario trajectories.
scenarios = [:baseline, :early_drive, :high_gain]
trajectories = Dict{Symbol, Tuple{Vector{Float64}, Vector{Float64}}}()
for s in scenarios
    trajectories[s] = run_scenario(s)
end

fig = Figure(size=(1600, 1100), backgroundcolor=:white)
Label(fig[0, 1:2], "LieControllability Results (Reproducible Composite Figure)", fontsize=26, font=:bold)

# A: Lie derivative and flow
panel_a = GridLayout(fig[1, 1])
ax_a = Axis(panel_a[1, 1], title="A. Drift field f8 and Lie derivative L_f h", xlabel="x1", ylabel="x2", aspect=DataAspect())
hm = heatmap!(ax_a, xs, ys, lie_values'; colormap=:vik)
arrows2d!(ax_a, qpoints, qdirs; color=(:black, 0.55), shaftwidth=1.2, tiplength=7, tipwidth=7)
Colorbar(panel_a[1, 2], hm, label="L_f h")

# B: Rank histogram
ax_b = Axis(fig[1, 2], title="B. Chow rank test over random samples", xlabel="rank", ylabel="count")
barplot!(ax_b, collect(rank_levels) .- 0.18, rank_counts_f1; width=0.32, color=:steelblue, label="f1 + g1")
barplot!(ax_b, collect(rank_levels) .+ 0.18, rank_counts_f4; width=0.32, color=:tomato, label="f4 + g1")
ax_b.xticks = (collect(rank_levels), string.(collect(rank_levels)))
axislegend(ax_b; position=:rt)
max_count = maximum(vcat(rank_counts_f1, rank_counts_f4))
text!(
    ax_b,
    1.05,
    0.96 * max_count,
    text="full-rank fraction\nf1+g1: $(round(chow_f1.full_rank_fraction, digits=2))\nf4+g1: $(round(chow_f4.full_rank_fraction, digits=2))",
    align=(:left, :top),
    fontsize=14,
    color=:black,
)

# C: Runtime sweep
ax_c = Axis(fig[2, 1], title="C. Dense-grid runtime sweep (median of 6 repeats)", xlabel="grid size n", ylabel="seconds")
lines!(ax_c, grid_sizes, loop_medians; color=:slategray, linewidth=3, label="loop")
scatter!(ax_c, grid_sizes, loop_medians; color=:slategray, markersize=10)
lines!(ax_c, grid_sizes, tullio_medians; color=:darkorange, linewidth=3, label="tullio")
scatter!(ax_c, grid_sizes, tullio_medians; color=:darkorange, markersize=10)
axislegend(ax_c; position=:lt)
text!(
    ax_c,
    first(grid_sizes),
    maximum(loop_medians) * 0.95,
    text="median speedup: $(round(median(speedups), digits=2))x",
    align=(:left, :top),
    fontsize=14,
)

# D: Trajectory comparison
ax_d = Axis(fig[2, 2], title="D. DBS readout trajectories", xlabel="time", ylabel="h_single(x)")
palette = Dict(:baseline => :royalblue, :early_drive => :seagreen, :high_gain => :firebrick)
for s in scenarios
    ts, ys_s = trajectories[s]
    lines!(ax_d, ts, ys_s; label=String(s), linewidth=3, color=palette[s])
end
axislegend(ax_d; position=:rb)

rowgap!(fig.layout, 14)
colgap!(fig.layout, 14)

png_path = joinpath(figdir, "publication_liecontrollability_results.png")
pdf_path = joinpath(figdir, "publication_liecontrollability_results.pdf")
save(png_path, fig; px_per_unit=2)
save(pdf_path, fig)

println("Saved PNG -> ", png_path)
println("Saved PDF -> ", pdf_path)
println("Chow min ranks: f1+g1=$(chow_f1.min_rank), f4+g1=$(chow_f4.min_rank)")
println("Median speedup across sweep: ", round(median(speedups), digits=2), "x")

