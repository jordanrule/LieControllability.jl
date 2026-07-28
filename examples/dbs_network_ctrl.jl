#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LieControllability
using Plots

# A lightweight, first-pass analogue of the Python DBS example.
# It keeps the same control-affine ingredients, but uses a simple
# deterministic graph layout instead of the original networkx/mayavi stack.

gr()

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

cs = control_system()
control_zero = disease_control(cs)
measure_zero = disease_measure(cs)
full_zero = full_control(cs)

println("Disease/control summary: control=$(control_zero), measure=$(measure_zero), full=$(full_zero)")

n = cs.n_elements
angles = collect(range(0, 2π, length=n + 1))[1:end-1]
rx = cos.(angles)
ry = sin.(angles)

control_idx = findall(!iszero, cs.g_ctrl(ones(n), cs.P))
readout_idx = findall(!iszero, h_single_vect(ones(n), cs.P))

network_plot = plot(aspect_ratio=:equal, legend=:topright, title="DBS control layout (first-pass Julia port)", xlabel="", ylabel="")
for i in 1:n
    for j in (i + 1):n
        if j <= n && cs.G[i, j] > 0
            plot!(network_plot, [rx[i], rx[j]], [ry[i], ry[j]], color=:gray, alpha=0.25, linewidth=1, label=false)
        end
    end
end
scatter!(network_plot, rx, ry, markersize=7, color=:black, label="nodes")
scatter!(network_plot, rx[control_idx], ry[control_idx], markersize=10, color=:red, label="control nodes")
scatter!(network_plot, rx[readout_idx], ry[readout_idx], markersize=10, color=:blue, label="readout nodes")
for i in 1:n
    annotate!(network_plot, rx[i], ry[i], text(string(i), 8, :white, :center))
end
savefig(network_plot, joinpath(figdir, "dbs_network_layout.png"))

laplacian_plot = heatmap(cs.L, aspect_ratio=:equal, title="Cycle Laplacian used by the DBS example", xlabel="node", ylabel="node", colorbar=false)
savefig(laplacian_plot, joinpath(figdir, "dbs_laplacian.png"))

assignment_plot = bar(1:n, cs.e_to_r, title="Element-to-region assignment", xlabel="element", ylabel="region id", legend=false)
savefig(assignment_plot, joinpath(figdir, "dbs_region_assignment.png"))

println("Saved figures to ", figdir)

