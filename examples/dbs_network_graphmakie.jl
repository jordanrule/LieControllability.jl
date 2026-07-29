#!/usr/bin/env julia
# dbs_network_graphmakie.jl — richer DBS network visualisation
# Uses Graphs.jl + GraphMakie.jl + CairoMakie.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LieControllability
using Graphs
using GraphMakie
using CairoMakie
figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)
cs = control_system()
n  = cs.n_elements
# Build Graphs.jl SimpleGraph from adjacency matrix
g = SimpleGraph(n)
for i in 1:n, j in (i+1):n
    cs.G[i, j] > 0 && add_edge!(g, i, j)
end
control_idx = findall(!iszero, cs.g_ctrl(ones(n), cs.P))
readout_idx = findall(!iszero, h_single_vect(ones(n), cs.P))
node_color = fill(:steelblue, n)
for i in control_idx; node_color[i] = :firebrick; end
for i in readout_idx;  node_color[i] = :gold;      end
node_size = fill(22, n)
for i in control_idx; node_size[i] = 32; end
for i in readout_idx; node_size[i] = 32; end
angles = range(0, 2pi, length=n+1)[1:end-1]
layout = Point2f.(cos.(angles), sin.(angles))
fig = Figure(size=(800, 780))
ax  = Axis(fig[1, 1];
    title="DBS Network — Graphs.jl / GraphMakie",
    aspect=DataAspect(),
    xgridvisible=false, ygridvisible=false,
    xticksvisible=false, yticksvisible=false,
    xticklabelsvisible=false, yticklabelsvisible=false)
graphplot!(ax, g;
    layout=(_)->layout,
    node_color=node_color,
    node_size=node_size,
    node_strokewidth=1.5,
    node_strokecolor=:white,
    edge_color=:gray70,
    edge_width=1.5,
    nlabels=string.(1:n),
    nlabels_color=:white,
    nlabels_align=(:center,:center),
    nlabels_textsize=11)
Legend(fig[1,2],
    [MarkerElement(color=:steelblue,marker=:circle,markersize=16),
     MarkerElement(color=:firebrick,marker=:circle,markersize=18),
     MarkerElement(color=:gold,     marker=:circle,markersize=18)],
    ["Network node","Control node (g_mono)","Readout node (h_single)"],
    tellheight=false)
ax2 = Axis(fig[2,1]; title="Cycle Laplacian", aspect=DataAspect(), xlabel="node", ylabel="node")
hm  = heatmap!(ax2, cs.L; colormap=:RdBu)
Colorbar(fig[2,2], hm; tellheight=false)
ax3 = Axis(fig[3,1:2]; title="Element-to-region assignment", xlabel="element", ylabel="region id")
barplot!(ax3, 1:n, cs.e_to_r; color=:mediumseagreen, strokewidth=0.5)
rowsize!(fig.layout, 1, Relative(0.55))
rowsize!(fig.layout, 2, Relative(0.25))
rowsize!(fig.layout, 3, Relative(0.20))
outpath = joinpath(figdir, "dbs_network_graphmakie.png")
save(outpath, fig; px_per_unit=2)
println("Saved -> ", outpath)
