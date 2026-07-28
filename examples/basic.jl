#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LieControllability
using Plots

gr()

figdir = joinpath(@__DIR__, "..", "figures")
mkpath(figdir)

h_scalar(x, args...) = 2 * x[1] + 3 * x[3]

field_plot = plot_field_slice(f8; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=19, normalize=true, title="Drift field f8 on z = 0")
savefig(field_plot, joinpath(figdir, "basic_vector_field.png"))

lie_derivative = L_d(h_scalar, f8)
lie_plot = plot_scalar_slice(lie_derivative; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=51, title="L_f h on z = 0")
savefig(lie_plot, joinpath(figdir, "basic_lie_derivative.png"))

sample = [1.0, 2.0, 0.5]
println("Sample L_f h at ", sample, " = ", lie_derivative(sample))
println("Saved figures to ", figdir)

