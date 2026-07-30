using Plots
using Tullio

_field_component(f, x, y, z, idx, args...) = f((x, y, z), args...)[idx]
_scalar_value(f, x, y, z, args...) = f((x, y, z), args...)

function meshgrid3(xs, ys, zs)
    xs = collect(xs)
    ys = collect(ys)
    zs = collect(zs)
    X = repeat(reshape(xs, :, 1, 1), 1, length(ys), length(zs))
    Y = repeat(reshape(ys, 1, :, 1), length(xs), 1, length(zs))
    Z = repeat(reshape(zs, 1, 1, :), length(xs), length(ys), 1)
    return X, Y, Z
end

function _slice_field_loop(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=21)
    xs = collect(range(xrange[1], xrange[2], length=n))
    ys = collect(range(yrange[1], yrange[2], length=n))
    U = zeros(Float64, length(ys), length(xs))
    V = zeros(Float64, length(ys), length(xs))
    W = zeros(Float64, length(ys), length(xs))
    for (j, y) in enumerate(ys)
        for (i, x) in enumerate(xs)
            value = f([x, y, z], args...)
            U[j, i] = value[1]
            V[j, i] = value[2]
            W[j, i] = value[3]
        end
    end
    return xs, ys, U, V, W
end

function _slice_field_tullio(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=21)
    xs = collect(range(xrange[1], xrange[2], length=n))
    ys = collect(range(yrange[1], yrange[2], length=n))
    X = repeat(reshape(xs, 1, :), length(ys), 1)
    Y = repeat(reshape(ys, :, 1), 1, length(xs))
    @tullio U[j, i] := _field_component(f, X[j, i], Y[j, i], z, 1, args...)
    @tullio V[j, i] := _field_component(f, X[j, i], Y[j, i], z, 2, args...)
    @tullio W[j, i] := _field_component(f, X[j, i], Y[j, i], z, 3, args...)
    return xs, ys, U, V, W
end

function _slice_field_tullio(::typeof(f8), bifur=[0, 0]; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=21)
    xs = collect(range(xrange[1], xrange[2], length=n))
    ys = collect(range(yrange[1], yrange[2], length=n))
    Y = repeat(reshape(ys, :, 1), 1, length(xs))
    b1 = Float64(bifur[1])
    b2 = Float64(bifur[2])
    z0 = Float64(z)
    @tullio U[j, i] := -(Y[j, i] * z0)
    @tullio V[j, i] := -(z0 * (b1 + Y[j, i] - z0))
    @tullio W[j, i] := -(Y[j, i] * (-b2 + z0 + Y[j, i]))
    return xs, ys, U, V, W
end

function slice_field(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=21, backend::Symbol=:loop)
    if backend == :loop
        return _slice_field_loop(f, args...; z=z, xrange=xrange, yrange=yrange, n=n)
    elseif backend == :tullio
        return _slice_field_tullio(f, args...; z=z, xrange=xrange, yrange=yrange, n=n)
    end
    throw(ArgumentError("Unsupported backend $(backend). Use :loop or :tullio."))
end

function slice_field_dense(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=161, backend::Symbol=:tullio)
    return slice_field(f, args...; z=z, xrange=xrange, yrange=yrange, n=n, backend=backend)
end

function _slice_scalar_loop(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=41)
    xs = collect(range(xrange[1], xrange[2], length=n))
    ys = collect(range(yrange[1], yrange[2], length=n))
    values = zeros(Float64, length(ys), length(xs))
    for (j, y) in enumerate(ys)
        for (i, x) in enumerate(xs)
            values[j, i] = f([x, y, z], args...)
        end
    end
    return xs, ys, values
end

function _slice_scalar_tullio(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=41)
    xs = collect(range(xrange[1], xrange[2], length=n))
    ys = collect(range(yrange[1], yrange[2], length=n))
    X = repeat(reshape(xs, 1, :), length(ys), 1)
    Y = repeat(reshape(ys, :, 1), 1, length(xs))
    @tullio values[j, i] := _scalar_value(f, X[j, i], Y[j, i], z, args...)
    return xs, ys, values
end

function slice_scalar(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=41, backend::Symbol=:loop)
    if backend == :loop
        return _slice_scalar_loop(f, args...; z=z, xrange=xrange, yrange=yrange, n=n)
    elseif backend == :tullio
        return _slice_scalar_tullio(f, args...; z=z, xrange=xrange, yrange=yrange, n=n)
    end
    throw(ArgumentError("Unsupported backend $(backend). Use :loop or :tullio."))
end

function slice_scalar_dense(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=201, backend::Symbol=:tullio)
    return slice_scalar(f, args...; z=z, xrange=xrange, yrange=yrange, n=n, backend=backend)
end

function plot_field_slice(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=21, normalize=false, title="Vector field slice", backend::Symbol=:loop)
    xs, ys, U, V, W = slice_field(f, args...; z=z, xrange=xrange, yrange=yrange, n=n, backend=backend)
    uflat = Float64[]
    vflat = Float64[]
    xflat = Float64[]
    yflat = Float64[]
    mags = Float64[]
    for (j, y) in enumerate(ys)
        for (i, x) in enumerate(xs)
            u = U[j, i]
            v = V[j, i]
            w = W[j, i]
            mag = sqrt(u^2 + v^2 + w^2)
            if normalize && mag > 0
                u /= mag
                v /= mag
            end
            push!(xflat, x)
            push!(yflat, y)
            push!(uflat, u)
            push!(vflat, v)
            push!(mags, mag)
        end
    end
    plt = quiver(xflat, yflat, quiver=(uflat, vflat), aspect_ratio=:equal, legend=false, title=title, color=:steelblue)
    scatter!(plt, xflat, yflat, markersize=1, markerstrokewidth=0, alpha=0.0)
    return plt
end

function plot_scalar_slice(f, args...; z=0.0, xrange=(-2.0, 2.0), yrange=(-2.0, 2.0), n=41, title="Scalar slice", clims=nothing, backend::Symbol=:loop)
    xs, ys, values = slice_scalar(f, args...; z=z, xrange=xrange, yrange=yrange, n=n, backend=backend)
    plt = heatmap(xs, ys, values, aspect_ratio=:equal, title=title, xlabel="x", ylabel="y", legend=false)
    if clims !== nothing
        plt = heatmap(xs, ys, values, aspect_ratio=:equal, title=title, xlabel="x", ylabel="y", legend=false, clims=clims)
    end
    return plt
end

