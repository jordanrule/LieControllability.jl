function h1(x, args...)
    return [0.0, 0.0, 1.0] .* x
end

function f1(x, args...)
    return [
        -x[2]^2 + x[3],
        -x[1]^3,
        -x[3]^2 + x[2],
    ]
end

function f2(x, args...)
    return [
        -sin(x[2]),
        -5 * x[1]^2,
        -sin(x[3] - x[2]),
    ]
end

function f3(x, args...)
    return -[
        x[1] * (x[1] - 2) * (x[1] + 2),
        x[2]^2,
        x[3]^2,
    ]
end

function f4(x, args...)
    return -[x[1], x[2], x[3]]
end

function f5(x, args...)
    return -[sin(x[3]), sin(x[1]), sin(x[2])]
end

function f6(x, bifur=[0, 1])
    return -[bifur[1] * x[1], bifur[2] * x[2], x[3]]
end

function f7(x, bifur=[0, 1])
    return -[
        x[1]^3 - bifur[1] * x[1]^2,
        x[3]^3 - 2 * x[2]^2,
        x[2]^3 - bifur[2] * x[3],
    ]
end

function f8(x, bifur=[0, 0])
    return -[
        x[2] * x[3],
        x[3] * (bifur[1] + x[2] - x[3]),
        x[2] * (-bifur[2] + x[3] + x[2]),
    ]
end

function f9(x)
    return -[
        x[2] * x[3],
        x[3] * (x[2] - x[3]),
        x[2] * (x[3] + x[2]),
    ]
end

function g1(x, args...)
    return [x[1], 0.0, x[2]]
end

function f_main(x)
    return [
        -x[1]  -x[2] * x[1]   0.0;
        -x[2] * x[3]  -x[2]   0.0;
        0.0  -x[1] * x[2]   x[3];
    ]
end

function f_k(x, D)
    x_1 = D' * x
    x_2 = sin.(x_1)
    x_3 = D * x_2
    return x_3
end

function cycle_adjacency(n::Int)
    A = zeros(Float64, n, n)
    for i in 1:n
        for offset in (1, 2)
            j = mod1(i + offset, n)
            k = mod1(i - offset, n)
            A[i, j] = 1.0
            A[j, i] = 1.0
            A[i, k] = 1.0
            A[k, i] = 1.0
        end
    end
    return A
end

function Xi_1(x, P)
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] .* x
end

function h_single_vect(x, P)
    measure_vect = zeros(eltype(x), length(x))
    readout_idx = _readout_index(P, length(x))
    measure_vect[readout_idx] = sin(x[readout_idx])
    return measure_vect
end

function h_single(x, P)
    measure_vect = zeros(eltype(x), length(x))
    readout_idx = _readout_index(P, length(x))
    measure_vect[readout_idx] = 1.0
    return dot(measure_vect, sin.(x))
end

function g_mono(x, P)
    test = zeros(eltype(x), length(x), length(x))
    resid = zeros(eltype(x), length(x))
    control_idx = _control_index(P, length(x))
    test[control_idx, control_idx] = 1.0
    ret_vec = test * x .+ resid
    return ret_vec
end

function f_trivial(x, P)
    if P isa AbstractMatrix
        filt_x = zeros(eltype(x), length(x), length(x))
        if length(x) >= 2
            filt_x[2, 2] = 1.0
        end
        if length(x) >= 4
            filt_x[4, 4] = 1.0
        end
        if length(x) >= 10
            filt_x[8, 10] = 1.0
        end
        return filt_x' * x
    end
    L = hasproperty(P, :L) ? P.L : Matrix{Float64}(I, length(x), length(x))
    drift_gain = hasproperty(P, :drift_gain) ? P.drift_gain : 0.08
    nonlinear_gain = hasproperty(P, :nonlinear_gain) ? P.nonlinear_gain : 0.0
    return -drift_gain .* (L * x) .+ nonlinear_gain .* tanh.(x)
end

function u_step(t, P)
    drive_onset = hasproperty(P, :drive_onset) ? P.drive_onset : 5.0
    drive_amplitude = hasproperty(P, :drive_amplitude) ? P.drive_amplitude : 1.0
    return t > drive_onset ? drive_amplitude : 0.0
end

@inline _bounded_index(idx::Int, n::Int) = clamp(idx, 1, n)

function _control_index(P, n::Int)
    if hasproperty(P, :control_node)
        return _bounded_index(Int(getproperty(P, :control_node)), n)
    end
    return _bounded_index(8, n)
end

function _readout_index(P, n::Int)
    if hasproperty(P, :readout_node)
        return _bounded_index(Int(getproperty(P, :readout_node)), n)
    end
    return _bounded_index(4, n)
end

function cycle_laplacian(n::Int)
    A = zeros(Float64, n, n)
    for i in 1:n
        for offset in (1, 2)
            j = mod1(i + offset, n)
            k = mod1(i - offset, n)
            A[i, j] = 1.0
            A[j, i] = 1.0
            A[i, k] = 1.0
            A[k, i] = 1.0
        end
    end
    degrees = vec(sum(A, dims=2))
    return Diagonal(degrees) - A
end

mutable struct ControlSystem
    f_drift::Function
    g_ctrl::Function
    u::Function
    h::Function
    G::Matrix{Float64}
    L::Matrix{Float64}
    D::Matrix{Float64}
    e_to_r::Vector{Int}
    Xi::Function
    P::Any
    x_state::Matrix{Float64}
    n_regions::Int
    n_symp::Int
    n_elements::Int
    control_sets::Vector{Vector{Float64}}
    measure_sets::Vector{Float64}
    full_sets::Vector{Vector{Float64}}
end

function ControlSystem(; n_elements::Int=10, seed::Int=1,
    control_node::Int=8, readout_node::Int=4,
    drift_gain::Float64=0.08, nonlinear_gain::Float64=0.01,
    drive_onset::Float64=5.0, drive_amplitude::Float64=1.0,
    use_parameterized::Bool=false)
    n_regions = Int(floor(n_elements / 2))
    L = cycle_laplacian(n_elements)
    G = cycle_adjacency(n_elements)
    D = Matrix{Float64}(I, n_elements, n_elements)
    rng = MersenneTwister(seed)
    e_to_r = rand(rng, 1:n_regions, n_elements)
    Xi = Xi_1
    control_node = _bounded_index(control_node, n_elements)
    readout_node = _bounded_index(readout_node, n_elements)
    P = use_parameterized ? (
        L=L,
        control_node=control_node,
        readout_node=readout_node,
        drift_gain=drift_gain,
        nonlinear_gain=nonlinear_gain,
        drive_onset=drive_onset,
        drive_amplitude=drive_amplitude,
    ) : L
    x_state = rand(rng, 1000, 1)
    n_symp = 2
    return ControlSystem(
        f_trivial,
        g_mono,
        u_step,
        h_single,
        G,
        L,
        D,
        e_to_r,
        Xi,
        P,
        x_state,
        n_regions,
        n_symp,
        n_elements,
        Vector{Vector{Float64}}(),
        Float64[],
        Vector{Vector{Float64}}(),
    )
end

control_system(; kwargs...) = ControlSystem(; kwargs...)

Base.@kwdef struct DBSScenario
    name::Symbol = :baseline
    n_elements::Int = 10
    seed::Int = 1
    control_node::Int = 8
    readout_node::Int = 4
    drift_gain::Float64 = 0.08
    nonlinear_gain::Float64 = 0.01
    drive_onset::Float64 = 5.0
    drive_amplitude::Float64 = 1.0
    tspan::Tuple{Float64, Float64} = (0.0, 20.0)
    dt::Float64 = 0.02
end

function dbs_scenario(name::Symbol=:baseline; kwargs...)
    if name == :baseline
        return DBSScenario(; name=name, kwargs...)
    elseif name == :early_drive
        return DBSScenario(; name=name, drive_onset=2.5, kwargs...)
    elseif name == :high_gain
        return DBSScenario(; name=name, drift_gain=0.12, nonlinear_gain=0.03, drive_amplitude=1.4, kwargs...)
    end
    throw(ArgumentError("Unknown DBS scenario $(name). Supported presets: :baseline, :early_drive, :high_gain"))
end

function control_system(s::DBSScenario)
    return ControlSystem(
        n_elements=s.n_elements,
        seed=s.seed,
        control_node=s.control_node,
        readout_node=s.readout_node,
        drift_gain=s.drift_gain,
        nonlinear_gain=s.nonlinear_gain,
        drive_onset=s.drive_onset,
        drive_amplitude=s.drive_amplitude,
        use_parameterized=true,
    )
end

function _control_effect(g, u)
    if u isa Number
        return g .* u
    elseif g isa AbstractMatrix
        return g * u
    elseif g isa AbstractVector && u isa AbstractVector && length(g) == length(u)
        return g .* u
    end
    throw(ArgumentError("Unsupported control/vector-field dimensions for control application."))
end

_system_rhs(cs::ControlSystem, x, t, u) = cs.f_drift(x, cs.P) .+ _control_effect(cs.g_ctrl(x, cs.P), u)

function simulate_trajectory(cs::ControlSystem, x0;
    tspan::Tuple{Float64, Float64}=(0.0, 20.0),
    dt::Float64=0.01,
    method::Symbol=:rk4,
    control=nothing,
    process_noise::Float64=0.0,
    rng=Random.default_rng())
    dt <= 0 && throw(ArgumentError("dt must be > 0"))
    t0, tf = tspan
    tf <= t0 && throw(ArgumentError("tspan must satisfy tf > t0"))
    n_steps = Int(floor((tf - t0) / dt)) + 1
    t = collect(range(t0, step=dt, length=n_steps))
    x = Float64.(collect(x0))
    X = zeros(Float64, length(x), n_steps)
    U = zeros(Float64, n_steps)
    X[:, 1] .= x
    control_fn = control === nothing ? ((tt, xx, ccs) -> ccs.u(tt, ccs.P)) : control
    for k in 1:(n_steps - 1)
        tk = t[k]
        uk = control_fn(tk, x, cs)
        U[k] = uk isa Number ? Float64(uk) : norm(uk)
        if method == :euler
            x_new = x .+ dt .* _system_rhs(cs, x, tk, uk)
        elseif method == :rk4
            # Standard RK4 step for non-stiff control-affine dynamics.
            k1 = _system_rhs(cs, x, tk, uk)
            k2 = _system_rhs(cs, x .+ 0.5 .* dt .* k1, tk + 0.5 * dt, uk)
            k3 = _system_rhs(cs, x .+ 0.5 .* dt .* k2, tk + 0.5 * dt, uk)
            k4 = _system_rhs(cs, x .+ dt .* k3, tk + dt, uk)
            x_new = x .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
        else
            throw(ArgumentError("Unsupported method $(method). Use :rk4 or :euler."))
        end
        if process_noise > 0
            x_new .+= process_noise * sqrt(dt) .* randn(rng, length(x))
        end
        x = x_new
        X[:, k + 1] .= x
    end
    U[end] = U[max(1, end - 1)]
    return (t=t, x=X, u=U, method=method, dt=dt)
end

function disease_control(cs::ControlSystem; rng=Random.default_rng())
    rand_checks = rand(rng, cs.n_elements, cs.n_elements) .* 20 .- 10
    is_zero = Bool[]
    cs.control_sets = Vector{Vector{Float64}}()
    for ii in 1:cs.n_elements
        x = rand_checks[ii, :]
        jac_g = ForwardDiff.jacobian(y -> cs.g_ctrl(y, cs.P), x)
        xi = cs.Xi(x, cs.P)
        push!(cs.control_sets, vec(jac_g * xi))
        push!(is_zero, all(iszero, cs.control_sets[end]))
    end
    result = all(is_zero)
    println("Control-Disease interaction is zero: $(result)")
    return result
end

function disease_measure(cs::ControlSystem; rng=Random.default_rng())
    rand_checks = rand(rng, cs.n_elements, cs.n_elements) .* 20 .- 10
    is_zero = Bool[]
    cs.measure_sets = Float64[]
    for ii in 1:cs.n_elements
        x = rand_checks[ii, :]
        grad_h = ForwardDiff.gradient(y -> cs.h(y, cs.P), x)
        xi = cs.Xi(x, cs.P)
        value = dot(grad_h, xi)
        push!(cs.measure_sets, value)
        push!(is_zero, iszero(value))
    end
    result = all(is_zero)
    println("Measurement-Disease interaction is zero: $(result)")
    return result
end

function full_control(cs::ControlSystem; rng=Random.default_rng())
    rand_checks = rand(rng, cs.n_elements, cs.n_elements) .* 20 .- 10
    is_zero = Bool[]
    cs.full_sets = Vector{Vector{Float64}}()
    for ii in 1:cs.n_elements
        x = rand_checks[ii, :]
        grad_f = ForwardDiff.jacobian(y -> cs.f_drift(y, cs.P), x)
        grad_g = ForwardDiff.jacobian(y -> cs.g_ctrl(y, cs.P), x)
        xi = cs.Xi(x, cs.P)
        value = vec(grad_f * xi .+ grad_g * xi)
        push!(cs.full_sets, value)
        push!(is_zero, all(iszero, value))
    end
    result = all(is_zero)
    println("Dyn+Ctrl is zero: $(result)")
    return result
end

