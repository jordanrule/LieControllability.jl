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
    measure_vect[4] = sin(x[4])
    return measure_vect
end

function h_single(x, P)
    measure_vect = zeros(eltype(x), length(x))
    measure_vect[4] = 1.0
    return dot(measure_vect, sin.(x))
end

function g_mono(x, P)
    test = zeros(eltype(x), length(x), length(x))
    resid = zeros(eltype(x), length(x))
    test[8, 8] = 1.0
    ret_vec = test * x .+ resid
    return ret_vec
end

function f_trivial(x, P)
    filt_x = zeros(eltype(x), length(x), length(x))
    filt_x[2, 2] = 1.0
    filt_x[4, 4] = 1.0
    filt_x[8, 10] = 1.0
    return filt_x' * x
end

function u_step(t, P)
    return t > 5 ? 1.0 : 0.0
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
    P::Matrix{Float64}
    x_state::Matrix{Float64}
    n_regions::Int
    n_symp::Int
    n_elements::Int
    control_sets::Vector{Vector{Float64}}
    measure_sets::Vector{Float64}
    full_sets::Vector{Vector{Float64}}
end

function ControlSystem(; n_elements::Int=10, seed::Int=1)
    n_regions = Int(floor(n_elements / 2))
    L = cycle_laplacian(n_elements)
    G = cycle_adjacency(n_elements)
    D = Matrix{Float64}(I, n_elements, n_elements)
    rng = MersenneTwister(seed)
    e_to_r = rand(rng, 1:n_regions, n_elements)
    Xi = Xi_1
    P = L
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

control_system() = ControlSystem()

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

