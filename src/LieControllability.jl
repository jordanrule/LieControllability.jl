module LieControllability

using LinearAlgebra
using ForwardDiff
using Random

export Operable, operable, L_d, L_dot, L_bracket,
       meshgrid3, slice_field, slice_scalar, plot_field_slice, plot_scalar_slice,
       f1, h1, f4, f8, f9, g1, f_main,
       f_trivial, g_mono, Xi_1, h_single, h_single_vect,
       ControlSystem, control_system, disease_control, disease_measure, full_control,
       u_step, cycle_laplacian

struct Operable{F}
    f::F
end

(o::Operable)(x, args...) = o.f(x, args...)
operable(f) = Operable(f)

function _combine_operables(op, a::Operable, b::Operable)
    Operable((x, args...) -> op(a(x, args...), b(x, args...)))
end

Base.:+(a::Operable, b::Operable) = _combine_operables(+, a, b)
Base.:-(a::Operable, b::Operable) = _combine_operables(-, a, b)
Base.:*(a::Operable, b::Operable) = _combine_operables(*, a, b)
Base.:/(a::Operable, b::Operable) = _combine_operables(/, a, b)
Base.:^(a::Operable, b::Operable) = _combine_operables(^, a, b)
Base.:-(a::Operable) = Operable((x, args...) -> -a(x, args...))

function _directional_derivative(d::Function, f::Function, x, args...)
    xvec = Float64.(collect(x))
    fx = f(xvec, args...)
    value = d(xvec, args...)
    if value isa Number
        grad = ForwardDiff.gradient(y -> d(y, args...), xvec)
        return dot(grad, fx)
    elseif value isa AbstractVector
        jac = ForwardDiff.jacobian(y -> d(y, args...), xvec)
        return jac * fx
    else
        throw(ArgumentError("L_d currently supports scalar or vector-valued functions only."))
    end
end

function L_d(d::Function, f::Function, order::Integer=1)
    order < 1 && throw(ArgumentError("order must be >= 1"))
    current = d
    for _ in 1:order
        prev = current
        current = (x, args...) -> _directional_derivative(prev, f, x, args...)
    end
    return current
end

function L_dot(h::Function, f::Function, order::Integer=1)
    ld = L_d(h, f, order)
    return (x, args...) -> sum(ld(x, args...))
end

function L_bracket(f::Function, g::Function, x0=nothing, args...)
    cf = (x, a...) -> ForwardDiff.jacobian(y -> f(y, a...), Float64.(collect(x))) * g(x, a...)
    cb = (x, a...) -> ForwardDiff.jacobian(y -> g(y, a...), Float64.(collect(x))) * f(x, a...)
    if x0 === nothing
        return cf, cb
    end
    return cf(x0, args...) .- cb(x0, args...)
end

include("dynamics.jl")
include("plotting.jl")

end # module

