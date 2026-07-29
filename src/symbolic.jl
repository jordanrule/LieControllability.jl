"""
    Symbolic Lie derivative and Lie bracket codepaths via Symbolics.jl.

Exported API
------------
- `sym_L_d(h_expr, f_exprs, vars)` — symbolic Lie derivative Lf(h)
- `sym_L_bracket(f_exprs, g_exprs, vars)` — symbolic Lie bracket [f,g]
- `sym_controllability_matrix(f_exprs, g_list, vars; depth)` — iterated brackets
- `sym_to_numeric(expr, vars)` — compile a symbolic expression to a Julia function
"""

using Symbolics
using LinearAlgebra

export sym_L_d, sym_L_bracket, sym_controllability_matrix, sym_to_numeric

"""
    sym_L_d(h_expr, f_exprs, vars) -> Num

Compute the symbolic Lie derivative of scalar `h_expr` along vector field `f_exprs`.

# Arguments
- `h_expr` : symbolic scalar expression (a `Num`)
- `f_exprs`: symbolic vector of length n (the drift / control field)
- `vars`   : symbolic variable vector, e.g. `@variables x[1:3]` flattened

# Returns the symbolic scalar `∇h · f`.

# Example
```julia
@variables x[1:3]
xv = collect(x)
h  = 2*xv[1] + 3*xv[3]
f  = [-xv[2]*xv[3], -xv[3]*(xv[2]-xv[3]), -xv[2]*(xv[3]+xv[2])]
Lf_h = sym_L_d(h, f, xv)
```
"""
function sym_L_d(h_expr, f_exprs, vars)
    grad_h = [Symbolics.derivative(h_expr, v) for v in vars]
    return sum(grad_h[i] * f_exprs[i] for i in eachindex(vars))
end

"""
    sym_L_bracket(f_exprs, g_exprs, vars) -> Vector{Num}

Compute the symbolic Lie bracket [f, g] = (∂g/∂x)f − (∂f/∂x)g.

Returns a vector of n symbolic expressions.
"""
function sym_L_bracket(f_exprs, g_exprs, vars)
    n = length(vars)
    # Jacobians
    Jg = [Symbolics.derivative(g_exprs[i], vars[j]) for i in 1:n, j in 1:n]
    Jf = [Symbolics.derivative(f_exprs[i], vars[j]) for i in 1:n, j in 1:n]
    # [f,g] = Jg * f - Jf * g
    result = Vector{Any}(undef, n)
    for i in 1:n
        result[i] = sum(Jg[i, j] * f_exprs[j] for j in 1:n) -
                    sum(Jf[i, j] * g_exprs[j] for j in 1:n)
    end
    return Symbolics.simplify.(result)
end

"""
    sym_controllability_matrix(f_exprs, g_list, vars; depth=nothing) -> Matrix{Num}

Build the controllability distribution matrix whose columns are
`g`, `[f,g]`, `[f,[f,g]]`, ... up to `depth` iterations (default: `length(vars)-1`).

Multiple control fields may be passed as a vector-of-vectors in `g_list`.
"""
function sym_controllability_matrix(f_exprs, g_list, vars; depth=nothing)
    n = length(vars)
    d = depth === nothing ? n - 1 : depth
    cols = Vector{Any}()
    for g0 in g_list
        current = g0
        push!(cols, copy(current))
        for _ in 1:d
            current = sym_L_bracket(f_exprs, current, vars)
            push!(cols, copy(current))
        end
    end
    # assemble into matrix
    M = Matrix{Any}(undef, n, length(cols))
    for (j, col) in enumerate(cols)
        for i in 1:n
            M[i, j] = col[i]
        end
    end
    return M
end

"""
    sym_to_numeric(expr_or_vec, vars) -> Function

Compile a symbolic expression or vector of expressions into a callable
`f(x::AbstractVector) -> scalar | Vector`.

```julia
@variables x[1:3]
xv = collect(x)
f_sym = [-xv[2]*xv[3], xv[1], xv[2]]
f_num = sym_to_numeric(f_sym, xv)
f_num([1.0, 2.0, 3.0])  # => [-6.0, 1.0, 2.0]
```
"""
function sym_to_numeric(expr, vars)
    if expr isa AbstractVector
        fns = [Symbolics.build_function(e, vars...; expression=Val{false}) for e in expr]
        return (x::AbstractVector) -> [fn(x...) for fn in fns]
    else
        fn = Symbolics.build_function(expr, vars...; expression=Val{false})
        return (x::AbstractVector) -> fn(x...)
    end
end

