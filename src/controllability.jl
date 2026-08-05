using LinearAlgebra
using Random

export lie_bracket_field, controllability_distribution, controllability_rank, chow_rank_test

lie_bracket_field(f::Function, g::Function) = (x, args...) -> L_bracket(f, g, x, args...)

_as_field_list(g_field::Function) = [g_field]
_as_field_list(g_fields::AbstractVector{<:Function}) = collect(g_fields)

function _evaluate_field(field::Function, x::AbstractVector, field_args::Tuple)
    return field_args == () ? field(x) : field(x, field_args...)
end

function _lie_algebra_fields(
    f::Function,
    g_fields;
    max_depth::Int,
    include_drift::Bool=true,
    max_fields::Int=256,
)
    max_depth < 0 && throw(ArgumentError("max_depth must be >= 0"))
    max_fields < 1 && throw(ArgumentError("max_fields must be >= 1"))

    controls = Function[_as_field_list(g_fields)...]
    generators = include_drift ? Function[f, controls...] : Function[controls...]
    isempty(generators) && throw(ArgumentError("At least one generator field is required."))

    all_fields = Function[generators...]
    level_fields = Function[generators...]

    for _ in 1:max_depth
        new_level = Function[]
        for base in generators
            for current in level_fields
                push!(new_level, lie_bracket_field(base, current))
                if length(all_fields) + length(new_level) >= max_fields
                    append!(all_fields, new_level)
                    return all_fields
                end
            end
        end
        append!(all_fields, new_level)
        level_fields = new_level
        isempty(level_fields) && break
    end

    return all_fields
end

function _sample_state(rng::AbstractRNG, n_state::Int, sample_bounds)
    if sample_bounds isa Tuple && length(sample_bounds) == 2
        lo, hi = sample_bounds
        return lo .+ (hi - lo) .* rand(rng, n_state)
    elseif sample_bounds isa AbstractVector && length(sample_bounds) == n_state
        lo = first.(sample_bounds)
        hi = last.(sample_bounds)
        return lo .+ (hi - lo) .* rand(rng, n_state)
    end
    throw(ArgumentError("sample_bounds must be (lo, hi) or a length-n_state vector of (lo, hi) tuples."))
end

function controllability_distribution(
    f::Function,
    g_fields,
    x::AbstractVector;
    max_depth::Int=length(x) - 1,
    include_drift::Bool=true,
    max_fields::Int=256,
    field_args::Tuple=(),
)
    fields = _lie_algebra_fields(f, g_fields; max_depth=max_depth, include_drift=include_drift, max_fields=max_fields)
    cols = [_evaluate_field(field, x, field_args) for field in fields]
    n_state = length(x)
    M = zeros(Float64, n_state, length(cols))
    for (j, col) in enumerate(cols)
        length(col) == n_state || throw(ArgumentError("Field output length does not match state dimension."))
        M[:, j] .= Float64.(col)
    end
    return M
end

function controllability_rank(
    f::Function,
    g_fields,
    x::AbstractVector;
    rank_tol::Float64=1e-8,
    kwargs...,
)
    M = controllability_distribution(f, g_fields, x; kwargs...)
    return rank(M; atol=rank_tol)
end

function chow_rank_test(
    f::Function,
    g_fields;
    n_state::Int,
    n_samples::Int=64,
    sample_bounds=(-1.0, 1.0),
    max_depth::Int=n_state - 1,
    include_drift::Bool=true,
    max_fields::Int=256,
    rank_tol::Float64=1e-8,
    rng::AbstractRNG=Random.default_rng(),
    field_args::Tuple=(),
)
    n_samples < 1 && throw(ArgumentError("n_samples must be >= 1"))

    sample_points = zeros(Float64, n_state, n_samples)
    ranks = zeros(Int, n_samples)

    for k in 1:n_samples
        x = _sample_state(rng, n_state, sample_bounds)
        sample_points[:, k] .= x
        ranks[k] = controllability_rank(
            f,
            g_fields,
            x;
            rank_tol=rank_tol,
            max_depth=max_depth,
            include_drift=include_drift,
            max_fields=max_fields,
            field_args=field_args,
        )
    end

    min_rank = minimum(ranks)
    full_rank_fraction = count(==(n_state), ranks) / n_samples

    return (
        is_controllable=min_rank == n_state,
        ranks=ranks,
        min_rank=min_rank,
        full_rank_fraction=full_rank_fraction,
        sample_points=sample_points,
        n_state=n_state,
        n_samples=n_samples,
        max_depth=max_depth,
        include_drift=include_drift,
        rank_tol=rank_tol,
    )
end

