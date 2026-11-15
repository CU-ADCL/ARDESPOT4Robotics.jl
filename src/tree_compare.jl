#=
Comparing two search trees.

The promise of the parallel options is that they build exactly the tree the serial search
builds, so the way to check that a model is thread-safe is to build the tree twice and
compare. Plain `isequal` is not enough for that: a state type that holds an array compares
its array fields by identity, so two separately built trees look different even when every
number in them is the same. `deep_isequal` descends instead.
=#

"""
    deep_isequal(a, b)

Like `isequal`, but descends into arrays and into the fields of structs instead of comparing
them by identity.

Two values are deeply equal if they have the same type and either `isequal` holds (numbers,
symbols, strings, ...), or they are arrays with the same axes whose elements are deeply
equal, or they are structs whose fields are deeply equal. Unassigned array slots and
undefined fields compare equal to each other and unequal to anything else.

Structs are compared field by field even if the type defines its own `isequal`. That is
deliberate: this is a verification tool, and a custom `isequal` that ignores a field would
hide exactly the kind of difference it is meant to find.
"""
function deep_isequal(a, b)
    a === b && return true
    T = typeof(a)
    T === typeof(b) || return false
    if a isa AbstractArray
        axes(a) == axes(b) || return false
        for i in eachindex(a)
            ia, ib = isassigned(a, i), isassigned(b, i)
            ia == ib || return false
            ia || continue # both slots unassigned
            deep_isequal(a[i], b[i]) || return false
        end
        return true
    end
    if isstructtype(T) && !(a isa Function) && fieldcount(T) > 0
        for i in 1:fieldcount(T)
            da, db = isdefined(a, i), isdefined(b, i)
            da == db || return false
            da || continue
            deep_isequal(getfield(a, i), getfield(b, i)) || return false
        end
        return true
    end
    return isequal(a, b)
end

"""
    tree_mismatch(D1::DESPOT, D2::DESPOT)

Return the name of the first field in which the two trees differ, or `nothing` if they are
identical. Useful for working out *what* differs when a model turns out not to be
thread-safe: `:scenarios` points at the generative model, `:l`/`:mu` at the bounds or the
rollout policy.

Fields are compared with [`deep_isequal`](@ref), so trees over states that hold arrays are
compared by value. The root's observation slot is never assigned and is skipped.
"""
function tree_mismatch(D1::DESPOT, D2::DESPOT)
    typeof(D1) === typeof(D2) || return :type
    for f in fieldnames(typeof(D1))
        x, y = getfield(D1, f), getfield(D2, f)
        if f === :obs # D.obs[1], the root, is never assigned
            length(x) == length(y) || return f
            all(deep_isequal(x[i], y[i]) for i in 2:length(x)) || return f
        else
            deep_isequal(x, y) || return f
        end
    end
    return nothing
end

"""
    trees_equal(D1::DESPOT, D2::DESPOT)

Whether two search trees are identical, comparing by value all the way down; see
[`tree_mismatch`](@ref), which also says which field differs.

Build the same tree serially and in parallel and compare them to check that a model is safe
to use with the parallel options:

```julia
function tree(parallel)
    sol = DESPOTSolver(bounds=bounds, K=K, random_source=HashedSource(),
                       parallel=parallel, parallel_scenarios=parallel,
                       parallel_min_scenarios=1)
    planner = solve(sol, pomdp)
    Random.seed!(planner.rs, 42)   # same scenarios either way
    return ARDESPOT4Robotics.build_despot(planner, b0)
end
ARDESPOT4Robotics.trees_equal(tree(false), tree(true))
```

Run it a few times and with several seeds: a data race does not show up on every run.
"""
trees_equal(D1::DESPOT, D2::DESPOT) = tree_mismatch(D1, D2) === nothing
