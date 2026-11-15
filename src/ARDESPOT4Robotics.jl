module ARDESPOT4Robotics

using POMDPs
using Parameters
using CPUTime
using ParticleFilters
using D3Trees
using Random
using Printf
using POMDPTools

using BasicPOMCP # for ExceptionRethrow and NoDecision
import BasicPOMCP.default_action

import Random.rand

export
    DESPOTSolver,
    DESPOTPlanner,

    DESPOTRandomSource,
    MemorizingSource,
    MemorizingRNG,
    HashedSource,

    ScenarioBelief,
    previous_obs,

    default_action,
    NoGap,

    IndependentBounds,
    FullyObservableValueUB,
    DefaultPolicyLB,
    bounds,
    init_bounds,
    lbound,
    ubound,
    init_bound,

    ReportWhenUsed

# include("random.jl")
include("random_2.jl")
include("hashed_source.jl")

"""
    DESPOTSolver(<keyword arguments>)

Implementation of the ARDESPOT solver trying to closely match the pseudo code of:

http://bigbird.comp.nus.edu.sg/m2ap/wordpress/wp-content/uploads/2017/08/jair14.pdf

Each field may be set via keyword argument. The fields that correspond to algorithm
parameters match the definitions in the paper exactly.

# Fields
- `epsilon_0`
- `xi`
- `K`
- `D`
- `lambda`
- `T_max`
- `max_trials`
- `bounds`
- `default_action`
- `rng`
- `parallel`
- `parallel_min_scenarios`
- `parallel_scenarios`
- `random_source`
- `bounds_warnings`
- `tree_in_info`

Further information can be found in the field docstrings (e.g.
`?DESPOTSolver.xi`)
"""
@with_kw mutable struct DESPOTSolver <: Solver
    "The target gap between the upper and the lower bound at the root of the partial DESPOT."
    epsilon_0::Float64                      = 0.0

    "The rate of target gap reduction."
    xi::Float64                             = 0.95

    "The number of sampled scenarios."
    K::Int                                  = 500

    "The maximum depth of the DESPOT."
    D::Int                                  = 90

    "Reguluarization constant."
    lambda::Float64                         = 0.01

    "The maximum online planning time per step."
    T_max::Float64                          = 1.0

    "The maximum number of trials of the planner."
    max_trials::Int                         = typemax(Int)

    "A representation for the upper and lower bound on the discounted value (e.g. `IndependentBounds`)."
    bounds::Any                             = IndependentBounds(-1e6, 1e6)

    """A default action to use if algorithm fails to provide an action because of an error.
   
    This can either be an action object, i.e. `default_action=1` if `actiontype(pomdp)==Int` or a function `f(pomdp, b, ex)` where b is the belief and ex is the exception that caused the planner to fail.
    """
    default_action::Any                     = ExceptionRethrow()

    "A random number generator for the internal sampling processes."
    rng::MersenneTwister                    = MersenneTwister(rand(UInt32))

    """If true, the actions of a belief node are expanded on separate threads (start Julia with `-t N`).

    The tree is the same one that `parallel=false` builds from the same random numbers. The POMDP, the bounds and any default policy are then called from several threads at once, so they must not share mutable state (an internal rng, a cache, ...). Requires a `random_source` that implements `ARDESPOT.worker_source`, i.e. a `HashedSource`.
    """
    parallel::Bool                          = false

    """With `parallel=true`, belief nodes with fewer scenarios than this are still expanded serially.

    Handing an action to another thread costs a few microseconds, which is more than expanding it takes when the node has few scenarios and the model and bounds are cheap. Lower this (down to 1) if a single generative model step or bound evaluation is expensive.
    """
    parallel_min_scenarios::Int             = 20

    """If true, the work *inside* one action expansion is itself spread over threads: the scenario steps are chunked across tasks (chunks of at least `parallel_min_scenarios` scenarios), and the initial bounds of the new child nodes are computed in parallel (children chunked by scenario count).

    Independent of `parallel`; they can be combined. The tree is still exactly the serial tree. The same thread-safety requirements as for `parallel` apply. Useful when the model or the bounds are expensive but there are fewer actions than threads.
    """
    parallel_scenarios::Bool                = false

    "A source for random numbers in scenario rollout"
    random_source::DESPOTRandomSource       = (parallel || parallel_scenarios) ? HashedSource() : MemorizingSource(K, D, rng)

    "If true, sanity checks on the provided bounds are performed."
    bounds_warnings::Bool                   = true

    "If true, a reprenstation of the constructed DESPOT is returned by POMDPTools.action_info."
    tree_in_info::Bool                      = false
end

include("scenario_belief.jl")
include("default_policy_sim.jl")
include("bounds.jl")

struct DESPOTPlanner{P<:POMDP, B, RS<:DESPOTRandomSource, RNG<:AbstractRNG} <: Policy
    sol::DESPOTSolver
    pomdp::P
    bounds::B
    rs::RS
    rng::RNG
end

function DESPOTPlanner(sol::DESPOTSolver, pomdp::POMDP)
    bounds = init_bounds(sol.bounds, pomdp, sol)
    rng = deepcopy(sol.rng)
    rs = deepcopy(sol.random_source)
    Random.seed!(rs, rand(rng, UInt32))
    if sol.parallel || sol.parallel_scenarios
        worker_source(rs) # throws an informative error if rs cannot be shared between threads
        if Threads.nthreads() == 1
            @warn("DESPOTSolver(parallel=true) will not be faster because Julia is running with a single thread. Start Julia with e.g. `julia -t 4`.", maxlog=1)
        end
    end
    return DESPOTPlanner(deepcopy(sol), pomdp, bounds, rs, rng)
end

include("tree.jl")
include("parallel.jl")
include("planner.jl")
include("pomdps_glue.jl")

include("visualization.jl")
include("exceptions.jl")

end # module
