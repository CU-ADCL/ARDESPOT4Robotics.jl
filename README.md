# ARDESPOT4Robotics

[![CI](https://github.com/CU-ADCL/ARDESPOT4Robotics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/CU-ADCL/ARDESPOT4Robotics.jl/actions/workflows/CI.yml)

An implementation of the AR-DESPOT (Anytime Regularized DEterminized Sparse Partially Observable Tree) online POMDP Solver, **with multithreaded tree search for expensive robotics models**.

This package is a derivative of [JuliaPOMDP/ARDESPOT.jl](https://github.com/JuliaPOMDP/ARDESPOT.jl) by Zachary Sunberg and contributors (MIT licensed). It adds opt-in parallelism at three levels of the search — see [Multithreaded tree expansion](#multithreaded-tree-expansion-experimental). **With the same random numbers, the parallel search builds exactly the same tree as the serial search**, node for node and bit for bit, so enabling threads changes how long planning takes and nothing else. With the parallel options off (the default), the solver behaves exactly like upstream ARDESPOT.jl.

Tried to match the pseudocode from this paper: http://bigbird.comp.nus.edu.sg/m2ap/wordpress/wp-content/uploads/2017/08/jair14.pdf as closely as possible. Look there for definitions of all symbols.

Problems use the [POMDPs.jl generative interface](https://github.com/JuliaPOMDP/POMDPs.jl).

If you are trying to use this package and require more documentation, please file an issue!

## Installation

This package is not in the Julia General registry; install it from GitHub:

```julia
using Pkg
Pkg.add("POMDPs")
Pkg.add(url="https://github.com/CU-ADCL/ARDESPOT4Robotics.jl")
```

It can be installed alongside upstream `ARDESPOT` — they are separate packages with separate UUIDs.

## Usage

```julia
using POMDPs, POMDPModels, ARDESPOT4Robotics
using POMDPTools

pomdp = TigerPOMDP()

solver = DESPOTSolver(bounds=(-20.0, 0.0))
planner = solve(solver, pomdp)

for (s, a, o) in stepthrough(pomdp, planner, "s,a,o", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
```

For minimal examples of problem implementations, see [this notebook](https://github.com/JuliaPOMDP/BasicPOMCP.jl/blob/master/notebooks/Minimal_Example.ipynb) and [the POMDPs.jl generative docs](http://juliapomdp.github.io/POMDPs.jl/latest/generative/).

## Solver Options

Solver options can be found in the `DESPOTSolver` docstring and accessed using [Julia's built in documentation system](https://docs.julialang.org/en/v1/manual/documentation/#Accessing-Documentation-1) (or directly in the [Solver source code](/src/ARDESPOT4Robotics.jl)). Each option has its own docstring and can be set with a keyword argument in the `DESPOTSolver` constructor. The definitions of the parameters match as closely as possible to the corresponding definition in the pseudocode of [this paper](http://bigbird.comp.nus.edu.sg/m2ap/wordpress/wp-content/uploads/2017/08/jair14.pdf).

### Bounds

#### Independent bounds

In most cases, the recommended way to specify bounds is with an `IndependentBounds` object, i.e.
```julia
DESPOTSolver(bounds=IndependentBounds(lower, upper))
```
where `lower` and `upper` are either a number or a function (see below).

Often, the lower bound is calculated with a default policy, this can be accomplished using a `DefaultPolicyLB` with any `Solver` or `Policy`.

If `lower` or `upper` is a function, it should handle two arguments. The first is the `POMDP` object and the second is the `ScenarioBelief`. To access the state particles in a `ScenairoBelief` `b`, use `particles(b)` (or `collect(particles(b))` to get a vector).

In most cases, the `check_terminal` and `consistency_fix_thresh` keyword arguments of `IndependentBounds` should be used to add robustness (see the `IndependentBounds` docstring for more info).

##### Example

For the `BabyPOMDP` from `POMDPModels`, bounds setup might look like this:
```julia
using POMDPModels
using POMDPTools
always_feed = FunctionPolicy(b->true)
lower = DefaultPolicyLB(always_feed)

function upper(pomdp::BabyPOMDP, b::ScenarioBelief)
    if all(s==true for s in particles(b)) # all particles are hungry
        return pomdp.r_hungry # the baby is hungry this time, but then becomes full magically and stays that way forever
    else
        return 0.0 # the baby magically stays full forever
    end
end

solver = DESPOTSolver(bounds=IndependentBounds(lower, upper))
```

#### Non-Independent bounds

Bounds need not be calculated independently; a single function that takes in the `POMDP` and `ScenarioBelief` and returns a tuple containing the lower and upper bounds can be passed to the `bounds` argument.

### Multithreaded tree expansion (experimental)

When a belief node is expanded, each action's work (stepping all scenarios and computing the initial bounds of the new child nodes) is independent of the other actions', so it can run on separate threads:

```julia
solver = DESPOTSolver(bounds=bounds, parallel=true) # and start Julia with `julia -t 4` or similar
```

The resulting tree is *exactly* the tree that `parallel=false` would build with the same random numbers (`random_source=HashedSource()`, the default when `parallel=true`, gives each scenario and depth a random stream that does not depend on the order in which the threads ask for it; the shared-memory `MemorizingSource` cannot be used with `parallel=true`).

Requirements and caveats:

- **The POMDP's generative model, the bounds, and any rollout policy inside the bounds are called from several threads at once, so they must be thread-safe**: they must not write to shared mutable state such as an internal `RNG` (use only the `rng` passed to `gen`/the `ScenarioBelief`'s random source), a preallocated scratch buffer, a memoization cache, or a lazily initialized field.
- Handing an action to another thread costs a few microseconds, so it only pays off when expanding one action is more expensive than that (an expensive model/bounds, or many scenarios). Nodes with fewer scenarios than `parallel_min_scenarios` (default 20) are expanded serially; lower it towards 1 if a single model step or bound evaluation is expensive.
- Only the node expansion is parallel; the rest of the search (selecting nodes, backing up bounds) stays serial, so the achievable speedup is capped by the fraction of time spent in expansion and by the number of actions.
- `parallel_scenarios=true` additionally (or on its own) parallelizes the inside of one action's expansion — the scenario steps are chunked across tasks, and the initial bounds of the new child nodes are computed in parallel. The tree is still exactly the serial tree, and the same thread-safety requirements apply. It helps when the model or bounds are expensive and the number of actions is smaller than the number of threads.
- `DefaultPolicyLB(policy, parallel=true, parallel_min_scenarios=n)` parallelizes the inside of a single default-policy bound evaluation: once the simulated scenarios split up over different observations, the branches run on separate threads. The bound value is exactly the serial one. This is the only way to speed up the bound of the *root* belief, which runs before the search loop and can dominate on expensive models.

### Making your POMDP thread-safe

With any of the parallel options on, your `gen`, your bounds and your rollout policy are called from several threads at the same time. Anything they write to that is shared between those calls is a data race: the search then silently computes wrong values, usually in a small fraction of calls, and the tree stops matching the serial one. The three problems below are the ones that actually show up in robotics models.

**1. A scratch buffer stored in the POMDP struct.** A common optimization is to preallocate a working array once and reuse it on every `gen` call:

```julia
struct MyPOMDP <: POMDP{S,A,O}
    path_cache::MVector{6,Tuple{Float64,Float64,Float64}}  # written by every gen call
end
```

Two threads then write to the same array at the same time. Allocate it inside the function instead and return it, or give each task its own copy (see 3).

**2. A rollout policy that remembers something between steps.** A guide that keeps a workspace, a cache, or a memory of what it did at the previous depth has exactly the same problem, one level deeper: the simulations of two different belief nodes overwrite each other's memory. If the state is genuinely per-rollout, give every task its own:

```julia
function policy_state(guide)
    get!(task_local_storage(), (:my_rollout_state, objectid(guide))) do
        MyWorkspace(guide)        # built once per task, on first use
    end::MyWorkspace
end
```

This reproduces the serial behaviour exactly, because each task runs one simulation path at a time, just as the serial code does.

**3. An RNG inside the model.** Use only the `rng` that is passed to `gen` (or the belief's random source). A `MersenneTwister` stored in the POMDP struct is shared mutable state like any other, and it also breaks reproducibility.

Also watch out for memoization caches and fields that are initialized lazily on first use — both are writes to shared state.

**How to check.** Build the tree twice with the same seed, once serial and once parallel, and compare. If anything in your model is not thread-safe, the trees differ:

```julia
function tree(parallel)
    sol = DESPOTSolver(bounds=bounds, K=K, random_source=HashedSource(),
                       parallel=parallel, parallel_scenarios=parallel, parallel_min_scenarios=1)
    planner = solve(sol, pomdp)
    Random.seed!(planner.rs, 42)       # same scenarios either way
    return ARDESPOT4Robotics.build_despot(planner, b0)
end
ARDESPOT4Robotics.trees_equal(tree(false), tree(true))
```

`ARDESPOT4Robotics.tree_mismatch` returns the name of the first field that differs instead of a `Bool`, which tells you where to look: `:scenarios` points at the generative model, `:l` or `:mu` at the bounds or the rollout policy. Both compare by value all the way down (`deep_isequal`), because plain `isequal` compares a `Vector` field of your state type by identity and would report a difference between any two separately built trees.

Run the check a few times and with several seeds: a race does not show up on every run.

## Visualization

[D3Trees.jl](https://github.com/sisl/D3Trees.jl) can be used to visualize the search tree, for example

```julia
using POMDPs, POMDPModels, D3Trees, ARDESPOT4Robotics
using POMDPTools

pomdp = TigerPOMDP()

solver = DESPOTSolver(bounds=(-20.0, 0.0), tree_in_info=true)
planner = solve(solver, pomdp)
b0 = initialstate_distribution(pomdp)

a, info = action_info(planner, b0)
inchrome(D3Tree(info[:tree], init_expand=5))
```
will create an interactive tree that looks like this:

![DESPOT tree](img/tree.png)

## Relationship to DESPOT.jl

[DESPOT.jl](https://github.com/JuliaPOMDP/DESPOT.jl) was designed to exactly emulate the [C++ code](https://github.com/AdaCompNUS/despot) released by the original DESPOT developers. This implementation was designed to be as close to the pseudocode from the journal paper as possible for the sake of readability. ARDESPOT4Robotics has a few more features (for example DESPOT.jl does not implement regularization and pruning), and has more compatibility with a wider range of POMDPs.jl problems because it does not emulate the C++ code.
