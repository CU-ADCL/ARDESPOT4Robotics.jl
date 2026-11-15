# Parallel version of the loop over actions in `expand!`.
#
# For each action, `expand!` steps every scenario of the belief node once, groups the results
# by observation, and computes the initial bounds of the resulting child nodes. None of that
# depends on what the other actions produce, so `expand_action` does it without touching the
# tree, one task per action, and `insert_action!` then adds the results to the tree in action
# order on the calling task. The node numbering and every stored value are therefore the same
# as in the serial loop.

struct ActionExpansion{S,A,O}
    a::A
    rsum::Float64
    obs::Vector{O}
    scenarios::Vector{Vector{Pair{Int,S}}} # scenarios[i] reached obs[i]
    L_0::Vector{Float64}
    U_0::Vector{Float64}
end

function expand_parallel!(D::DESPOT{S,A,O}, b::Int, p::DESPOTPlanner) where {S,A,O}
    belief = get_belief(D, b, p.rs)
    as = actions(p.pomdp, belief)
    isempty(as) && return nothing
    scenarios = D.scenarios[b]
    depth = D.Delta[b]

    if !p.sol.parallel
        # parallel_scenarios only: actions stay in order on this task, each expansion is
        # internally parallel over scenarios and child bounds.
        for a in as
            insert_action!(D, b, expand_one_action(p, p.rs, scenarios, depth, a), p)
        end
        return nothing
    end

    # The first action runs on this task with the planner's own random source. Every other
    # action gets a task and a private copy of the random source.
    tasks = map(Iterators.drop(as, 1)) do a
        rs = worker_source(p.rs)
        Threads.@spawn expand_one_action(p, rs, scenarios, depth, a)
    end
    local first_expansion::ActionExpansion{S,A,O}
    try
        first_expansion = expand_one_action(p, p.rs, scenarios, depth, first(as))
    finally
        # never return while a task is still running, even if this action failed
        for t in tasks
            try
                wait(t)
            catch ex
                ex isa TaskFailedException || rethrow()
            end
        end
    end

    insert_action!(D, b, first_expansion, p)
    for t in tasks
        insert_action!(D, b, fetch_result(t)::ActionExpansion{S,A,O}, p)
    end
    return nothing
end

# One action's expansion, possibly itself parallel over scenarios (see the bottom of this file).
function expand_one_action(p::DESPOTPlanner, rs::DESPOTRandomSource,
                           scenarios::Vector{Pair{Int,S}}, depth::Int, a) where S
    if p.sol.parallel_scenarios
        return expand_action_parscen(p, rs, scenarios, depth, a)
    else
        return expand_action(p, rs, scenarios, depth, a)
    end
end

# Like `fetch`, but if the task failed, throw the exception it failed with (as the serial loop
# would have) instead of a TaskFailedException wrapping it.
function fetch_result(t::Task)
    try
        return fetch(t)
    catch ex
        ex isa TaskFailedException ? throw(ex.task.result) : rethrow()
    end
end

function expand_action(p::DESPOTPlanner, rs::DESPOTRandomSource,
                       scenarios::Vector{Pair{Int,S}}, depth::Int, a) where S
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)
    odict = Dict{O, Int}()
    obs = O[]
    child_scenarios = Vector{Pair{Int,S}}[]
    rsum = 0.0

    for scen in scenarios
        rng = get_rng(rs, first(scen), depth)
        s = last(scen)
        if !isterminal(p.pomdp, s)
            sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, rng)
            rsum += r
            i = get(odict, o, 0)
            if i == 0
                push!(child_scenarios, Vector{Pair{Int, S}}())
                i = length(child_scenarios)
                odict[o] = i
                push!(obs, o)
            end
            push!(child_scenarios[i], first(scen)=>sp)
        end
    end

    L_0s = Vector{Float64}(undef, length(obs))
    U_0s = Vector{Float64}(undef, length(obs))
    for i in eachindex(obs)
        scenario_belief = ScenarioBelief(child_scenarios[i], rs, depth+1, obs[i])
        L_0, U_0 = bounds(p.bounds, p.pomdp, scenario_belief)

        if p.sol.bounds_warnings
            bounds_sanity_check(p.pomdp, scenario_belief, L_0, U_0)
        end

        L_0s[i] = L_0
        U_0s[i] = U_0
    end

    return ActionExpansion{S,A,O}(a, rsum, obs, child_scenarios, L_0s, U_0s)
end

function insert_action!(D::DESPOT, b::Int, x::ActionExpansion, p::DESPOTPlanner)
    nbps = length(x.obs)
    first_bp = length(D.scenarios) + 1
    append!(D.scenarios, x.scenarios)

    push!(D.ba_children, collect(first_bp:first_bp+nbps-1))
    ba = length(D.ba_children)
    push!(D.ba_action, x.a)
    push!(D.children[b], ba)
    rho = x.rsum*discount(p.pomdp)^D.Delta[b]/p.sol.K - p.sol.lambda
    push!(D.ba_rho, rho)
    push!(D.ba_Rsum, x.rsum)

    resize!(D, length(D.children) + nbps)
    for i in 1:nbps
        bp = first_bp + i - 1

        D.obs[bp] = x.obs[i]
        D.children[bp] = Int[]
        D.parent_b[bp] = b
        D.parent[bp] = ba
        D.Delta[bp] = D.Delta[b]+1

        L_0 = x.L_0[i]
        U_0 = x.U_0[i]
        l_0 = length(D.scenarios[bp])/p.sol.K * discount(p.pomdp)^D.Delta[bp] * L_0
        mu_0 = max(l_0, length(D.scenarios[bp])/p.sol.K * discount(p.pomdp)^D.Delta[bp] * U_0 - p.sol.lambda)

        D.mu[bp] = mu_0
        D.U[bp] = U_0
        D.l[bp] = l_0
        D.l_0[bp] = l_0
    end

    push!(D.ba_mu, D.ba_rho[ba] + sum(D.mu[bp] for bp in D.ba_children[ba]))
    return nothing
end

# ---------------------------------------------------------------------------------------
# Scenario-level parallelism (DESPOTSolver(parallel_scenarios=true), independent of
# `parallel`): inside one action's expansion, phase 1 steps the scenarios in contiguous
# chunks (one task each, own worker_source), and phase 2 computes the child nodes' initial
# bounds in contiguous child blocks of roughly equal scenario mass. The tree stays exactly
# the serial tree:
#   - every random stream is a pure function of (seed, scenario, depth), so results do not
#     depend on which task runs what;
#   - chunks are contiguous scenario ranges merged in order, so the observation
#     first-appearance order and the within-child scenario order match the serial loop;
#   - per-scenario rewards are kept and summed in scenario order during the merge (chunk
#     subtotals would change the floating-point summation order and the last ulp of rsum);
#   - each child's bound is one bounds() call, exactly as in the serial loop (the scenarios
#     of a single child are never split: the rollout policy acts on the group belief, so
#     splitting them would change the algorithm).

struct ScenarioStepChunk{S,O}
    obs::Vector{O}                          # first-appearance order within the chunk
    groups::Vector{Vector{Pair{Int,S}}}     # groups[i] reached obs[i]
    rewards::Vector{Float64}                # per-scenario rewards, in scenario order (skipping terminals)
end

function step_scenarios(p::DESPOTPlanner, rs::DESPOTRandomSource,
                        scenarios::Vector{Pair{Int,S}}, depth::Int, a, range::UnitRange{Int}) where S
    O = obstype(p.pomdp)
    odict = Dict{O,Int}()
    obs = O[]
    groups = Vector{Pair{Int,S}}[]
    rewards = Float64[]
    for j in range
        scen = scenarios[j]
        rng = get_rng(rs, first(scen), depth)
        s = last(scen)
        if !isterminal(p.pomdp, s)
            sp, o, r = @gen(:sp, :o, :r)(p.pomdp, s, a, rng)
            push!(rewards, r)
            i = get(odict, o, 0)
            if i == 0
                push!(groups, Vector{Pair{Int,S}}())
                i = length(groups)
                odict[o] = i
                push!(obs, o)
            end
            push!(groups[i], first(scen)=>sp)
        end
    end
    return ScenarioStepChunk{S,O}(obs, groups, rewards)
end
# (field renamed: merge below reads c.rewards)

function bounds_range!(L_0s::Vector{Float64}, U_0s::Vector{Float64}, p::DESPOTPlanner,
                       rs::DESPOTRandomSource, child_scenarios::Vector{Vector{Pair{Int,S}}},
                       obs, depth::Int, range::UnitRange{Int}) where S
    for i in range
        scenario_belief = ScenarioBelief(child_scenarios[i], rs, depth+1, obs[i])
        L_0, U_0 = bounds(p.bounds, p.pomdp, scenario_belief)
        if p.sol.bounds_warnings
            bounds_sanity_check(p.pomdp, scenario_belief, L_0, U_0)
        end
        L_0s[i] = L_0
        U_0s[i] = U_0
    end
    return nothing
end

# split 1:n into at most k contiguous nearly-equal ranges (first range returned first)
function chunk_ranges(n::Int, k::Int)
    k = max(min(k, n), 1)
    q, r = divrem(n, k)
    ranges = Vector{UnitRange{Int}}(undef, k)
    lo = 1
    for i in 1:k
        hi = lo + q - 1 + (i <= r ? 1 : 0)
        ranges[i] = lo:hi
        lo = hi + 1
    end
    return ranges
end

# split the children (in order) into at most k contiguous blocks of roughly equal
# total scenario count (bound cost is roughly proportional to it)
function mass_ranges(child_scenarios::Vector{<:Vector}, k::Int)
    n = length(child_scenarios)
    k = max(min(k, n), 1)
    total = sum(length, child_scenarios)
    ranges = UnitRange{Int}[]
    lo = 1
    acc = 0
    for i in 1:n
        acc += length(child_scenarios[i])
        # close block number length(ranges)+1 once its share of the mass is reached,
        # but keep at least one child for the final block
        if length(ranges) < k - 1 && i < n && acc*k >= total*(length(ranges)+1)
            push!(ranges, lo:i)
            lo = i + 1
        end
    end
    push!(ranges, lo:n)
    return ranges
end

function expand_action_parscen(p::DESPOTPlanner, rs::DESPOTRandomSource,
                               scenarios::Vector{Pair{Int,S}}, depth::Int, a) where S
    A = actiontype(p.pomdp)
    O = obstype(p.pomdp)
    K = length(scenarios)
    # every chunk gets at least parallel_min_scenarios scenarios
    nchunks = min(Threads.nthreads(), max(1, K ÷ max(p.sol.parallel_min_scenarios, 1)))

    # phase 1: step the scenarios, chunked; first chunk on this task
    ranges = chunk_ranges(K, nchunks)
    step_tasks = map(ranges[2:end]) do r
        ws = worker_source(rs)
        Threads.@spawn step_scenarios(p, ws, scenarios, depth, a, r)
    end
    local first_chunk::ScenarioStepChunk{S,O}
    try
        first_chunk = step_scenarios(p, rs, scenarios, depth, a, ranges[1])
    finally
        for t in step_tasks
            try
                wait(t)
            catch ex
                ex isa TaskFailedException || rethrow()
            end
        end
    end
    chunks = ScenarioStepChunk{S,O}[first_chunk]
    for t in step_tasks
        push!(chunks, fetch_result(t)::ScenarioStepChunk{S,O})
    end

    # merge in scenario order (chunks are contiguous ranges in order)
    odict = Dict{O,Int}()
    obs = O[]
    child_scenarios = Vector{Pair{Int,S}}[]
    rsum = 0.0
    for c in chunks
        for r in c.rewards
            rsum += r
        end
        for (o, grp) in zip(c.obs, c.groups)
            i = get(odict, o, 0)
            if i == 0
                push!(child_scenarios, grp) # take ownership of the chunk's vector
                i = length(child_scenarios)
                odict[o] = i
                push!(obs, o)
            else
                append!(child_scenarios[i], grp)
            end
        end
    end

    # phase 2: initial bounds of the children, chunked by scenario mass
    nbps = length(obs)
    L_0s = Vector{Float64}(undef, nbps)
    U_0s = Vector{Float64}(undef, nbps)
    if nbps > 0
        branges = mass_ranges(child_scenarios, min(Threads.nthreads(), nbps))
        btasks = map(branges[2:end]) do r
            ws = worker_source(rs)
            Threads.@spawn bounds_range!(L_0s, U_0s, p, ws, child_scenarios, obs, depth, r)
        end
        try
            bounds_range!(L_0s, U_0s, p, rs, child_scenarios, obs, depth, branges[1])
        finally
            for t in btasks
                try
                    wait(t)
                catch ex
                    ex isa TaskFailedException || rethrow()
                end
            end
        end
        for t in btasks
            fetch_result(t)
        end
    end

    return ActionExpansion{S,A,O}(a, rsum, obs, child_scenarios, L_0s, U_0s)
end

# ---------------------------------------------------------------------------------------
# Parallel default-policy simulation (DefaultPolicyLB(policy, parallel=true)).
#
# branching_sim steps the whole scenario group once, splits it by observation, and recurses
# into each branch. The branches are independent: a branch's rollout actions depend only on
# its own belief, and every random stream is a pure function of (seed, scenario, depth), so
# the sub-simulations may run on separate tasks. The returned value is exactly the serial
# one: r_sum is accumulated in scenario order on the calling task, every branch value is the
# same number the serial recursion computes (branches below the threshold literally run the
# serial code), and the branch values are summed in observation first-appearance order, just
# like the serial loop. The one thing that must stay serial is the pre-split simulation of a
# group: the policy acts on the whole group belief, so splitting it would change the actions.

sim_branch_parallel(pomdp::POMDP, policy::Policy, bp::ScenarioBelief, steps::Integer, fval, min_scenarios::Int) =
    length(bp.scenarios) == 1 ? rollout(pomdp, policy, bp, steps, fval) :
                                branching_sim_parallel(pomdp, policy, bp, steps, fval, min_scenarios)

function branching_sim_parallel(pomdp::POMDP, policy::Policy, b::ScenarioBelief, steps::Integer, fval, min_scenarios::Int)
    S = statetype(pomdp)
    O = obstype(pomdp)
    odict = Dict{O, Vector{Pair{Int, S}}}()
    olist = O[]

    if steps <= 0
        return length(b.scenarios)*fval(pomdp, b)
    end

    a = action(policy, b)

    r_sum = 0.0
    for (k, s) in b.scenarios
        if !isterminal(pomdp, s)
            rng = get_rng(b.random_source, k, b.depth)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)

            if haskey(odict, o)
                push!(odict[o], k=>sp)
            else
                odict[o] = [k=>sp]
                push!(olist, o)
            end

            r_sum += r
        end
    end

    # Branches with at least min_scenarios scenarios get their own task (and a private copy
    # of the random source); the rest are simulated inline with the serial code while the
    # tasks run. Values are gathered and summed in olist order, like the serial loop.
    n = length(olist)
    results = Vector{Float64}(undef, n)
    tasks = Task[]
    task_idx = Int[]
    try
        for i in 1:n
            scenarios = odict[olist[i]]
            if length(scenarios) >= min_scenarios
                ws = worker_source(b.random_source)
                bp = ScenarioBelief(scenarios, ws, b.depth+1, olist[i])
                push!(tasks, Threads.@spawn sim_branch_parallel(pomdp, policy, bp, steps-1, fval, min_scenarios))
                push!(task_idx, i)
            end
        end
        for i in 1:n
            scenarios = odict[olist[i]]
            if length(scenarios) < min_scenarios
                bp = ScenarioBelief(scenarios, b.random_source, b.depth+1, olist[i])
                if length(scenarios) == 1
                    results[i] = rollout(pomdp, policy, bp, steps-1, fval)
                else
                    results[i] = branching_sim(pomdp, policy, bp, steps-1, fval)
                end
            end
        end
    finally
        # never return while a task is still running, even if the inline work failed
        for t in tasks
            try
                wait(t)
            catch ex
                ex isa TaskFailedException || rethrow()
            end
        end
    end
    for (t, i) in zip(tasks, task_idx)
        results[i] = fetch_result(t)::Float64
    end

    next_r = 0.0
    for i in 1:n
        next_r += results[i]
    end

    return r_sum + discount(pomdp)*next_r
end
