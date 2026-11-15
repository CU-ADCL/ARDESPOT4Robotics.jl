# Tests for DESPOTSolver(parallel=true): HashedSource, expand_parallel!, worker_source.
# Run the suite both single- and multi-threaded (e.g. julia -t 4); the parallel code paths
# are exercised either way, but races only have a chance to show up with threads.

using ARDESPOT4Robotics: get_rng, worker_source, build_despot, expand_action

# A POMDP whose generative model throws for one action, to check that an exception inside
# a worker task surfaces as the original exception (not a TaskFailedException).
struct ParTestThrowingPOMDP <: POMDP{Bool, Int, Bool} end
struct ParTestError <: Exception end
POMDPs.actions(::ParTestThrowingPOMDP) = 1:3
POMDPs.discount(::ParTestThrowingPOMDP) = 0.9
POMDPs.isterminal(::ParTestThrowingPOMDP, s::Bool) = false
POMDPs.initialstate(::ParTestThrowingPOMDP) = Deterministic(true)
function POMDPs.gen(::ParTestThrowingPOMDP, s::Bool, a::Int, rng)
    a == 2 && throw(ParTestError())
    return (sp=s, o=true, r=1.0)
end

# A state type that holds an array, like the robotics models this package is meant for:
# `isequal` compares the array field by identity, `deep_isequal` by value.
struct ParTestBoxedState
    v::Vector{Float64}
    n::Int
end

partest_same_tree(D1, D2) = ARDESPOT4Robotics.tree_mismatch(D1, D2)

@testset "parallel" begin
    @testset "HashedSource determinism" begin
        hs = HashedSource(3)
        draw(src, k, d) = (rng = get_rng(src, k, d); [rand(rng) for _ in 1:5])
        a = draw(hs, 7, 3)
        _ = draw(hs, 2, 5) # using other streams in between must not matter
        @test draw(hs, 7, 3) == a
        @test draw(hs, 7, 4) != a
        @test draw(hs, 8, 3) != a
        @test draw(HashedSource(4), 7, 3) != a
        hs2 = Random.seed!(HashedSource(), 3)
        @test draw(hs2, 7, 3) == a

        ws = worker_source(hs)
        @test ws !== hs && ws.rng !== hs.rng # private generator
        @test draw(ws, 7, 3) == a
        for k in 1:10, d in 0:5
            @test draw(worker_source(hs), k, d) == draw(hs, k, d)
        end

        rng = @inferred get_rng(hs, 1, 0)
        @test @inferred(rand(rng)) isa Float64
        @test @inferred(rand(rng, UInt64)) isa UInt64
        @test @inferred(randn(rng)) isa Float64
        @test rand(rng, 1:10) in 1:10
        @test rand(rng, Bool) isa Bool
    end

    @testset "MemorizingSource is rejected" begin
        @test_throws ArgumentError worker_source(MemorizingSource(10, 5, MersenneTwister(1)))
        sol = DESPOTSolver(bounds=(-20.0, 0.0), parallel=true,
                           random_source=MemorizingSource(500, 90, MersenneTwister(1)))
        @test_throws ArgumentError solve(sol, TigerPOMDP())
    end

    @testset "serial and parallel trees are identical" begin
        for (m, bds) in ((TigerPOMDP(), (-20.0, 0.0)),
                         (BabyPOMDP(), IndependentBounds(DefaultPolicyLB(FeedWhenCrying()), 0.0)))
            # (parallel, parallel_scenarios): serial reference, actions only, scenarios only, both
            trees = map(((false, false), (true, false), (false, true), (true, true))) do (par, parscen)
                sol = DESPOTSolver(bounds=bds, T_max=Inf, max_trials=100, rng=MersenneTwister(7),
                                   random_source=HashedSource(), parallel=par,
                                   parallel_scenarios=parscen, parallel_min_scenarios=1)
                pl = solve(sol, m)
                Random.seed!(pl.rs, 99)
                build_despot(pl, initialstate(m))
            end
            for i in 2:4
                @test partest_same_tree(trees[1], trees[i]) === nothing
            end
            @test length(trees[1].children) > 100 # the check means something
        end
    end

    @testset "worker exceptions surface unchanged" begin
        m = ParTestThrowingPOMDP()
        sol = DESPOTSolver(bounds=(0.0, 10.0), parallel=true, parallel_min_scenarios=1,
                           T_max=Inf, max_trials=5, bounds_warnings=false)
        pl = solve(sol, m)
        @test_throws ParTestError build_despot(pl, initialstate(m))
    end

    @testset "parallel DefaultPolicyLB" begin
        m = BabyPOMDP()
        pol = FeedWhenCrying()
        rng = MersenneTwister(3)
        scenarios = [i => rand(rng, initialstate(m)) for i in 1:40]
        for depth in (0, 5), seed in (3, 11)
            b = ScenarioBelief(scenarios, Random.seed!(HashedSource(), seed), depth, missing)
            ref = ARDESPOT4Robotics.lbound(DefaultPolicyLB(pol, 50, (m,x)->0.0), m, b)
            for ms in (1, 3, 20)
                bp = ScenarioBelief(scenarios, Random.seed!(HashedSource(), seed), depth, missing)
                @test ARDESPOT4Robotics.lbound(DefaultPolicyLB(pol, 50, (m,x)->0.0, true, ms), m, bp) == ref
            end
        end

        # whole tree with a parallel lower bound == fully serial tree
        trees = map((false, true)) do par
            lb = DefaultPolicyLB(pol; parallel=par, parallel_min_scenarios=1)
            sol = DESPOTSolver(bounds=IndependentBounds(lb, 0.0), T_max=Inf, max_trials=100,
                               rng=MersenneTwister(7), random_source=HashedSource(),
                               parallel=par, parallel_scenarios=par, parallel_min_scenarios=1)
            pl = solve(sol, m)
            Random.seed!(pl.rs, 99)
            build_despot(pl, initialstate(m))
        end
        @test partest_same_tree(trees...) === nothing

        # a MemorizingSource is rejected as soon as a branch would be handed to a task
        ms_src = Random.seed!(MemorizingSource(40, 50, MersenneTwister(1)), 3)
        bm = ScenarioBelief(scenarios, ms_src, 0, missing)
        @test_throws ArgumentError ARDESPOT4Robotics.lbound(DefaultPolicyLB(pol, 50, (m,x)->0.0, true, 1), m, bm)
    end

    @testset "deep_isequal and tree comparison" begin
        deep_isequal = ARDESPOT4Robotics.deep_isequal
        # the case plain isequal gets wrong: equal contents, different arrays
        a, b = ParTestBoxedState([1.0, 2.0], 3), ParTestBoxedState([1.0, 2.0], 3)
        @test !isequal(a, b)          # compares the vector field by identity
        @test deep_isequal(a, b)      # compares it by value
        @test deep_isequal([a, a], [b, b])
        @test !deep_isequal(a, ParTestBoxedState([1.0, 2.5], 3))
        @test !deep_isequal(a, ParTestBoxedState([1.0, 2.0], 4))
        @test !deep_isequal(a, ParTestBoxedState([1.0], 3))
        # scalars keep isequal's semantics, which is what bitwise comparison needs
        @test deep_isequal(NaN, NaN)
        @test !deep_isequal(0.0, -0.0)
        @test !deep_isequal(1, 1.0)   # different types
        @test deep_isequal((1, [2.0]), (1, [2.0]))
        @test deep_isequal(Dict(:a => [1.0]), Dict(:a => [1.0]))
        # unassigned slots, as in a tree's root observation
        u, w = Vector{Vector{Float64}}(undef, 2), Vector{Vector{Float64}}(undef, 2)
        u[2] = [1.0]; w[2] = [1.0]
        @test deep_isequal(u, w)
        w[1] = [0.0]
        @test !deep_isequal(u, w)

        m = BabyPOMDP()
        mktree(par) = begin
            sol = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FeedWhenCrying()), 0.0),
                               K=30, T_max=Inf, max_trials=30, random_source=HashedSource(),
                               parallel=par, parallel_scenarios=par, parallel_min_scenarios=1)
            pl = solve(sol, m)
            Random.seed!(pl.rs, 5)
            build_despot(pl, initialstate(m))
        end
        D1, D2 = mktree(false), mktree(true)
        @test ARDESPOT4Robotics.tree_mismatch(D1, D2) === nothing
        @test ARDESPOT4Robotics.trees_equal(D1, D2)
        D2.l[2] += 1.0 # a tree that really does differ is reported, with the field
        @test ARDESPOT4Robotics.tree_mismatch(D1, D2) === :l
        @test !ARDESPOT4Robotics.trees_equal(D1, D2)
    end

    @testset "expand_action is inferrable" begin
        m = BabyPOMDP()
        sol = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FeedWhenCrying()), 0.0),
                           parallel=true, parallel_min_scenarios=1, T_max=Inf, max_trials=5)
        pl = solve(sol, m)
        scenarios = [i => rand(MersenneTwister(1), initialstate(m)) for i in 1:10]
        @test @inferred(expand_action(pl, pl.rs, scenarios, 0, true)) isa ARDESPOT4Robotics.ActionExpansion{Bool, Bool, Bool}
    end
end
