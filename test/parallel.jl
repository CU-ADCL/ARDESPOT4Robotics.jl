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

# Field-by-field tree comparison; D.obs[1] (the root) is never assigned.
function partest_same_tree(D1, D2)
    for f in fieldnames(typeof(D1))
        x, y = getfield(D1, f), getfield(D2, f)
        if f == :obs
            (length(x) == length(y) && all(isequal(x[i], y[i]) for i in 2:length(x))) || return f
        else
            isequal(x, y) || return f
        end
    end
    return nothing
end

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
    @testset "expand_action is inferrable" begin
        m = BabyPOMDP()
        sol = DESPOTSolver(bounds=IndependentBounds(DefaultPolicyLB(FeedWhenCrying()), 0.0),
                           parallel=true, parallel_min_scenarios=1, T_max=Inf, max_trials=5)
        pl = solve(sol, m)
        scenarios = [i => rand(MersenneTwister(1), initialstate(m)) for i in 1:10]
        @test @inferred(expand_action(pl, pl.rs, scenarios, 0, true)) isa ARDESPOT4Robotics.ActionExpansion{Bool, Bool, Bool}
    end
end
