# The SplitMix64 output function: a bijection on UInt64 that mixes all the bits.
@inline function mix64(x::UInt64)
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

"""
    SplitMixRNG(s)

The SplitMix64 generator (Steele, Lea & Flood 2014): the `i`-th output is `mix64(s + i*0x9e3779b97f4a7c15)`.

Unlike generators that carry their stream in a large evolving state, its output is a pure function of the counter `s`, so `get_rng(::HashedSource, scenario, depth)` only has to store a hash of `(seed, scenario, depth)` in it. That makes it a few times cheaper per `get_rng` call than reseeding a `Xoshiro` or `MersenneTwister`.
"""
mutable struct SplitMixRNG <: AbstractRNG
    s::UInt64
end

@inline next(r::SplitMixRNG) = mix64(r.s += 0x9e3779b97f4a7c15)

Random.rng_native_52(::SplitMixRNG) = UInt64
rand(r::SplitMixRNG, ::SamplerType{UInt64}) = next(r)
rand(r::SplitMixRNG, ::SamplerType{Int64}) = next(r) % Int64
rand(r::SplitMixRNG, ::SamplerTrivial{UInt52Raw{UInt64}}) = next(r) # "Raw": callers ignore the high 12 bits
rand(r::SplitMixRNG, T::SamplerUnion(Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32)) = next(r) % T[]

"""
    HashedSource(seed=0)

A `DESPOTRandomSource` in which the random numbers for a scenario at a depth are a pure function of `(seed, scenario, depth)`.

`MemorizingSource` hands out one shared, lazily filled stream per `(scenario, depth)`, so it cannot be used by two tasks at once. `HashedSource` keeps no shared table: `get_rng` sets its private generator to a counter derived from a hash of `(seed, scenario, depth)`, and `worker_source` makes a copy with its own private generator that yields exactly the same numbers. This is what makes `DESPOTSolver(parallel=true)` possible, and it is the default random source in that mode.

It can also be used with `parallel=false`; the numbers differ from those of a `MemorizingSource` with the same solver `rng`, but the algorithm is the same.
"""
mutable struct HashedSource <: DESPOTRandomSource
    seed::UInt64
    rng::SplitMixRNG # its counter is set by every get_rng call, so it may only be used by one task at a time
end

HashedSource(seed::Integer=0) = HashedSource(seed % UInt64, SplitMixRNG(0))

function Random.seed!(s::HashedSource, seed::Integer)
    s.seed = seed % UInt64
    return s
end

function get_rng(s::HashedSource, scenario::Int, depth::Int)
    # Distinct (scenario, depth) pairs give distinct counters as long as both fit in 32 bits,
    # because mix64 is a bijection.
    s.rng.s = mix64(s.seed + mix64((scenario % UInt64) << 32 ⊻ (depth % UInt64)))
    return s.rng
end

"""
    worker_source(rs::DESPOTRandomSource)

Return a random source that gives exactly the same numbers as `rs` for every scenario and depth, and that another task can use while `rs` itself is in use.

A custom random source has to implement this to be usable with `DESPOTSolver(parallel=true)`.
"""
function worker_source(rs::DESPOTRandomSource)
    throw(ArgumentError("""
        DESPOTSolver(parallel=true) cannot use a $(nameof(typeof(rs))), because it shares mutable random streams between the actions of a node.

        Use `random_source=HashedSource()` (the default when `parallel=true`) instead.
        """))
end

worker_source(s::HashedSource) = HashedSource(s.seed, SplitMixRNG(0))
