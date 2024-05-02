"""
Code to generate synthetic storms to test prediction models.
"""
module SyntheticStorms
using StaticArrays
using Random
using Distributions
using CSV
using DataFrames
using Tables
using StatsBase
using HDF5
using FileIO
using OrderedCollections

struct Flash{T}
    t::T
    r::SVector{2, T}
end

struct Storm{T}
    "Flash peak time"
    tpeak::T

    "Storm duration as stdev of times"
    tsigma::T
    
    "Velocity"
    v::SVector{2, T}

    "Flash peak location"
    rpeak::SVector{2, T}

    "Storm extension as std dev of flash locations"
    rsigma::T
    
    "Expected number of total flashes"
    n::T
end

Base.@kwdef struct StormDistribution{T1, T2, T3, T4, T5, T6}
    "Distribution of peak times"
    tpeak::T1

    "Distribution of durations"
    tsigma::T2

    "Distribution of each v component"
    v::T3

    "Distribution of each rpeak component"
    rpeak::T4

    "Distribution of storm extensions"
    rsigma::T5
    
    "Distribution of n"
    n::T6
end

"""
Sample a single storm from the distribution of storms in `dist`.
"""
function Random.rand(rng::AbstractRNG,
                     d::Random.SamplerTrivial{<:StormDistribution})
    tpeak = rand(rng, d[].tpeak)
    tsigma = rand(rng, d[].tsigma)
    v = SA[rand(rng, d[].v), rand(rng, d[].v)]
    rpeak = SA[rand(rng, d[].rpeak), rand(rng, d[].rpeak)]
    rsigma = rand(rng, d[].rsigma)
    n = rand(rng, d[].n)
    
    return Storm(tpeak, tsigma, v, rpeak, rsigma, n)
end


"""
Sample a single flash from a storm.
"""
function Random.rand(rng::AbstractRNG, d::Random.SamplerTrivial{Storm{T}}) where T
    (;tpeak, tsigma, v, rpeak, rsigma) = d[]
    dt = tsigma * randn(rng)
    t = tpeak + dt
    rcentroid = rpeak .+ v .* dt
    r = rcentroid .+ rsigma .* @SVector(randn(rng, 2))

    return Flash(t, r)
end

"""
Save all flashes in a csv file ignoring everything outside the rectangle (0, `a`)^2 and for
t not in (0, tend).
"""
function tocsv(fname, v::Vector{<:Flash}; a=100.0, tend=100.0)
    v1 = filter(f -> (0 < f.r[1] < a && 0 < f.r[2] < a && 0 < f.t < tend), v)
    sort!(v1, by=f->f.t)
    df = DataFrame(x=[f.r[1] for f in v1],
                   y=[f.r[2] for f in v1],
                   t=[f.t for f in v1])
    @info "number of generated flashes: $(length(v1))"
    
    CSV.write(fname, df; compress=true)
end

"""
Generate flashes from the storm distribution and save them in `outfile`. `tend` is the final sampling
time and `freq` the number of storms per unit time.  There will be about `tend` * `freq` storms.
"""
function generate(outfile, tend, freq=10)
    dist = StormDistribution(
        # Longer than saving interval to reduce boundary effects
        tpeak  = Uniform(0, tend * 1.2),
        tsigma = LogNormal(log(0.5), log(2)),
        v      = Normal(0.0, 20.0),

        # Larger than saved space to reduce boundary effects
        rpeak  = Uniform(0, 200.0),
        rsigma = LogNormal(log(2.0), log(5.0)),
        n      = LogNormal(log(1000.0), log(10.0)))

    # First we sample storms from the storm distribution
    storms = rand(dist, round(Int, Int(rand(Poisson(tend * freq)))))

    # Then for each storm we sample flashes
    flashes = Flash{Float64}[]
    for storm in storms
        nflashes = Int(rand(Poisson(storm.n)))
        append!(flashes, rand(storm, nflashes))
    end
    
    # Save the resulting flashes
    tocsv(outfile, flashes; a=100, tend)
end

function main(;folder=expanduser("~/data/glm/synthetic/"))
    # Training data
    outfile = joinpath(folder, "synstorm_train.csv.gz")
    tend = 2000.0
    generate(outfile, tend)

    # Validation data
    outfile = joinpath(folder, "synstorm_valid.csv.gz")
    tend = 200.0
    generate(outfile, tend)    

    # Test data
    outfile = joinpath(folder, "synstorm_test.csv.gz")
    tend = 200.0
    generate(outfile, tend)    
end

function make_histograms(;folder=expanduser("~/data/glm/synthetic/"))
    # Bins per unit time
    bput = 5
    h = histogram(joinpath(folder, "synstorm_train.csv.gz"); tend=2000.0, nt=bput * 2000)
    save(joinpath(folder, "synstorm_train.h5"), OrderedDict("flashes" => h.weights))

    h = histogram(joinpath(folder, "synstorm_valid.csv.gz"); tend=200.0, nt=bput * 200)
    save(joinpath(folder, "synstorm_valid.h5"), OrderedDict("flashes" => h.weights))

    h = histogram(joinpath(folder, "synstorm_test.csv.gz"); tend=200.0, nt=bput * 200)
    save(joinpath(folder, "synstorm_test.h5"), OrderedDict("flashes" => h.weights))    
end


"""
Create a histogram of flashes read from file in `fname`. `a` is the lateral size of the square domain,
`tend` is the upper bound of time, `nx` and `nt` are respectively the number of bins in the x/y and t
directions.
"""
function histogram(fname; a=100.0, tend=100.0, nx=64, nt=100)
    edges = (LinRange(0, a, nx + 1), LinRange(0, a, nx + 1), LinRange(0, tend, nt + 1))
    
    df = CSV.read(fname, DataFrame)
    h = fit(Histogram, (df.x, df.y, df.t), edges)
    return h
end


end # module SyntheticStorms

if abspath(PROGRAM_FILE) == @__FILE__
    SyntheticStorms.main()
end
