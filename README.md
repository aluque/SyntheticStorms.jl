# Generate synthetic storms
This code is used to sample synthetic flashes with simple distributions to test ML algorithms.

## Install
To install the code, first install [julia](https://julialang.org/). Then, from a julia prompt:
```julia
julia> using Pkg; Pkg.add(url="https://github.com/aluque/SyntheticStorms.jl")
```

## Run
To create the data for synthetic flashes in `"output/folder"`:
```julia
julia> using SyntheticStorms
julia> SyntheticStorms.main(;folder="output/folder");
```


