# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Julia 1.8.2
#     language: julia
#     name: julia-1.8
# ---

using Distributed
using BenchmarkTools
using Random
using Test
rmprocs(workers())
addprocs(4)
nworkers()

# +
using Random, Plots, Sobol, BenchmarkTools

nx = 100
np = 10_000_000
xmin, xmax = -6, 6

xp = randn(np);

function serial_deposition( xp, xmin, xmax, nx )
    np = length(xp)
    rho = zeros(Float64, nx)
    for i in eachindex(xp)
        x_norm = (xp[i]-xmin) / (xmax - xmin) # normalized position
        ip = trunc(Int,  x_norm * nx) + 1 # nearest grid point
        rho[ip] += 1
    end

    rho ./ sum(rho .* (xmax - xmin) / nx)

end

@time serial_deposition(xp, xmin, xmax, nx);

# +
@everywhere begin
    
using SharedArrays

function distributed_deposition(xp, xmin, xmax, nx)
    np = length(xp)
    rho = SharedArray(zeros(Float64, nx))
    
    @sync @distributed (+) for i in 1:length(rho)
            x_norm = (xp[i]-xmin) / (xmax - xmin)
            ip = trunc(Int,  x_norm * nx)+1
            rho[ip] += 1
        end

    rho ./ sum(rho .* (xmax - xmin) / nx)

end
end

@time distributed_deposition(xp, xmin, xmax, nx);

# -

@time distributed_deposition(xp, xmin, xmax, nx);


workers()


