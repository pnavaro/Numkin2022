# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
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

# +
using Random, Plots, Sobol, BenchmarkTools

nx = 100
np = 1_000_000
xmin, xmax = -6, 6

xp = randn(np);

histogram(xp, normalize=true, bins=nx)
plot!(x-> (exp(-x^2/2))/sqrt(2π), -6, 6)

# +
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

plot(LinRange(xmin, xmax, nx), serial_deposition(xp, xmin , xmax, nx), lw = 2)
plot!(x-> (exp(-x^2/2))/sqrt(2π), -6, 6)

# +
using .Threads

function parallel_deposition( xp, xmin, xmax, nx )

    Lx = xmax - xmin
    rho = zeros(Float64, nx)
    ntid = nthreads()
    rho_local = [zero(rho) for _ in 1:ntid]
    chunks = Iterators.partition(1:np, np÷ntid)

    @sync for chunk in chunks
        @spawn begin
            tid = threadid()
            for i in chunk
                x_norm = (xp[i]-xmin) / Lx
                ip = trunc(Int,  x_norm * nx)+1
                rho_local[tid][ip] += 1
            end
        end
    end

    rho .= reduce(+,rho_local)

    rho ./ sum(rho .* (xmax - xmin) / nx)

end

plot(LinRange(xmin, xmax, nx), parallel_deposition(xp, xmin, xmax, nx), lw = 2)
plot!(x-> (exp(-x^2/2))/sqrt(2π), -6, 6)
# -

@btime serial_deposition($xp, $xmin, $xmax, $nx);

@btime parallel_deposition($xp, $xmin, $xmax, $nx);


