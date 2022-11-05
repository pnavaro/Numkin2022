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
#     display_name: Julia 1.8.0
#     language: julia
#     name: julia-1.8
# ---

# # Gray-Scott system
#
# The reaction-diffusion system described here involves two generic chemical species, whose concentration at a given point in space is referred to by variables u and v. As the term implies, they react with each other, and they diffuse through the medium. Therefore the concentration of U and V at any given location changes with time and can differ from that at other locations.
#
# The overall behavior of the system is described by the following formula, two equations which describe three sources of increase and decrease for each of the two chemicals:
#
#
# $$
# \begin{array}{l}
# \displaystyle \frac{\partial u}{\partial t} = D_u \Delta u - uv^2 + F(1-u) \\
# \displaystyle \frac{\partial v}{\partial t} = D_v \Delta v + uv^2 - (F+k)v
# \end{array}
# $$
#
# The laplacian is computed with the following numerical scheme
#
# $$
# \Delta u_{i,j} \approx u_{i,j-1} + u_{i-1,j} -4u_{i,j} + u_{i+1, j} + u_{i, j+1}
# $$
#
# The classic Euler scheme is used to integrate the time derivative.
#
# $u$ is $1$ everywhere et $v$ is $0$ in the domain except in a square zone where $v = 0.25$ and $ u = 0.5$. This square located in the center of the domain is  $[0, 1]\times[0,1]$ with a size of $0.2$.
#

const Dᵤ = .1
const Dᵥ = .05
const F = 0.0545
const k = 0.062

function init(n)
 
    u = ones((n+2,n+2))
    v = zeros((n+2,n+2))
    
    x, y = LinRange(0, 1, n+2), LinRange(0, 1, n+2)

    for j in eachindex(y), i in eachindex(x)
        if (0.4<x[i]) && (x[i]<0.6) && (0.4<y[j]) && (y[j]<0.6)
    
            u[i,j] = 0.50
            v[i,j] = 0.25
        end
    end
        
    return u, v
end



using .Threads
a = zeros(12)
@threads for i = eachindex(a)
    a[i] = threadid()
end
println(a)

# +
using .Threads

function grayscott!( u, v, Δu, Δv)

    n = size(Δu,1)

    @threads for c = 1:n
        c1 = c + 1
        c2 = c + 2
        for r = 1:n
            r1 = r + 1
            r2 = r + 2
            @inbounds Δu[r,c] = u[r1,c2] + u[r1,c] + u[r2,c1] + u[r,c1] - 4*u[r1,c1]
            @inbounds Δv[r,c] = v[r1,c2] + v[r1,c] + v[r2,c1] + v[r,c1] - 4*v[r1,c1]
        end
    end

    @threads for c = 1:n
        c1 = c + 1
        for r = 1:n
            r1 = r + 1  
            @inbounds uvv = u[r1,c1]*v[r1,c1]*v[r1,c1]
            @inbounds u[r1,c1] +=  Dᵤ * Δu[r,c] - uvv + F*(1 - u[r1,c1])
            @inbounds v[r1,c1] +=  Dᵥ * Δv[r,c] + uvv - (F + k)*v[r1,c1]
        end
    end

end

function run_simulation( n = 300, maxiter = 10_000)
    u, v = init(n)
    Δu = zeros(n, n)
    Δv = zeros(n, n)
    for t in 1:maxiter
        grayscott!(u, v, Δu, Δv)
    end
    return u, v
end

@time u, v = run_simulation(500)

# +
using Plots
options = (aspect_ratio = :equal, axis = nothing, legend = :none, framestyle = :none)

heatmap(u; options...)

# +
using Plots

u, v = init(300)

options = (aspect_ratio = :equal, axis = nothing, legend = :none, framestyle = :none)

heatmap(v; options...)

# +
n = 300
u, v = init(n)
Δu = zeros(n, n)
Δv = zeros(n, n)

@gif for i in 1:1_000
    grayscott!(u, v, Δu, Δv)
    heatmap(v; options...)
end every 50
# -


# -*- coding: utf-8 -*-
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

# + [markdown] name="A slide " slideshow={"slide_type": "fragment"}
# ![ParallelStencil](./figures/parallelstencil.png)
#
#
# [https://github.com/omlins/ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)

# + [markdown] name="A slide " slideshow={"slide_type": "slide"}
# ## Setting up the environment
#
# Before we start, let's activate the environment:
# -

using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()
Pkg.status()

# And add the package(s) we will use

using Plots, CUDA, BenchmarkTools

# ### 1. Array programming on CPU

# +
function grayscott(; n=300, maxiter = 5000)

    # parameters
    Dᵤ, Dᵥ = .1, .05
    F, k = 0.0545, 0.062

    u = ones((n+2,n+2))
    v = zeros((n+2,n+2))

    x, y = LinRange(0, 1, n+2), LinRange(0, 1, n+2)

    for j in eachindex(y), i in eachindex(x)
        if (0.4<x[i]) && (x[i]<0.6) && (0.4<y[j]) && (y[j]<0.6)
            u[i,j] = 0.50
            v[i,j] = 0.25
        end
    end
    
    Δu = zeros(n, n)
    Δv = zeros(n, n)
    uvv = zeros(n,n)
    
    options = (aspect_ratio = :equal, axis = nothing, 
        legend = :none, framestyle = :none)

    @gif for t in 1:maxiter
        
            @views Δu .= u[1:end-2,2:end-1] .+ u[2:end-1,1:end-2] .+ u[2:end-1, 3:end] .+ u[3:end,2:end-1] .- 4*u[2:end-1, 2:end-1]
            @views Δv .= v[1:end-2,2:end-1] .+ v[2:end-1,1:end-2] .+ v[2:end-1, 3:end] .+ v[3:end,2:end-1] .- 4*v[2:end-1, 2:end-1]
            @views uvv .= u[2:end-1,2:end-1] .* v[2:end-1,2:end-1] .* v[2:end-1,2:end-1]
            @views u[2:end-1,2:end-1] .+=  Dᵤ .* Δu .- uvv .+ F * (1 .- u[2:end-1,2:end-1])
            @views v[2:end-1,2:end-1] .+=  Dᵥ .* Δv .+ uvv .- (F + k) .* v[2:end-1,2:end-1]
         
            heatmap(v; options...)

    end every 100

end

@time grayscott()

# +
using CUDA

function grayscott(; n=300, maxiter = 5000)

    Dᵤ, Dᵥ = .1, .05
    F, k = 0.0545, 0.062

    u = CUDA.ones((n+2,n+2))
    v = CUDA.zeros((n+2,n+2))

    x, y = LinRange(0, 1, n+2), LinRange(0, 1, n+2)

    for j in eachindex(y), i in eachindex(x)
        if (0.4<x[i]) && (x[i]<0.6) && (0.4<y[j]) && (y[j]<0.6)
            u[i,j] = 0.50
            v[i,j] = 0.25
        end
    end
    
    Δu = CUDA.zeros(n, n)
    Δv = CUDA.zeros(n, n)
    uvv = CUDA.zeros(n,n)
    
    options = (aspect_ratio = :equal, axis = nothing, legend = :none, framestyle = :none)

    @gif for t in 1:maxiter
    
        @views Δu .= u[1:end-2,2:end-1] .+ u[2:end-1,1:end-2] .+ u[2:end-1, 3:end] .+ u[3:end,2:end-1] .- 4*u[2:end-1, 2:end-1]
        @views Δv .= v[1:end-2,2:end-1] .+ v[2:end-1,1:end-2] .+ v[2:end-1, 3:end] .+ v[3:end,2:end-1] .- 4*v[2:end-1, 2:end-1]
        @views uvv .= u[2:end-1,2:end-1] .* v[2:end-1,2:end-1] .* v[2:end-1,2:end-1]
        @views u[2:end-1,2:end-1] .+=  Dᵤ .* Δu .- uvv .+ F * (1 .- u[2:end-1,2:end-1])
        @views v[2:end-1,2:end-1] .+=  Dᵥ .* Δv .+ uvv .- (F + k) .* v[2:end-1,2:end-1]
         
        heatmap(Array(v); options...)

    end every 100


end

@time grayscott()
# -

# ### CPU vs GPU array programming performance
#
# For this, we can isolate the physics computation into a function that we will evaluate for benchmarking

# +
function update_uv!( u, v, Δu, Δv, uvv, Dᵤ, Dᵥ, F, k)
    
    @views Δu .= u[1:end-2,2:end-1] .+ u[2:end-1,1:end-2] .+ u[2:end-1, 3:end] .+ u[3:end,2:end-1] .- 4*u[2:end-1, 2:end-1]
    @views Δv .= v[1:end-2,2:end-1] .+ v[2:end-1,1:end-2] .+ v[2:end-1, 3:end] .+ v[3:end,2:end-1] .- 4*v[2:end-1, 2:end-1]
       
    @views uvv .= u[2:end-1,2:end-1] .* v[2:end-1,2:end-1] .* v[2:end-1,2:end-1]
    @views u[2:end-1,2:end-1] .+=  Dᵤ .* Δu .- uvv .+ F * (1 .- u[2:end-1,2:end-1])
    @views v[2:end-1,2:end-1] .+=  Dᵥ .* Δv .+ uvv .- (F + k) .* v[2:end-1,2:end-1]
    
end
# -

Dᵤ, Dᵥ = .1, .05
F, k = 0.0545, 0.062
n = 2048
u = rand(Float64, n+2, n+2)
v = rand(Float64, n+2, n+2)
Δu = zeros(n, n)
Δv = zeros(n, n)
uvv = zeros(n,n)
t_it = @belapsed begin update_uv!($u, $v, $Δu, $Δv, $uvv, $Dᵤ, $Dᵥ, $F, $k); end
T_eff_cpu = 23 * 1e-9 * n^2 * sizeof(Float64) / t_it
println("T_eff = $(T_eff_cpu) GiB/s using CPU array programming")

# Let's repeat the experiment using the GPU

Dᵤ, Dᵥ = .1, .05
F, k = 0.0545, 0.062
n = 4096
u = CUDA.rand(Float64, n+2, n+2)
v = CUDA.rand(Float64, n+2, n+2)
Δu = CUDA.zeros(n, n)
Δv = CUDA.zeros(n, n)
uvv = CUDA.zeros(n,n)
t_it = @belapsed begin update_uv!($u, $v, $Δu, $Δv, $uvv, $Dᵤ, $Dᵥ, $F, $k); synchronize(); end
T_eff_gpu = 23 * 1e-9 * n^2 * sizeof(Float64) / t_it
println("T_eff = $(T_eff_gpu) GiB/s using GPU array programming")

# We see some improvement from performing the computations on the GPU, however, $T_\mathrm{eff}$ is not yet close to GPU's peak memory bandwidth
#
# How to improve? Now it's time for ParallelStencil
#
# ### 3. Kernel programming using ParallelStencil
#
# In this first example, we'll use the `FiniteDifferences` module to enable math-close notation and the `CUDA` "backend". We could simply switch the backend to `Threads` if we want the same code to run on multiple CPU threads using Julia's native multi-threading capabilities. But for time issues, we won't investigate this today.

USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(0)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

Dᵤ, Dᵥ = .1, .05
F, k = 0.0545, 0.062
n = 2048
u = @rand(n+2, n+2)
v = @rand(n+2, n+2)
Δu = @zeros( n, n)
Δv = @zeros(n, n)
uvv = @zeros( n,n);

# Using math-close notations from the `FiniteDifferences2D` module, our update kernel can be re-written as following:

@parallel function update_uv_ps!( u, v, Δu, Δv, uvv, Dᵤ, Dᵥ, F, k)
    
    @all(Δu) = @d2_xi(u) + @d2_yi(u)
    @all(Δv) = @d2_xi(u) + @d2_yi(v)
       
    @all(uvv) = @inn(u) * @inn(v) * @inn(v)
    @inn(u) = @inn(u) + Dᵤ * @all(Δu) - @all(uvv) + F * (1 - @inn(u))
    @inn(v) = @inn(v) + Dᵥ * @all(Δv) + @all(uvv) - (F + k) * @inn(v)
    return
end

?@inn

?@d2_xi

t_it = @belapsed begin @parallel update_uv_ps!($u, $v, $Δu, $Δv, $uvv, $Dᵤ, $Dᵥ, $F, $k);  end
T_eff_ps = 23 * 1e-9 * n^2 * sizeof(Float64)/t_it
println("T_eff = $(T_eff_ps) GiB/s using ParallelStencil on GPU and the FiniteDifferences2D module")

# And sample again our performance on the GPU using ParallelStencil this time:

# - Julia and ParallelStencil permit to solve the two-language problem
# - ParallelStencil and Julia GPU permit to exploit close to GPUs' peak memory throughput
#
# ![parallelstencil](./figures/parallelstencil.png)
#
# More hungry? Check out [https://github.com/omlins/ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) and the [miniapps](https://github.com/omlins/ParallelStencil.jl#concise-singlemulti-xpu-miniapps)
#
# Advance features not covered today:
# - Using shared-memory and 2.5D blocking (see [here](https://github.com/omlins/ParallelStencil.jl#support-for-architecture-agnostic-low-level-kernel-programming) with [example](https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion2D_shmem_novis.jl))
# - Multi-GPU with communication-computation overlap combining ParallelStencil and [ImplicitGlobalGrid]()
# - Stay tuned, AMDGPU support is coming soon 🚀
#
# _contact: luraess@ethz.ch_
