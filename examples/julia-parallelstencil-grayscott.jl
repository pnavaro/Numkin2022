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
