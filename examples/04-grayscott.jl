using BenchmarkTools
using Plots
using ProgressMeter

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


function grayscott!( u, v, Δu, Δv)

    n = size(Δu,1)

    for c = 1:n
        c1 = c + 1
        c2 = c + 2
        for r = 1:n
            r1 = r + 1
            r2 = r + 2
            @inbounds Δu[r,c] = u[r1,c2] + u[r1,c] + u[r2,c1] + u[r,c1] - 4*u[r1,c1]
            @inbounds Δv[r,c] = v[r1,c2] + v[r1,c] + v[r2,c1] + v[r,c1] - 4*v[r1,c1]
        end
    end

    for c = 1:n
        c1 = c + 1
        for r = 1:n
            r1 = r + 1  
            @inbounds uvv = u[r1,c1]*v[r1,c1]*v[r1,c1]
            @inbounds u[r1,c1] +=  Dᵤ * Δu[r,c] - uvv + F*(1 - u[r1,c1])
            @inbounds v[r1,c1] +=  Dᵥ * Δv[r,c] + uvv - (F + k)*v[r1,c1]
        end
    end

end



function run_simulation_loops( update_uv!, n, maxiter = 10_000)
    u, v = init(n)
    Δu = zeros(n, n)
    Δv = zeros(n, n)
    @showprogress 1 for t in 1:maxiter
        update_uv!(u, v, Δu, Δv)
    end
    return u, v
end

@time u, v = run_simulation_loops(grayscott!, 500)

# -

using .Threads
nthreads()

# +
function grayscott_threads!( u, v, Δu, Δv)

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

@time u, v = run_simulation_loops(grayscott_threads!, 500)

# +
function grayscott_vectorized!(u, v, Δu, Δv, uvv)
        
    @views Δu .= (u[1:end-2,2:end-1] .+ u[2:end-1,1:end-2] .+ u[2:end-1, 3:end] 
            .+ u[3:end,2:end-1] .- 4 .* u[2:end-1, 2:end-1] )

    @views Δv .= (v[1:end-2,2:end-1] .+ v[2:end-1,1:end-2] .+ v[2:end-1, 3:end] 
            .+ v[3:end,2:end-1] .- 4 .* v[2:end-1, 2:end-1] )

    @views uvv .= u[2:end-1,2:end-1] .* v[2:end-1,2:end-1] .* v[2:end-1,2:end-1]

    @views u[2:end-1,2:end-1] .+=  Dᵤ .* Δu .- uvv .+ F * (1 .- u[2:end-1,2:end-1])

    @views v[2:end-1,2:end-1] .+=  Dᵥ .* Δv .+ uvv .- (F + k) .* v[2:end-1,2:end-1]
                 
end


# +
function run_simulation_vectorized( n, maxiter = 10_000)
    u, v = init(n)
    Δu = zeros(n, n)
    Δv = zeros(n, n)
    uvv = zeros(n ,n )
    @showprogress 1 for t in 1:maxiter
        grayscott_vectorized!(u, v, Δu, Δv, uvv)
    end
    return u, v
end

@time u, v = run_simulation_vectorized(500)


# +
using CUDA

function run_on_gpu( n = 500, maxiter = 10_000)

    u0, v0 = init(n)

    u = CuArray(u0)
    v = CuArray(v0)

    Δu = CUDA.zeros(n, n)
    Δv = CUDA.zeros(n, n)
    uvv = CUDA.zeros(n,n)
    
    for t in 1:maxiter
    
        grayscott_vectorized!(u, v, Δu, Δv, uvv)

    end

    return Array(u), Array(v)

end

CUDA.@time u, v = run_on_gpu( 500)

USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(0)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

@parallel function grayscott_ps!( u, v, Δu, Δv, uvv)
    
    @all(Δu) = @d2_xi(u) + @d2_yi(u)
    @all(Δv) = @d2_xi(u) + @d2_yi(v)
       
    @all(uvv) = @inn(u) * @inn(v) * @inn(v)
    @inn(u) = @inn(u) + Dᵤ * @all(Δu) - @all(uvv) + F * (1 - @inn(u))
    @inn(v) = @inn(v) + Dᵥ * @all(Δv) + @all(uvv) - (F + k) * @inn(v)
    return
end

function run_with_ps( n = 500, maxiter = 10_000)

    u0, v0 = init(n)

    u = Data.Array(v0)
    v = Data.Array(v0)

    Δu = @zeros(n, n)
    Δv = @zeros(n, n)
    uvv = @zeros(n,n)
    
    for t in 1:maxiter
    
        @parallel grayscott_ps!(u, v, Δu, Δv, uvv)

    end
    
    u, v

end

@time run_with_ps(500)
