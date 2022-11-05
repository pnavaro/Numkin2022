using MPI


MPI.Init()

comm = MPI.COMM_WORLD
prank = MPI.Comm_rank(comm)
psize = MPI.Comm_size(comm)
print("Hello world, I am rank $(prank) of $(psize)\n")

xmin, xmax = -6, 6
nx = 100
np = 1_000_000

if MPI.Comm_rank(comm) == 0
    xp = randn(np)
else
    xp = zeros(np)
end

MPI.Bcast!(xp, 0, comm)

chunks = collect(Iterators.partition(1:np, np ÷ psize))

function deposition( xp, xmin, xmax, nx )
    rho = zeros(Float64, nx)
    for i in eachindex(xp)
        x_norm = (xp[i]-xmin) / (xmax - xmin) # normalized position
        ip = trunc(Int,  x_norm * nx) + 1 # nearest grid point
        rho[ip] += 1
    end

    rho 

end

rho_local = deposition( view(xp,chunks[prank+1]), xmin, xmax, nx )

rho = MPI.Reduce(rho_local, +, comm) 

MPI.Barrier(comm)

if MPI.Comm_rank(comm) == 0
    println( deposition( xp, xmin, xmax, nx ) ≈ rho)
end
