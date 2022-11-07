using MPI, PencilArrays
MPI.Init()
comm = MPI.COMM_WORLD
Nx, Ny, Nz = (256, 64, 128)  # global domain dimensions
# Automatically generate a decomposition over all processes
pen = Pencil((Nx, Ny, Nz), comm)
# Construct a distributed array
u = PencilArray{Float64}(undef, pen)
println(size_local(pen))
