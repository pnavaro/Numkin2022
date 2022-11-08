using MPI, PencilArrays
MPI.Init()
comm = MPI.COMM_WORLD
Nx, Ny, Nz = (256, 64, 128)  # global domain dimensions
# Automatically generate a decomposition over all processes
pen = Pencil((Nx, Ny, Nz), comm)
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println(size_local(pen))
end
