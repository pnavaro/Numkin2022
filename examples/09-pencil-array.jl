using MPI, PencilArrays, Random
MPI.Init()
dims = (32, 8, 16); comm = MPI.COMM_WORLD;
pen = Pencil(dims, comm);
u = PencilArray{Float64}(undef, pen);
rand!(u);  # fill with random values (requires `using Random`)

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
   println(summary(u))
   println(summary(parent(u)) )
end
