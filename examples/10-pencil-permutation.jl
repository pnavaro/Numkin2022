using MPI, PencilArrays, Random

MPI.Init()

dims = (32, 8, 16); comm = MPI.COMM_WORLD;
pen = Pencil(dims, comm; permute = Permutation(3, 2, 1));
u = PencilArray{Float64}(undef, pen); randn!(u);

if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println(size_local(u))
    println(size_local(u, MemoryOrder()))
end

