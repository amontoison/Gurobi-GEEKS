using LinearAlgebra, SparseArrays
using CUDA, CUDA.CUSPARSE, CUDSS
using SuiteSparseMatrixCollection

include("utils.jl")

# Download a matrix from the SuiteSparse Matrix Collection
path_pb = fetch_ssmc("AMD", "G3_circuit"; format="MM") # http://sparse.tamu.edu/AMD/G3_circuit
path_A = joinpath(path_pb, "G3_circuit.mtx")
path_b = joinpath(path_pb, "G3_circuit.mtx")

# path_pb = fetch_ssmc("Schmid", "thermal2"; format="MM") # http://sparse.tamu.edu/Schmid/thermal2
# path_A = joinpath(path_pb, "thermal2.mtx")
# path_b = joinpath(path_pb, "thermal2_b.mtx")

A_cpu = load_matrix(path_A)
m, n = size(A_cpu)
b_cpu = isfile(path_b) ? Vector(load_matrix(path_b)[:,1]) : ones(Float64, m)
x_cpu = similar(b_cpu)

# GPU
if CUDA.functional()
    A_gpu = CuSparseMatrixCSR(A_cpu)
    b_gpu = CuVector(b_cpu)
    x_gpu = CuVector(x_cpu)

    # "SPD" (LLᵀ) / "S" (LDLᵀ) / "G" (LU)
    # "F" (Full) / "L" (lower triangle) / "U" (Upper triangle)
    solver = CudssSolver(A_gpu, "SPD", 'F')

    time_analysis = CUDA.@elapsed CUDA.@sync begin
        cudss("analysis", solver, x_gpu, b_gpu)
    end
    println("Analysis: $(time_analysis) seconds.")

    time_factorization = CUDA.@elapsed CUDA.@sync begin
        cudss("factorization", solver, x_gpu, b_gpu)
    end
    println("Factorization: $(time_factorization) seconds.")

    time_backsolve = CUDA.@elapsed CUDA.@sync begin
        cudss("solve", solver, x_gpu, b_gpu)
    end
    println("Triangular solves: $(time_backsolve) seconds.")

    r_gpu = b_gpu - A_gpu * x_gpu
    rNorm = norm(r_gpu)
    println("Residual norm ‖b - Ax‖: $rNorm.")
end

# CPU
A_cholmod = SparseArrays.CHOLMOD.Sparse(A_cpu)

time_analysis = @elapsed begin
    solver = SparseArrays.CHOLMOD.symbolic(A_cholmod)
end
println("Analysis: $(time_analysis) seconds.")

time_factorization = @elapsed begin
    SparseArrays.CHOLMOD.cholesky!(solver, A_cpu; check=false)
end
println("Factorization: $(time_factorization) seconds.")

time_backsolve = @elapsed begin
    b_cholmod = SparseArrays.CHOLMOD.Dense(b_cpu)
    x_cholmod = SparseArrays.CHOLMOD.solve(SparseArrays.CHOLMOD.CHOLMOD_A, solver, b_cholmod)
    copyto!(x_cpu, x_cholmod)
end
println("Triangular solves: $(time_backsolve) seconds.")

r_cpu = b_cpu - A_cpu * x_cpu
rNorm = norm(r_cpu)
println("Residual norm ‖b - Ax‖: $rNorm.")
