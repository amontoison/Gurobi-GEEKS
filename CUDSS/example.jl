using LinearAlgebra, SparseArrays
using CUDA, CUDA.CUSPARSE, CUDSS

include("utils.jl")

path_A = "./data/K.mtx"
path_b = "./data/rhs.txt"
# path2 = "./data/1138_bus.mtx"
# path2 = "./data/1138_bus.rb"
# path3 = "./data/1138_bus.mat"

A = load_matrix(path_A)
b = load_rhs(path_b)
x = similar(b)

if CUDA.functional()
    A_gpu = CuSparseMatrixCSR(A)
    b_gpu = CuVector(b)
    x_gpu = CuVector(x)

    # "SPD" (LLᵀ) / "S" (LDLᵀ) / "G" (LU)
    # "F" (Full) / "L" (lower triangle) / "U" (Upper triangle)
    solver = CudssSolver(A_gpu, "S", 'F')

    cudss("analysis", solver, x_gpu, b_gpu)
    cudss("factorization", solver, x_gpu, b_gpu)
    cudss("solve", solver, x_gpu, b_gpu)

    r_gpu = b_gpu - A_gpu * x_gpu
    norm(r_gpu)
end
