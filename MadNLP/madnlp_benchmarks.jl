using ExaModelsExamples, MadNLP, CUDA, MadNLPGPU

list_problems = [
    "pglib_opf_case10000_goc.m",
    "pglib_opf_case10192_epigrids.m",
    "pglib_opf_case10480_goc.m",
    "pglib_opf_case1354_pegase.m",
    "pglib_opf_case13659_pegase.m",
    "pglib_opf_case179_goc.m",
    "pglib_opf_case19402_goc.m",
    "pglib_opf_case2000_goc.m",
    "pglib_opf_case20758_epigrids.m",
    "pglib_opf_case2312_goc.m",
    "pglib_opf_case2742_goc.m",
    "pglib_opf_case2869_pegase.m",
    "pglib_opf_case30000_goc.m",
    "pglib_opf_case3022_goc.m",
    "pglib_opf_case3970_goc.m",
    "pglib_opf_case4020_goc.m",
    "pglib_opf_case4601_goc.m",
    "pglib_opf_case4619_goc.m",
    "pglib_opf_case4837_goc.m",
    "pglib_opf_case4917_goc.m",
    "pglib_opf_case500_goc.m",
    "pglib_opf_case5658_epigrids.m",
    "pglib_opf_case7336_epigrids.m",
    "pglib_opf_case78484_epigrids.m",
    "pglib_opf_case793_goc.m",
    "pglib_opf_case8387_pegase.m",
    "pglib_opf_case89_pegase.m",
    "pglib_opf_case9241_pegase.m",
    "pglib_opf_case9591_goc.m",
]

opf1 = "pglib_opf_case4020_goc.m"

# GPU
if CUDA.functional()
  println("--- GPU ---")
  # It will automatically download the case file for you.
  model = ac_power_model(opf1; backend=CUDABackend())
  s = MadNLPSolver(model; tol=1e-6)
  solve!(s)
  println()
end

# CPU
println("--- CPU ---")
model = ac_power_model(opf1)
s = MadNLPSolver(model; tol=1e-6)
solve!(s)

opf2 = "pglib_opf_case78484_epigrids.m"

# GPU
if CUDA.functional()
  println("--- GPU ---")
  # It will automatically download the case file for you.
  model = ac_power_model(opf2; backend=CUDABackend())
  s = MadNLPSolver(model; tol=1e-6)
  solve!(s)
  println()
end

# CPU
println("--- CPU ---")
model = ac_power_model(opf2)
s = MadNLPSolver(model; tol=1e-6)
solve!(s)
