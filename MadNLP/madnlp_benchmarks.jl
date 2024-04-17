using ExaModelsExamples, MadNLP, CUDA, MadNLPGPU

# CPU
model = ac_power_model("pglib_opf_case78484_epigrids.m"; tol=1e-8)   # It will automatically download the case file for you.
s = MadNLPSolver(model; tol=1e-8)
solve!(s)

# GPU
if CUDA.functional()
  model = ac_power_model("pglib_opf_case78484_epigrids.m"; backend=CUDABackend(), tol=1e-8)
  s = MadNLPSolver(model; tol=1e-8)
  solve!(s)
end
