import DelimitedFiles, MatrixMarket, MAT, HarwellRutherfordBoeing

function load_matrix(path::String)
    ext = splitext(path)[2]
    if ext == ".mtx"
        M = MatrixMarket.mmread(path)
        return M
    elseif ext == ".rra"
        M = HarwellRutherfordBoeing.HarwellBoeingMatrix(path)
        return M
    elseif ext == ".rb"
        rb = HarwellRutherfordBoeing.RutherfordBoeingData(path)
        M = rb.data
        return M
    elseif ext == ".mat"
        dict = MAT.matread(path)
        M = dict["Problem"]["A"]
        return M
    else
        error("The format of this file is not supported.")
    end
end

function load_rhs(path::String)
    rhs = DelimitedFiles.readdlm(path, Float64)[:]
    return rhs
end
