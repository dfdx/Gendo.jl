import Statistics
using LinearAlgebra
using Espresso
using Distributions
using ChainRulesCore
using CUDA


include("utils.jl")
include("helpers.jl")
include("devices.jl")
include("tape.jl")
include("tapeutils.jl")
include("trace/trace.jl")
include("diffrules/diffrules.jl")
include("grad.jl")
include("compile.jl")
include("update.jl")
include("transform.jl")
include("onnx/onnx.jl")
include("cuda.jl")


# function __init__()
#     @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("cuda.jl")
# end

const BEST_AVAILABLE_DEVICE = Ref{AbstractDevice}(CPU())

if CUDA.functional()
    try
        BEST_AVAILABLE_DEVICE[] = GPU(1)        
    catch ex
        # something is wrong with the user's set-up (or there's a bug in CuArrays)
        @warn "CUDA is installed, but not working properly" exception=(ex,catch_backtrace())

    end
end


best_available_device() = BEST_AVAILABLE_DEVICE[]
