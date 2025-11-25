module FFInterp

using StaticArrays
using JlrsCore.Wrap

# Detect platform
# Build full path relative to this file
const libname = Sys.islinux() ? "libffinterp.so" :
                 Sys.isapple() ? "libffinterp.dylib" :
                 Sys.iswindows() ? "ffinterp.dll" : error("Unsupported OS")

const libpath = joinpath(@__DIR__, "..", "..", "..", "..", "target", "release", libname)

@wrapmodule(libpath, :ffinterp_init)

function __init__()
    @initjlrs
end

function interpolate(
    pts::Vector{SVector{N, T}}, 
    method, 
    f0, 
    niter::Integer
) where {N, T}
    ptr = Ptr{T}(pointer(pts))
    len = length(pts) * N
    
    flat_view = unsafe_wrap(Array, ptr, len; own=false)
    
    GC.@preserve pts begin
        return interpolate(flat_view, method, f0, niter)
    end
end

function (interpolant::Interpolant{T})(x::T) where T
    return evaluate(interpolant, x)
end

function (interpolant::Interpolant{T})(x::Vector{T}) where T
    return evaluate_many(interpolant, x)
end


export interpolate

end
