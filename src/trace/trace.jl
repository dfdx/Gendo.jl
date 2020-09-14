function __new__(T, args...)
    # @show T
    # @show args
    # note: we also add __new__() to the list of primitives so it's not overdubbed recursively
    if T <: NamedTuple
        return T(args)
    else
        return T(args...)
    end
end


__tuple__(args...) = tuple(args...)
__getfield__(args...) = getfield(args...)


function module_functions(modl)
    res = Vector{Function}()
    for s in Base.names(modl; all=true)
        isdefined(modl, s) || continue
        fn = getfield(modl, s)
        if fn isa Function # && match(r"^[a-z#]+$", string(s)) != nothing
            push!(res, fn)
        end
    end
    return res
end


# TODO: this accounts to a significant part of loading time.
# Most of this time is taken by Set(long-function-list),
# maybe because of long calculation of function hash,
# maybe for another reason. Anyway, I don't see a quick and
# reliable way to fix it right now, so let's come back
# to this later
const PRIMITIVES = Set{Any}(vcat(
    module_functions(Base),
    module_functions(Core),
    module_functions(Core.Intrinsics),
    [Broadcast.materialize, Broadcast.broadcasted, Colon(), (:),
     Base.not_int,
     # our own special functions
     __new__, __tuple__, __getfield__, namedtuple, guess_device]));


include("cassette.jl")
# include("interp.jl")
include("irtools.jl")


trace = irtrace
