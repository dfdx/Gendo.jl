module onnx
  const _ProtoBuf_Top_ = @static isdefined(parentmodule(@__MODULE__), :_ProtoBuf_Top_) ? (parentmodule(@__MODULE__))._ProtoBuf_Top_ : parentmodule(@__MODULE__)
  include("onnx_pb.jl")
end

using .onnx
using ProtoBuf


rawproto(io::IO) = readproto(io, onnx.ModelProto())
rawproto(path::String) = open(rawproto, path)


################################################################################
#                                 Data Loading                                 #
################################################################################

# Based on: https://github.com/FluxML/ONNX.jl/blob/master/src/convert.jl

const DATA_TYPE_CODES = Dict(
    1 => Float32,
    6 => Int32,
    7 => Int64,
    9 => Int8,
    10 => Float16,
    11 => Float64,
)


"""
Get the array from a TensorProto object.
"""
function get_array(x::onnx.TensorProto)
    if (x.data_type == 1)
        if !isempty(x.float_data)
            x = reshape(reinterpret(Float32, x.float_data), reverse(x.dims)...)
        else
            x = reshape(reinterpret(Float32, x.raw_data), reverse(x.dims)...)
        end
        return x
    end
    if x.data_type == 7
        if !isempty(x.raw_data)
            x = reshape(reinterpret(Int64, x.raw_data), reverse(x.dims)...)
        else
            x = reshape(reinterpret(Int64, x.int64_data), reverse(x.dims)...)
        end
        return x
    end
    if x.data_type == 9
        x = reshape(reinterpret(Int8, x.raw_data), reverse(x.dims)...)
        return x
    end
    if x.data_type == 6
         x = reshape(reinterpret(Int32, x.raw_data), reverse(x.dims)...)
        return x
    end
    if x.data_type == 11
        if !isempty(x.raw_data)
            x = reshape(reinterpret(Float64, x.raw_data), reverse(x.dims)...)
        else
            x = Base.convert(Array{Float32, N} where N, reshape(x.double_data , reverse(x.dims)...))
        end
        return x
    end
    if x.data_type == 10
        x = reshape(reinterpret(Float16, x.raw_data), reverse(x.dims)...)
        return x
    end
end


permutedims_from_onnx(a::AbstractArray{T, 2}) where T = transpose(a)
permutedims_from_onnx(a::AbstractArray{T, N}) where {T, N} = a


################################################################################
#                                Loaders                                       #
################################################################################

"""
Generic model that can hold any nested properties. 

    m = GenericModel()
    m.foo = rand(5)
    println(m.foo)
    setproperty!(m, rand(10), :bar, :baz)  # TODO: setrecursive!()?
    println(m.bar.baz)
"""
mutable struct GenericModel
    fields::NamedTuple
end

GenericModel() = GenericModel(namedtuple((), ()))


function combine_tuples(t1::NamedTuple{F1}, t2::NamedTuple{F2}) where {F1, F2}
    fields = Symbol[f for f in F1]
    vals = Any[v for v in t1]
    for (f, v) in zip(F2, t2)
        idx = findfirst(old_f -> old_f == f, fields)
        if idx !== nothing
            # field exists -> replace
            vals[idx] = v
        else
            push!(fields, f)
            push!(vals, v)
        end
    end
    return namedtuple((fields...,), vals)
end


function Base.getproperty(m::GenericModel, f::Symbol)
    flds = getfield(m, :fields)
    return getproperty(flds, f)
end


function Base.setproperty!(m::GenericModel, f::Symbol, v::Any)
    old = getfield(m, :fields)
    new = combine_tuples(old, NamedTuple{(f,)}((v,)))
    setfield!(m, :fields, new)
end


function setrecursive!(m::GenericModel, v::Any, fs...)
    if length(fs) == 1
        f = fs[1]
        setproperty!(m, f, v)
    else
        f, rest = fs[1], fs[2:end]
        if hasproperty(getfield(m, :fields), f)
            m_next = getproperty(m, f)
            setrecursive!(m_next, v, rest...)
        else
            m_next = GenericModel()
            setrecursive!(m_next, v, rest...)
            setproperty!(m, f, m_next)
        end
    end    
end


function setrecursive!(m::GenericModel, v::Any, path::String)
    fs = map(Symbol, split(path, "."))
    setrecursive!(m, v, fs...)
end


function load_inputs!(graph::onnx.GraphProto, tape::Tape, name2id::Dict{String, Int})
    # graph.initializer, usually, model parameters
    for init in graph.initializer
        # TODO: initializer may also specify constant
        val = get_array(init) |> permutedims_from_onnx
        id = record!(tape, Input, val)
        name2id[init.name] = id
    end
    # graph.input - model inputs
    for inp in graph.input
        tt = inp._type.tensor_type
        eltyp = DATA_TYPE_CODES[tt.elem_type]
        shape = reverse([d.dim_value for d in tt.shape.dim])
        val = zeros(eltyp, shape...)
        id = record!(tape, Input, val)
        name2id[inp.name] = id
    end
end


function load_nodes!(graph::onnx.GraphProto, tape::Tape, name2id::Dict{String, Int})
    for nd in graph.node

    end
end



function load_tape(path)
    proto = rawproto(path);
    graph = proto.graph;
    tape = Tape()
    name2id = Dict{String, Int}()   # mapping from name in graph to tape ID
    load_inputs!(graph, tape, name2id)
    return tape
end
