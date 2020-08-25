# syntax: proto3
using ProtoBuf
import ProtoBuf.meta

struct __enum_Version <: ProtoEnum
    _START_VERSION::Int32
    IR_VERSION_2017_10_10::Int32
    IR_VERSION_2017_10_30::Int32
    IR_VERSION_2017_11_3::Int32
    IR_VERSION_2019_1_22::Int32
    IR_VERSION_2019_3_18::Int32
    IR_VERSION_2019_9_19::Int32
    IR_VERSION::Int32
    __enum_Version() = new(0,1,2,3,4,5,6,7)
end #struct __enum_Version
const Version = __enum_Version()

mutable struct StringStringEntryProto <: ProtoType
    key::AbstractString
    value::AbstractString
    StringStringEntryProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct StringStringEntryProto

mutable struct TensorAnnotation <: ProtoType
    tensor_name::AbstractString
    quant_parameter_tensor_names::Base.Vector{StringStringEntryProto}
    TensorAnnotation(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorAnnotation

struct __enum_TensorProto_DataType <: ProtoEnum
    UNDEFINED::Int32
    FLOAT::Int32
    UINT8::Int32
    INT8::Int32
    UINT16::Int32
    INT16::Int32
    INT32::Int32
    INT64::Int32
    STRING::Int32
    BOOL::Int32
    FLOAT16::Int32
    DOUBLE::Int32
    UINT32::Int32
    UINT64::Int32
    COMPLEX64::Int32
    COMPLEX128::Int32
    BFLOAT16::Int32
    __enum_TensorProto_DataType() = new(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
end #struct __enum_TensorProto_DataType
const TensorProto_DataType = __enum_TensorProto_DataType()

struct __enum_TensorProto_DataLocation <: ProtoEnum
    DEFAULT::Int32
    EXTERNAL::Int32
    __enum_TensorProto_DataLocation() = new(0,1)
end #struct __enum_TensorProto_DataLocation
const TensorProto_DataLocation = __enum_TensorProto_DataLocation()

mutable struct TensorProto_Segment <: ProtoType
    _begin::Int64
    _end::Int64
    TensorProto_Segment(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorProto_Segment

mutable struct TensorProto <: ProtoType
    dims::Base.Vector{Int64}
    data_type::Int32
    segment::TensorProto_Segment
    float_data::Base.Vector{Float32}
    int32_data::Base.Vector{Int32}
    string_data::Base.Vector{Array{UInt8,1}}
    int64_data::Base.Vector{Int64}
    name::AbstractString
    doc_string::AbstractString
    raw_data::Array{UInt8,1}
    external_data::Base.Vector{StringStringEntryProto}
    data_location::Int32
    double_data::Base.Vector{Float64}
    uint64_data::Base.Vector{UInt64}
    TensorProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorProto
const __fnum_TensorProto = Int[1,2,3,4,5,6,7,8,12,9,13,14,10,11]
const __pack_TensorProto = Symbol[:dims,:float_data,:int32_data,:int64_data,:double_data,:uint64_data]
meta(t::Type{TensorProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TensorProto, ProtoBuf.DEF_VAL, true, __pack_TensorProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct SparseTensorProto <: ProtoType
    values::TensorProto
    indices::TensorProto
    dims::Base.Vector{Int64}
    SparseTensorProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct SparseTensorProto
const __pack_SparseTensorProto = Symbol[:dims]
meta(t::Type{SparseTensorProto}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, __pack_SparseTensorProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct TensorShapeProto_Dimension <: ProtoType
    dim_value::Int64
    dim_param::AbstractString
    denotation::AbstractString
    TensorShapeProto_Dimension(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorShapeProto_Dimension
const __oneofs_TensorShapeProto_Dimension = Int[1,1,0]
const __oneof_names_TensorShapeProto_Dimension = [Symbol("value")]
meta(t::Type{TensorShapeProto_Dimension}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_TensorShapeProto_Dimension, __oneof_names_TensorShapeProto_Dimension, ProtoBuf.DEF_FIELD_TYPES)

mutable struct TensorShapeProto <: ProtoType
    dim::Base.Vector{TensorShapeProto_Dimension}
    TensorShapeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TensorShapeProto

mutable struct OperatorSetIdProto <: ProtoType
    domain::AbstractString
    version::Int64
    OperatorSetIdProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct OperatorSetIdProto

struct __enum_AttributeProto_AttributeType <: ProtoEnum
    UNDEFINED::Int32
    FLOAT::Int32
    INT::Int32
    STRING::Int32
    TENSOR::Int32
    GRAPH::Int32
    SPARSE_TENSOR::Int32
    FLOATS::Int32
    INTS::Int32
    STRINGS::Int32
    TENSORS::Int32
    GRAPHS::Int32
    SPARSE_TENSORS::Int32
    __enum_AttributeProto_AttributeType() = new(0,1,2,3,4,5,11,6,7,8,9,10,12)
end #struct __enum_AttributeProto_AttributeType
const AttributeProto_AttributeType = __enum_AttributeProto_AttributeType()

mutable struct AttributeProto <: ProtoType
    name::AbstractString
    ref_attr_name::AbstractString
    doc_string::AbstractString
    _type::Int32
    f::Float32
    i::Int64
    s::Array{UInt8,1}
    t::TensorProto
    g::Base.Any
    sparse_tensor::SparseTensorProto
    floats::Base.Vector{Float32}
    ints::Base.Vector{Int64}
    strings::Base.Vector{Array{UInt8,1}}
    tensors::Base.Vector{TensorProto}
    graphs::Base.Any
    sparse_tensors::Base.Vector{SparseTensorProto}
    AttributeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct AttributeProto (has cyclic type dependency)
const __fnum_AttributeProto = Int[1,21,13,20,2,3,4,5,6,22,7,8,9,10,11,23]
const __pack_AttributeProto = Symbol[:floats,:ints]
const __ftype_AttributeProto = Dict(:g => "GraphProto", :graphs => "Base.Vector{GraphProto}")
meta(t::Type{AttributeProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_AttributeProto, ProtoBuf.DEF_VAL, true, __pack_AttributeProto, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_AttributeProto)

mutable struct ValueInfoProto <: ProtoType
    name::AbstractString
    _type::Base.Any
    doc_string::AbstractString
    ValueInfoProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ValueInfoProto (has cyclic type dependency)
const __ftype_ValueInfoProto = Dict(:_type => "TypeProto")
meta(t::Type{ValueInfoProto}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_ValueInfoProto)

mutable struct NodeProto <: ProtoType
    input::Base.Vector{AbstractString}
    output::Base.Vector{AbstractString}
    name::AbstractString
    op_type::AbstractString
    domain::AbstractString
    attribute::Base.Vector{AttributeProto}
    doc_string::AbstractString
    NodeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct NodeProto (has cyclic type dependency)
const __fnum_NodeProto = Int[1,2,3,4,7,5,6]
meta(t::Type{NodeProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_NodeProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct TrainingInfoProto <: ProtoType
    initialization::Base.Any
    algorithm::Base.Any
    initialization_binding::Base.Vector{StringStringEntryProto}
    update_binding::Base.Vector{StringStringEntryProto}
    TrainingInfoProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TrainingInfoProto (has cyclic type dependency)
const __ftype_TrainingInfoProto = Dict(:initialization => "GraphProto", :algorithm => "GraphProto")
meta(t::Type{TrainingInfoProto}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_TrainingInfoProto)

mutable struct ModelProto <: ProtoType
    ir_version::Int64
    opset_import::Base.Vector{OperatorSetIdProto}
    producer_name::AbstractString
    producer_version::AbstractString
    domain::AbstractString
    model_version::Int64
    doc_string::AbstractString
    graph::Base.Any
    metadata_props::Base.Vector{StringStringEntryProto}
    training_info::Base.Vector{TrainingInfoProto}
    ModelProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct ModelProto (has cyclic type dependency)
const __fnum_ModelProto = Int[1,8,2,3,4,5,6,7,14,20]
const __ftype_ModelProto = Dict(:graph => "GraphProto")
meta(t::Type{ModelProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_ModelProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_ModelProto)

mutable struct GraphProto <: ProtoType
    node::Base.Vector{NodeProto}
    name::AbstractString
    initializer::Base.Vector{TensorProto}
    sparse_initializer::Base.Vector{SparseTensorProto}
    doc_string::AbstractString
    input::Base.Vector{ValueInfoProto}
    output::Base.Vector{ValueInfoProto}
    value_info::Base.Vector{ValueInfoProto}
    quantization_annotation::Base.Vector{TensorAnnotation}
    GraphProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct GraphProto (has cyclic type dependency)
const __fnum_GraphProto = Int[1,2,5,15,10,11,12,13,14]
meta(t::Type{GraphProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_GraphProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, ProtoBuf.DEF_FIELD_TYPES)

mutable struct TypeProto_Tensor <: ProtoType
    elem_type::Int32
    shape::TensorShapeProto
    TypeProto_Tensor(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TypeProto_Tensor
hash(v::TypeProto_Tensor) = ProtoBuf.protohash(v)
isequal(v1::TypeProto_Tensor, v2::TypeProto_Tensor) = ProtoBuf.protoisequal(v1, v2)
==(v1::TypeProto_Tensor, v2::TypeProto_Tensor) = ProtoBuf.protoeq(v1, v2)

mutable struct TypeProto_Sequence <: ProtoType
    elem_type::Base.Any
    TypeProto_Sequence(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TypeProto_Sequence (has cyclic type dependency)
const __ftype_TypeProto_Sequence = Dict(:elem_type => "TypeProto")
meta(t::Type{TypeProto_Sequence}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_TypeProto_Sequence)

mutable struct TypeProto_Map <: ProtoType
    key_type::Int32
    value_type::Base.Any
    TypeProto_Map(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TypeProto_Map (has cyclic type dependency)
const __ftype_TypeProto_Map = Dict(:value_type => "TypeProto")
meta(t::Type{TypeProto_Map}) = meta(t, ProtoBuf.DEF_REQ, ProtoBuf.DEF_FNUM, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, ProtoBuf.DEF_ONEOFS, ProtoBuf.DEF_ONEOF_NAMES, __ftype_TypeProto_Map)

mutable struct TypeProto <: ProtoType
    tensor_type::TypeProto_Tensor
    sequence_type::TypeProto_Sequence
    map_type::TypeProto_Map
    denotation::AbstractString
    TypeProto(; kwargs...) = (o=new(); fillunset(o); isempty(kwargs) || ProtoBuf._protobuild(o, kwargs); o)
end #mutable struct TypeProto (has cyclic type dependency)
const __fnum_TypeProto = Int[1,4,5,6]
const __oneofs_TypeProto = Int[1,1,1,0]
const __oneof_names_TypeProto = [Symbol("value")]
meta(t::Type{TypeProto}) = meta(t, ProtoBuf.DEF_REQ, __fnum_TypeProto, ProtoBuf.DEF_VAL, true, ProtoBuf.DEF_PACK, ProtoBuf.DEF_WTYPES, __oneofs_TypeProto, __oneof_names_TypeProto, ProtoBuf.DEF_FIELD_TYPES)

export Version, AttributeProto_AttributeType, AttributeProto, ValueInfoProto, NodeProto, TrainingInfoProto, ModelProto, StringStringEntryProto, TensorAnnotation, GraphProto, TensorProto_DataType, TensorProto_DataLocation, TensorProto_Segment, TensorProto, SparseTensorProto, TensorShapeProto_Dimension, TensorShapeProto, TypeProto_Tensor, TypeProto_Sequence, TypeProto_Map, TypeProto, OperatorSetIdProto, AttributeProto_AttributeType, AttributeProto, ValueInfoProto, NodeProto, TrainingInfoProto, ModelProto, GraphProto, TypeProto_Sequence, TypeProto_Map, TypeProto
