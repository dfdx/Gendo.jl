module onnx
  const _ProtoBuf_Top_ = @static isdefined(parentmodule(@__MODULE__), :_ProtoBuf_Top_) ? (parentmodule(@__MODULE__))._ProtoBuf_Top_ : parentmodule(@__MODULE__)
  include("onnx_pb.jl")
end

using .onnx
using ProtoBuf


rawproto(io::IO) = readproto(io, onnx.ModelProto())
rawproto(path::String) = open(rawproto, path)
