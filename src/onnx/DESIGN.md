## Intended usage

1. Load pure tape (low-level API?):

```julia
tape = load_tape("/path/to/model.onnx")
compile!(tape)
play!(tape, ??, inp1, inp2)
```
^ this is good to generate reverse pass operations and continue training

2. Create generic model with API similar to normal models:

```julia
m = ONNX.load_model("/path/to/model.onnx")
m(inp1, inp2)
```
In this case:

 * all non-constant initializers are put into a complosed model-like object `m`
 * `m()` is overloaded with code generated from the tape