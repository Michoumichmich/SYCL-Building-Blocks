# SYCL Building Blocks

Header-based SYCL reusable algorithms and data structures.

## Runtime Index Wrapper

Functions used to access arrays-based variables with a dynamic/runtime index. This allows to force the registerization of these arrays which is not possible otherwise, on GPU. The benchmarks give a performance of **19**
billion iterations per second with the regular indexing. With our method we're achieving about **270** billion iterations per second. Sometimes Writes still sends the array to the stack-frame. But we're still getting 3X
better performance using the wrapper (see commented lines in benchmark)

The wrapper has a specialisation for `std::array`, `sycl::vec`, `C-style arrays` and `sycl::id`. It also accepts any type that has a subscript operator, but then the used must put the maximum accessed index in the first
template parameter.

#### Use example

```
int array[10] = {0};
runtime_index_wrapper(array, i % 10, j) // performs array[i%10]=j
assert(j == runtime_index_wrapper(array, i % 10)); // reads the value
```

## Prefix Scan

### Decoupled lookback prefix scan

Implements a modified version of the *Single-pass Parallel Prefix Scan with Decoupled Look-back* with variable stragegy: *scan-then-reduce* or *reduce-then-scan*. The strategy depends on the previous partition
descriptors to hide latencies. Performs at speeds close to `memcpy` and has only *2n* memory movements. Forward guarantees progress required.

### Cooperative Prefix Scan

Implements a *radix-N scan-then-propagate* strategy using Kogge-Stone group-scans and propagation fans. This implementation demonstrates the use of Cooperative Groups and is thus experimental. The computation is
performed in a single kernel launch and using the whole device. The bottleneck is currently the implementation of the SYCL group algorithms.

### Reduction

Parallel reduction algorithm using SYCL reductors and a recursive approach to unroll the loops. Memory bound, performance is thus equivalent to CUB.

## Intrinsics

## [Experimental] Cooperative groups

Device wide synchronisation functions. Kernel ranges are bound to ensure forward progress, If the Nvidia GPU is using AMS this might not be enough.


