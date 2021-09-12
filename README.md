# SYCL Building Blocks

Header-based SYCL reusable algorithms and data structures.

## Runtime Index Wrapper

Functions used to access arrays-based variables with a dynamic/runtime index. This allows to force the registerization of these arrays which is not possible otherwise, on GPU. The benchmarks give a performance of **12**
billion iterations per second with the regular indexing. With our method we're achieving about **470** billion iterations per second.

When writing, if all threads access the same index only, there is a ligher/faster version available by defining `RUNTIME_IDX_STORE_USE_SWITCH`. Sometimes registerization won't happen using. Read speed is not affected.

There is a read version, `runtime_index_wrapper_log` that uses a binary search. It reduces the number of steps, but often is slower because of thread divergence if not all the threads access the same value.

The wrapper has a specialisation for `std::array`, `sycl::vec`, `C-style arrays` and `sycl::id`. It also accepts any type that has a subscript operator, but then the used must put the maximum accessed index in the first
template parameter.

#### Benchmarks

With Google benchmark on a GTX 1660 Ti:

```
Benchmark                       Time      CPU  Iterations     UserCounters...
registerized_array/268435456 57.2 ms  57.1 ms  12 items_per_second=469.731G/s
stack_array/268435456        2216 ms  2214 ms   1 items_per_second=12.1234G/s
```

#### Use example

```C++
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


