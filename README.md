# SYCL Building Blocks

Header-based SYCL reusable algorithms and data structures.

## Register bit array
Dynamically addressable bit array which will be kept in GPU registers. The API is based on the one offered by `std::bitset` (C++20).

## Prefix Scans

### Decoupled lookback prefix scan

Implements a modified version of the *Single-pass Parallel Prefix Scan with Decoupled Look-back* with variable stragegy: *scan-then-reduce* or *reduce-then-scan*. The strategy depends on the previous partition
descriptors to hide latencies. Performs at speeds close to `memcpy` and has only *2n* memory movements. Forward guarantees progress required.

### Cooperative Prefix Scan

Implements a *radix-N scan-then-propagate* strategy using Kogge-Stone group-scans and propagation fans. This implementation demonstrates the use of Cooperative Groups and is thus experimental. The computation is
performed in a single kernel launch and using the whole device. The bottleneck is currently the implementation of the SYCL group algorithms.

### Reduction

Parallel reduction algorithm using SYCL reductors and a recursive approach to unroll the loops. Memory bound, performance is thus equivalent to CUB.

## Intrinsics

CUDA intrinsics missing in SYCL such as *bit-reversal* *funnel-shifter* and many more. See [intrinsics.hpp](include/intrinsics.hpp) for a list of implemented functions.

## Runtime Index Wrapper

Functions and Classes used to access arrays-based types with a dynamic/runtime index. This allows to force the registerization of these arrays which is not possible otherwise, on GPU (annd even CPU!).

This allows also the compiler to see nex optimisations (see my SYCL Summer Session Talk). For best performance, if addressing bytes, pack them in 32/64 bit words and extract the byte yourself and use the library to
address words.

When writing to the array, if all threads access the same index only, there is a ligher/faster version available by defining `RUNTIME_IDX_STORE_USE_SWITCH`. Sometimes registerization won't happen using. Read speed is not
affected.

There is a read version, `runtime_index_wrapper_log` that uses a binary search. It reduces the number of steps, but often is slower because of thread divergence if not all the threads access the same value.

The wrapper has a specialisation for `std::array`, `sycl::vec`, `C-style arrays` and `sycl::id`. It also accepts any type that has a subscript operator, but then the used must put the maximum accessed index in the first
template parameter.

#### Benchmarks

With Google benchmark on a GTX 1660 Ti:

```
Benchmark                       Time      CPU  Iterations     UserCounters...
registerized_array/1073741824 57.2 ms  57.1 ms  12 items_per_second=1040G/s
stack_array/1073741824        2216 ms  2214 ms   1 items_per_second=37.3725G/s
```

And on a CPU:

```
Benchmark                       Time      CPU  Iterations     UserCounters...
registerized_array/1073741824  3.98 ms   3.95 ms   178 items_per_second=27.1994T/s
stack_array/1073741824         7.02 ms   6.99 ms   100 items_per_second=15.3549T/s
```

#### Use example

```C++
int array[10] = {0};
runtime_index_wrapper(array, i % 10, j) // performs array[i%10]=j and returns J
assert(j == runtime_index_wrapper(array, i % 10)); // reads the value
```

#### Class interface

```C++
std::array<size_t, 10> vec = {};
sycl::ext::runtime_wrapper acc(vec);

acc.write(i % 10, j) // performs array[i%10]=j and returns J
assert(j == acc[i % 10]); // reads the value
```

For types that have a subscript operator, but where the implementation is not able to extract the maximum accessed index, the user must provide the size when using read/write.

```
std::vector<int> vec ... ;
sycl::ext::runtime_wrapper acc(vec);

acc.write<vec_size>(i, val) // performs array[i%10]=j and returns J
assert(j == acc.read<vec_size>(i % 10)); // reads the value
```

## runtime_byte_array

This class solves the issue previously mentionned of storing bytes. It allows to increase the look-up time by packing the bytes in bigger words (default is 32 bits words).

```C++
runtime_byte_array<4> array{'S', 'Y', 'C', 'L', ..}; // Creates an array stored in a 32 bit word, by default

runtime_byte_array<256, uint64_t> array2{}; // Create an array of 256 0-intialised bytes stored in 32 64-bit words.

```

#### Some values

With an array of 12 bytes:

```
registerized_and_optimised_byte_array/1073741824       2266 ms         2264 ms            1 items_per_second=47.4268G/s
registerized_byte_array/1073741824                     2921 ms         2918 ms            1 items_per_second=36.8014G/s
stack_byte_array/1073741824                            3433 ms         3429 ms            1 items_per_second=31.3109G/s
```

For arrays of 32 bytes we're still faster that both the previous wrapper and a stack-frame array. For bigger arrays (>64), the regular stack-frame array is faster because our look-up time is too big. For smaller
arrays (<16), the byte selection from the words introduces overhead and the regular wrapper performs better. Benchmark your application.

## [Experimental] Cooperative groups

Device wide synchronisation functions. Kernel ranges are bound to ensure forward progress, If the Nvidia GPU is using AMS this might not be enough.


