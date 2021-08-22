# SYCL Building Blocks

Header-based SYCL reusable algorithms and data structures.

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

## Parallel Primitives
