# SYCL Building Blocks

Header-based SYCL reusable algorithms and data structures.

## Intrinsics

## [Experimental] Cooperative groups

Device wide synchronisation library. Kernel ranges are bound to ensure forward progress, If the Nvidia GPU is using AMS this might not be enough.

## Parallel Primitives

### Prefix Scan

### Cooperative Prefix Scan

Implements a radix-N scan-then-propagate strategy using Kogge-Stone group-scans and propagation fans. This implementation demonstrates the use of Cooperative Groups and is thus experimenntal. The computation is performed
in a single kernel launch and using the whole device. The bottleneck is currently the implementation of the SYCL group algorithms.

