#include <parallel_primitives/reduction.hpp>
#include <benchmark/benchmark.h>

/**
 * Sum should converge to PI
 */
void basel_problem_reduce(benchmark::State &state) {
    static sycl::queue q{sycl::gpu_selector{}};
    using namespace parallel_primitives;
    auto in = sycl::malloc_device<half>(state.range(0), q);

    for (auto _ : state) {
        reduce_device<sycl::plus<>>(q, in, state.range(0));
    }
    state.SetItemsProcessed(state.range(0));
    sycl::free(in, q);
}


BENCHMARK(basel_problem_reduce)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1'000'000, 2'000'000'000);

// Run benchmark
BENCHMARK_MAIN();
