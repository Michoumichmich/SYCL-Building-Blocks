#include <parallel_primitives/reduction.hpp>
#include <benchmark/benchmark.h>

void reduce_benchmark(benchmark::State &state) {
    static sycl::queue q{sycl::cpu_selector{}};
    auto size = static_cast<size_t>(state.range(0));
    using namespace parallel_primitives;
    auto in = sycl::malloc_device<int32_t>(size, q);

    q.fill(in, 1, size).wait();

    int32_t res;
    for (auto _ : state) {
        res = reduce_device<sycl::plus<>>(q, in, size);
    }
    state.SetItemsProcessed(state.range(0));
    std::stringstream str;
    str << "Result: " << res;
    state.SetLabel(str.str());
    sycl::free(in, q);
}


BENCHMARK(reduce_benchmark)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1, 2'000'000'000);

// Run benchmark
BENCHMARK_MAIN();
