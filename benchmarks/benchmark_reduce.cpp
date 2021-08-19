#include <parallel_primitives/reduction.hpp>
#include <benchmark/benchmark.h>

void reduce_benchmark(benchmark::State &state) {
    using T = float;
    static sycl::queue q{sycl::gpu_selector{}};
    auto size = static_cast<size_t>(state.range(0));
    using namespace parallel_primitives;
    auto in = sycl::malloc_device<T>(size, q);

    q.fill(in, T(1), size).wait();

    T res;
    for (auto _ : state) {
        res = reduce_device<sycl::plus<>>(q, in, size);
    }

    state.SetBytesProcessed(sizeof(T) * state.iterations() * state.range(0));
    std::stringstream str;
    str << "Result: " << res;
    state.SetLabel(str.str());
    sycl::free(in, q);
}


BENCHMARK(reduce_benchmark)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1, 1'300'000'000);

// Run benchmark
BENCHMARK_MAIN();
