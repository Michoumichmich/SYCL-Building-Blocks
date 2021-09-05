#include <parallel_primitives/reduction.hpp>
#include <benchmark/benchmark.h>
#include <usm_smart_ptr.hpp>

using namespace usm_smart_ptr;
using namespace parallel_primitives;


void reduce_benchmark(benchmark::State &state) {
    using T = uint32_t;
    static sycl::queue q{sycl::gpu_selector{}};
    auto in = usm_unique_ptr<T, alloc::device>(state.range(0), q);

    q.fill(in.get(), T(1), in.size()).wait();

    T res;
    for (auto _: state) {
        res = reduce_device<sycl::plus<>>(q, in.get_span());
    }

    // reduce<sycl::plus<>>(q, in.get_span());

    state.SetBytesProcessed(state.iterations() * in.size_bytes());
    std::stringstream str;
    str << "Result: " << res;
    state.SetLabel(str.str());
}


void reduce_benchmark2(benchmark::State &state) {
    using T = uint64_t;
    static sycl::queue q{sycl::gpu_selector{}};
    auto size = static_cast<size_t>(state.range(0));
    auto in = usm_unique_ptr<T, alloc::shared>(size, q);

    std::iota(in.get(), in.get() + in.size(), 0);

    T res;
    for (auto _: state) {
        res = reduce_device<sycl::plus<>>(q, in.get_span());
    }

    state.SetBytesProcessed(state.iterations() * in.size_bytes());
    std::stringstream str;
    str << "Result: " << res << " expected: " << (size * (size - 1)) / 2;
    state.SetLabel(str.str());
}

BENCHMARK(reduce_benchmark)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1, 1'300'000'000);
BENCHMARK(reduce_benchmark2)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1, 1'300'000'000);

// Run benchmark
BENCHMARK_MAIN();
