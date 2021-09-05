#include <local_mem_alignement_checker.hpp>
#include <sycl/sycl.hpp>
#include <benchmark/benchmark.h>

struct bad_align {
    float a, b, c, d;
};

struct better_align {
    float a, b, c, d;
    float pad;
};

template<typename T>
struct local_mem_benchmark_kernel;

template<typename T>
size_t benchmark_local_mem(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    auto local_mem_size = q.get_device().get_info<sycl::info::device::local_mem_size>() / sizeof(T);
    sycl::nd_range<1> range({1024 * 24}, {1024});
    q.submit([&](sycl::handler &cgh) {
        sycl::accessor<T, 1, sycl::access_mode::read_write, sycl::access::target::local> acc({local_mem_size}, cgh);
        q.parallel_for<local_mem_benchmark_kernel<T>>(range, [=](sycl::nd_item<1> it) {
            auto id = it.get_local_linear_id();
            for (auto i = 0; i < size; ++i) {
                auto src = (i + id) % local_mem_size;
                auto dst = (i + id + 32) % local_mem_size;
                acc[src].a = acc[dst].b;
                acc[src].c = acc[dst].d;
                acc[src].b = acc[dst].c;
                acc[src].d = acc[dst].a;
            }
        });
    }).wait();
    return 8 * range.get_global_range().size() * sizeof(float) * size;
}


void bad_align_bench(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_bytes = 0;
    for (auto _: state) {
        processed_bytes += benchmark_local_mem<bad_align>(size);
    }
    state.SetBytesProcessed(static_cast<int64_t>(processed_bytes));
}

void better_align_bench(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    assert_local_alignement<better_align>();
    size_t processed_bytes = 0;
    for (auto _: state) {
        processed_bytes += benchmark_local_mem<better_align>(size);
    }
    state.SetBytesProcessed(static_cast<int64_t>(processed_bytes));
}

BENCHMARK(better_align_bench)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(500'000, 50'000'000);
BENCHMARK(bad_align_bench)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(500'000, 50'000'000);


BENCHMARK_MAIN();