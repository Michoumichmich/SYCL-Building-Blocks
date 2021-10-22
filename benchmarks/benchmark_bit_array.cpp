#define CONSTEVAL_REGISTER_SHORTCUT

#include <benchmark/benchmark.h>
#include <register_bit_array.hpp>
#include <numeric>


constexpr uint a = 1140671485;
constexpr uint c = 12820163;
constexpr uint m = 1 << 24;
constexpr int array_size = 128;
constexpr int iter = 200;


size_t benchmark_runtime_bit_array(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    uint *ptr = sycl::malloc_device<uint>(size, q);
    q.parallel_for<class runtime_bit_array_kernel_optimised>(sycl::range<1>(size), [=](sycl::id<1> id) {
        uint rand_num = id.get(0);
        register_bit_array<array_size> arr{};

        for (int i = 0; i < iter; ++i) {
            rand_num = (a * rand_num + c) % m;
            auto write_idx = rand_num % array_size;
            auto read_idx = (i * rand_num) % array_size;
            auto flip_idx = (i + rand_num) % array_size;

#pragma unroll
            for (int j = 0; j < array_size / 2; ++j) {
                arr.swap(j, array_size - j - 1);
                //arr.swap(array_size - j - 1, j);
            }

            arr.write(write_idx, arr[read_idx]);
            arr.reset(read_idx);
            arr.flip(flip_idx);
        }
        ptr[id.get(0)] = arr.count();
    }).wait();
    //std::cout << std::accumulate(ptr, ptr + size, 0) << std::endl;
    sycl::free(ptr, q);
    return size * iter;
}

size_t benchmark_runtime_bit_array_stack(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    uint *ptr = sycl::malloc_device<uint>(size, q);
    q.parallel_for<class runtime_bit_array_stack_kernel>(sycl::range<1>(size), [=](sycl::id<1> id) {
        uint rand_num = id.get(0);
        std::array<bool, array_size> arr{};
        arr.fill(false);
        for (int i = 0; i < iter; ++i) {
            rand_num = (a * rand_num + c) % m;
            auto write_idx = rand_num % array_size;
            auto read_idx = (i * rand_num) % array_size;
            auto flip_idx = (i + rand_num) % array_size;

#pragma unroll
            for (int j = 0; j < array_size / 2; ++j) {
                std::swap(arr[j], arr[array_size - j - 1]);
                //std::swap(arr[array_size - j - 1], arr[j]);
            }

            arr[write_idx] = arr[read_idx];
            arr[read_idx] = false;
            arr[flip_idx] ^= 1;
        }

        ptr[id.get(0)] = std::count(arr.begin(), arr.end(), true);
    }).wait();
    //std::cout << std::accumulate(ptr, ptr + size, 0) << std::endl;
    sycl::free(ptr, q);
    return size * iter;
}

void registerized_and_optimised_bit_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_runtime_bit_array(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}


void stack_bit_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_runtime_bit_array_stack(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}


BENCHMARK(registerized_and_optimised_bit_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 67108864);
BENCHMARK(stack_bit_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 67108864);


BENCHMARK_MAIN();