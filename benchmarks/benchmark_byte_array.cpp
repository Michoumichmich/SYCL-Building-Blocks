#include <runtime_byte_array.hpp>
#include <benchmark/benchmark.h>


constexpr uint a = 1140671485;
constexpr uint c = 12820163;
constexpr uint m = 1 << 24;
constexpr int array_size = 16;
constexpr int iter = 200;


size_t benchmark_runtime_byte_array(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<class runtime_byte_array_kernel_optimised>(sycl::range<1>{size}, [=](sycl::id<1> id) {
        uint rand_num = id.get(0);
        runtime_byte_array<array_size> arr{static_cast<unsigned char>(*ptr), static_cast<unsigned char>(*ptr)};

        for (int i = 0; i < iter; ++i) {
            rand_num = (a * rand_num + c) % m;
            auto write_idx = rand_num % array_size;
            auto read_idx = (i * rand_num) % array_size;
            auto flip_idx = (i + rand_num) % array_size;

            arr.write(rand_num % 2, arr[read_idx]);
            arr.write(rand_num % 2, arr[write_idx]);
            arr.write(flip_idx, rand_num);
            arr.write(read_idx, arr[rand_num % 4]);
        }
        *ptr = arr[*ptr % array_size];
    }).wait();
    return size * iter;
}

size_t benchmark_runtime_byte_array_non_specialised(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<class runtime_byte_array_non_specialised_kernel>(sycl::range<1>{size}, [=](sycl::id<1> id) {
        uint rand_num = id.get(0);
        std::array<uint8_t, array_size> arr_storage{static_cast<unsigned char>(*ptr), static_cast<unsigned char>(*ptr)};
        sycl::ext::runtime_wrapper arr(arr_storage);

        for (int i = 0; i < iter; ++i) {
            rand_num = (a * rand_num + c) % m;
            auto write_idx = rand_num % array_size;
            auto read_idx = (i * rand_num) % array_size;
            auto flip_idx = (i + rand_num) % array_size;

            arr.write(rand_num % 2, arr[read_idx]);
            arr.write(rand_num % 2, arr[write_idx]);
            arr.write(flip_idx, rand_num);
            arr.write(read_idx, arr[rand_num % 4]);
        }
        *ptr = arr[*ptr % array_size];
    }).wait();
    return size * iter;
}

size_t benchmark_runtime_byte_array_stack(size_t size) {
    sycl::queue q{sycl::gpu_selector{}};
    uint *ptr = sycl::malloc_device<uint>(1, q);
    q.parallel_for<class runtime_byte_array_stack_kernel>(sycl::range<1>{size}, [=](sycl::id<1> id) {
        uint rand_num = id.get(0);
        std::array<uint8_t, array_size> arr{static_cast<unsigned char>(*ptr), static_cast<unsigned char>(*ptr)};

        for (int i = 0; i < iter; ++i) {
            rand_num = (a * rand_num + c) % m;
            auto write_idx = rand_num % array_size;
            auto read_idx = (i * rand_num) % array_size;
            auto flip_idx = (i + rand_num) % array_size;

            arr[rand_num % 2] = arr[read_idx];
            arr[rand_num % 2] = arr[write_idx];
            arr[flip_idx] = rand_num;
            arr[read_idx] = arr[rand_num % 4];
        }
        *ptr = arr[*ptr % array_size];
    }).wait();
    return size * iter;
}

void registerized_and_optimised_byte_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_runtime_byte_array(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}

void registerized_byte_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_runtime_byte_array_non_specialised(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}


void stack_byte_array(benchmark::State &state) {
    auto size = static_cast<size_t>(state.range(0));
    size_t processed_items = 0;
    for (auto _: state) {
        processed_items += benchmark_runtime_byte_array_stack(size);
    }
    state.SetItemsProcessed(static_cast<int64_t>(processed_items));
}


BENCHMARK(registerized_and_optimised_byte_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 33554432);
BENCHMARK(registerized_byte_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 33554432);
BENCHMARK(stack_byte_array)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(3'000'000, 33554432);



BENCHMARK_MAIN();