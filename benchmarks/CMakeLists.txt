cmake_minimum_required(VERSION 3.0)
project(SYCL_Intrinsics LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include_directories(../include)

# If you're using the DPCPP compiler, these flags will be used. Set here the devices you want to target
set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xcuda-ptxas -v -DSYCL_USE_NATIVE_FP_ATOMICS -ffast-math -fcuda-approx-transcendentals -ffp-contract=fast -fcuda-flush-denormals-to-zero -fsycl-id-queries-fit-in-int -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_75)
#set(DPCPP_FLAGS -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda)
include(../cmake/FindSYCL.cmake)

# Default C++ Flags for warnings and optimisation
set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -fno-common -Winit-self -Wuninitialized -Woverloaded-virtual")
set(EXTRA_W_FLAGS "-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5") #-Wnoexcept -Wold-style-cast -Wstrict-null-sentinel -switch-default -Wlogical-op
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-inline-namespace-reopened-noninline -Wno-undef -Wno-unused")
set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")

# Adding the flags to the targets
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS} ${DISABLED_WARNINGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} ${EXTRA_W_FLAGS} ${DISABLED_WARNINGS}")

enable_testing()

include(FetchContent)

# Externally provided libraries
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG master)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG master) # need master for benchmark::benchmark

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)

add_executable(benchmark_scan benchmark_scan.cpp)
target_link_libraries(benchmark_scan PRIVATE benchmark::benchmark)
add_sycl_to_target(TARGET benchmark_scan SOURCES benchmark_scan.cpp)

add_executable(benchmark_reduce benchmark_reduce.cpp)
target_link_libraries(benchmark_reduce PRIVATE benchmark::benchmark)
add_sycl_to_target(TARGET benchmark_reduce SOURCES benchmark_reduce.cpp)

add_executable(benchmark_alignement benchmark_local_alignement.cpp)
target_link_libraries(benchmark_alignement PRIVATE benchmark::benchmark)
add_sycl_to_target(TARGET benchmark_alignement SOURCES benchmark_local_alignement.cpp)

add_executable(benchmark_registers benchmark_array_register.cpp)
target_link_libraries(benchmark_registers PRIVATE benchmark::benchmark)
add_sycl_to_target(TARGET benchmark_registers SOURCES benchmark_array_register.cpp)

add_executable(benchmark_byte_array benchmark_byte_array.cpp)
target_link_libraries(benchmark_byte_array PRIVATE benchmark::benchmark)
add_sycl_to_target(TARGET benchmark_byte_array SOURCES benchmark_byte_array.cpp)

add_executable(benchmark_bit_array benchmark_bit_array.cpp)
target_link_libraries(benchmark_bit_array PRIVATE benchmark::benchmark)
add_sycl_to_target(TARGET benchmark_bit_array SOURCES benchmark_bit_array.cpp)


