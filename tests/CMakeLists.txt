cmake_minimum_required(VERSION 3.9)
project(SYCL_Intrinsics LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)


include_directories(../include)

# If you're using the DPCPP compiler, these flags will be used. Set here the devices you want to target
set(DPCPP_FLAGS -fsycl -fsycl-targets=nvptx64-nvidia-cuda--sm_75 -DSYCL_USE_NATIVE_FP_ATOMICS -fcuda-flush-denormals-to-zero -ffast-math -fcuda-approx-transcendentals -ffp-contract=fast -fcuda-flush-denormals-to-zero -fcuda-approx-transcendentals -fsycl-id-queries-fit-in-int)
#set(DPCPP_FLAGS -fsycl -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice)
include(../cmake/FindSYCL.cmake)

# Default C++ Flags for warnings and optimisation
set(WARNING_FLAGS "-Wall -Wextra -Wshadow -Wdouble-promotion -fno-common -Winit-self -Wuninitialized -Woverloaded-virtual")
set(EXTRA_W_FLAGS "-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5") #-Wnoexcept -Wold-style-cast -Wstrict-null-sentinel -switch-default -Wlogical-op
set(DISABLED_WARNINGS "-Wno-c++20-extensions -Wno-inline-namespace-reopened-noninline -Wno-undef -Wno-unused")
set(OPT_FLAGS "-march=native -mtune=native -Ofast -fomit-frame-pointer")

# Adding the flags to the targets
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS} ${DISABLED_WARNINGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${WARNING_FLAGS} ${EXTRA_W_FLAGS} ${DISABLED_WARNINGS} -v")


include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


enable_testing()

set(all_sources
        test_reduction.cpp
        test_intrinsics.cpp
        test_alignement.cpp
        test_cooperative_groups.cpp
        test_pointers.cpp
        test_queue_helpers.cpp
        test_scan.cpp
        )


add_executable(
        tests
        ${all_sources}
)

add_sycl_to_target(TARGET tests SOURCES ${all_sources})

target_link_libraries(tests PUBLIC gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)