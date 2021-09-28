set(all_sources
        tests/test_reduction.cpp
        tests/test_intrinsics.cpp
        tests/test_alignement.cpp
        tests/test_cooperative_groups.cpp
        tests/test_pointers.cpp
        tests/test_queue_helpers.cpp
        tests/test_scan.cpp
        tests/test_runtime_index_wrapper.cpp
        )

add_executable(
        tests
        ${all_sources}
)

add_sycl_to_target(TARGET tests SOURCES ${all_sources})

target_link_libraries(tests PUBLIC gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)