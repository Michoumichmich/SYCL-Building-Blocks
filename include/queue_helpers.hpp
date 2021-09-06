#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

/**
 * Selects a CUDA device (but returns sometimes an invalid one)
 */
class cuda_selector : public sycl::device_selector {
public:
    int operator()(const sycl::device &device) const override {
        //return device.get_platform().get_backend() == sycl::backend::cuda && device.get_info<sycl::info::device::is_available>() ? 1 : -1;
        return device.is_gpu() && (device.get_info<sycl::info::device::driver_version>().find("CUDA") != std::string::npos) ? 1 : -1;
    }
};

/**
 * Tries to get a queue from a selector else returns the host device
 * @tparam strict if true will check whether the queue can run a trivial task which implied
 * that the translation unit needs to be compiler with support for the device you're selecting.
 */
template<bool strict = true, typename T>
inline sycl::queue try_get_queue(const T &selector) {
    auto exception_handler = [](const sycl::exception_list &exceptions) {
        for (std::exception_ptr const &e: exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
            }
            catch (std::exception const &e) {
                std::cout << "Caught asynchronous STL exception: " << e.what() << std::endl;
            }
        }
    };

    try {
        sycl::device dev = sycl::device(selector);
        sycl::queue q = sycl::queue(dev, exception_handler);
        if constexpr (strict) {
            try {
                if (dev.is_cpu() || dev.is_gpu()) { //Only CPU and GPU not host, dsp, fpga, ?...
                    q.submit([](sycl::handler &cgh) {
                        cgh.single_task([]() { (void) 0; });
                    }).wait_and_throw();
                }
            } catch (...) {
                std::cerr << "Warning: " << dev.get_info<sycl::info::device::name>() << " found but not working! Fall back on: ";
                q = sycl::queue(sycl::host_selector{}, exception_handler);
                std::cerr << q.get_device().get_info<sycl::info::device::name>() << '\n';
                return q;
            }
        }
        return q;
    }
    catch (...) {
        auto q = sycl::queue(sycl::host_selector{}, exception_handler);
        std::cerr << "Warning: Expected device not found! Fall back on: " << q.get_device().get_info<sycl::info::device::name>() << '\n';
        return q;
    }

}


#include <sys/mman.h>
#include <unistd.h>

/**
 * Checks whether a pointer was allocated on the host device
 * @see http://si-head.nl/articles/msync
 */
template<typename T>
inline bool valid_pointer(T *p) {
    auto pagesz = (size_t) sysconf(_SC_PAGESIZE); // Get page size and calculate page mask
    size_t pagemask = ~(pagesz - 1);
    void *base = (void *) (((size_t) p) & pagemask); // Calculate base address
    return msync(base, sizeof(T), MS_ASYNC) == 0;
}


/**
 * Checks whether a pointer is usable on a queue to perform computation.
 * @tparam T
 * @tparam debug
 * @param ptr
 * @param q
 * @return
 */
template<typename T, bool debug = false>
inline bool is_ptr_usable(const T *ptr, const sycl::queue &q) {
    if (q.get_device().is_host()) {
        return valid_pointer(ptr);
    }

    try {
        sycl::get_pointer_device(ptr, q.get_context());
        sycl::usm::alloc alloc_type = sycl::get_pointer_type(ptr, q.get_context());
        if constexpr(debug) {
            std::cerr << "Allocated on:" << q.get_device().get_info<sycl::info::device::name>() << " USM type: ";
            switch (alloc_type) {
                case sycl::usm::alloc::host:
                    std::cerr << "alloc::host" << '\n';
                    break;
                case sycl::usm::alloc::device:
                    std::cerr << "alloc::device" << '\n';
                    break;
                case sycl::usm::alloc::shared:
                    std::cerr << "alloc::shared" << '\n';
                    break;
                case sycl::usm::alloc::unknown:
                    std::cerr << "alloc::unknown" << '\n';
                    break;
            }
        }
        return alloc_type == sycl::usm::alloc::shared // Shared memory is ok
               || alloc_type == sycl::usm::alloc::device // Device memory is ok
               || alloc_type == sycl::usm::alloc::host;
    } catch (...) {
        if constexpr (debug) {
            std::cerr << "Not allocated on:" << q.get_device().get_info<sycl::info::device::name>() << '\n';
        }
        return false;
    }
}


/**
 * Usefull for memory bound computation.
 * Returns CPU devices that represents different numa nodes.
 * @return
 */
inline std::vector<sycl::device> get_cpu_runners_numa() {
    try {
        sycl::device d{sycl::cpu_selector{}};
        auto numa_nodes = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);
        return numa_nodes;
    }
    catch (...) {
        return {sycl::device{sycl::host_selector{}}};
    }
}


