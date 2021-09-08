#pragma once

#include <sycl/sycl.hpp>
#include <memory>
#include <utility>


namespace usm_smart_ptr {
    using namespace sycl::usm;

    /**
     *  SYCL USM Deleter. The std::unique_ptr deleter takes only the pointer
     *  to delete as an argument so that's the only work-around.
     * @tparam T
     */
    template<typename T>
    struct usm_deleter {
        const sycl::queue q_;

        explicit usm_deleter(sycl::queue q) : q_(std::move(q)) {}

        void operator()(T *ptr) const noexcept {
            if (ptr) {
                sycl::free(ptr, q_);
            }
        }
    };


    template<typename sycl::usm::alloc location>
    static constexpr sycl::access::address_space get_space() {
        if constexpr(location == sycl::usm::alloc::shared) {
            return sycl::access::address_space::global_space;
        } else if constexpr(location == sycl::usm::alloc::device) {
            return sycl::access::address_space::global_device_space;
        } else if constexpr(location == sycl::usm::alloc::host) {
            return sycl::access::address_space::global_host_space;
        }
    }


    /**
     * Wrapper for a std::unique_ptr that calls the SYCL deleter (sycl::free).
     * Also holds the number of elements allocated.
     * @tparam T
     * @tparam location
     */
    template<typename T, sycl::usm::alloc location>
    class usm_unique_ptr : public std::unique_ptr<T, usm_deleter<T>> {
    private:
        const size_t count_{};
    public:
        [[nodiscard]] usm_unique_ptr(size_t count, sycl::queue q)
                : std::unique_ptr<T, usm_deleter<T>>(sycl::malloc<T>(count, q, location), usm_deleter<T>{q}),
                  count_(count) {}

        [[nodiscard]] explicit usm_unique_ptr(sycl::queue q)
                : usm_unique_ptr(1, q) {}

        [[nodiscard]] inline size_t size_bytes() const noexcept { return count_ * sizeof(T); }

        [[nodiscard]] inline size_t size() const noexcept { return count_; }

        [[nodiscard]] inline sycl::multi_ptr<T, get_space<location>()> get_multi() const noexcept {
            return {std::unique_ptr<T, usm_deleter<T>>::get()};
        }

        [[nodiscard]] inline sycl::span<T> get_span() const noexcept {
            return {std::unique_ptr<T, usm_deleter<T>>::get(), count_};
        }

    };


    /**
     * Same interface as usm_unique_ptr
     * @tparam T
     * @tparam location
     */
    template<typename T, sycl::usm::alloc location>
    class usm_shared_ptr : public std::shared_ptr<T> {
    private:
        const size_t count_{};

    public:
        [[nodiscard]] usm_shared_ptr(size_t count, sycl::queue q)
                : std::shared_ptr<T>(sycl::malloc<T>(count, q, location), usm_deleter<T>{q}),
                  count_(count) {}

        [[nodiscard]] explicit usm_shared_ptr(sycl::queue q)
                : usm_shared_ptr(1, q) {}

        [[nodiscard]] inline size_t size_bytes() const noexcept { return count_ * sizeof(T); }

        [[nodiscard]] inline size_t size() const noexcept { return count_; }

        [[nodiscard]] inline sycl::span<T> get_span() const noexcept {
            return {std::shared_ptr<T>::get(), count_};
        }

        [[nodiscard]] inline sycl::multi_ptr<T, get_space<location>()> get_multi() const noexcept {
            return {std::shared_ptr<T>::get()};
        }

    };

}