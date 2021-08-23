#pragma once

#include <sycl/sycl.hpp>

#ifndef ATOMIC_REF_NAMESPACE
#ifdef USING_DPCPP
#define ATOMIC_REF_NAMESPACE sycl::ext::oneapi
#else
#define ATOMIC_REF_NAMESPACE sycl
#endif
#endif


namespace parallel_primitives::decoupled_lookback_internal {

    enum class status : char {
        aggregate_available,
        prefix_available,
        invalid
    };

    template<typename T, typename func>
    struct data {
        T value_ = get_init<T, func>();
        status status_flag_ = status::invalid;
    };

    template<typename T, typename func, bool use_atomics>
    class partition_descriptor_impl;

    template<typename T, typename func>
    class partition_descriptor_impl<T, func, true> {
        using atomic_ref_t = ATOMIC_REF_NAMESPACE::atomic_ref<
                uint64_t,
                ATOMIC_REF_NAMESPACE::memory_order::relaxed,
                ATOMIC_REF_NAMESPACE::memory_scope::device,
                sycl::access::address_space::global_space
        >;

    private:

        union atomic_union_storage {
            uint64_t storage;
            data<T, func> data{};
        } packed{};

    public:
        inline void set_aggregate(const T &aggregate) {
            atomic_ref_t ref(packed.storage);
            ref.store(atomic_union_storage{.data={.value_ = aggregate, .status_flag_=status::aggregate_available}}.storage);
        }

        inline void set_prefix(const T &prefix) {
            atomic_ref_t ref(packed.storage);
            ref.store(atomic_union_storage{.data={.value_ = prefix, .status_flag_=status::prefix_available}}.storage);
        }

        static T run_look_back(partition_descriptor_impl *ptr_base, const size_t &partition_id) {
            T tmp = get_init<T, func>();
            const func op{};
            for (auto partition = partition_id; partition_id > 0;) {
                partition--;
                atomic_ref_t ref(ptr_base[partition].packed.storage);
                atomic_union_storage data{.storage = ref.load()};

                while (data.data.status_flag_ == status::invalid) {/* wait */
                    data.storage = ref.load();
                }

                if (data.data.status_flag_ == status::prefix_available) {
                    return op(tmp, data.data.value_);
                }
                //if (ptr_base[partition].status_flag_ == status::aggregate_available) {
                tmp = op(tmp, data.data.value_);
                //}
            }
            return tmp;
        }

        static std::optional<T> is_ready(partition_descriptor_impl *ptr_base, const size_t &partition_id) {
            if (partition_id == 0) {
                return get_init<T, func>();
            }
            atomic_ref_t ref(ptr_base[partition_id - 1].packed.storage);
            atomic_union_storage data{.storage = ref.load()};
            if (data.data.status_flag_ == status::prefix_available) {
                return data.data.value_;
            } else {
                return std::nullopt;
            }
        }

    };

    template<typename T, typename func>
    class partition_descriptor_impl<T, func, false> {
    private:
        T inclusive_prefix_ = get_init<T, func>();
        T aggregate_ = get_init<T, func>();
        status status_flag_ = status::invalid;

    public:
        inline void set_aggregate(const T &aggregate) {
            aggregate_ = aggregate;
            sycl::atomic_fence(sycl::memory_order_seq_cst, sycl::memory_scope_work_group);
            status_flag_ = status::aggregate_available;
        }

        inline void set_prefix(const T &prefix) {
            inclusive_prefix_ = prefix;
            sycl::atomic_fence(sycl::memory_order_seq_cst, sycl::memory_scope_work_group);
            status_flag_ = status::prefix_available;
        }

        static T run_look_back(volatile partition_descriptor_impl *ptr_base, const size_t &partition_id) {
            T tmp = get_init<T, func>();
            const func op{};
            for (auto partition = partition_id; partition_id > 0;) {
                partition--;
                while (ptr_base[partition].status_flag_ == status::invalid) {/* wait */}
                if (ptr_base[partition].status_flag_ == status::prefix_available) {
                    return op(tmp, ptr_base[partition].inclusive_prefix_);
                }
                //if (ptr_base[partition].status_flag_ == status::aggregate_available) {
                tmp = op(tmp, ptr_base[partition].aggregate_);
                //}
            }
            return tmp;
        }

        static std::optional<T> is_ready(const partition_descriptor_impl *ptr_base, const size_t &partition_id) {
            if (partition_id == 0) {
                return get_init<T, func>();
            } else if (ptr_base[partition_id - 1].status_flag_ == status::prefix_available) {
                return ptr_base[partition_id - 1].inclusive_prefix_;
            } else {
                return std::nullopt;
            }
        }
    };
}
