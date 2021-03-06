#pragma once

#include "../../intrinsics.hpp"
#include "common.h"

namespace parallel_primitives::decoupled_lookback_internal {

    using internal::get_init;

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
    private:

        using storage_type = typename sycl::ext::smallest_storage_t<data<T, func>>::type;

        union atomic_union_storage {
            storage_type storage;
            data<T, func> data{};
        } packed{};

        using atomic_ref_t = sycl::atomic_ref<
                storage_type,
                sycl::memory_order::relaxed,
                sycl::memory_scope::work_group,
                sycl::access::address_space::global_space
        >;

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
                //  sycl::ext::prefetch_constant(ptr_base + partition - 1);
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
            //sycl::ext::prefetch_constant(this);
        }

        inline void set_prefix(const T &prefix) {
            inclusive_prefix_ = prefix;
            sycl::atomic_fence(sycl::memory_order_seq_cst, sycl::memory_scope_work_group);
            status_flag_ = status::prefix_available;
            //      sycl::ext::prefetch_constant(this);
        }


        static T run_look_back(volatile partition_descriptor_impl *ptr_base, const size_t &partition_id) {
            T tmp = get_init<T, func>();
            const func op{};
            for (auto partition = partition_id; partition_id > 0;) {
                partition--;
                sycl::ext::prefetch(ptr_base + partition - 1);
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
