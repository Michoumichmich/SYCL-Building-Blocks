/**
    Copyright 2021 Codeplay Software Ltd.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use these files except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    For your convenience, a copy of the License has been included in this
    repository.

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <numeric>

#ifndef ATOMIC_REF_NAMESPACE
#ifdef USING_DPCPP //SYCL_IMPLEMENTATION_ONEAPI
#define ATOMIC_REF_NAMESPACE sycl::ext::oneapi
#else
#define ATOMIC_REF_NAMESPACE sycl
#endif
#endif

template<int dim>
class nd_range_barrier {
private:
    using mask_t = uint64_t;
    mask_t groups_waiting_ = 0;
    const mask_t barrier_mask_ = 0;
    mask_t reached_ = 0;

    static mask_t compute_barrier_mask(size_t group_count, const std::vector<size_t> &cooperating_groups) {
        mask_t out = 0;
        if (cooperating_groups.empty()) {
            for (size_t gr = 0; gr < group_count; ++gr) {
                out |= mask_t(1) << gr;
            }
        } else {
            for (auto e: cooperating_groups) {
                if (e >= group_count) throw std::out_of_range("Making barrier on out of range group");
                out |= mask_t(1) << e;
            }
        }
        return out;
    }

    template<typename func>
    static mask_t compute_barrier_mask(size_t group_count, func &&predicate) {
        mask_t out = 0;
        for (size_t gr = 0; gr < group_count; ++gr) {
            if (predicate(gr)) {
                out |= mask_t(1) << gr;
            }
        }
        return out;
    }

    static mask_t compute_item_mask(const sycl::nd_item<dim> &this_item) {
        return mask_t(1) << this_item.get_group_linear_id();
    }


    void perform_check(sycl::queue &q, const sycl::nd_range<dim> &kernel_range) {
        if (kernel_range.get_group_range().size() > sizeof(mask_t) * 8) {
            throw std::runtime_error("Not implemented.");
        }
        if (kernel_range.get_group_range().size() > q.get_device().get_info<sycl::info::device::max_compute_units>()) {
            throw std::runtime_error("Too much groups requested on cooperative barrier. Forward progress not guaranteed.");
        }

        if (kernel_range.get_local_range().size() > q.get_device().get_info<sycl::info::device::max_work_group_size>()) {
            throw std::runtime_error("Too much items per group. Forward progress not guaranteed.");
        }
    }

    nd_range_barrier(sycl::queue q, const sycl::nd_range<dim> &kernel_range, const std::vector<size_t> &cooperating_groups)
            : barrier_mask_(compute_barrier_mask(kernel_range.get_group_range().size(), cooperating_groups)) {
        perform_check(q, kernel_range);
    }

    template<typename func>
    nd_range_barrier(sycl::queue q, const sycl::nd_range<dim> &kernel_range, func &&predicate)
            :barrier_mask_(compute_barrier_mask(kernel_range.get_group_range().size(), predicate)) {
        perform_check(q, kernel_range);
    }

public:

    /**
     * Constructor helpers
     */
    template<typename func>
    static nd_range_barrier<dim> *make_barrier(sycl::queue &q, const sycl::nd_range<dim> &kernel_range, const func &predicate) {
        auto barrier = sycl::malloc_shared<nd_range_barrier<dim>>(1, q);
        return new(barrier) nd_range_barrier<dim>(q, kernel_range, predicate);
    }


    static nd_range_barrier<dim> *make_barrier(sycl::queue &q, const sycl::nd_range<dim> &kernel_range, const std::vector<size_t> &cooperating_groups = {}) {
        auto barrier = sycl::malloc_shared<nd_range_barrier<dim>>(1, q);
        return new(barrier) nd_range_barrier<dim>(q, kernel_range, cooperating_groups);
    }

    void wait(sycl::nd_item<dim> this_item) {
        const mask_t this_group_mask = compute_item_mask(this_item);

        if ((this_group_mask & barrier_mask_) == 0) return;

        this_item.barrier(sycl::access::fence_space::local_space);
        /* Choosing one work item to perform the work */
        if (this_item.get_local_linear_id() == 0) {
            using atomic_ref_t = ATOMIC_REF_NAMESPACE::atomic_ref<
                    mask_t,
                    ATOMIC_REF_NAMESPACE::memory_order::relaxed, //TODO acq_rel
                    ATOMIC_REF_NAMESPACE::memory_scope::device,
                    sycl::access::address_space::global_space
            >;
            atomic_ref_t groups_waiting_ref(groups_waiting_);
            atomic_ref_t barrier_reached_ref(reached_);

            /* Waiting before entering the barrier */
            while (barrier_reached_ref.load() != 0) {}

            /* Registring this group at the barrier. */
            groups_waiting_ref.fetch_or(this_group_mask);

            if (groups_waiting_ref.load() == barrier_mask_) {
                barrier_reached_ref.store(1);
            } else {
                while (barrier_reached_ref.load() != 1) {}
            }

            /* This group leaves the barrier. */
            groups_waiting_ref.fetch_and(~this_group_mask);

            if (groups_waiting_ref.load() == 0) {
                barrier_reached_ref.store(0);
            } else {
                while (barrier_reached_ref.load() != 0) {}
            }
        }

        this_item.barrier(sycl::access::fence_space::local_space);
    }
};


template<typename KernelName>
sycl::nd_range<1> get_max_occupancy(sycl::queue &q, size_t local_mem = 0) {
    (void) local_mem;
    sycl::kernel_id id = sycl::get_kernel_id<KernelName>();
    auto kernel = sycl::get_kernel_bundle<sycl::bundle_state::executable>(q.get_context()).get_kernel(id);
    // size_t private_mem_size = kernel.get_info<sycl::info::kernel_device_specific::private_mem_size>(q.get_device());
    // size_t registers = private_mem_size / 2; // find a way to get that value
    // printf("Kernel private mem size: %lu", private_mem_size);

    size_t max_items = kernel.get_info<sycl::info::kernel_device_specific::work_group_size>(q.get_device());
    size_t max_groups = (uint32_t) q.get_device().get_info<sycl::info::device::max_compute_units>();

    return {sycl::range<1>(max_items * max_groups), sycl::range<1>(max_items)};
}



/*
    class cooperative_demo;
    static inline void cooperative_group_demo(sycl::queue q) {
    auto kernel_param = sycl::nd_range<1>({1024 * 16}, {1024});
    auto grid_barrier = nd_range_barrier<1>::make_barrier(q, kernel_param);
    auto pair_barrier = nd_range_barrier<1>::make_barrier(q, kernel_param, {0, 10});
    q.submit([&](sycl::handler &cgh) {
        sycl::stream os(1024, 256, cgh);
        cgh.parallel_for<class cooperative_demo>(
                kernel_param,
                [=](sycl::nd_item<1> it) {
                    if (it.get_local_linear_id() == 0) os << "Pos 0: " << it.get_group_linear_id() << sycl::endl;
                    grid_barrier->wait(it);
                    if (it.get_local_linear_id() == 0) os << "Pos 1: " << it.get_group_linear_id() << sycl::endl;
                    pair_barrier->wait(it);
                    if (it.get_local_linear_id() == 0) os << "Pos 2: " << it.get_group_linear_id() << sycl::endl;
                    grid_barrier->wait(it);
                    if (it.get_local_linear_id() == 0) os << "Pos 3: " << it.get_group_linear_id() << sycl::endl;
                });
    }).wait();
}*/