#pragma once

// This file provides two functions to help write GPU elementwise kernels:
//
//   gpu_kernel(TensorIterator iter, <lambda>)
//   gpu_kernel_with_scalars(TensorIterator iter, <lambda>)
//
// The gpu_kernel_with_scalars generates specializations that support a
// single scalar CPU argument, such as from `cuda_tensor + 5`. The CPU scalar
// is lifted to a kernel parameter instead of copying to device memory.
// This should be  used in conjunction with TensorIterator::allow_cpu_scalars_,
// which is the default for TensorIterator::binary_op. Otherwise, all inputs
// and the output must be on the GPU.
//
// For example, to write a reciprocal kernel for GPU float Tensors:
//
//   gpu_kernel(iter, []GPU_LAMBDA(float a) {
//    return 1.0f / a;
//   });
//
// To write a multiplication kernel for GPU float Tensors where one argument
// may be a CPU scalar:
//
//   gpu_kernel_with_scalars(iter, []GPU_LAMBDA(float a, float b) {
//     return a * b;
//   });
//
// See BinaryOpsKernel.cu for the complete implementation
//

#include <iostream>
#include <tuple>
#include <type_traits>

#include <ATen/core/Array.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <c10/util/TypeCast.h>
#include <fstream>
#include <error.h>
#include <stdio.h>
#include <unistd.h>
#include <unordered_set>
#ifdef __NVCC__
#define ASSERT_HOST_DEVICE_LAMBDA(type)                       \
  static_assert(                                              \
      __nv_is_extended_host_device_lambda_closure_type(type), \
      #type " must be a __host__ __device__ lambda")
#else
#define ASSERT_HOST_DEVICE_LAMBDA(type)
#endif

namespace at {
namespace native {

template <int vec_size, typename func_t, typename array_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits = function_traits<func_t>;
  int remaining = N - block_work_size() * blockIdx.x;

  if (remaining < block_work_size()) { // if this block handles the reminder,
                                       // just do a naive unrolled loop
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    auto output_calc = TrivialOffsetCalculator<1>();
    auto loader = memory::LoadWithoutCast();
    auto storer = memory::StoreWithoutCast();
    auto policy = memory::policies::unroll<
        array_t,
        decltype(input_calc),
        decltype(output_calc),
        memory::LoadWithoutCast,
        memory::StoreWithoutCast>(
        data, remaining, input_calc, output_calc, loader, storer);
    elementwise_kernel_helper(f, policy);
  } else { // if this block has a full `block_work_size` data to handle, use
           // vectorized memory access
    elementwise_kernel_helper(
        f, memory::policies::vectorized<vec_size, array_t>(data));
  }
}

template <
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unrolled_elementwise_kernel(
    int N,
    func_t f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  int remaining = N - block_work_size() * blockIdx.x;
  auto policy = memory::policies::
      unroll<array_t, inp_calc_t, out_calc_t, loader_t, storer_t>(
          data, remaining, ic, oc, l, s);
  elementwise_kernel_helper(f, policy);
}

// this function assume trivial 1d and no dynamic casting
template <typename func_t, typename array_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  int vec_size = memory::can_vectorize_up_to<func_t>(data);
 switch (vec_size) {
    case 4: {
        cudaEvent_t start_4, stop_4;
        cudaEventCreate(&start_4);
        cudaEventCreate(&stop_4);
        cudaEventRecord(start_4, 0);
        vectorized_elementwise_kernel<4, func_t, array_t>
            <<<grid, num_threads(), 0, stream>>>(N, f, data);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        cudaEventRecord(stop_4, 0);
        cudaEventSynchronize(stop_4);
        float execution_time_4 = 0;
        cudaEventElapsedTime(&execution_time_4, start_4, stop_4);
        std::ofstream log_file_4("/home/zixi/workspace/log/vectorized_elementwise_kernel_4.txt", std::ios_base::app);
        if (log_file_4.is_open()) {
            log_file_4 << "vectorized_elementwise_kernel_4 execution time: " << execution_time_4 << " ms" << std::endl;
            log_file_4.close();
        } else {
            std::cerr << "Unable to open log file" << std::endl;
        }
        break;
    }
    case 2: {
        cudaEvent_t start_2, stop_2;
        cudaEventCreate(&start_2);
        cudaEventCreate(&stop_2);
        cudaEventRecord(start_2, 0);

        vectorized_elementwise_kernel<2, func_t, array_t>
            <<<grid, num_threads(), 0, stream>>>(N, f, data);

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cudaEventRecord(stop_2, 0);
        cudaEventSynchronize(stop_2);
        float execution_time_2 = 0;
        cudaEventElapsedTime(&execution_time_2, start_2, stop_2);
        std::ofstream log_file_2("/home/zixi/workspace/log/vectorized_elementwise_kernel_2.txt", std::ios_base::app);
        if (log_file_2.is_open()) {
            log_file_2 << "vectorized_elementwise_kernel_2 execution time: " << execution_time_2 << " ms" << std::endl;
            log_file_2.close();
        } else {
            std::cerr << "Unable to open log file" << std::endl;
        }
        break;
    }
    case 1: {
        auto input_calc = TrivialOffsetCalculator<traits::arity>();
        auto output_calc = TrivialOffsetCalculator<1>();
        auto loader = memory::LoadWithoutCast();
        auto storer = memory::StoreWithoutCast();
        unrolled_elementwise_kernel<func_t, array_t>
            <<<grid, num_threads(), 0, stream>>>(
                N, f, data, input_calc, output_calc, loader, storer);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        break;
    }
    default:
        TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

template <
    typename func_t,
    typename array_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void launch_unrolled_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    inp_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  unrolled_elementwise_kernel<func_t, array_t>
      <<<grid, num_threads(), 0, stream>>>(N, f, data, ic, oc, l, s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel(int N, func_t f, uint8_t* smid_arr) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;

  //--my code----------
  int smid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x == 0){
    // smid_arr[blockIdx.x] = smid;
    // printf("smid: %d\n", smid);
  }
  //----------------------------
#pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template <int nt, int vt, typename func_t>
static void launch_legacy_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));


  auto stream = at::cuda::getCurrentCUDAStream();
  size_t total_size = nt * (N + block.x * vt - 1) / (block.x * vt);
  uint8_t *smids_native_d, *smids_native_h;
  cudaMalloc(&smids_native_d, total_size);
  if (!(smids_native_h = (uint8_t*)malloc(total_size)))
    error(1, errno, "Unable to allocate memory for test");
  // std::cout<<"stream_id: "<<stream.id()<<std::endl;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f, smids_native_d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float excution_time = 0;
  cudaEventElapsedTime(&excution_time, start, stop);
  std::ofstream log_file("/home/zixi/workspace/log/elementwise_kernel.txt", std::ios_base::app);
  if (log_file.is_open()) {
    log_file << "elementwise_kernel execution time: " << excution_time << " ms" << std::endl;
    log_file.close();
  } else {
    std::cerr << "Unable to open log file" << std::endl;
  }

  cudaMemcpy(smids_native_h, smids_native_d, total_size, cudaMemcpyDeviceToHost);
  // printf("Printing smids_native_h for elementwise_kernel:\n");
  // std::unordered_set<int> unique_elements;
  // for (size_t i = 0; i < total_size; i++) {
  //     int element = smids_native_h[i];
  //     if (unique_elements.find(element) == unique_elements.end()) {
  //         printf("%d ", element);
  //         unique_elements.insert(element);
  //     }
  // }
  // printf("\n");
  // std::cout << "------------------------- " << std::endl;
}

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(c10::load<typename traits::template arg<INDEX>::type>(
      data[INDEX] + i * strides[INDEX])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i,
    std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(
      dtypes[I], data[I] + i * strides[I])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const ScalarType dtypes[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}

template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    return launch_vectorized_kernel(numel, f, data);
  }
  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
  constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 2 : 4;
  launch_legacy_kernel<128, unroll_factor>(numel, [=] GPU_LAMBDA(int idx) {
    auto offsets = offset_calc.get(idx);
    arg0_t* out = (arg0_t*)(data[0] + offsets[0]);
    *out = invoke(f, &data.data[1], &offsets.data[1], 1);
  });
}

template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  if (!needs_dynamic_casting<func_t>::check(iter)) {
    return gpu_kernel_impl_nocast(iter, f);
  }
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
#ifdef USE_ROCM
    at::detail::Array<ScalarType, ntensors> dtypes;
    auto inner_strides = iter.get_inner_strides();
    at::detail::Array<int, ntensors> strides;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
      strides[i] = inner_strides[i];
    }
    launch_legacy_kernel<512, 1>(numel, [=]GPU_LAMBDA(int idx) {
      void* out = data[0] + strides[0] * idx;
      arg0_t result = invoke(f, &data.data[1], &strides.data[1], &dtypes.data[1], idx);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
#else
    auto loader = memory::LoadWithCast<traits::arity>(iter);
    auto storer = memory::StoreWithCast<1>(iter);
    auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
    auto output_offset_calculator = TrivialOffsetCalculator<1>();
    launch_unrolled_kernel(
        numel,
        f,
        data,
        input_offset_calculator,
        output_offset_calculator,
        loader,
        storer);
#endif
  } else {
    at::detail::Array<ScalarType, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
    }
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
    launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
      auto offsets = offset_calc.get(idx);
      void* out = data[0] + offsets[0];
      arg0_t result = invoke(f, &data.data[1], &offsets.data[1], &dtypes.data[1], 1);
      c10::cast_and_store<arg0_t>(dtypes[0], out, result);
    });
  }
}

} // namespace native
} // namespace at
