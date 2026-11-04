/*
 * Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
 *
 * Hygon modifications to this file are licensed under the Apache License,
 * Version 2.0 (the "License"); you may not use these modifications except
 * in compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ int64_t ceil_div(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

__device__ int64_t safe_min(int64_t a, int64_t b) {
  return a < b ? a : b;
}

__global__ void launch_alloc_decode_kernel(
    const int64_t* seq_lens_ptr,
    const int32_t* last_loc_ptr,
    const int64_t* free_page_ptr,
    int64_t* out_indices,
    int64_t bs,
    int64_t page_size) {
  int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;

  if (pid >= bs) return;

  int64_t seq_len = seq_lens_ptr[pid];
  int64_t pre_len = seq_len - 1;

  int64_t num_page_start_loc_self = ceil_div(seq_len, page_size) - ceil_div(pre_len, page_size);

  int64_t sum_num_new_pages = 0;
  for (int64_t i = 0; i <= pid; i++) {
    int64_t other_seq_len = seq_lens_ptr[i];
    int64_t other_pre_len = (i <= pid) ? (other_seq_len - 1) : other_seq_len;

    int64_t other_num_pages_after = ceil_div(other_seq_len, page_size);
    int64_t other_num_pages_before = ceil_div(other_pre_len, page_size);
    int64_t other_num_new_pages = other_num_pages_after - other_num_pages_before;

    sum_num_new_pages += other_num_new_pages;
  }
  int64_t new_page_start_loc = sum_num_new_pages - num_page_start_loc_self;

  if (num_page_start_loc_self == 0) {
    int32_t last_loc = last_loc_ptr[pid];
    out_indices[pid] = last_loc + 1;
  } else {
    int64_t page = free_page_ptr[new_page_start_loc];
    out_indices[pid] = page * page_size;
  }
}

__global__ void launch_alloc_extend_kernel(
    const int64_t* pre_lens_ptr,
    const int64_t* seq_lens_ptr,
    const int64_t* last_loc_ptr,
    const int64_t* free_page_ptr,
    int64_t* out_indices,
    int64_t bs,
    int64_t page_size) {
  int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;

  if (pid >= bs) return;

  int64_t seq_len = seq_lens_ptr[pid];
  int64_t pre_len = pre_lens_ptr[pid];
  int64_t extend_len = seq_len - pre_len;

  int64_t sum_extend_lens = 0;
  for (int64_t i = 0; i <= pid; i++) {
    int64_t other_seq_len = seq_lens_ptr[i];
    int64_t other_pre_len = pre_lens_ptr[i];
    int64_t other_extend_len = other_seq_len - other_pre_len;
    sum_extend_lens += other_extend_len;
  }

  int64_t output_start_loc = sum_extend_lens - extend_len;
  int64_t num_page_start_loc_self = ceil_div(seq_len, page_size) - ceil_div(pre_len, page_size);

  int64_t sum_num_new_pages = 0;
  for (int64_t i = 0; i <= pid; i++) {
    int64_t other_seq_len = seq_lens_ptr[i];
    int64_t other_pre_len = pre_lens_ptr[i];

    int64_t other_num_pages_after = ceil_div(other_seq_len, page_size);
    int64_t other_num_pages_before = ceil_div(other_pre_len, page_size);
    int64_t other_num_new_pages = other_num_pages_after - other_num_pages_before;

    sum_num_new_pages += other_num_new_pages;
  }
  int64_t new_page_start_loc = sum_num_new_pages - num_page_start_loc_self;

  int64_t last_loc = last_loc_ptr[pid];
  int64_t num_part1 = safe_min(seq_len, ceil_div(pre_len, page_size) * page_size) - pre_len;

  for (int64_t offset = 0; offset < num_part1 && offset < page_size; offset++) {
    int64_t output_idx = output_start_loc + offset;
    out_indices[output_idx] = last_loc + 1 + offset;
  }

  if (pre_len + num_part1 == seq_len) {
    return;
  }

  int64_t num_part2 = (seq_len / page_size) * page_size - ceil_div(pre_len, page_size) * page_size;
  for (int64_t offset = 0; offset < num_part2; offset++) {
    int64_t page_idx = new_page_start_loc + offset / page_size;
    int64_t page_start = free_page_ptr[page_idx];
    int64_t output_idx = output_start_loc + num_part1 + offset;
    out_indices[output_idx] = page_start * page_size + offset % page_size;
  }

  if (pre_len + num_part1 + num_part2 == seq_len) {
    return;
  }

  int64_t num_part3 = seq_len - (seq_len / page_size) * page_size;
  int64_t last_page_idx = new_page_start_loc + num_page_start_loc_self - 1;
  int64_t start_loc = free_page_ptr[last_page_idx];

  for (int64_t offset = 0; offset < num_part3 && offset < page_size; offset++) {
    int64_t output_idx = output_start_loc + num_part1 + num_part2 + offset;
    out_indices[output_idx] = start_loc * page_size + offset;
  }
}

void hcu_alloc_decode_kernel(
    const at::Tensor seq_lens_ptr,
    const at::Tensor last_loc_ptr,
    const at::Tensor free_page_ptr,
    at::Tensor out_indices,
    int64_t bs,
    int64_t page_size) {
  const int64_t* seq_lens_ptr1 = static_cast<const int64_t*>(seq_lens_ptr.data_ptr());
  const int32_t* last_loc_ptr1 = static_cast<const int32_t*>(last_loc_ptr.data_ptr());
  const int64_t* free_page_ptr1 = static_cast<const int64_t*>(free_page_ptr.data_ptr());
  int64_t* out_indices1 = static_cast<int64_t*>(out_indices.data_ptr());

  int64_t block_size = 64;
  int64_t grid_size = (bs + block_size - 1) / block_size;
  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  launch_alloc_decode_kernel<<<grid_size, block_size, 0, torch_current_stream>>>(
      seq_lens_ptr1, last_loc_ptr1, free_page_ptr1, out_indices1, bs, page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hcu_alloc_extend_kernel(
    const at::Tensor pre_lens_ptr,
    const at::Tensor seq_lens_ptr,
    const at::Tensor last_loc_ptr,
    const at::Tensor free_page_ptr,
    at::Tensor out_indices,
    int64_t bs,
    int64_t page_size) {
  const int64_t* pre_lens_ptr1 = static_cast<const int64_t*>(pre_lens_ptr.data_ptr());
  const int64_t* seq_lens_ptr1 = static_cast<const int64_t*>(seq_lens_ptr.data_ptr());
  const int64_t* last_loc_ptr1 = static_cast<const int64_t*>(last_loc_ptr.data_ptr());
  const int64_t* free_page_ptr1 = static_cast<const int64_t*>(free_page_ptr.data_ptr());
  int64_t* out_indices1 = static_cast<int64_t*>(out_indices.data_ptr());

  int64_t block_size = 64;
  int64_t grid_size = (bs + block_size - 1) / block_size;
  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  launch_alloc_extend_kernel<<<grid_size, block_size, 0, torch_current_stream>>>(
      pre_lens_ptr1, seq_lens_ptr1, last_loc_ptr1, free_page_ptr1, out_indices1, bs, page_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void launch_assign_req_to_token_pool(
    const int64_t* req_pool_indices_ptr,
    int32_t* req_to_token_ptr,
    const int32_t* allocate_lens_ptr,
    const int32_t* new_allocate_lens,
    int64_t* out_cache_loc_ptr,
    int64_t shape,
    int64_t bs) {
  int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= bs) return;

  int32_t kv_start = allocate_lens_ptr[pid];
  int32_t kv_end = new_allocate_lens[pid];
  int64_t pool_idx = req_pool_indices_ptr[pid];
  int32_t* token_pool = req_to_token_ptr + pool_idx * shape;

  int64_t sum_out_offset = 0;
  for (int64_t length_offset = 0; length_offset < pid; length_offset++) {
    sum_out_offset += (int64_t)(new_allocate_lens[length_offset] - allocate_lens_ptr[length_offset]);
  }
  int64_t* out_cache_ptr = out_cache_loc_ptr + sum_out_offset;

  int32_t copy_length = kv_end - kv_start;
#pragma unroll(32)
  for (int32_t out_cache_index = 0; out_cache_index < copy_length; out_cache_index++) {
    token_pool[kv_start + out_cache_index] = static_cast<int32_t>(out_cache_ptr[out_cache_index]);
  }
}

void hcu_assign_req_to_token_pool(
    const at::Tensor req_pool_indices_ptr,
    at::Tensor req_to_token_ptr,
    const at::Tensor allocate_lens_ptr,
    at::Tensor new_allocate_lens,
    at::Tensor out_cache_loc_ptr,
    int64_t shape,
    int64_t bs) {
  const int64_t* req_pool_indices_ptr1 = static_cast<const int64_t*>(req_pool_indices_ptr.data_ptr());
  int32_t* req_to_token_ptr1 = static_cast<int32_t*>(req_to_token_ptr.data_ptr());
  const int32_t* allocate_lens_ptr1 = static_cast<const int32_t*>(allocate_lens_ptr.data_ptr());
  int32_t* new_allocate_lens1 = static_cast<int32_t*>(new_allocate_lens.data_ptr());
  int64_t* out_cache_loc_ptr1 = static_cast<int64_t*>(out_cache_loc_ptr.data_ptr());

  int64_t block_size = 64;
  int64_t grid_size = (bs + block_size - 1) / block_size;
  cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();
  launch_assign_req_to_token_pool<<<grid_size, block_size, 0, torch_current_stream>>>(
      req_pool_indices_ptr1, req_to_token_ptr1, allocate_lens_ptr1, new_allocate_lens1, out_cache_loc_ptr1, shape, bs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void get_last_loc_kernel(
    const int32_t* req_to_token,
    const int64_t* req_pool_indices_tensor,
    const int32_t* prefix_lens_tensor,
    int64_t* result,
    int64_t num_tokens,
    int32_t req_to_token_stride) {
  int64_t pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= num_tokens) return;

  int32_t pre_len = prefix_lens_tensor[pid];
  if (pre_len > 0) {
    int64_t req_idx = req_pool_indices_tensor[pid];
    int64_t token_idx = req_idx * req_to_token_stride + (pre_len - 1);
    result[pid] = static_cast<int64_t>(req_to_token[token_idx]);
  } else {
    result[pid] = static_cast<int64_t>(-1);
  }
}

at::Tensor
hcu_get_last_loc(const at::Tensor req_to_token, const at::Tensor req_pool_indices, const at::Tensor prefix_lens) {
  // TORCH_CHECK(req_to_token.device().is_cuda(), "req_to_token must be CUDA tensor");
  // TORCH_CHECK(req_pool_indices.device().is_cuda(), "req_pool_indices must be CUDA tensor");
  // TORCH_CHECK(prefix_lens.device().is_cuda(), "prefix_lens must be CUDA tensor");

  // TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be 2D tensor [batch, seq_len]");
  // TORCH_CHECK(prefix_lens.dim() == 1, "prefix_lens must be 1D");
  // TORCH_CHECK(req_pool_indices.dim() == 1, "req_pool_indices must be 1D");

  int32_t num_tokens = prefix_lens.numel();
  // TORCH_CHECK(req_pool_indices.numel() == num_tokens, "req_pool_indices must have same length as prefix_lens");

  int32_t req_to_token_stride = req_to_token.stride(0);

  const int32_t* req_to_token_ptr = static_cast<const int32_t*>(req_to_token.data_ptr());
  const int64_t* req_pool_indices_ptr = static_cast<const int64_t*>(req_pool_indices.data_ptr());
  const int32_t* prefix_lens_ptr = static_cast<const int32_t*>(prefix_lens.data_ptr());

  // auto req_to_token_c = req_to_token.contiguous();
  // auto req_pool_indices_c = req_pool_indices.contiguous();
  // auto prefix_lens_c   = prefix_lens.contiguous();

  // const int32_t* req_to_token_ptr = req_to_token_c.data_ptr<int32_t>();
  // const int64_t* req_pool_indices_ptr = req_pool_indices_c.data_ptr<int64_t>();
  // const int32_t* prefix_lens_ptr  = prefix_lens_c.data_ptr<int32_t>();

  auto req_pool_indices_c = req_pool_indices.contiguous();
  auto result = at::empty_like(req_pool_indices_c);
  int64_t* result_ptr = static_cast<int64_t*>(result.data_ptr());

  const int64_t block_size = 64;
  const int64_t grid_size = (num_tokens + block_size - 1) / block_size;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  get_last_loc_kernel<<<grid_size, block_size, 0, stream>>>(
      req_to_token_ptr, req_pool_indices_ptr, prefix_lens_ptr, result_ptr, num_tokens, req_to_token_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("hcu_alloc_decode_kernel", &hcu_alloc_decode_kernel);
  module.def("hcu_alloc_extend_kernel", &hcu_alloc_extend_kernel);
  module.def("hcu_assign_req_to_token_pool", &hcu_assign_req_to_token_pool);
  module.def("hcu_get_last_loc", &hcu_get_last_loc);
}
