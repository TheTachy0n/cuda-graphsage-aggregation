#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

// Baseline: each thread processes every feature of one destination node.
// Threads in a warp generally traverse different neighbor lists, so loads are
// less coalesced and each thread does more serial work.
__global__ void mean_naive_kernel(const int64_t* offsets, const int64_t* indices,
                                  const float* x, float* out, int64_t nodes, int features) {
  int64_t node = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node >= nodes) return;
  int64_t start = offsets[node], end = offsets[node + 1];
  int64_t degree = end - start;
  for (int f = 0; f < features; ++f) {
    float sum = 0.0f;
    for (int64_t edge = start; edge < end; ++edge)
      sum += x[indices[edge] * features + f];
    out[node * features + f] = degree ? sum / static_cast<float>(degree) : 0.0f;
  }
}

// Optimized mapping: one block owns a destination node; threads split feature
// dimensions. For one neighbor, adjacent lanes read adjacent feature values,
// producing coalesced global-memory accesses.
__global__ void mean_coalesced_kernel(const int64_t* offsets, const int64_t* indices,
                                      const float* x, float* out, int features) {
  int64_t node = blockIdx.x;
  int64_t start = offsets[node], end = offsets[node + 1];
  int64_t degree = end - start;
  for (int f = threadIdx.x; f < features; f += blockDim.x) {
    float sum = 0.0f;
    for (int64_t edge = start; edge < end; ++edge)
      sum += x[indices[edge] * features + f];
    out[node * features + f] = degree ? sum / static_cast<float>(degree) : 0.0f;
  }
}

void check_inputs(const torch::Tensor& offsets, const torch::Tensor& indices,
                  const torch::Tensor& features) {
  TORCH_CHECK(offsets.is_cuda() && indices.is_cuda() && features.is_cuda(), "all inputs must be CUDA tensors");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64 && indices.scalar_type() == torch::kInt64, "CSR tensors must be int64");
  TORCH_CHECK(features.scalar_type() == torch::kFloat32 && features.dim() == 2, "features must be a 2D float32 tensor");
  TORCH_CHECK(offsets.is_contiguous() && indices.is_contiguous() && features.is_contiguous(), "inputs must be contiguous");
}
}  // namespace

torch::Tensor mean_aggregate_naive_cuda(torch::Tensor offsets, torch::Tensor indices, torch::Tensor features) {
  check_inputs(offsets, indices, features);
  auto out = torch::zeros_like(features);
  int64_t nodes = features.size(0);
  constexpr int threads = 256;
  mean_naive_kernel<<<(nodes + threads - 1) / threads, threads>>>(offsets.data_ptr<int64_t>(), indices.data_ptr<int64_t>(), features.data_ptr<float>(), out.data_ptr<float>(), nodes, features.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor mean_aggregate_coalesced_cuda(torch::Tensor offsets, torch::Tensor indices, torch::Tensor features) {
  check_inputs(offsets, indices, features);
  auto out = torch::zeros_like(features);
  mean_coalesced_kernel<<<features.size(0), 256>>>(offsets.data_ptr<int64_t>(), indices.data_ptr<int64_t>(), features.data_ptr<float>(), out.data_ptr<float>(), features.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
