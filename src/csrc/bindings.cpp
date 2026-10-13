#include <torch/extension.h>

torch::Tensor mean_aggregate_naive_cuda(torch::Tensor row_offsets,
                                        torch::Tensor col_indices,
                                        torch::Tensor features);
torch::Tensor mean_aggregate_coalesced_cuda(torch::Tensor row_offsets,
                                            torch::Tensor col_indices,
                                            torch::Tensor features);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mean_aggregate_naive", &mean_aggregate_naive_cuda,
        "CSR mean aggregation: one thread per node (CUDA)");
  m.def("mean_aggregate_coalesced", &mean_aggregate_coalesced_cuda,
        "CSR mean aggregation: one block per node (CUDA)");
}
