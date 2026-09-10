#include "test_cuda_helpers.hpp"
#include "gpu_presolver/presolve/gpu_presolve_kernels.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using gpu_presolver_test::check;
using gpu_presolver_test::copy_to_device;
using gpu_presolver_test::copy_to_host;

}  // namespace

int main() {
  using gpu_presolver::presolve::DeviceCsrMatrix;
  const double inf = std::numeric_limits<double>::infinity();

  std::int32_t* row_ptr = copy_to_device<std::int32_t>({0, 0, 1, 3});
  std::int32_t* col_val = copy_to_device<std::int32_t>({2, 0, 1});
  double* nz_val = copy_to_device<double>({4.0, 5.0, 6.0});

  DeviceCsrMatrix matrix{3, 3, 3, row_ptr, col_val, nz_val};

  double* l = copy_to_device<double>({-1.0, -inf, 2.0});
  double* u = copy_to_device<double>({3.0, 5.0, inf});
  double* row_min_fin = nullptr;
  double* row_max_fin = nullptr;
  std::int32_t* row_min_neg_inf_count = nullptr;
  std::int32_t* row_max_pos_inf_count = nullptr;
  check(cudaMalloc(&row_min_fin, sizeof(double) * 3), "cudaMalloc row_min_fin");
  check(cudaMalloc(&row_max_fin, sizeof(double) * 3), "cudaMalloc row_max_fin");
  check(cudaMalloc(&row_min_neg_inf_count, sizeof(std::int32_t) * 3), "cudaMalloc row_min_neg_inf_count");
  check(cudaMalloc(&row_max_pos_inf_count, sizeof(std::int32_t) * 3), "cudaMalloc row_max_pos_inf_count");

  gpu_presolver::presolve::compute_row_activity_summary(
      row_min_fin,
      row_max_fin,
      row_min_neg_inf_count,
      row_max_pos_inf_count,
      matrix,
      l,
      u,
      1.0e-12);
  check(cudaDeviceSynchronize(), "activity sync");

  const std::vector<double> h_row_min_fin = copy_to_host(row_min_fin, 3);
  const std::vector<double> h_row_max_fin = copy_to_host(row_max_fin, 3);
  const std::vector<std::int32_t> h_neg = copy_to_host(row_min_neg_inf_count, 3);
  const std::vector<std::int32_t> h_pos = copy_to_host(row_max_pos_inf_count, 3);
  assert(h_row_min_fin[0] == 0.0);
  assert(h_row_max_fin[0] == 0.0);
  assert(h_neg[0] == 0 && h_pos[0] == 0);
  assert(h_row_min_fin[1] == 8.0);
  assert(h_row_max_fin[1] == 0.0);
  assert(h_neg[1] == 0 && h_pos[1] == 1);
  assert(h_row_min_fin[2] == -5.0);
  assert(h_row_max_fin[2] == 45.0);
  assert(h_neg[2] == 1);
  assert(h_pos[2] == 0);

  std::int32_t* at_row_ptr = copy_to_device<std::int32_t>({0, 1, 2, 3});
  std::int32_t* at_col_val = copy_to_device<std::int32_t>({2, 2, 1});
  double* at_nz_val = copy_to_device<double>({5.0, -6.0, 4.0});
  DeviceCsrMatrix transpose{3, 3, 3, at_row_ptr, at_col_val, at_nz_val};
  double* col_max_abs = nullptr;
  check(cudaMalloc(&col_max_abs, sizeof(double) * 3), "cudaMalloc col_max_abs");
  gpu_presolver::presolve::compute_col_max_abs(col_max_abs, transpose);
  check(cudaDeviceSynchronize(), "col max sync");
  const std::vector<double> h_col_max_abs = copy_to_host(col_max_abs, 3);
  assert((h_col_max_abs == std::vector<double>{5.0, 6.0, 4.0}));

  cudaFree(row_ptr);
  cudaFree(col_val);
  cudaFree(nz_val);
  cudaFree(l);
  cudaFree(u);
  cudaFree(row_min_fin);
  cudaFree(row_max_fin);
  cudaFree(row_min_neg_inf_count);
  cudaFree(row_max_pos_inf_count);
  cudaFree(at_row_ptr);
  cudaFree(at_col_val);
  cudaFree(at_nz_val);
  cudaFree(col_max_abs);

  std::cout << "test_gpu_presolve_kernels passed\n";
  return 0;
}
