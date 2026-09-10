#include "presolve_helpers.hpp"
#include "gpu_presolver/presolve/fixed_col_tape.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

using detail::throw_if_cuda_error;

constexpr int GPU_PRESOLVE_THREADS = 256;

template <typename T>
std::size_t reserve_scratch(std::size_t& bytes, std::size_t count) {
  const std::size_t alignment = alignof(T);
  bytes = (bytes + alignment - 1) & ~(alignment - 1);
  const std::size_t offset = bytes;
  bytes += sizeof(T) * count;
  return offset;
}

template <typename T>
T* scratch_at(void* storage, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(storage) + offset);
}

__device__ bool fixed_col_selected(const std::uint8_t* fixed_mask,
                                   const std::uint8_t* excluded_mask,
                                   std::int32_t col) {
  return fixed_mask[col] != std::uint8_t{0} &&
         (excluded_mask == nullptr || excluded_mask[col] == std::uint8_t{0});
}

__global__ void fixed_col_tape_counts(std::int32_t* selected_scan,
                                      std::int32_t* active_nnz_scan,
                                      const std::uint8_t* fixed_mask,
                                      const std::uint8_t* excluded_mask,
                                      const std::uint8_t* keep_row,
                                      const std::int32_t* at_row_ptr,
                                      const std::int32_t* at_col_val,
                                      std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n) {
    return;
  }
  const bool selected = fixed_col_selected(fixed_mask, excluded_mask, col);
  selected_scan[col] = selected ? 1 : 0;
  std::int32_t active_nnz = 0;
  if (selected) {
    for (std::int32_t p = at_row_ptr[col]; p < at_row_ptr[col + 1]; ++p) {
      active_nnz += keep_row[at_col_val[p]] != std::uint8_t{0} ? 1 : 0;
    }
  }
  active_nnz_scan[col] = active_nnz;
}

__global__ void pack_fixed_col_tape(std::int32_t* packed_cols,
                                    double* packed_fixed_vals,
                                    double* packed_c,
                                    std::int32_t* packed_nnz_starts,
                                    std::int32_t* packed_rows,
                                    double* packed_coeffs,
                                    const std::int32_t* selected_scan,
                                    const std::int32_t* active_nnz_scan,
                                    const std::uint8_t* fixed_mask,
                                    const std::uint8_t* excluded_mask,
                                    const double* fixed_val,
                                    const std::uint8_t* keep_row,
                                    const double* c,
                                    const std::int32_t* at_row_ptr,
                                    const std::int32_t* at_col_val,
                                    const double* at_nz_val,
                                    std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || !fixed_col_selected(fixed_mask, excluded_mask, col)) {
    return;
  }

  const std::int32_t record = selected_scan[col] - 1;
  const std::int32_t nnz_start = col == 0 ? 0 : active_nnz_scan[col - 1];
  const std::int32_t nnz_stop = active_nnz_scan[col];
  packed_cols[record] = col;
  packed_fixed_vals[record] = fixed_val[col];
  packed_c[record] = c[col];
  packed_nnz_starts[record + 1] = nnz_stop;

  std::int32_t out = nnz_start;
  for (std::int32_t p = at_row_ptr[col]; p < at_row_ptr[col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (keep_row[row] != std::uint8_t{0}) {
      packed_rows[out] = row;
      packed_coeffs[out] = at_nz_val[p];
      ++out;
    }
  }
}

}  // namespace

double append_compact_fixed_col_tape_from_device(PostsolveTape* tape,
                                               const std::uint8_t* fixed_mask,
                                               const std::uint8_t* excluded_mask,
                                               const double* fixed_val,
                                               const std::uint8_t* keep_row,
                                               const double* c,
                                               const DeviceCsrMatrix& AT,
                                               std::int32_t n,
                                               const char* context) {
  if (n <= 0) {
    return 0.0;
  }

  void* scan_slab = nullptr;
  throw_if_cuda_error(cudaMalloc(&scan_slab,
                                 2 * sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  auto* selected_scan = static_cast<std::int32_t*>(scan_slab);
  auto* active_nnz_scan = selected_scan + n;

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  fixed_col_tape_counts<<<blocks_n, GPU_PRESOLVE_THREADS>>>(selected_scan,
                                                           active_nnz_scan,
                                                           fixed_mask,
                                                           excluded_mask,
                                                           keep_row,
                                                           AT.rowPtr,
                                                           AT.colVal,
                                                           n);
  throw_if_cuda_error(cudaGetLastError(), context);
  void* scan_temp = nullptr;
  std::size_t scan_temp_bytes = 0;
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(
                          scan_temp, scan_temp_bytes, selected_scan, selected_scan, n),
                      context);
  throw_if_cuda_error(cudaMalloc(&scan_temp, scan_temp_bytes), context);
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(
                          scan_temp, scan_temp_bytes, selected_scan, selected_scan, n),
                      context);
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(
                          scan_temp, scan_temp_bytes, active_nnz_scan, active_nnz_scan, n),
                      context);
  cudaFree(scan_temp);

  std::int32_t record_count = 0;
  std::int32_t active_nnz = 0;
  throw_if_cuda_error(cudaMemcpy(&record_count, selected_scan + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(&active_nnz, active_nnz_scan + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (record_count == 0) {
    cudaFree(scan_slab);
    return 0.0;
  }

  std::size_t packed_bytes = 0;
  const std::size_t cols_offset = reserve_scratch<std::int32_t>(packed_bytes, record_count);
  const std::size_t fixed_vals_offset = reserve_scratch<double>(packed_bytes, record_count);
  const std::size_t c_offset = reserve_scratch<double>(packed_bytes, record_count);
  const std::size_t starts_offset = reserve_scratch<std::int32_t>(packed_bytes, record_count + 1);
  const std::size_t rows_offset = reserve_scratch<std::int32_t>(packed_bytes, active_nnz);
  const std::size_t coeffs_offset = reserve_scratch<double>(packed_bytes, active_nnz);
  void* packed_slab = nullptr;
  throw_if_cuda_error(cudaMalloc(&packed_slab, packed_bytes), context);
  auto* packed_cols = scratch_at<std::int32_t>(packed_slab, cols_offset);
  auto* packed_fixed_vals = scratch_at<double>(packed_slab, fixed_vals_offset);
  auto* packed_c = scratch_at<double>(packed_slab, c_offset);
  auto* packed_nnz_starts = scratch_at<std::int32_t>(packed_slab, starts_offset);
  auto* packed_rows = scratch_at<std::int32_t>(packed_slab, rows_offset);
  auto* packed_coeffs = scratch_at<double>(packed_slab, coeffs_offset);
  throw_if_cuda_error(cudaMemset(packed_nnz_starts, 0, sizeof(std::int32_t)), context);

  pack_fixed_col_tape<<<blocks_n, GPU_PRESOLVE_THREADS>>>(packed_cols,
                                                          packed_fixed_vals,
                                                          packed_c,
                                                          packed_nnz_starts,
                                                          packed_rows,
                                                          packed_coeffs,
                                                          selected_scan,
                                                          active_nnz_scan,
                                                          fixed_mask,
                                                          excluded_mask,
                                                          fixed_val,
                                                          keep_row,
                                                          c,
                                                          AT.rowPtr,
                                                          AT.colVal,
                                                          AT.nzVal,
                                                          n);
  throw_if_cuda_error(cudaGetLastError(), context);

  std::vector<std::int32_t> host_cols(
      tape == nullptr ? 0 : static_cast<std::size_t>(record_count));
  std::vector<double> host_fixed_vals(static_cast<std::size_t>(record_count));
  std::vector<double> host_c(static_cast<std::size_t>(record_count));
  std::vector<std::int32_t> host_nnz_starts(
      tape == nullptr ? 0 : static_cast<std::size_t>(record_count + 1));
  std::vector<std::int32_t> host_rows(
      tape == nullptr ? 0 : static_cast<std::size_t>(active_nnz));
  std::vector<double> host_coeffs(
      tape == nullptr ? 0 : static_cast<std::size_t>(active_nnz));
  if (tape != nullptr) {
    throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(record_count),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  throw_if_cuda_error(cudaMemcpy(host_fixed_vals.data(), packed_fixed_vals,
                                 sizeof(double) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_c.data(), packed_c,
                                 sizeof(double) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (tape != nullptr) {
    throw_if_cuda_error(cudaMemcpy(host_nnz_starts.data(), packed_nnz_starts,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(record_count + 1),
                                   cudaMemcpyDeviceToHost),
                        context);
    if (active_nnz > 0) {
      throw_if_cuda_error(cudaMemcpy(host_rows.data(), packed_rows,
                                     sizeof(std::int32_t) * static_cast<std::size_t>(active_nnz),
                                     cudaMemcpyDeviceToHost),
                          context);
      throw_if_cuda_error(cudaMemcpy(host_coeffs.data(), packed_coeffs,
                                     sizeof(double) * static_cast<std::size_t>(active_nnz),
                                     cudaMemcpyDeviceToHost),
                          context);
    }
  }

  long double objective_delta = 0.0L;
  for (std::int32_t rec = 0; rec < record_count; ++rec) {
    objective_delta +=
        static_cast<long double>(host_c[static_cast<std::size_t>(rec)]) *
        static_cast<long double>(host_fixed_vals[static_cast<std::size_t>(rec)]);
    if (tape == nullptr) {
      continue;
    }
    tape->types.push_back(static_cast<std::int32_t>(PostsolveReductionType::FixedCol));
    tape->indices.push_back(host_cols[static_cast<std::size_t>(rec)]);
    tape->vals.push_back(host_fixed_vals[static_cast<std::size_t>(rec)]);
    tape->vals.push_back(host_c[static_cast<std::size_t>(rec)]);
    const std::int32_t start = host_nnz_starts[static_cast<std::size_t>(rec)];
    const std::int32_t stop = host_nnz_starts[static_cast<std::size_t>(rec + 1)];
    tape->indices.insert(
        tape->indices.end(), host_rows.begin() + start, host_rows.begin() + stop);
    tape->vals.insert(
        tape->vals.end(), host_coeffs.begin() + start, host_coeffs.begin() + stop);
    tape->index_starts.push_back(static_cast<std::int32_t>(tape->indices.size()));
    tape->value_starts.push_back(static_cast<std::int32_t>(tape->vals.size()));
    tape->dual_modes.push_back(static_cast<std::uint8_t>(PostsolveDualMode::Minimal));
  }

  cudaFree(scan_slab);
  cudaFree(packed_slab);
  return static_cast<double>(objective_delta);
}

}  // namespace gpu_presolver::presolve
