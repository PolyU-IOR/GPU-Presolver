#include "presolve_helpers.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"

#include "gpu_presolver/presolve/rules/rule_antipodal_components.hpp"
#include "gpu_presolver/presolve/rules/rule_bounded_two_row_projection.hpp"
#include "gpu_presolver/presolve/rules/rule_covering_cost_dominance.hpp"
#include "gpu_presolver/presolve/rules/rule_infeasible_redundant_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_infeasible_fixed_variables.hpp"
#include "gpu_presolver/presolve/rules/rule_doubleton_equations.hpp"
#include "gpu_presolver/presolve/rules/rule_dual_fix.hpp"
#include "gpu_presolver/presolve/rules/rule_empty_cols.hpp"
#include "gpu_presolver/presolve/rules/rule_empty_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_linf_components.hpp"
#include "gpu_presolver/presolve/rules/rule_duplicate_columns.hpp"
#include "gpu_presolver/presolve/rules/rule_duplicate_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_orphan_mccormick_projection.hpp"
#include "gpu_presolver/presolve/rules/rule_implied_variable_bounds.hpp"
#include "gpu_presolver/presolve/rules/rule_redundant_bounds.hpp"
#include "gpu_presolver/presolve/rules/rule_column_singletons.hpp"
#include "gpu_presolver/presolve/rules/rule_singleton_rows.hpp"
#include "gpu_presolver/presolve/rules/rule_structural_l1_substitution.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

using detail::throw_if_cuda_error;
using detail::env_enabled;

constexpr std::int32_t SPARSE_FIXED_BOUND_RECORD_THRESHOLD = 65536;
constexpr std::int32_t COMPACT_LONG_ROW_THRESHOLD = 1024;
constexpr std::int32_t PARALLEL_IDENTITY_FILL_THRESHOLD = 1 << 20;
constexpr unsigned PARALLEL_IDENTITY_FILL_MAX_THREADS = 8;

void throw_if_cusparse_error(cusparseStatus_t status, const char* context) {
  if (status == CUSPARSE_STATUS_SUCCESS) {
    return;
  }
  throw std::runtime_error(std::string(context) + ": cusparse status " +
                           std::to_string(static_cast<int>(status)));
}

template <class T>
cudaError_t stream_ordered_malloc(T** pointer, std::size_t bytes) {
  return cudaMallocAsync(reinterpret_cast<void**>(pointer), bytes, nullptr);
}

cudaError_t stream_ordered_free(void* pointer) {
  return pointer == nullptr ? cudaSuccess : cudaFreeAsync(pointer, nullptr);
}

void inclusive_scan_i32(std::int32_t* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(
      cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
      context);
  throw_if_cuda_error(stream_ordered_malloc(&temp_storage, temp_bytes), context);
  throw_if_cuda_error(
      cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
      context);
  stream_ordered_free(temp_storage);
}

void append_rule_name(std::string& rules, const char* name) {
  if (!rules.empty()) {
    rules += ",";
  }
  rules += name;
}

__global__ void _kernel_fill_u8(std::uint8_t* data, std::uint8_t value, std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    data[j] = value;
  }
}

__global__ void _kernel_update_row_degrees_from_removed_cols(
    std::int32_t* row_nnz,
    const std::uint8_t* keep_col_before,
    const std::uint8_t* keep_col_after,
    const std::uint8_t* keep_row_after,
    const std::int32_t* at_row_ptr,
    const std::int32_t* at_col_val,
    std::int32_t n) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || keep_col_before[col] == std::uint8_t{0} ||
      keep_col_after[col] != std::uint8_t{0}) {
    return;
  }
  for (std::int32_t p = at_row_ptr[col]; p < at_row_ptr[col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (keep_row_after[row] != std::uint8_t{0}) {
      atomicSub(&row_nnz[row], 1);
    }
  }
}

__global__ void _kernel_update_col_degrees_from_removed_rows(
    std::int32_t* col_nnz,
    const std::uint8_t* keep_row_before,
    const std::uint8_t* keep_row_after,
    const std::uint8_t* keep_col_after,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    std::int32_t m) {
  const std::int32_t row =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m || keep_row_before[row] == std::uint8_t{0} ||
      keep_row_after[row] != std::uint8_t{0}) {
    return;
  }
  for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    const std::int32_t col = col_val[p];
    if (keep_col_after[col] != std::uint8_t{0}) {
      atomicSub(&col_nnz[col], 1);
    }
  }
}

__global__ void _kernel_refresh_singleton_rows_from_degrees(
    std::uint8_t* singleton_mask,
    std::int32_t* column_singleton,
    double* singleton_val,
    const std::int32_t* row_nnz,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t m) {
  const std::int32_t row =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) return;
  const bool is_singleton = keep_row[row] != std::uint8_t{0} && row_nnz[row] == 1;
  singleton_mask[row] = is_singleton ? std::uint8_t{1} : std::uint8_t{0};
  column_singleton[row] = -1;
  singleton_val[row] = 0.0;
  if (!is_singleton) return;
  for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    const std::int32_t col = col_val[p];
    if (keep_col[col] != std::uint8_t{0}) {
      column_singleton[row] = col;
      singleton_val[row] = nz_val[p];
      return;
    }
  }
}

__global__ void _kernel_refresh_column_singletons_from_degrees(
    std::uint8_t* singleton_mask,
    std::int32_t* singleton_row,
    double* singleton_val,
    const std::int32_t* col_nnz,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const std::int32_t* at_row_ptr,
    const std::int32_t* at_col_val,
    const double* at_nz_val,
    std::int32_t n) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n) return;
  const bool is_singleton = keep_col[col] != std::uint8_t{0} && col_nnz[col] == 1;
  singleton_mask[col] = is_singleton ? std::uint8_t{1} : std::uint8_t{0};
  singleton_row[col] = -1;
  singleton_val[col] = 0.0;
  if (!is_singleton) return;
  for (std::int32_t p = at_row_ptr[col]; p < at_row_ptr[col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    if (keep_row[row] != std::uint8_t{0}) {
      singleton_row[col] = row;
      singleton_val[col] = at_nz_val[p];
      return;
    }
  }
}

__global__ void _kernel_mark_changed_dual_cols(
    std::uint8_t* candidate_mask,
    const std::uint8_t* keep_col,
    const double* c,
    const double* l,
    const double* u,
    const double* c_seen,
    const double* l_seen,
    const double* u_seen,
    std::int32_t n) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n) {
    candidate_mask[col] =
        keep_col[col] != std::uint8_t{0} &&
                (c[col] != c_seen[col] || l[col] != l_seen[col] ||
                 u[col] != u_seen[col])
            ? std::uint8_t{1}
            : std::uint8_t{0};
  }
}

__global__ void _kernel_mark_dual_cols_from_changed_rows(
    std::uint8_t* candidate_mask,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const double* AL,
    const double* AU,
    const std::uint8_t* keep_row_seen,
    const double* AL_seen,
    const double* AU_seen,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    std::int32_t m) {
  const std::int32_t row =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m ||
      (keep_row[row] == keep_row_seen[row] && AL[row] == AL_seen[row] &&
       AU[row] == AU_seen[row])) {
    return;
  }
  for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    const std::int32_t col = col_val[p];
    if (keep_col[col] != std::uint8_t{0}) {
      candidate_mask[col] = std::uint8_t{1};
    }
  }
}

__global__ void _kernel_compute_csr_row_stats(std::int32_t* row_nnz,
                                              std::uint8_t* empty_row_mask,
                                              std::uint8_t* singleton_row_mask,
                                              std::int32_t* singleton_row_col,
                                              double* singleton_row_val,
                                              const std::int32_t* row_ptr,
                                              const std::int32_t* col_val,
                                              const double* nz_val,
                                              std::int32_t rows) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < rows) {
    const std::int32_t first = row_ptr[i];
    const std::int32_t count = row_ptr[i + 1] - first;
    if (row_nnz != nullptr) row_nnz[i] = count;
    if (empty_row_mask != nullptr) {
      empty_row_mask[i] = count == 0 ? std::uint8_t{1} : std::uint8_t{0};
    }
    if (singleton_row_mask != nullptr) {
      singleton_row_mask[i] = count == 1 ? std::uint8_t{1} : std::uint8_t{0};
    }
    if (singleton_row_col != nullptr) {
      singleton_row_col[i] = count == 1 ? col_val[first] : -1;
    }
    if (singleton_row_val != nullptr) {
      singleton_row_val[i] = count == 1 ? nz_val[first] : 0.0;
    }
  }
}

__global__ void _kernel_mask_to_prefix_i32(std::int32_t* out,
                                           const std::uint8_t* mask,
                                           std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    out[j] = mask[j] != std::uint8_t{0} ? 1 : 0;
  }
}

__global__ void _kernel_old_to_new_from_prefix(std::int32_t* old_to_new,
                                               const std::uint8_t* mask,
                                               const std::int32_t* prefix,
                                               std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    old_to_new[j] = mask[j] != std::uint8_t{0} ? prefix[j] - 1 : -1;
  }
}

__global__ void _kernel_count_compacted_rows(std::int32_t* row_counts_new,
                                             const std::int32_t* row_old_to_new,
                                             const std::int32_t* col_old_to_new,
                                             const std::int32_t* row_ptr,
                                             const std::int32_t* col_val,
                                             std::int32_t* long_rows,
                                             std::int32_t* long_row_count,
                                             std::int32_t long_row_threshold,
                                             std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m_old) {
    const std::int32_t row_new = row_old_to_new[row];
    if (row_new < 0) {
      return;
    }
    if (row_ptr[row + 1] - row_ptr[row] > long_row_threshold) {
      const std::int32_t slot = atomicAdd(long_row_count, 1);
      long_rows[slot] = row;
      return;
    }
    std::int32_t count = 0;
    for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      if (col_old_to_new[col_val[p]] >= 0) {
        ++count;
      }
    }
    row_counts_new[row_new] = count;
  }
}

__global__ void _kernel_count_compacted_long_rows(
    std::int32_t* row_counts_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const std::int32_t* long_rows,
    std::int32_t long_row_count) {
  const std::int32_t long_index = static_cast<std::int32_t>(blockIdx.x);
  if (long_index >= long_row_count) {
    return;
  }
  const std::int32_t row = long_rows[long_index];
  std::int32_t count = 0;
  for (std::int32_t p = row_ptr[row] + static_cast<std::int32_t>(threadIdx.x);
       p < row_ptr[row + 1]; p += static_cast<std::int32_t>(blockDim.x)) {
    count += col_old_to_new[col_val[p]] >= 0 ? 1 : 0;
  }
  __shared__ std::int32_t shared_counts[256];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  shared_counts[tid] = count;
  __syncthreads();
  for (std::int32_t stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_counts[tid] += shared_counts[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    row_counts_new[row_old_to_new[row]] = shared_counts[0];
  }
}

__global__ void _kernel_count_compacted_rows_block(
    std::int32_t* row_counts_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= m_old) {
    return;
  }
  const std::int32_t row_new = row_old_to_new[row];
  if (row_new < 0) {
    return;
  }
  std::int32_t local_count = 0;
  for (std::int32_t p = row_ptr[row] + static_cast<std::int32_t>(threadIdx.x);
       p < row_ptr[row + 1]; p += static_cast<std::int32_t>(blockDim.x)) {
    local_count += col_old_to_new[col_val[p]] >= 0 ? 1 : 0;
  }
  using BlockReduce = cub::BlockReduce<std::int32_t, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const std::int32_t count = BlockReduce(reduce_storage).Sum(local_count);
  if (threadIdx.x == 0) {
    row_counts_new[row_new] = count;
  }
}

__global__ void _kernel_row_ptr_from_prefix(std::int32_t* row_ptr,
                                            const std::int32_t* prefix,
                                            std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i == 0) {
    row_ptr[0] = 0;
  }
  if (i < m) {
    row_ptr[i + 1] = prefix[i];
  }
}

__global__ void _kernel_copy_compacted_rows(std::int32_t* col_val_new,
                                            double* nz_val_new,
                                            const std::int32_t* row_ptr_new,
                                            const std::int32_t* row_old_to_new,
                                            const std::int32_t* col_old_to_new,
                                            const std::int32_t* row_ptr,
                                            const std::int32_t* col_val,
                                            const double* nz_val,
                                            std::int32_t long_row_threshold,
                                            std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m_old) {
    const std::int32_t row_new = row_old_to_new[row];
    if (row_new < 0) {
      return;
    }
    if (row_ptr[row + 1] - row_ptr[row] > long_row_threshold) {
      return;
    }
    std::int32_t write = row_ptr_new[row_new];
    for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      const std::int32_t col_new = col_old_to_new[col_val[p]];
      if (col_new >= 0) {
        col_val_new[write] = col_new;
        nz_val_new[write] = nz_val[p];
        ++write;
      }
    }
  }
}

__global__ void _kernel_copy_compacted_long_rows(
    std::int32_t* col_val_new,
    double* nz_val_new,
    const std::int32_t* row_ptr_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const std::int32_t* long_rows,
    std::int32_t long_row_count) {
  const std::int32_t long_index = static_cast<std::int32_t>(blockIdx.x);
  if (long_index >= long_row_count) {
    return;
  }
  const std::int32_t row = long_rows[long_index];
  const std::int32_t row_new = row_old_to_new[row];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  __shared__ std::int32_t scan[256];
  __shared__ std::int32_t base;
  if (tid == 0) {
    base = 0;
  }
  __syncthreads();
  for (std::int32_t chunk = row_ptr[row]; chunk < row_ptr[row + 1]; chunk += 256) {
    const std::int32_t p = chunk + tid;
    std::int32_t col_new = -1;
    if (p < row_ptr[row + 1]) {
      col_new = col_old_to_new[col_val[p]];
    }
    scan[tid] = col_new >= 0 ? 1 : 0;
    __syncthreads();
    for (std::int32_t offset = 1; offset < 256; offset <<= 1) {
      const std::int32_t value =
          scan[tid] + (tid >= offset ? scan[tid - offset] : 0);
      __syncthreads();
      scan[tid] = value;
      __syncthreads();
    }
    if (col_new >= 0) {
      const std::int32_t write = row_ptr_new[row_new] + base + scan[tid] - 1;
      col_val_new[write] = col_new;
      nz_val_new[write] = nz_val[p];
    }
    const std::int32_t chunk_count = scan[255];
    __syncthreads();
    if (tid == 0) {
      base += chunk_count;
    }
    __syncthreads();
  }
}

__global__ void _kernel_copy_compacted_rows_block(
    std::int32_t* col_val_new,
    double* nz_val_new,
    const std::int32_t* row_ptr_new,
    const std::int32_t* row_old_to_new,
    const std::int32_t* col_old_to_new,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t m_old) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
  if (row >= m_old) {
    return;
  }
  const std::int32_t row_new = row_old_to_new[row];
  if (row_new < 0) {
    return;
  }
  using BlockScan = cub::BlockScan<std::int32_t, 256>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ std::int32_t row_write_offset;
  if (threadIdx.x == 0) {
    row_write_offset = 0;
  }
  __syncthreads();
  const std::int32_t row_stop = row_ptr[row + 1];
  for (std::int32_t chunk = row_ptr[row]; chunk < row_stop;
       chunk += 256) {
    const std::int32_t p = chunk + static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t col_new =
        p < row_stop ? col_old_to_new[col_val[p]] : -1;
    const std::int32_t keep = col_new >= 0 ? 1 : 0;
    std::int32_t prefix = 0;
    std::int32_t chunk_count = 0;
    BlockScan(scan_storage).ExclusiveSum(keep, prefix, chunk_count);
    if (keep != 0) {
      const std::int32_t write =
          row_ptr_new[row_new] + row_write_offset + prefix;
      col_val_new[write] = col_new;
      nz_val_new[write] = nz_val[p];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      row_write_offset += chunk_count;
    }
    __syncthreads();
  }
}

__global__ void _kernel_gather_by_old_to_new(double* dst,
                                             const double* src,
                                             const std::int32_t* old_to_new,
                                             std::int32_t n_old) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n_old) {
    const std::int32_t mapped = old_to_new[j];
    if (mapped >= 0) {
      dst[mapped] = src[j];
    }
  }
}

__global__ void _kernel_identity_red2org_from_old_to_new(
    std::int32_t* red2org,
    std::int32_t* removed_fixed_count,
    const std::int32_t* old_to_new,
    const double* l,
    const double* u,
    double bound_tol,
    std::int32_t n_old) {
  using BlockReduce = cub::BlockReduce<std::int32_t, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const std::int32_t old_index =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  std::int32_t removed_fixed = 0;
  if (old_index < n_old) {
    const std::int32_t new_index = old_to_new[old_index];
    if (new_index >= 0) {
      red2org[new_index] = old_index;
    } else if (isfinite(l[old_index]) && isfinite(u[old_index]) &&
               fabs(l[old_index] - u[old_index]) <= bound_tol) {
      removed_fixed = 1;
    }
  }
  const std::int32_t block_count = BlockReduce(reduce_storage).Sum(removed_fixed);
  if (threadIdx.x == 0 && block_count > 0) {
    atomicAdd(removed_fixed_count, block_count);
  }
}

struct StatsRequirements {
  bool row_nnz = false;
  bool singleton_rows = false;
  bool col_nnz = false;
  bool empty_cols = false;
  bool column_singletons = false;
};

void allocate_stats(PresolveStatsGpu& stats,
                    std::int32_t m,
                    std::int32_t n,
                    const StatsRequirements& needs) {
  auto align_up = [](std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  };
  struct SideOffsets {
    std::size_t nnz = 0;
    std::size_t empty_mask = 0;
    std::size_t singleton_mask = 0;
    std::size_t singleton_index = 0;
    std::size_t singleton_value = 0;
  };
  auto layout_side = [&](std::size_t count,
                         bool need_nnz,
                         bool need_empty,
                         bool need_singleton,
                         SideOffsets& side,
                         std::size_t& offset) {
    if (need_nnz) {
      offset = align_up(offset, alignof(std::int32_t));
      side.nnz = offset;
      offset += sizeof(std::int32_t) * count;
    }
    if (need_empty) {
      side.empty_mask = offset;
      offset += count;
    }
    if (need_singleton) {
      side.singleton_mask = offset;
      offset += count;
      offset = align_up(offset, alignof(std::int32_t));
      side.singleton_index = offset;
      offset += sizeof(std::int32_t) * count;
      offset = align_up(offset, alignof(double));
      side.singleton_value = offset;
      offset += sizeof(double) * count;
    }
  };

  const std::size_t row_count = static_cast<std::size_t>(m);
  const std::size_t col_count = static_cast<std::size_t>(n);
  SideOffsets row_offsets;
  SideOffsets col_offsets;
  std::size_t total_bytes = 0;
  const bool need_rows = needs.row_nnz || needs.singleton_rows;
  const bool need_cols = needs.col_nnz || needs.empty_cols || needs.column_singletons;
  if (need_rows && row_count > 0) {
    layout_side(row_count,
                needs.row_nnz,
                false,
                needs.singleton_rows,
                row_offsets,
                total_bytes);
  }
  if (need_cols && col_count > 0) {
    layout_side(col_count,
                needs.col_nnz,
                needs.empty_cols,
                needs.column_singletons,
                col_offsets,
                total_bytes);
  }
  if (total_bytes == 0) {
    return;
  }

  throw_if_cuda_error(stream_ordered_malloc(&stats.contiguous_storage, total_bytes),
                      "cudaMalloc stats contiguous storage");
  auto* storage = static_cast<unsigned char*>(stats.contiguous_storage);
  if (need_rows && row_count > 0) {
    if (needs.row_nnz) {
      stats.row_nnz = reinterpret_cast<std::int32_t*>(storage + row_offsets.nnz);
    }
    if (needs.singleton_rows) {
      stats.singleton_row_mask = storage + row_offsets.singleton_mask;
      stats.singleton_row_col =
          reinterpret_cast<std::int32_t*>(storage + row_offsets.singleton_index);
      stats.singleton_row_val =
          reinterpret_cast<double*>(storage + row_offsets.singleton_value);
    }
  }
  if (need_cols && col_count > 0) {
    if (needs.col_nnz) {
      stats.col_nnz = reinterpret_cast<std::int32_t*>(storage + col_offsets.nnz);
    }
    if (needs.empty_cols) stats.empty_col_mask = storage + col_offsets.empty_mask;
    if (needs.column_singletons) {
      stats.column_singleton_mask = storage + col_offsets.singleton_mask;
      stats.column_singleton_row =
          reinterpret_cast<std::int32_t*>(storage + col_offsets.singleton_index);
      stats.column_singleton_val =
          reinterpret_cast<double*>(storage + col_offsets.singleton_value);
    }
  }
}

void free_stats(PresolveStatsGpu& stats) {
  if (stats.contiguous_storage != nullptr) {
    stream_ordered_free(stats.contiguous_storage);
  } else {
    stream_ordered_free(stats.row_nnz);
    stream_ordered_free(stats.singleton_row_mask);
    stream_ordered_free(stats.singleton_row_col);
    stream_ordered_free(stats.singleton_row_val);
    stream_ordered_free(stats.col_nnz);
    stream_ordered_free(stats.empty_col_mask);
    stream_ordered_free(stats.column_singleton_mask);
    stream_ordered_free(stats.column_singleton_row);
    stream_ordered_free(stats.column_singleton_val);
  }
}

void recompute_row_stats_from_csr(PresolveStatsGpu& stats, const LPInfoGpu& lp) {
  if (lp.A.rows <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int row_blocks = (lp.A.rows + threads - 1) / threads;
  _kernel_compute_csr_row_stats<<<row_blocks, threads>>>(
      stats.row_nnz,
      nullptr,
      stats.singleton_row_mask,
      stats.singleton_row_col,
      stats.singleton_row_val,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      lp.A.rows);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_csr_row_stats row");
}

void recompute_col_stats_from_csr(PresolveStatsGpu& stats, const LPInfoGpu& lp) {
  if (lp.A.cols <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int col_blocks = (lp.A.cols + threads - 1) / threads;
  _kernel_compute_csr_row_stats<<<col_blocks, threads>>>(
      stats.col_nnz,
      stats.empty_col_mask,
      stats.column_singleton_mask,
      stats.column_singleton_row,
      stats.column_singleton_val,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      lp.A.cols);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_compute_csr_row_stats col");
}

struct PlanRequirements {
  bool objective = true;
  bool col_bounds = true;
  bool row_bounds = true;
};

void allocate_plan(PresolvePlanGpu& plan,
                   const LPInfoGpu& lp,
                   const PlanRequirements& needs) {
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  const bool profile_alloc = env_enabled("GPUPRESOLVER_PRESOLVE_ALLOC_PROFILE");
  const auto alloc_start = std::chrono::steady_clock::now();
  auto after_mask_alloc = alloc_start;
  auto after_vector_alloc = alloc_start;
  auto after_mask_fill = alloc_start;
  auto after_row_copy = alloc_start;
  auto after_col_copy = alloc_start;
  const std::size_t row_mask_bytes = static_cast<std::size_t>(m);
  const std::size_t col_mask_bytes = static_cast<std::size_t>(n);
  const std::size_t row_vector_bytes = sizeof(double) * static_cast<std::size_t>(m);
  const std::size_t col_vector_bytes = sizeof(double) * static_cast<std::size_t>(n);
  auto align_up = [](std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  };
  std::size_t offset = row_mask_bytes + col_mask_bytes;
  std::size_t c_offset = 0;
  std::size_t l_offset = 0;
  std::size_t u_offset = 0;
  std::size_t AL_offset = 0;
  std::size_t AU_offset = 0;
  auto reserve_vector = [&](bool required, std::size_t bytes, std::size_t& result) {
    if (!required) return;
    offset = align_up(offset, alignof(double));
    result = offset;
    offset += bytes;
  };
  reserve_vector(needs.objective, col_vector_bytes, c_offset);
  reserve_vector(needs.col_bounds, col_vector_bytes, l_offset);
  reserve_vector(needs.col_bounds, col_vector_bytes, u_offset);
  reserve_vector(needs.row_bounds, row_vector_bytes, AL_offset);
  reserve_vector(needs.row_bounds, row_vector_bytes, AU_offset);
  const std::size_t total_bytes = offset;
  if (total_bytes > 0) {
    throw_if_cuda_error(stream_ordered_malloc(&plan.contiguous_storage, total_bytes),
                        "cudaMalloc plan contiguous storage");
    auto* storage = static_cast<unsigned char*>(plan.contiguous_storage);
    plan.keep_row_mask = storage;
    plan.keep_col_mask = storage + row_mask_bytes;
    if (needs.objective) plan.new_c = reinterpret_cast<double*>(storage + c_offset);
    if (needs.col_bounds) {
      plan.new_l = reinterpret_cast<double*>(storage + l_offset);
      plan.new_u = reinterpret_cast<double*>(storage + u_offset);
    }
    if (needs.row_bounds) {
      plan.new_AL = reinterpret_cast<double*>(storage + AL_offset);
      plan.new_AU = reinterpret_cast<double*>(storage + AU_offset);
    }
  }
  if (profile_alloc) {
    after_mask_alloc = std::chrono::steady_clock::now();
  }
  if (profile_alloc) {
    after_vector_alloc = std::chrono::steady_clock::now();
  }
  constexpr int threads = 256;
  if (m > 0) {
    const int blocks_m = (m + threads - 1) / threads;
    _kernel_fill_u8<<<blocks_m, threads>>>(plan.keep_row_mask, std::uint8_t{1}, m);
    throw_if_cuda_error(cudaGetLastError(), "fill keep_row_mask");
  }
  if (n > 0) {
    const int blocks_n = (n + threads - 1) / threads;
    _kernel_fill_u8<<<blocks_n, threads>>>(plan.keep_col_mask, std::uint8_t{1}, n);
    throw_if_cuda_error(cudaGetLastError(), "fill keep_col_mask");
  }
  if (profile_alloc) {
    throw_if_cuda_error(cudaDeviceSynchronize(), "profile plan mask fill synchronize");
    after_mask_fill = std::chrono::steady_clock::now();
  }
  if (m > 0 && needs.row_bounds) {
    throw_if_cuda_error(cudaMemcpy(plan.new_AL, lp.AL, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan AL");
    throw_if_cuda_error(cudaMemcpy(plan.new_AU, lp.AU, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan AU");
  }
  if (profile_alloc) {
    after_row_copy = std::chrono::steady_clock::now();
  }
  if (n > 0 && needs.objective) {
    throw_if_cuda_error(cudaMemcpy(plan.new_c, lp.c, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan c");
  }
  if (n > 0 && needs.col_bounds) {
    throw_if_cuda_error(cudaMemcpy(plan.new_l, lp.l, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan l");
    throw_if_cuda_error(cudaMemcpy(plan.new_u, lp.u, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                        "cudaMemcpy plan u");
  }
  if (profile_alloc) {
    after_col_copy = std::chrono::steady_clock::now();
    const std::chrono::duration<double> mask_alloc_elapsed = after_mask_alloc - alloc_start;
    const std::chrono::duration<double> vector_alloc_elapsed = after_vector_alloc - after_mask_alloc;
    const std::chrono::duration<double> mask_fill_elapsed = after_mask_fill - after_vector_alloc;
    const std::chrono::duration<double> row_copy_elapsed = after_row_copy - after_mask_fill;
    const std::chrono::duration<double> col_copy_elapsed = after_col_copy - after_row_copy;
    const std::chrono::duration<double> total_elapsed = after_col_copy - alloc_start;
    std::cerr << ">>> [GPU Presolve C++ alloc] rows=" << m << " cols=" << n
              << " total=" << total_elapsed.count() << "s"
              << " mask_malloc=" << mask_alloc_elapsed.count() << "s"
              << " vector_malloc=" << vector_alloc_elapsed.count() << "s"
              << " mask_fill=" << mask_fill_elapsed.count() << "s"
              << " row_copy=" << row_copy_elapsed.count() << "s"
              << " col_copy=" << col_copy_elapsed.count() << "s\n";
  }
}

void free_plan(PresolvePlanGpu& plan) {
  const bool profile_alloc = env_enabled("GPUPRESOLVER_PRESOLVE_ALLOC_PROFILE");
  const auto free_start = std::chrono::steady_clock::now();
  if (plan.contiguous_storage != nullptr) {
    stream_ordered_free(plan.contiguous_storage);
  } else {
    stream_ordered_free(plan.keep_row_mask);
    stream_ordered_free(plan.keep_col_mask);
    stream_ordered_free(plan.new_c);
    stream_ordered_free(plan.new_l);
    stream_ordered_free(plan.new_u);
    stream_ordered_free(plan.new_AL);
    stream_ordered_free(plan.new_AU);
  }
  if (plan.has_new_A) {
    stream_ordered_free(plan.new_A.rowPtr);
    stream_ordered_free(plan.new_A.colVal);
    stream_ordered_free(plan.new_A.nzVal);
  }
  if (profile_alloc) {
    throw_if_cuda_error(cudaDeviceSynchronize(), "profile plan free synchronize");
    const std::chrono::duration<double> free_elapsed =
        std::chrono::steady_clock::now() - free_start;
    std::cerr << ">>> [GPU Presolve C++ alloc] free=" << free_elapsed.count() << "s\n";
  }
}

std::int32_t build_prefix_from_mask(std::int32_t* prefix,
                                    const std::uint8_t* mask,
                                    std::int32_t n) {
  if (n == 0) {
    return 0;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_mask_to_prefix_i32<<<blocks, threads>>>(prefix, mask, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mask_to_prefix_i32");
  inclusive_scan_i32(prefix, n, "cub inclusive scan mask");
  std::int32_t count = 0;
  throw_if_cuda_error(cudaMemcpy(&count, prefix + n - 1, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy prefix count");
  return count;
}

std::int32_t* build_old_to_new(const std::uint8_t* mask,
                               std::int32_t n,
                               std::int32_t* count_out) {
  if (n == 0) {
    *count_out = 0;
    return nullptr;
  }
  std::int32_t* prefix = nullptr;
  std::int32_t* old_to_new = nullptr;
  throw_if_cuda_error(cudaMalloc(&prefix, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc prefix");
  throw_if_cuda_error(cudaMalloc(&old_to_new, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMalloc old_to_new");
  const std::int32_t count = build_prefix_from_mask(prefix, mask, n);
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_old_to_new_from_prefix<<<blocks, threads>>>(old_to_new, mask, prefix, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_old_to_new_from_prefix");
  cudaFree(prefix);
  *count_out = count;
  return old_to_new;
}

double* compact_vector(const double* src,
                       const std::int32_t* old_to_new,
                       std::int32_t n_old,
                       std::int32_t n_new) {
  double* dst = nullptr;
  if (n_new > 0) {
    throw_if_cuda_error(cudaMalloc(&dst, sizeof(double) * static_cast<std::size_t>(n_new)),
                        "cudaMalloc compact vector");
  }
  if (n_old > 0 && n_new > 0) {
    constexpr int threads = 256;
    const int blocks = (n_old + threads - 1) / threads;
    _kernel_gather_by_old_to_new<<<blocks, threads>>>(dst, src, old_to_new, n_old);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_gather_by_old_to_new");
  }
  return dst;
}

double* clone_device_vector(const double* src, std::int32_t n, const char* context) {
  double* dst = nullptr;
  if (n > 0) {
    throw_if_cuda_error(cudaMalloc(&dst, sizeof(double) * static_cast<std::size_t>(n)),
                        context);
    throw_if_cuda_error(cudaMemcpy(dst, src, sizeof(double) * static_cast<std::size_t>(n),
                                   cudaMemcpyDeviceToDevice),
                        context);
  }
  return dst;
}

template <class T>
std::vector<T> copy_device_array(const T* device, std::int32_t count, const char* context) {
  std::vector<T> values(static_cast<std::size_t>(count));
  if (count > 0) {
    throw_if_cuda_error(cudaMemcpy(values.data(), device, sizeof(T) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  return values;
}

std::vector<std::int32_t> compact_identity_red2org_on_gpu(
    const std::int32_t* old_to_new,
    const double* l,
    const double* u,
    std::int32_t n_old,
    std::int32_t n_new,
    std::int32_t* removed_fixed_count,
    const char* context) {
  if (n_new <= 0) {
    *removed_fixed_count = 0;
    return {};
  }
  std::int32_t* red2org_d = nullptr;
  std::int32_t* removed_fixed_count_d = nullptr;
  throw_if_cuda_error(
      stream_ordered_malloc(
          &red2org_d,
          sizeof(std::int32_t) * static_cast<std::size_t>(n_new)),
      context);
  throw_if_cuda_error(stream_ordered_malloc(&removed_fixed_count_d,
                                            sizeof(std::int32_t)),
                      context);
  throw_if_cuda_error(cudaMemset(removed_fixed_count_d, 0,
                                 sizeof(std::int32_t)),
                      context);
  constexpr int threads = 256;
  const int blocks = (n_old + threads - 1) / threads;
  _kernel_identity_red2org_from_old_to_new<<<blocks, threads>>>(
      red2org_d, removed_fixed_count_d, old_to_new, l, u, 1.0e-9, n_old);
  throw_if_cuda_error(cudaGetLastError(),
                      "_kernel_identity_red2org_from_old_to_new");
  std::vector<std::int32_t> red2org(static_cast<std::size_t>(n_new));
  throw_if_cuda_error(
      cudaMemcpy(red2org.data(), red2org_d,
                 sizeof(std::int32_t) * static_cast<std::size_t>(n_new),
                 cudaMemcpyDeviceToHost),
      context);
  throw_if_cuda_error(cudaMemcpy(removed_fixed_count, removed_fixed_count_d,
                                 sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  stream_ordered_free(red2org_d);
  stream_ordered_free(removed_fixed_count_d);
  return red2org;
}

__global__ void _kernel_removed_fixed_bound_record_counts(std::int32_t* selected_scan,
                                                          const std::uint8_t* keep_col,
                                                          const double* l,
                                                          const double* u,
                                                          double bound_tol,
                                                          std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n) {
    const bool removed = keep_col[col] == std::uint8_t{0};
    const bool fixed = isfinite(l[col]) && isfinite(u[col]) && fabs(l[col] - u[col]) <= bound_tol;
    selected_scan[col] = removed && fixed ? 1 : 0;
  }
}

__global__ void _kernel_pack_removed_fixed_bound_records(std::int32_t* packed_cols,
                                                         double* packed_vals,
                                                         const std::int32_t* selected_scan,
                                                         const std::uint8_t* keep_col,
                                                         const double* l,
                                                         const double* u,
                                                         double bound_tol,
                                                         std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n) {
    return;
  }
  const bool removed = keep_col[col] == std::uint8_t{0};
  const bool fixed = isfinite(l[col]) && isfinite(u[col]) && fabs(l[col] - u[col]) <= bound_tol;
  if (!removed || !fixed) {
    return;
  }
  const std::int32_t prev_selected = col == 0 ? 0 : selected_scan[col - 1];
  const std::int32_t out = selected_scan[col] - 1;
  if (selected_scan[col] == prev_selected || out < 0) {
    return;
  }
  packed_cols[out] = col;
  packed_vals[out] = 0.5 * (l[col] + u[col]);
}

__global__ void _kernel_gather_removed_bound_values(double* lower_out,
                                                    double* upper_out,
                                                    const std::int32_t* local_cols,
                                                    const double* l,
                                                    const double* u,
                                                    std::int32_t count) {
  const std::int32_t k = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < count) {
    const std::int32_t col = local_cols[k];
    lower_out[k] = l[col];
    upper_out[k] = u[col];
  }
}

__global__ void _kernel_removed_col_counts(std::int32_t* selected_scan,
                                           const std::uint8_t* keep_col,
                                           std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n) {
    selected_scan[col] = keep_col[col] == std::uint8_t{0} ? 1 : 0;
  }
}

__global__ void _kernel_pack_removed_cols(std::int32_t* packed_cols,
                                          const std::int32_t* selected_scan,
                                          const std::uint8_t* keep_col,
                                          std::int32_t n) {
  const std::int32_t col = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || keep_col[col] != std::uint8_t{0}) {
    return;
  }
  const std::int32_t prev_selected = col == 0 ? 0 : selected_scan[col - 1];
  const std::int32_t out = selected_scan[col] - 1;
  if (selected_scan[col] == prev_selected || out < 0) {
    return;
  }
  packed_cols[out] = col;
}

std::vector<std::int32_t> pack_removed_cols_sparse(const std::uint8_t* keep_col,
                                                   std::int32_t n,
                                                   std::int32_t removed_count,
                                                   const char* context) {
  if (n <= 0 || removed_count <= 0) {
    return {};
  }

  std::int32_t* selected_scan = nullptr;
  std::int32_t* packed_cols = nullptr;
  throw_if_cuda_error(cudaMalloc(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(removed_count)),
                      context);

  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  _kernel_removed_col_counts<<<blocks, threads>>>(selected_scan, keep_col, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_removed_col_counts");
  inclusive_scan_i32(selected_scan, n, context);
  _kernel_pack_removed_cols<<<blocks, threads>>>(packed_cols, selected_scan, keep_col, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_removed_cols");

  std::vector<std::int32_t> host_cols(static_cast<std::size_t>(removed_count));
  throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(removed_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  cudaFree(selected_scan);
  cudaFree(packed_cols);
  return host_cols;
}

void append_removed_fixed_bound_records_sparse(
    PresolveRecordGpu& record,
    const std::vector<std::int32_t>& removed_local_cols,
    const double* l,
    const double* u,
    const std::vector<std::int32_t>& old_col_red2org,
    const char* context) {
  const std::int32_t count = static_cast<std::int32_t>(removed_local_cols.size());
  if (count <= 0) {
    return;
  }

  std::int32_t* local_cols_d = nullptr;
  double* lower_d = nullptr;
  double* upper_d = nullptr;
  const std::size_t cols_bytes = sizeof(std::int32_t) * static_cast<std::size_t>(count);
  const std::size_t vals_bytes = sizeof(double) * static_cast<std::size_t>(count);
  throw_if_cuda_error(cudaMalloc(&local_cols_d, cols_bytes), context);
  throw_if_cuda_error(cudaMalloc(&lower_d, vals_bytes), context);
  throw_if_cuda_error(cudaMalloc(&upper_d, vals_bytes), context);
  throw_if_cuda_error(cudaMemcpy(local_cols_d, removed_local_cols.data(), cols_bytes,
                                 cudaMemcpyHostToDevice),
                      context);

  constexpr int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  _kernel_gather_removed_bound_values<<<blocks, threads>>>(
      lower_d, upper_d, local_cols_d, l, u, count);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_gather_removed_bound_values");

  std::vector<double> lower(static_cast<std::size_t>(count));
  std::vector<double> upper(static_cast<std::size_t>(count));
  throw_if_cuda_error(cudaMemcpy(lower.data(), lower_d, vals_bytes, cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(upper.data(), upper_d, vals_bytes, cudaMemcpyDeviceToHost),
                      context);

  for (std::int32_t k = 0; k < count; ++k) {
    const double lo = lower[static_cast<std::size_t>(k)];
    const double hi = upper[static_cast<std::size_t>(k)];
    if (!isfinite(lo) || !isfinite(hi) || fabs(lo - hi) > 1.0e-9) {
      continue;
    }
    const std::int32_t local_col = removed_local_cols[static_cast<std::size_t>(k)];
    if (local_col >= 0 && local_col < static_cast<std::int32_t>(old_col_red2org.size())) {
      record.fixed_idx.push_back(old_col_red2org[static_cast<std::size_t>(local_col)]);
      record.fixed_val.push_back(0.5 * (lo + hi));
    }
  }

  cudaFree(local_cols_d);
  cudaFree(lower_d);
  cudaFree(upper_d);
}

void append_removed_fixed_bound_records(PresolveRecordGpu& record,
                                        const std::uint8_t* keep_col,
                                        const double* l,
                                        const double* u,
                                        const std::vector<std::int32_t>& old_col_red2org,
                                        std::int32_t n,
                                        const char* context) {
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  std::int32_t* selected_scan = nullptr;
  throw_if_cuda_error(cudaMalloc(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      context);
  _kernel_removed_fixed_bound_record_counts<<<blocks, threads>>>(
      selected_scan, keep_col, l, u, 1.0e-9, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_removed_fixed_bound_record_counts");
  inclusive_scan_i32(selected_scan, n, context);
  std::int32_t count = 0;
  throw_if_cuda_error(cudaMemcpy(&count, selected_scan + n - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (count <= 0) {
    cudaFree(selected_scan);
    return;
  }

  std::int32_t* packed_cols = nullptr;
  double* packed_vals = nullptr;
  throw_if_cuda_error(cudaMalloc(&packed_cols, sizeof(std::int32_t) * static_cast<std::size_t>(count)),
                      context);
  throw_if_cuda_error(cudaMalloc(&packed_vals, sizeof(double) * static_cast<std::size_t>(count)),
                      context);
  _kernel_pack_removed_fixed_bound_records<<<blocks, threads>>>(
      packed_cols, packed_vals, selected_scan, keep_col, l, u, 1.0e-9, n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_removed_fixed_bound_records");

  std::vector<std::int32_t> host_cols(static_cast<std::size_t>(count));
  std::vector<double> host_vals(static_cast<std::size_t>(count));
  throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_vals.data(), packed_vals,
                                 sizeof(double) * static_cast<std::size_t>(count),
                                 cudaMemcpyDeviceToHost),
                      context);
  record.fixed_idx.reserve(record.fixed_idx.size() + static_cast<std::size_t>(count));
  record.fixed_val.reserve(record.fixed_val.size() + static_cast<std::size_t>(count));
  for (std::int32_t k = 0; k < count; ++k) {
    const std::int32_t local_col = host_cols[static_cast<std::size_t>(k)];
    if (local_col >= 0 && local_col < static_cast<std::int32_t>(old_col_red2org.size())) {
      record.fixed_idx.push_back(old_col_red2org[static_cast<std::size_t>(local_col)]);
      record.fixed_val.push_back(host_vals[static_cast<std::size_t>(k)]);
    }
  }
  cudaFree(selected_scan);
  cudaFree(packed_cols);
  cudaFree(packed_vals);
}

std::vector<std::int32_t> make_identity_i32(std::int32_t n) {
  std::vector<std::int32_t> values(static_cast<std::size_t>(n));
  if (values.size() <
      static_cast<std::size_t>(PARALLEL_IDENTITY_FILL_THRESHOLD)) {
    for (std::int32_t i = 0; i < n; ++i) {
      values[static_cast<std::size_t>(i)] = i;
    }
    return values;
  }

  const unsigned hardware_threads = std::thread::hardware_concurrency();
  const unsigned worker_count = std::min<unsigned>(
      PARALLEL_IDENTITY_FILL_MAX_THREADS,
      hardware_threads == 0 ? 1U : hardware_threads);
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  const std::int64_t chunk =
      (static_cast<std::int64_t>(n) + worker_count - 1) / worker_count;
  try {
    for (unsigned worker = 0; worker < worker_count; ++worker) {
      const std::int32_t begin = static_cast<std::int32_t>(
          std::min<std::int64_t>(static_cast<std::int64_t>(n),
                                 static_cast<std::int64_t>(worker) * chunk));
      const std::int32_t end = static_cast<std::int32_t>(
          std::min<std::int64_t>(static_cast<std::int64_t>(n),
                                 static_cast<std::int64_t>(worker + 1) * chunk));
      workers.emplace_back([&values, begin, end]() {
        for (std::int32_t i = begin; i < end; ++i) {
          values[static_cast<std::size_t>(i)] = i;
        }
      });
    }
  } catch (...) {
    for (std::thread& worker : workers) {
      worker.join();
    }
    throw;
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  return values;
}

PresolveRecordGpu make_initial_record(const LPInfoGpu& lp) {
  PresolveRecordGpu record;
  record.m0 = lp.A.rows;
  record.n0 = lp.A.cols;
  record.m1 = lp.A.rows;
  record.n1 = lp.A.cols;
  record.row_red2org = make_identity_i32(lp.A.rows);
  record.col_red2org = make_identity_i32(lp.A.cols);
  record.obj_constant_old = lp.obj_constant;
  record.obj_constant_new = lp.obj_constant;
  return record;
}

void rebuild_org2red(std::vector<std::int32_t>& org2red,
                     const std::vector<std::int32_t>& red2org,
                     std::int32_t original_count) {
  org2red.assign(static_cast<std::size_t>(original_count), -1);
  for (std::int32_t red = 0; red < static_cast<std::int32_t>(red2org.size()); ++red) {
    const std::int32_t org = red2org[static_cast<std::size_t>(red)];
    if (org >= 0 && org < original_count) {
      org2red[static_cast<std::size_t>(org)] = red;
    }
  }
}

StructuralL1PrimalRecoveryStep globalize_structural_recovery(
    const StructuralL1PrimalRecoveryStep& local,
    const std::vector<std::int32_t>& col_red2org) {
  StructuralL1PrimalRecoveryStep global = local;
  for (StructuralL1SplitRecovery& split : global.splits) {
    split.t_col = col_red2org[static_cast<std::size_t>(split.t_col)];
    split.e_col = col_red2org[static_cast<std::size_t>(split.e_col)];
  }
  for (StructuralOuterPairRecovery& pair : global.outer_pairs) {
    pair.bound_col = col_red2org[static_cast<std::size_t>(pair.bound_col)];
    pair.free_col = col_red2org[static_cast<std::size_t>(pair.free_col)];
  }
  for (StructuralLinkedSlackRecovery& slack : global.linked_slacks) {
    slack.slack_col = col_red2org[static_cast<std::size_t>(slack.slack_col)];
    slack.t_col = col_red2org[static_cast<std::size_t>(slack.t_col)];
  }
  for (StructuralMaxSlackRecovery& slack : global.max_slacks) {
    slack.slack_col = col_red2org[static_cast<std::size_t>(slack.slack_col)];
    for (std::int32_t& t_col : slack.t_cols) {
      t_col = col_red2org[static_cast<std::size_t>(t_col)];
    }
  }
  return global;
}

AntipodalComponentPrimalRecoveryStep globalize_antipodal_recovery(
    const AntipodalComponentPrimalRecoveryStep& local,
    const std::vector<std::int32_t>& col_red2org) {
  AntipodalComponentPrimalRecoveryStep global = local;
  const std::size_t count = global.plus_cols.size();
  for (std::size_t i = 0; i < count; ++i) {
    global.plus_cols[i] =
        col_red2org[static_cast<std::size_t>(global.plus_cols[i])];
    global.minus_cols[i] =
        col_red2org[static_cast<std::size_t>(global.minus_cols[i])];
    global.root_plus_cols[i] =
        col_red2org[static_cast<std::size_t>(global.root_plus_cols[i])];
    global.root_minus_cols[i] =
        col_red2org[static_cast<std::size_t>(global.root_minus_cols[i])];
  }
  return global;
}

std::int32_t map_local_col_to_global(std::int32_t col,
                                     const std::vector<std::int32_t>& col_red2org) {
  if (col < 0 || col >= static_cast<std::int32_t>(col_red2org.size())) {
    return col;
  }
  return col_red2org[static_cast<std::size_t>(col)];
}

std::int32_t map_local_row_to_global(std::int32_t row,
                                     const std::vector<std::int32_t>& row_red2org) {
  if (row < 0 || row >= static_cast<std::int32_t>(row_red2org.size())) {
    return row;
  }
  return row_red2org[static_cast<std::size_t>(row)];
}

void append_globalized_postsolve_tape(PostsolveTape& dest,
                                      const PostsolveTape& src,
                                      const std::vector<std::int32_t>& row_red2org,
                                      const std::vector<std::int32_t>& col_red2org) {
  if (src.types.empty()) {
    return;
  }
  const std::size_t old_records = dest.types.size();
  const std::size_t old_indices = dest.indices.size();
  const std::size_t old_values = dest.vals.size();
  const auto reserve_for_append = [](auto& values, std::size_t added) {
    const std::size_t required = values.size() + added;
    if (required <= values.capacity()) {
      return;
    }
    const std::size_t doubled = values.capacity() <=
            std::numeric_limits<std::size_t>::max() / 2
        ? values.capacity() * 2
        : std::numeric_limits<std::size_t>::max();
    values.reserve(std::max(required, doubled));
  };
  reserve_for_append(dest.types, src.types.size());
  reserve_for_append(dest.index_starts, src.types.size());
  reserve_for_append(dest.value_starts, src.types.size());
  reserve_for_append(dest.dual_modes, src.dual_modes.size());
  reserve_for_append(dest.indices, src.indices.size());
  reserve_for_append(dest.vals, src.vals.size());

  // Append each flat array once while preserving chronological record order.
  // This avoids repeated small insertions for singleton-heavy models.
  dest.types.insert(dest.types.end(), src.types.begin(), src.types.end());
  dest.dual_modes.insert(dest.dual_modes.end(),
                         src.dual_modes.begin(), src.dual_modes.end());
  dest.indices.insert(dest.indices.end(), src.indices.begin(), src.indices.end());
  dest.vals.insert(dest.vals.end(), src.vals.begin(), src.vals.end());

  dest.index_starts.resize(old_records + src.types.size() + 1);
  dest.value_starts.resize(old_records + src.types.size() + 1);
  for (std::size_t k = 1; k <= src.types.size(); ++k) {
    dest.index_starts[old_records + k] = static_cast<std::int32_t>(
        old_indices + static_cast<std::size_t>(src.index_starts[k]));
    dest.value_starts[old_records + k] = static_cast<std::int32_t>(
        old_values + static_cast<std::size_t>(src.value_starts[k]));
  }

  struct IndexSegment {
    std::vector<std::int32_t>& values;
    std::size_t offset;
    std::size_t count;

    bool empty() const { return count == 0; }
    std::size_t size() const { return count; }
    std::int32_t& operator[](std::size_t index) { return values[offset + index]; }
  };

  for (std::int32_t k = 0; k < static_cast<std::int32_t>(src.types.size()); ++k) {
    const std::int32_t idx0 = src.index_starts[static_cast<std::size_t>(k)];
    const std::int32_t idx1 = src.index_starts[static_cast<std::size_t>(k + 1)];
    const std::size_t index_offset =
        old_indices + static_cast<std::size_t>(idx0);
    IndexSegment indices{dest.indices, index_offset,
                         static_cast<std::size_t>(idx1 - idx0)};

    switch (static_cast<PostsolveReductionType>(src.types[static_cast<std::size_t>(k)])) {
      case PostsolveReductionType::FixedCol:
        if (!indices.empty()) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          for (std::size_t p = 1; p < indices.size(); ++p) {
            indices[p] = map_local_row_to_global(indices[p], row_red2org);
          }
        }
        break;
      case PostsolveReductionType::FixedColInf:
        if (indices.size() >= 2) {
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
          std::size_t p = 2;
          while (p < indices.size()) {
            const std::int32_t row_len = indices[p++];
            for (std::int32_t t = 0; t < row_len && p < indices.size(); ++t, ++p) {
              indices[p] = map_local_col_to_global(indices[p], col_red2org);
            }
          }
        }
        break;
      case PostsolveReductionType::SubCol:
        if (indices.size() >= 3) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_row_to_global(indices[1], row_red2org);
          const std::int32_t support_count = indices[2];
          for (std::int32_t t = 0; t < support_count && 3 + t < static_cast<std::int32_t>(indices.size()); ++t) {
            indices[static_cast<std::size_t>(3 + t)] =
                map_local_col_to_global(indices[static_cast<std::size_t>(3 + t)], col_red2org);
          }
        }
        break;
      case PostsolveReductionType::DuplicateColumn:
        if (indices.size() >= 2) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
        }
        break;
      case PostsolveReductionType::DuplicateRow:
        if (indices.size() >= 2) {
          indices[0] = map_local_row_to_global(indices[0], row_red2org);
          indices[1] = map_local_row_to_global(indices[1], row_red2org);
        }
        break;
      case PostsolveReductionType::DeletedRow:
        if (!indices.empty()) {
          indices[0] = map_local_row_to_global(indices[0], row_red2org);
          if (indices.size() >= 2) {
            indices[1] = map_local_col_to_global(indices[1], col_red2org);
          }
        }
        break;
      case PostsolveReductionType::LhsChange:
      case PostsolveReductionType::RhsChange:
      case PostsolveReductionType::EqToIneq:
        if (!indices.empty()) {
          indices[0] = map_local_row_to_global(indices[0], row_red2org);
        }
        break;
      case PostsolveReductionType::BoundChangeNoRow:
        if (!indices.empty()) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
        }
        break;
      case PostsolveReductionType::BoundChangeTheRow:
        if (indices.size() >= 2) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_row_to_global(indices[1], row_red2org);
        }
        break;
      case PostsolveReductionType::DoubletonEquation:
        if (indices.size() >= 4) {
          indices[0] = map_local_col_to_global(indices[0], col_red2org);
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
          indices[2] = map_local_row_to_global(indices[2], row_red2org);
          const std::int32_t support_count = indices[3];
          for (std::int32_t t = 0; t < support_count && 4 + t < static_cast<std::int32_t>(indices.size()); ++t) {
            indices[static_cast<std::size_t>(4 + t)] =
                map_local_row_to_global(indices[static_cast<std::size_t>(4 + t)], row_red2org);
          }
        }
        break;
      case PostsolveReductionType::FmeCol:
        if (indices.size() >= 3) {
          const std::int32_t involved_count = indices[0];
          indices[1] = map_local_col_to_global(indices[1], col_red2org);
          for (std::int32_t t = 0; t < involved_count && 2 + t < static_cast<std::int32_t>(indices.size()); ++t) {
            indices[static_cast<std::size_t>(2 + t)] =
                map_local_row_to_global(indices[static_cast<std::size_t>(2 + t)], row_red2org);
          }
          const std::int32_t side_count_pos = 2 + involved_count;
          if (side_count_pos < static_cast<std::int32_t>(indices.size())) {
            const std::int32_t side_col_count = indices[static_cast<std::size_t>(side_count_pos)];
            for (std::int32_t t = 0; t < side_col_count &&
                                     side_count_pos + 1 + t < static_cast<std::int32_t>(indices.size()); ++t) {
              indices[static_cast<std::size_t>(side_count_pos + 1 + t)] =
                  map_local_col_to_global(indices[static_cast<std::size_t>(side_count_pos + 1 + t)], col_red2org);
            }
          }
        }
        break;
      case PostsolveReductionType::AddedRow:
      case PostsolveReductionType::AddedRows:
        break;
    }

  }
}

void update_record_from_plan(PresolveRecordGpu& record,
                             const PresolvePlanGpu& plan,
                             const LPInfoGpu& old_lp,
                             std::int32_t m_new,
                             std::int32_t n_new,
                             double obj_constant_new,
                             const std::int32_t* col_old_to_new) {
  const bool profile_record = env_enabled("GPUPRESOLVER_PRESOLVE_RECORD_PROFILE");
  const auto record_start = std::chrono::steady_clock::now();
  const bool has_removed_rows = m_new < old_lp.A.rows;
  const bool has_removed_cols = n_new < old_lp.A.cols;
  const std::int32_t removed_row_count = has_removed_rows ? (old_lp.A.rows - m_new) : 0;
  const std::int32_t removed_col_count = has_removed_cols ? (old_lp.A.cols - n_new) : 0;
  const bool sparse_row_update =
      has_removed_rows && removed_row_count <= SPARSE_FIXED_BOUND_RECORD_THRESHOLD;
  const bool sparse_col_update =
      has_removed_cols && removed_col_count <= SPARSE_FIXED_BOUND_RECORD_THRESHOLD;
  const bool record_removed_indices = env_enabled("GPUPRESOLVER_RECORD_REMOVED_INDICES");
  const bool identity_gpu_col_update =
      has_removed_cols && n_new > 0 && old_lp.A.cols >= 1000000 &&
      removed_col_count >= old_lp.A.cols / 8 &&
      record.col_red2org.size() == static_cast<std::size_t>(record.n0) &&
      !record_removed_indices;
  const std::vector<std::uint8_t> keep_row = has_removed_rows && !sparse_row_update
      ? copy_device_array(plan.keep_row_mask, old_lp.A.rows, "cudaMemcpy record keep_row")
      : std::vector<std::uint8_t>{};
  const std::vector<std::uint8_t> keep_col =
      has_removed_cols && !sparse_col_update && !identity_gpu_col_update
      ? copy_device_array(plan.keep_col_mask, old_lp.A.cols, "cudaMemcpy record keep_col")
      : std::vector<std::uint8_t>{};
  const auto after_keep_copy = std::chrono::steady_clock::now();

  const std::vector<std::int32_t>& old_row_red2org = record.row_red2org;
  const std::vector<std::int32_t>& old_col_red2org = record.col_red2org;
  const auto after_old_mapping_copy = std::chrono::steady_clock::now();
  std::vector<std::int32_t> next_row_red2org;
  std::vector<std::int32_t> next_col_red2org;
  std::vector<std::int32_t> removed_local_rows;
  std::vector<std::int32_t> removed_local_cols;
  std::int32_t identity_removed_fixed_count = -1;
  if (has_removed_rows) {
    next_row_red2org.reserve(static_cast<std::size_t>(m_new));
    if (record_removed_indices) {
      record.removed_row_idx.reserve(record.removed_row_idx.size() +
                                     static_cast<std::size_t>(old_lp.A.rows - m_new));
    }
    if (sparse_row_update) {
      removed_local_rows.reserve(static_cast<std::size_t>(removed_row_count));
    }
  }
  if (has_removed_cols) {
    next_col_red2org.reserve(static_cast<std::size_t>(n_new));
    if (record_removed_indices) {
      record.removed_col_idx.reserve(record.removed_col_idx.size() +
                                     static_cast<std::size_t>(old_lp.A.cols - n_new));
    }
    if (sparse_col_update) {
      removed_local_cols.reserve(static_cast<std::size_t>(removed_col_count));
    }
  }

  if (has_removed_rows) {
    if (sparse_row_update) {
      removed_local_rows = pack_removed_cols_sparse(plan.keep_row_mask,
                                                    old_lp.A.rows,
                                                    removed_row_count,
                                                    "cudaMemcpy sparse removed rows");
      std::int32_t range_begin = 0;
      for (const std::int32_t local_row : removed_local_rows) {
        if (local_row < range_begin ||
            local_row >= static_cast<std::int32_t>(old_row_red2org.size())) {
          continue;
        }
        next_row_red2org.insert(next_row_red2org.end(),
                                old_row_red2org.begin() + range_begin,
                                old_row_red2org.begin() + local_row);
        if (record_removed_indices) {
          record.removed_row_idx.push_back(
              old_row_red2org[static_cast<std::size_t>(local_row)]);
        }
        range_begin = local_row + 1;
      }
      next_row_red2org.insert(next_row_red2org.end(),
                              old_row_red2org.begin() + range_begin,
                              old_row_red2org.end());
    } else {
      next_row_red2org.resize(static_cast<std::size_t>(m_new));
      std::size_t next_row = 0;
      for (std::int32_t row = 0; row < old_lp.A.rows; ++row) {
        const std::int32_t global =
            old_row_red2org[static_cast<std::size_t>(row)];
        if (keep_row[static_cast<std::size_t>(row)] != std::uint8_t{0}) {
          next_row_red2org[next_row++] = global;
        } else {
          if (record_removed_indices) {
            record.removed_row_idx.push_back(global);
          }
        }
      }
      if (next_row != static_cast<std::size_t>(m_new)) {
        throw std::runtime_error("row mapping compaction count mismatch");
      }
    }
  }
  if (has_removed_cols) {
    if (identity_gpu_col_update) {
      next_col_red2org = compact_identity_red2org_on_gpu(
          col_old_to_new,
          plan.new_l != nullptr ? plan.new_l : old_lp.l,
          plan.new_u != nullptr ? plan.new_u : old_lp.u,
          old_lp.A.cols, n_new, &identity_removed_fixed_count,
          "cudaMemcpy compact identity column mapping");
    } else if (sparse_col_update) {
      removed_local_cols = pack_removed_cols_sparse(plan.keep_col_mask,
                                                    old_lp.A.cols,
                                                    removed_col_count,
                                                    "cudaMemcpy sparse removed cols");
      std::int32_t range_begin = 0;
      for (const std::int32_t local_col : removed_local_cols) {
        if (local_col < range_begin ||
            local_col >= static_cast<std::int32_t>(old_col_red2org.size())) {
          continue;
        }
        next_col_red2org.insert(next_col_red2org.end(),
                                old_col_red2org.begin() + range_begin,
                                old_col_red2org.begin() + local_col);
        if (record_removed_indices) {
          record.removed_col_idx.push_back(old_col_red2org[static_cast<std::size_t>(local_col)]);
        }
        range_begin = local_col + 1;
      }
      next_col_red2org.insert(next_col_red2org.end(),
                              old_col_red2org.begin() + range_begin,
                              old_col_red2org.end());
    } else {
      next_col_red2org.resize(static_cast<std::size_t>(n_new));
      std::size_t next_col = 0;
      for (std::int32_t col = 0; col < old_lp.A.cols; ++col) {
        const std::int32_t global =
            old_col_red2org[static_cast<std::size_t>(col)];
        if (keep_col[static_cast<std::size_t>(col)] != std::uint8_t{0}) {
          next_col_red2org[next_col++] = global;
        } else {
          if (record_removed_indices) {
            record.removed_col_idx.push_back(global);
          }
        }
      }
      if (next_col != static_cast<std::size_t>(n_new)) {
        throw std::runtime_error("column mapping compaction count mismatch");
      }
    }
    if (identity_gpu_col_update && identity_removed_fixed_count == 0) {
      // The identity compaction kernel already checked every removed column.
      // A nonzero count falls through to the original stable packing path.
    } else if (sparse_col_update) {
      append_removed_fixed_bound_records_sparse(record,
                                               removed_local_cols,
                                               plan.new_l != nullptr ? plan.new_l : old_lp.l,
                                               plan.new_u != nullptr ? plan.new_u : old_lp.u,
                                               old_col_red2org,
                                               "cudaMemcpy sparse record fixed bounds");
    } else {
      append_removed_fixed_bound_records(record,
                                         plan.keep_col_mask,
                                         plan.new_l != nullptr ? plan.new_l : old_lp.l,
                                         plan.new_u != nullptr ? plan.new_u : old_lp.u,
                                         old_col_red2org,
                                         old_lp.A.cols,
                                         "cudaMemcpy record fixed bounds");
    }
  }
  const auto after_removed_mapping = std::chrono::steady_clock::now();

  const bool has_primal_recovery =
      plan.has_structural_primal_recovery ||
      plan.has_antipodal_component_recovery;
  if (plan.has_structural_primal_recovery &&
      plan.has_antipodal_component_recovery) {
    throw std::runtime_error(
        "a presolve plan cannot commit structural and antipodal primal "
        "recoveries at the same chronological position");
  }
  if (has_primal_recovery && !plan.tape.types.empty()) {
    throw std::runtime_error(
        "a primal-recovery plan must not also contain ordinary postsolve "
        "records; split it into chronological plans");
  }
  if (record.tape.types.size() > static_cast<std::size_t>(INT32_MAX)) {
    throw std::runtime_error("postsolve tape exceeds 32-bit checkpoint range");
  }
  const std::int32_t tape_position =
      static_cast<std::int32_t>(record.tape.types.size());
  if (plan.has_structural_primal_recovery) {
    if (record.structural_primal_recoveries.size() >=
        static_cast<std::size_t>(INT32_MAX)) {
      throw std::runtime_error(
          "structural recovery payload exceeds 32-bit checkpoint range");
    }
    const std::int32_t payload_index = static_cast<std::int32_t>(
        record.structural_primal_recoveries.size());
    record.structural_primal_recoveries.push_back(
        globalize_structural_recovery(plan.structural_primal_recovery,
                                      old_col_red2org));
    record.primal_recovery_timeline.push_back(PrimalRecoveryCheckpoint{
        PrimalRecoveryKind::StructuralL1, payload_index, tape_position});
  }
  if (plan.has_antipodal_component_recovery) {
    if (record.antipodal_component_recoveries.size() >=
        static_cast<std::size_t>(INT32_MAX)) {
      throw std::runtime_error(
          "antipodal recovery payload exceeds 32-bit checkpoint range");
    }
    const std::int32_t payload_index = static_cast<std::int32_t>(
        record.antipodal_component_recoveries.size());
    record.antipodal_component_recoveries.push_back(
        globalize_antipodal_recovery(plan.antipodal_component_recovery,
                                     old_col_red2org));
    record.primal_recovery_timeline.push_back(PrimalRecoveryCheckpoint{
        PrimalRecoveryKind::AntipodalComponent, payload_index,
        tape_position});
    record.has_primal_only_antipodal_reduction = true;
  }
  if (plan.has_covering_cost_dominance_reduction) {
    record.has_primal_only_covering_cost_dominance_reduction = true;
  }
  if (plan.has_projected_auxiliary_reduction) {
    record.has_primal_only_projected_auxiliary_reduction = true;
  }
  append_globalized_postsolve_tape(record.tape, plan.tape,
                                   old_row_red2org, old_col_red2org);
  // Append to the canonical host tape during presolve.  Upload it lazily
  // in postsolve_gpu to avoid repeatedly growing and copying a device mirror.
  const auto after_tape = std::chrono::steady_clock::now();

  if (has_removed_rows) {
    record.row_red2org = std::move(next_row_red2org);
  }
  if (has_removed_cols) {
    record.col_red2org = std::move(next_col_red2org);
  }
  record.m1 = m_new;
  record.n1 = n_new;
  record.obj_constant_new = obj_constant_new;
  const auto after_rebuild = std::chrono::steady_clock::now();
  if (profile_record) {
    const std::chrono::duration<double> keep_copy_elapsed = after_keep_copy - record_start;
    const std::chrono::duration<double> old_mapping_copy_elapsed = after_old_mapping_copy - after_keep_copy;
    const std::chrono::duration<double> removed_mapping_elapsed = after_removed_mapping - after_old_mapping_copy;
    const std::chrono::duration<double> tape_elapsed = after_tape - after_removed_mapping;
    const std::chrono::duration<double> rebuild_elapsed = after_rebuild - after_tape;
    const std::chrono::duration<double> total_elapsed = after_rebuild - record_start;
    std::cerr << ">>> [GPU Presolve C++ record-profile]"
              << " dims=(" << old_lp.A.rows << "," << old_lp.A.cols << ")->("
              << m_new << "," << n_new << ")"
              << " removed_rows=" << (old_lp.A.rows - m_new)
              << " removed_cols=" << (old_lp.A.cols - n_new)
              << " tape_records=" << plan.tape.types.size()
              << " tape_indices=" << plan.tape.indices.size()
              << " tape_vals=" << plan.tape.vals.size()
              << " keep_copy=" << keep_copy_elapsed.count() << "s"
              << " old_mapping_copy=" << old_mapping_copy_elapsed.count() << "s"
              << " removed_mapping=" << removed_mapping_elapsed.count() << "s"
              << " tape=" << tape_elapsed.count() << "s"
              << " rebuild=" << rebuild_elapsed.count() << "s"
              << " total=" << total_elapsed.count() << "s\n";
  }
}

DeviceCsrMatrix transpose_csr_gpu(const DeviceCsrMatrix& A) {
  DeviceCsrMatrix AT;
  AT.rows = A.cols;
  AT.cols = A.rows;
  AT.nnz = A.nnz;
  cusparseHandle_t handle = nullptr;
  void* buffer = nullptr;
  try {
    throw_if_cuda_error(cudaMalloc(&AT.rowPtr, sizeof(std::int32_t) * static_cast<std::size_t>(AT.rows + 1)),
                        "cudaMalloc transpose rowPtr");
    if (A.nnz == 0) {
      throw_if_cuda_error(cudaMemset(AT.rowPtr, 0, sizeof(std::int32_t) * static_cast<std::size_t>(AT.rows + 1)),
                          "cudaMemset empty transpose rowPtr");
      return AT;
    }
    throw_if_cuda_error(cudaMalloc(&AT.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(A.nnz)),
                        "cudaMalloc transpose colVal");
    throw_if_cuda_error(cudaMalloc(&AT.nzVal, sizeof(double) * static_cast<std::size_t>(A.nnz)),
                        "cudaMalloc transpose nzVal");

    throw_if_cusparse_error(cusparseCreate(&handle), "cusparseCreate transpose");
    std::size_t buffer_size = 0;
    constexpr cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
    throw_if_cusparse_error(
        cusparseCsr2cscEx2_bufferSize(handle,
                                       A.rows,
                                       A.cols,
                                       A.nnz,
                                       A.nzVal,
                                       A.rowPtr,
                                       A.colVal,
                                       AT.nzVal,
                                       AT.rowPtr,
                                       AT.colVal,
                                       CUDA_R_64F,
                                       CUSPARSE_ACTION_NUMERIC,
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       alg,
                                       &buffer_size),
        "cusparseCsr2cscEx2_bufferSize transpose");
    if (buffer_size > 0) {
      throw_if_cuda_error(cudaMalloc(&buffer, buffer_size),
                          "cudaMalloc cusparse transpose buffer");
    }
    throw_if_cusparse_error(cusparseCsr2cscEx2(handle,
                                              A.rows,
                                              A.cols,
                                              A.nnz,
                                              A.nzVal,
                                              A.rowPtr,
                                              A.colVal,
                                              AT.nzVal,
                                              AT.rowPtr,
                                              AT.colVal,
                                              CUDA_R_64F,
                                              CUSPARSE_ACTION_NUMERIC,
                                              CUSPARSE_INDEX_BASE_ZERO,
                                              alg,
                                              buffer),
                           "cusparseCsr2cscEx2 transpose");
    cudaFree(buffer);
    buffer = nullptr;
    throw_if_cusparse_error(cusparseDestroy(handle), "cusparseDestroy transpose");
    handle = nullptr;
    return AT;
  } catch (...) {
    cudaFree(buffer);
    if (handle != nullptr) {
      cusparseDestroy(handle);
    }
    cudaFree(AT.rowPtr);
    cudaFree(AT.colVal);
    cudaFree(AT.nzVal);
    throw;
  }
}

DeviceCsrMatrix compact_csr_by_masks_gpu(const DeviceCsrMatrix& A,
                                         const std::int32_t* row_old_to_new,
                                         const std::int32_t* col_old_to_new,
                                         std::int32_t m_new,
                                         std::int32_t n_new) {
  DeviceCsrMatrix out;
  out.rows = m_new;
  out.cols = n_new;
  throw_if_cuda_error(stream_ordered_malloc(&out.rowPtr, sizeof(std::int32_t) * static_cast<std::size_t>(m_new + 1)),
                      "cudaMalloc compact rowPtr");
  if (m_new == 0 || n_new == 0) {
    throw_if_cuda_error(cudaMemset(out.rowPtr, 0, sizeof(std::int32_t) * static_cast<std::size_t>(m_new + 1)),
                        "cudaMemset empty compact rowPtr");
    return out;
  }
  std::int32_t* row_counts = nullptr;
  std::int32_t* long_rows = nullptr;
  std::int32_t* long_row_count = nullptr;
  // Reserve full 256-thread blocks for dense rows to avoid excessive thread
  // allocation on matrices with many short rows.
  const bool use_block_per_row =
      static_cast<std::int64_t>(A.nnz) >
      32LL * static_cast<std::int64_t>(A.rows);
  const bool use_long_row_kernels = !use_block_per_row && A.nnz >= 10000000;
  const std::int32_t long_row_threshold =
      use_long_row_kernels ? COMPACT_LONG_ROW_THRESHOLD : INT_MAX;
  throw_if_cuda_error(stream_ordered_malloc(&row_counts, sizeof(std::int32_t) * static_cast<std::size_t>(m_new)),
                      "cudaMalloc compact row_counts");
  throw_if_cuda_error(cudaMemset(row_counts, 0, sizeof(std::int32_t) * static_cast<std::size_t>(m_new)),
                      "cudaMemset compact row_counts");
  if (use_long_row_kernels) {
    throw_if_cuda_error(stream_ordered_malloc(&long_rows, sizeof(std::int32_t) * static_cast<std::size_t>(A.rows)),
                        "cudaMalloc compact long_rows");
    throw_if_cuda_error(stream_ordered_malloc(&long_row_count, sizeof(std::int32_t)),
                        "cudaMalloc compact long_row_count");
    throw_if_cuda_error(cudaMemset(long_row_count, 0, sizeof(std::int32_t)),
                        "cudaMemset compact long_row_count");
  }
  constexpr int threads = 256;
  const int old_row_blocks = (A.rows + threads - 1) / threads;
  if (use_block_per_row) {
    _kernel_count_compacted_rows_block<<<A.rows, threads>>>(
        row_counts, row_old_to_new, col_old_to_new, A.rowPtr, A.colVal, A.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_count_compacted_rows_block");
  } else {
    _kernel_count_compacted_rows<<<old_row_blocks, threads>>>(
        row_counts, row_old_to_new, col_old_to_new, A.rowPtr, A.colVal,
        long_rows, long_row_count, long_row_threshold, A.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_count_compacted_rows");
  }
  std::int32_t long_row_count_host = 0;
  if (use_long_row_kernels) {
    throw_if_cuda_error(cudaMemcpy(&long_row_count_host, long_row_count,
                                   sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy compact long_row_count");
  }
  if (long_row_count_host > 0) {
    _kernel_count_compacted_long_rows<<<long_row_count_host, threads>>>(
        row_counts, row_old_to_new, col_old_to_new, A.rowPtr, A.colVal,
        long_rows, long_row_count_host);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_count_compacted_long_rows");
  }
  inclusive_scan_i32(row_counts, m_new, "cub inclusive scan compact");
  std::int32_t nnz_new = 0;
  throw_if_cuda_error(cudaMemcpy(&nnz_new, row_counts + m_new - 1, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy compact nnz");
  out.nnz = nnz_new;
  const int new_row_blocks = (m_new + threads - 1) / threads;
  _kernel_row_ptr_from_prefix<<<new_row_blocks, threads>>>(out.rowPtr, row_counts, m_new);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_row_ptr_from_prefix compact");
  if (nnz_new > 0) {
    throw_if_cuda_error(stream_ordered_malloc(&out.colVal, sizeof(std::int32_t) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc compact colVal");
    throw_if_cuda_error(stream_ordered_malloc(&out.nzVal, sizeof(double) * static_cast<std::size_t>(nnz_new)),
                        "cudaMalloc compact nzVal");
    if (use_block_per_row) {
      _kernel_copy_compacted_rows_block<<<A.rows, threads>>>(
          out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
          A.rowPtr, A.colVal, A.nzVal, A.rows);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_compacted_rows_block");
    } else {
      _kernel_copy_compacted_rows<<<old_row_blocks, threads>>>(
          out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
          A.rowPtr, A.colVal, A.nzVal, long_row_threshold, A.rows);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_compacted_rows");
      if (long_row_count_host > 0) {
        _kernel_copy_compacted_long_rows<<<long_row_count_host, threads>>>(
            out.colVal, out.nzVal, out.rowPtr, row_old_to_new, col_old_to_new,
            A.rowPtr, A.colVal, A.nzVal, long_rows, long_row_count_host);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_copy_compacted_long_rows");
      }
    }
  }
  stream_ordered_free(row_counts);
  stream_ordered_free(long_rows);
  stream_ordered_free(long_row_count);
  return out;
}

struct WorkingLp {
  LPInfoGpu lp;
  bool owns = false;

  ~WorkingLp() { free_owned(); }

  void free_owned() {
    if (!owns) {
      return;
    }
    cudaFree(lp.A.rowPtr);
    cudaFree(lp.A.colVal);
    cudaFree(lp.A.nzVal);
    cudaFree(lp.AT.rowPtr);
    cudaFree(lp.AT.colVal);
    cudaFree(lp.AT.nzVal);
    cudaFree(lp.c);
    cudaFree(lp.AL);
    cudaFree(lp.AU);
    cudaFree(lp.l);
    cudaFree(lp.u);
    owns = false;
  }

  void free_owned_vectors() {
    if (!owns) {
      return;
    }
    cudaFree(lp.c);
    cudaFree(lp.AL);
    cudaFree(lp.AU);
    cudaFree(lp.l);
    cudaFree(lp.u);
    lp.c = nullptr;
    lp.AL = nullptr;
    lp.AU = nullptr;
    lp.l = nullptr;
    lp.u = nullptr;
  }
};

void apply_plan_to_working_lp(WorkingLp& current,
                              const PresolvePlanGpu& plan,
                              PresolveRecordGpu* record) {
  const bool profile_apply = env_enabled("GPUPRESOLVER_PRESOLVE_APPLY_PROFILE");
  const auto apply_start = std::chrono::steady_clock::now();
  auto sync_for_profile = [&](const char* context) {
    if (profile_apply) {
      throw_if_cuda_error(cudaDeviceSynchronize(), context);
    }
  };
  std::int32_t m_new = 0;
  std::int32_t n_new = 0;
  std::int32_t* row_old_to_new =
      build_old_to_new(plan.keep_row_mask, current.lp.A.rows, &m_new);
  std::int32_t* col_old_to_new =
      build_old_to_new(plan.keep_col_mask, current.lp.A.cols, &n_new);
  sync_for_profile("profile apply build_old_to_new synchronize");
  const auto after_old_to_new = std::chrono::steady_clock::now();
  const bool structural_unchanged =
      current.owns && !plan.has_new_A && m_new == current.lp.A.rows && n_new == current.lp.A.cols;
  const double* c_source = plan.new_c != nullptr ? plan.new_c : current.lp.c;
  const double* l_source = plan.new_l != nullptr ? plan.new_l : current.lp.l;
  const double* u_source = plan.new_u != nullptr ? plan.new_u : current.lp.u;
  const double* AL_source = plan.new_AL != nullptr ? plan.new_AL : current.lp.AL;
  const double* AU_source = plan.new_AU != nullptr ? plan.new_AU : current.lp.AU;
  if (structural_unchanged) {
    double* c_new = clone_device_vector(c_source, current.lp.A.cols, "cudaMalloc/copy no-struct c");
    double* l_new = clone_device_vector(l_source, current.lp.A.cols, "cudaMalloc/copy no-struct l");
    double* u_new = clone_device_vector(u_source, current.lp.A.cols, "cudaMalloc/copy no-struct u");
    double* AL_new = clone_device_vector(AL_source, current.lp.A.rows, "cudaMalloc/copy no-struct AL");
    double* AU_new = clone_device_vector(AU_source, current.lp.A.rows, "cudaMalloc/copy no-struct AU");
    sync_for_profile("profile apply clone no-struct vectors synchronize");
    const auto after_vectors = std::chrono::steady_clock::now();
    if (record != nullptr) {
      update_record_from_plan(*record, plan, current.lp, m_new, n_new,
                              current.lp.obj_constant + plan.obj_constant_delta,
                              col_old_to_new);
    }
    sync_for_profile("profile apply no-struct record synchronize");
    const auto after_record = std::chrono::steady_clock::now();
    cudaFree(row_old_to_new);
    cudaFree(col_old_to_new);
    current.free_owned_vectors();
    sync_for_profile("profile apply no-struct free synchronize");
    const auto after_free = std::chrono::steady_clock::now();
    current.lp.c = c_new;
    current.lp.l = l_new;
    current.lp.u = u_new;
    current.lp.AL = AL_new;
    current.lp.AU = AU_new;
    current.lp.obj_constant += plan.obj_constant_delta;
    current.owns = true;
    if (profile_apply) {
      const std::chrono::duration<double> old_to_new_elapsed = after_old_to_new - apply_start;
      const std::chrono::duration<double> vectors_elapsed = after_vectors - after_old_to_new;
      const std::chrono::duration<double> record_elapsed = after_record - after_vectors;
      const std::chrono::duration<double> free_elapsed = after_free - after_record;
      const std::chrono::duration<double> total_elapsed = after_free - apply_start;
      std::cerr << ">>> [GPU Presolve C++ apply-profile] structural_unchanged=true"
                << " dims=(" << current.lp.A.rows << "," << current.lp.A.cols << ")->("
                << m_new << "," << n_new << ")"
                << " old_to_new=" << old_to_new_elapsed.count() << "s"
                << " vectors=" << vectors_elapsed.count() << "s"
                << " record=" << record_elapsed.count() << "s"
                << " free=" << free_elapsed.count() << "s"
                << " total=" << total_elapsed.count() << "s\n";
    }
    return;
  }
  const DeviceCsrMatrix& source_A = plan.has_new_A ? plan.new_A : current.lp.A;
  DeviceCsrMatrix A_new = compact_csr_by_masks_gpu(
      source_A, row_old_to_new, col_old_to_new, m_new, n_new);
  sync_for_profile("profile apply compact A synchronize");
  const auto after_compact_a = std::chrono::steady_clock::now();
  DeviceCsrMatrix AT_new = transpose_csr_gpu(A_new);
  sync_for_profile("profile apply transpose synchronize");
  const auto after_transpose = std::chrono::steady_clock::now();
  double* c_new = compact_vector(c_source, col_old_to_new, current.lp.A.cols, n_new);
  double* l_new = compact_vector(l_source, col_old_to_new, current.lp.A.cols, n_new);
  double* u_new = compact_vector(u_source, col_old_to_new, current.lp.A.cols, n_new);
  double* AL_new = compact_vector(AL_source, row_old_to_new, current.lp.A.rows, m_new);
  double* AU_new = compact_vector(AU_source, row_old_to_new, current.lp.A.rows, m_new);
  sync_for_profile("profile apply compact vectors synchronize");
  const auto after_vectors = std::chrono::steady_clock::now();
  if (record != nullptr) {
    update_record_from_plan(*record, plan, current.lp, m_new, n_new,
                            current.lp.obj_constant + plan.obj_constant_delta,
                            col_old_to_new);
  }
  sync_for_profile("profile apply record synchronize");
  const auto after_record = std::chrono::steady_clock::now();
  cudaFree(row_old_to_new);
  cudaFree(col_old_to_new);

  current.free_owned();
  sync_for_profile("profile apply free synchronize");
  const auto after_free = std::chrono::steady_clock::now();
  current.lp.A = A_new;
  current.lp.AT = AT_new;
  current.lp.c = c_new;
  current.lp.l = l_new;
  current.lp.u = u_new;
  current.lp.AL = AL_new;
  current.lp.AU = AU_new;
  current.lp.obj_constant += plan.obj_constant_delta;
  current.owns = true;
  if (profile_apply) {
    const std::chrono::duration<double> old_to_new_elapsed = after_old_to_new - apply_start;
    const std::chrono::duration<double> compact_a_elapsed = after_compact_a - after_old_to_new;
    const std::chrono::duration<double> transpose_elapsed = after_transpose - after_compact_a;
    const std::chrono::duration<double> vectors_elapsed = after_vectors - after_transpose;
    const std::chrono::duration<double> record_elapsed = after_record - after_vectors;
    const std::chrono::duration<double> free_elapsed = after_free - after_record;
    const std::chrono::duration<double> total_elapsed = after_free - apply_start;
    std::cerr << ">>> [GPU Presolve C++ apply-profile] structural_unchanged=false"
              << " dims=(" << source_A.rows << "," << source_A.cols << ")->("
              << m_new << "," << n_new << ")"
              << " nnz=" << source_A.nnz << "->" << A_new.nnz
              << " old_to_new=" << old_to_new_elapsed.count() << "s"
              << " compact_A=" << compact_a_elapsed.count() << "s"
              << " transpose=" << transpose_elapsed.count() << "s"
              << " vectors=" << vectors_elapsed.count() << "s"
              << " record=" << record_elapsed.count() << "s"
              << " free=" << free_elapsed.count() << "s"
              << " total=" << total_elapsed.count() << "s\n";
  }
}

}  // namespace

bool _has_good_nnz_progress(std::int32_t nnz_before,
                            std::int32_t nnz_after,
                            double ratio) {
  if (nnz_before <= 0) {
    return false;
  }
  return static_cast<double>(nnz_after) < ratio * static_cast<double>(nnz_before);
}

namespace {

GpuPresolveSummary run_gpu_presolve_impl(const LPInfoGpu& lp,
                                         const PresolveParams& input_params,
                                         bool keep_reduced_lp) {
  PresolveParams effective_params = input_params;
  if (!effective_params.enable_structure_specific_rules) {
    effective_params.enable_covering_cost_dominance = false;
    effective_params.enable_linf_components = false;
    effective_params.enable_antipodal_components = false;
    effective_params.enable_bounded_two_row_projection = false;
    effective_params.enable_orphan_mccormick_projection = false;
    effective_params.enable_structural_l1_substitution = false;
  }
  const PresolveParams& params = effective_params;

  struct RuleWorkspaceReleaseGuard {
    ~RuleWorkspaceReleaseGuard() {
      release_implied_variable_bounds_workspace();
      release_column_singletons_workspace();
    }
  } rule_workspace_release_guard;

  const bool profile_record =
      env_enabled("GPUPRESOLVER_PRESOLVE_RECORD_PROFILE");
  const auto initial_record_start = std::chrono::steady_clock::now();
  GpuPresolveSummary summary;
  summary.original_rows = lp.A.rows;
  summary.original_cols = lp.A.cols;
  summary.record = make_initial_record(lp);
  if (profile_record) {
    const std::chrono::duration<double> initial_record_elapsed =
        std::chrono::steady_clock::now() - initial_record_start;
    std::cerr << ">>> [GPU Presolve C++ record-profile]"
              << " initial_record=" << initial_record_elapsed.count()
              << "s rows=" << lp.A.rows << " cols=" << lp.A.cols << "\n";
  }

  WorkingLp current;
  current.lp = lp;

  bool terminal = false;
  const bool profile_rules = env_enabled("GPUPRESOLVER_PRESOLVE_RULE_PROFILE");
  const auto presolve_start = std::chrono::steady_clock::now();
  const bool has_time_limit = params.max_time > 0.0 && std::isfinite(params.max_time);
  bool column_singletons_dual_clean = false;
  bool column_singletons_eq_clean = false;
  auto invalidate_column_singleton_clean = [&]() {
    column_singletons_dual_clean = false;
    column_singletons_eq_clean = false;
  };
  auto time_exceeded = [&]() -> bool {
    if (!has_time_limit) {
      return false;
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - presolve_start;
    return elapsed.count() >= params.max_time;
  };

  auto run_row_phase = [&](bool empty_rows,
                           bool singleton_rows,
                           bool infeasible_redundant_rows,
                           bool implied_variable_bounds,
                           bool duplicate_rows) -> bool {
    if (terminal || time_exceeded()) {
      return false;
    }
    const auto phase_start = std::chrono::steady_clock::now();
    PresolvePlanGpu plan;
    PresolveStatsGpu stats;
    const bool run_empty_rows = empty_rows && params.enable_empty_rows;
    const bool run_singleton_rows = singleton_rows && params.enable_singleton_rows;
    const bool run_infeasible_redundant_rows = infeasible_redundant_rows && params.enable_infeasible_redundant_rows;
    const bool run_implied_variable_bounds =
        implied_variable_bounds && params.enable_implied_variable_bounds;
    const bool run_duplicate_rows = duplicate_rows && params.enable_duplicate_rows;
    PlanRequirements plan_requirements;
    plan_requirements.objective = run_implied_variable_bounds;
    plan_requirements.col_bounds =
        run_singleton_rows || run_infeasible_redundant_rows || run_implied_variable_bounds;
    plan_requirements.row_bounds = run_empty_rows || run_singleton_rows ||
                                   run_infeasible_redundant_rows || run_implied_variable_bounds ||
                                   run_duplicate_rows;
    allocate_plan(plan, current.lp, plan_requirements);
    StatsRequirements stats_requirements;
    stats_requirements.row_nnz =
        (empty_rows && params.enable_empty_rows) ||
        (infeasible_redundant_rows && params.enable_infeasible_redundant_rows) ||
        (implied_variable_bounds && params.enable_implied_variable_bounds);
    stats_requirements.singleton_rows = singleton_rows && params.enable_singleton_rows;
    const bool needs_row_stats =
        stats_requirements.row_nnz || stats_requirements.singleton_rows;
    allocate_stats(stats, current.lp.A.rows, current.lp.A.cols, stats_requirements);
    if (needs_row_stats) {
      recompute_row_stats_from_csr(stats, current.lp);
    }
    auto after_stats = std::chrono::steady_clock::now();
    const std::int32_t m_before = current.lp.A.rows;
    const std::int32_t n_before = current.lp.A.cols;
    const std::int32_t nnz_before = current.lp.A.nnz;
    std::string rules;
    if (empty_rows && params.enable_empty_rows) append_rule_name(rules, "empty_rows");
    if (singleton_rows && params.enable_singleton_rows) append_rule_name(rules, "singleton_rows");
    if (infeasible_redundant_rows && params.enable_infeasible_redundant_rows) append_rule_name(rules, "infeasible_redundant_rows");
    if (implied_variable_bounds && params.enable_implied_variable_bounds) append_rule_name(rules, "implied_variable_bounds");
    if (duplicate_rows && params.enable_duplicate_rows) append_rule_name(rules, "duplicate_rows");

    if (!time_exceeded() && empty_rows && params.enable_empty_rows) apply_rule_empty_rows(plan, current.lp, stats, params);
    if (!time_exceeded() && singleton_rows && params.enable_singleton_rows) apply_rule_singleton_rows(plan, current.lp, stats, params);
    if (!time_exceeded() && infeasible_redundant_rows && params.enable_infeasible_redundant_rows) apply_rule_infeasible_redundant_rows(plan, current.lp, stats, params);
    if (!time_exceeded() && implied_variable_bounds && params.enable_implied_variable_bounds) apply_rule_implied_variable_bounds(plan, current.lp, stats, params);
    if (!time_exceeded() && duplicate_rows && params.enable_duplicate_rows) apply_rule_duplicate_rows(plan, current.lp, stats, params);
    auto after_plan = std::chrono::steady_clock::now();

    summary.has_infeasible = plan.has_infeasible;
    summary.has_unbounded = plan.has_unbounded;
    terminal = plan.has_infeasible || plan.has_unbounded;
    const bool changed = !terminal && plan.has_change;
    if (changed) {
      apply_plan_to_working_lp(current, plan, &summary.record);
      invalidate_column_singleton_clean();
    }
    auto after_apply = std::chrono::steady_clock::now();
    if (profile_rules) {
      throw_if_cuda_error(cudaDeviceSynchronize(), "profile row phase synchronize");
      const std::chrono::duration<double> phase_elapsed =
          std::chrono::steady_clock::now() - phase_start;
      const std::chrono::duration<double> stats_elapsed = after_stats - phase_start;
      const std::chrono::duration<double> plan_elapsed = after_plan - after_stats;
      const std::chrono::duration<double> apply_elapsed = after_apply - after_plan;
      std::cerr << ">>> [GPU Presolve C++ profile] phase=row rules=[" << rules << "] changed="
                << (changed ? "true" : "false") << " dims=(" << m_before << "," << n_before
                << ")->(" << current.lp.A.rows << "," << current.lp.A.cols << ") nnz="
                << nnz_before << "->" << current.lp.A.nnz
                << " time=" << phase_elapsed.count() << "s stats=" << stats_elapsed.count()
                << "s plan=" << plan_elapsed.count() << "s apply=" << apply_elapsed.count()
                << "s\n";
    }
    free_stats(stats);
    free_plan(plan);
    return changed;
  };

  auto run_col_phase = [&](bool infeasible_fixed_variables,
                           bool empty_cols,
                           bool column_singletons_dual,
                           bool column_singletons_eq,
                           bool doubleton_equations,
                           bool structural_l1,
                           bool dual_fix,
                           bool duplicate_columns,
                           bool redundant_bounds) -> bool {
    if (terminal || time_exceeded()) {
      return false;
    }
    const auto phase_start = std::chrono::steady_clock::now();
    const bool structural_only =
        !infeasible_fixed_variables && !empty_cols && !column_singletons_dual && !column_singletons_eq &&
        !doubleton_equations && structural_l1 && !dual_fix && !duplicate_columns && !redundant_bounds &&
        params.enable_structural_l1_substitution;
    if (structural_only && !structural_l1_prefix_screen_passes(current.lp)) {
      if (profile_rules) {
        throw_if_cuda_error(cudaDeviceSynchronize(), "profile structural prefix screen synchronize");
        const std::chrono::duration<double> phase_elapsed =
            std::chrono::steady_clock::now() - phase_start;
        std::cerr << ">>> [GPU Presolve C++ profile] phase=col rules=[structural_l1_substitution] changed=false"
                  << " dims=(" << current.lp.A.rows << "," << current.lp.A.cols << ")->("
                  << current.lp.A.rows << "," << current.lp.A.cols << ") nnz="
                  << current.lp.A.nnz << "->" << current.lp.A.nnz
                  << " time=" << phase_elapsed.count() << "s\n";
      }
      return false;
    }
    PresolvePlanGpu plan;
    PresolveStatsGpu stats;
    const bool run_infeasible_fixed_variables = infeasible_fixed_variables && params.enable_infeasible_fixed_variables;
    const bool run_empty_cols = empty_cols && params.enable_empty_cols;
    const bool run_column_singletons_dual =
        column_singletons_dual && params.enable_column_singletons_dual_infer;
    const bool run_column_singletons_eq =
        column_singletons_eq && params.enable_column_singletons_eq;
    const bool run_doubleton_equations = doubleton_equations && params.enable_doubleton_equations;
    const bool run_structural_l1 =
        structural_l1 && params.enable_structural_l1_substitution;
    const bool run_dual_fix = dual_fix && params.enable_dual_fix;
    const bool run_duplicate_columns = duplicate_columns && params.enable_duplicate_columns;
    const bool run_redundant_bounds =
        redundant_bounds && (params.enable_redundant_bounds ||
                             params.enable_zero_cost_redundant_box_bounds);
    PlanRequirements plan_requirements;
    plan_requirements.objective = run_infeasible_fixed_variables || run_empty_cols ||
                                  run_column_singletons_dual || run_column_singletons_eq ||
                                  run_doubleton_equations || run_structural_l1 || run_dual_fix ||
                                  run_duplicate_columns;
    plan_requirements.col_bounds = plan_requirements.objective || run_redundant_bounds;
    plan_requirements.row_bounds = run_infeasible_fixed_variables || run_column_singletons_dual ||
                                   run_column_singletons_eq || run_doubleton_equations ||
                                   run_structural_l1 || run_dual_fix ||
                                   run_duplicate_columns || run_redundant_bounds;
    allocate_plan(plan, current.lp, plan_requirements);
    StatsRequirements stats_requirements;
    stats_requirements.empty_cols = empty_cols && params.enable_empty_cols;
    stats_requirements.column_singletons =
        (column_singletons_dual && params.enable_column_singletons_dual_infer) ||
        (column_singletons_eq && params.enable_column_singletons_eq);
    const bool needs_col_stats =
        stats_requirements.empty_cols || stats_requirements.column_singletons;
    allocate_stats(stats, current.lp.A.rows, current.lp.A.cols, stats_requirements);
    if (needs_col_stats) {
      recompute_col_stats_from_csr(stats, current.lp);
    }
    auto after_stats = std::chrono::steady_clock::now();
    const std::int32_t m_before = current.lp.A.rows;
    const std::int32_t n_before = current.lp.A.cols;
    const std::int32_t nnz_before = current.lp.A.nnz;
    std::string rules;
    if (infeasible_fixed_variables && params.enable_infeasible_fixed_variables) append_rule_name(rules, "infeasible_fixed_variables");
    if (empty_cols && params.enable_empty_cols) append_rule_name(rules, "empty_cols");
    if (column_singletons_dual && params.enable_column_singletons_dual_infer) append_rule_name(rules, "column_singletons_dual_infer");
    if (column_singletons_eq && params.enable_column_singletons_eq) append_rule_name(rules, "column_singletons_eq");
    if (doubleton_equations && params.enable_doubleton_equations) append_rule_name(rules, "doubleton_equations");
    if (structural_l1 && params.enable_structural_l1_substitution) append_rule_name(rules, "structural_l1_substitution");
    if (dual_fix && params.enable_dual_fix) append_rule_name(rules, "dual_fix");
    if (duplicate_columns && params.enable_duplicate_columns) append_rule_name(rules, "duplicate_columns");
    if (run_redundant_bounds) {
      append_rule_name(rules,
                       params.enable_redundant_bounds
                           ? "redundant_bounds"
                           : "zero_cost_redundant_box_bounds");
    }

    if (!time_exceeded() && infeasible_fixed_variables && params.enable_infeasible_fixed_variables) apply_rule_infeasible_fixed_variables(plan, current.lp, params);
    if (!time_exceeded() && empty_cols && params.enable_empty_cols) apply_rule_empty_cols(plan, current.lp, stats, params);
    if (!time_exceeded() && column_singletons_dual && params.enable_column_singletons_dual_infer) {
      apply_rule_column_singletons_dual_infer(plan, current.lp, stats, params);
    }
    if (!time_exceeded() && column_singletons_eq && params.enable_column_singletons_eq) {
      apply_rule_column_singletons_eq(plan, current.lp, stats, params);
    }
    if (!time_exceeded() && doubleton_equations && params.enable_doubleton_equations) apply_rule_doubleton_equations(plan, current.lp, stats, params);
    if (!time_exceeded() && structural_l1 && params.enable_structural_l1_substitution) {
      apply_rule_structural_l1_substitution(plan, current.lp, stats, params);
    }
    if (!time_exceeded() && dual_fix && params.enable_dual_fix) apply_rule_dual_fix(plan, current.lp, stats, params);
    if (!time_exceeded() && duplicate_columns && params.enable_duplicate_columns) apply_rule_duplicate_columns(plan, current.lp, stats, params);
    if (!time_exceeded() && run_redundant_bounds) {
      apply_rule_redundant_bounds(plan, current.lp, stats, params);
    }
    auto after_plan = std::chrono::steady_clock::now();

    summary.has_infeasible = plan.has_infeasible;
    summary.has_unbounded = plan.has_unbounded;
    terminal = plan.has_infeasible || plan.has_unbounded;
    const bool changed = !terminal && plan.has_change;
    if (changed) {
      apply_plan_to_working_lp(current, plan, &summary.record);
      invalidate_column_singleton_clean();
    } else {
      const bool singleton_dual_only =
          !infeasible_fixed_variables && !empty_cols && column_singletons_dual && !column_singletons_eq &&
          !doubleton_equations && !structural_l1 && !dual_fix && !duplicate_columns && !redundant_bounds &&
          params.enable_column_singletons_dual_infer;
      const bool singleton_eq_only =
          !infeasible_fixed_variables && !empty_cols && !column_singletons_dual && column_singletons_eq &&
          !doubleton_equations && !structural_l1 && !dual_fix && !duplicate_columns && !redundant_bounds &&
          params.enable_column_singletons_eq;
      if (singleton_dual_only) {
        column_singletons_dual_clean = true;
      }
      if (singleton_eq_only) {
        column_singletons_eq_clean = true;
      }
    }
    auto after_apply = std::chrono::steady_clock::now();
    if (profile_rules) {
      throw_if_cuda_error(cudaDeviceSynchronize(), "profile col phase synchronize");
      const std::chrono::duration<double> phase_elapsed =
          std::chrono::steady_clock::now() - phase_start;
      const std::chrono::duration<double> stats_elapsed = after_stats - phase_start;
      const std::chrono::duration<double> plan_elapsed = after_plan - after_stats;
      const std::chrono::duration<double> apply_elapsed = after_apply - after_plan;
      std::cerr << ">>> [GPU Presolve C++ profile] phase=col rules=[" << rules << "] changed="
                << (changed ? "true" : "false") << " dims=(" << m_before << "," << n_before
                << ")->(" << current.lp.A.rows << "," << current.lp.A.cols << ") nnz="
                << nnz_before << "->" << current.lp.A.nnz
                << " time=" << phase_elapsed.count() << "s stats=" << stats_elapsed.count()
                << "s plan=" << plan_elapsed.count() << "s apply=" << apply_elapsed.count()
                << "s\n";
    }
    free_stats(stats);
    free_plan(plan);
    return changed;
  };

  auto run_singleton_rows_to_exhaustion = [&]() -> bool {
    bool changed_any = false;
    while (!terminal && !time_exceeded()) {
      const bool changed = run_row_phase(false, true, false, false, false);
      changed_any = changed_any || changed;
      if (!changed) {
        break;
      }
    }
    return changed_any;
  };

  auto run_trivial_cleanup_recirculation = [&]() -> bool {
    bool changed_any = false;
    while (!terminal && !time_exceeded()) {
      bool changed_pass = false;
      changed_pass = run_col_phase(true, true, false, false, false, false, false, false, false) || changed_pass;
      changed_pass = run_row_phase(true, true, false, false, false) || changed_pass;
      changed_pass = run_col_phase(false, true, false, false, false, false, false, false, false) || changed_pass;
      changed_any = changed_pass || changed_any;
      if (!changed_pass) {
        break;
      }
    }
    return changed_any;
  };

  auto run_cleanup = [&]() -> bool {
    bool changed_any = false;
    changed_any = run_col_phase(true, true, false, false, false, false, false, false, false) || changed_any;
    changed_any = run_col_phase(false, false, false, false, false, false, true, false, false) || changed_any;
    changed_any = run_singleton_rows_to_exhaustion() || changed_any;
    changed_any = run_row_phase(true, false, false, false, false) || changed_any;
    changed_any = run_col_phase(false, true, false, false, false, false, false, false, false) || changed_any;
    return changed_any;
  };

  auto run_repeated_doubleton = [&]() -> bool {
    bool changed_any = false;
    while (!terminal && !time_exceeded()) {
      const bool changed = run_col_phase(false, false, false, false, true, false, false, false, false);
      changed_any = changed_any || changed;
      if (!changed || params.doubleton_equations_single_batch_per_iter) {
        break;
      }
    }
    return changed_any;
  };

  // Batch the column-singleton -> dual-fix -> singleton-row cascade in one
  // plan. Live statistics honor the masks, so A and AT need only one
  // compaction while reduction and postsolve tape order are preserved.
  auto run_batched_singleton_cleanup_closure = [&]() -> bool {
    constexpr std::int32_t kMinNnzForBatchedClosure = 10000000;
    if (terminal || time_exceeded() ||
        current.lp.A.nnz < kMinNnzForBatchedClosure ||
        !params.enable_column_singletons_eq || !params.enable_dual_fix ||
        !params.enable_singleton_rows) {
      return false;
    }

    const auto phase_start = std::chrono::steady_clock::now();
    const std::int32_t m_before = current.lp.A.rows;
    const std::int32_t n_before = current.lp.A.cols;
    const std::int32_t nnz_before = current.lp.A.nnz;
    PresolvePlanGpu plan;
    PresolveStatsGpu stats;
    PlanRequirements plan_requirements;
    allocate_plan(plan, current.lp, plan_requirements);
    StatsRequirements stats_requirements;
    stats_requirements.row_nnz = true;
    stats_requirements.col_nnz = true;
    stats_requirements.singleton_rows = true;
    stats_requirements.column_singletons = true;
    allocate_stats(stats, current.lp.A.rows, current.lp.A.cols, stats_requirements);

    recompute_row_stats_from_csr(stats, current.lp);
    recompute_col_stats_from_csr(stats, current.lp);
    std::uint8_t* keep_row_before = nullptr;
    std::uint8_t* keep_col_before = nullptr;
    std::uint8_t* dual_candidate_mask = nullptr;
    std::uint8_t* dual_keep_row_seen = nullptr;
    double* dual_c_seen = nullptr;
    double* dual_l_seen = nullptr;
    double* dual_u_seen = nullptr;
    double* dual_AL_seen = nullptr;
    double* dual_AU_seen = nullptr;
    throw_if_cuda_error(cudaMalloc(&keep_row_before,
                                   static_cast<std::size_t>(current.lp.A.rows)),
                        "cudaMalloc batched closure previous row mask");
    throw_if_cuda_error(cudaMalloc(&keep_col_before,
                                   static_cast<std::size_t>(current.lp.A.cols)),
                        "cudaMalloc batched closure previous col mask");
    throw_if_cuda_error(cudaMalloc(&dual_candidate_mask,
                                   static_cast<std::size_t>(current.lp.A.cols)),
                        "cudaMalloc batched closure dual candidates");
    throw_if_cuda_error(cudaMalloc(&dual_keep_row_seen,
                                   static_cast<std::size_t>(current.lp.A.rows)),
                        "cudaMalloc batched closure dual row mask snapshot");
    throw_if_cuda_error(cudaMalloc(&dual_c_seen,
                                   sizeof(double) *
                                       static_cast<std::size_t>(current.lp.A.cols)),
                        "cudaMalloc batched closure dual c snapshot");
    throw_if_cuda_error(cudaMalloc(&dual_l_seen,
                                   sizeof(double) *
                                       static_cast<std::size_t>(current.lp.A.cols)),
                        "cudaMalloc batched closure dual l snapshot");
    throw_if_cuda_error(cudaMalloc(&dual_u_seen,
                                   sizeof(double) *
                                       static_cast<std::size_t>(current.lp.A.cols)),
                        "cudaMalloc batched closure dual u snapshot");
    throw_if_cuda_error(cudaMalloc(&dual_AL_seen,
                                   sizeof(double) *
                                       static_cast<std::size_t>(current.lp.A.rows)),
                        "cudaMalloc batched closure dual AL snapshot");
    throw_if_cuda_error(cudaMalloc(&dual_AU_seen,
                                   sizeof(double) *
                                       static_cast<std::size_t>(current.lp.A.rows)),
                        "cudaMalloc batched closure dual AU snapshot");
    throw_if_cuda_error(cudaMemcpy(keep_row_before,
                                   plan.keep_row_mask,
                                   static_cast<std::size_t>(current.lp.A.rows),
                                   cudaMemcpyDeviceToDevice),
                        "cudaMemcpy batched closure previous row mask");
    throw_if_cuda_error(cudaMemcpy(keep_col_before,
                                   plan.keep_col_mask,
                                   static_cast<std::size_t>(current.lp.A.cols),
                                   cudaMemcpyDeviceToDevice),
                        "cudaMemcpy batched closure previous col mask");
    auto snapshot_dual_state = [&]() {
      throw_if_cuda_error(cudaMemcpy(dual_keep_row_seen,
                                     plan.keep_row_mask,
                                     static_cast<std::size_t>(current.lp.A.rows),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy dual row mask snapshot");
      throw_if_cuda_error(cudaMemcpy(dual_c_seen,
                                     plan.new_c,
                                     sizeof(double) *
                                         static_cast<std::size_t>(current.lp.A.cols),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy dual c snapshot");
      throw_if_cuda_error(cudaMemcpy(dual_l_seen,
                                     plan.new_l,
                                     sizeof(double) *
                                         static_cast<std::size_t>(current.lp.A.cols),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy dual l snapshot");
      throw_if_cuda_error(cudaMemcpy(dual_u_seen,
                                     plan.new_u,
                                     sizeof(double) *
                                         static_cast<std::size_t>(current.lp.A.cols),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy dual u snapshot");
      throw_if_cuda_error(cudaMemcpy(dual_AL_seen,
                                     plan.new_AL,
                                     sizeof(double) *
                                         static_cast<std::size_t>(current.lp.A.rows),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy dual AL snapshot");
      throw_if_cuda_error(cudaMemcpy(dual_AU_seen,
                                     plan.new_AU,
                                     sizeof(double) *
                                         static_cast<std::size_t>(current.lp.A.rows),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy dual AU snapshot");
    };
    snapshot_dual_state();

    auto mark_incremental_dual_candidates = [&]() {
      constexpr int threads = 256;
      const int row_blocks = (current.lp.A.rows + threads - 1) / threads;
      const int col_blocks = (current.lp.A.cols + threads - 1) / threads;
      _kernel_mark_changed_dual_cols<<<col_blocks, threads>>>(
          dual_candidate_mask,
          plan.keep_col_mask,
          plan.new_c,
          plan.new_l,
          plan.new_u,
          dual_c_seen,
          dual_l_seen,
          dual_u_seen,
          current.lp.A.cols);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_changed_dual_cols");
      _kernel_mark_dual_cols_from_changed_rows<<<row_blocks, threads>>>(
          dual_candidate_mask,
          plan.keep_row_mask,
          plan.keep_col_mask,
          plan.new_AL,
          plan.new_AU,
          dual_keep_row_seen,
          dual_AL_seen,
          dual_AU_seen,
          current.lp.A.rowPtr,
          current.lp.A.colVal,
          current.lp.A.rows);
      throw_if_cuda_error(cudaGetLastError(),
                          "_kernel_mark_dual_cols_from_changed_rows");
    };

    auto refresh_incremental_stats = [&]() {
      constexpr int threads = 256;
      const int row_blocks = (current.lp.A.rows + threads - 1) / threads;
      const int col_blocks = (current.lp.A.cols + threads - 1) / threads;
      _kernel_update_row_degrees_from_removed_cols<<<col_blocks, threads>>>(
          stats.row_nnz,
          keep_col_before,
          plan.keep_col_mask,
          plan.keep_row_mask,
          current.lp.AT.rowPtr,
          current.lp.AT.colVal,
          current.lp.A.cols);
      throw_if_cuda_error(cudaGetLastError(),
                          "_kernel_update_row_degrees_from_removed_cols");
      _kernel_update_col_degrees_from_removed_rows<<<row_blocks, threads>>>(
          stats.col_nnz,
          keep_row_before,
          plan.keep_row_mask,
          plan.keep_col_mask,
          current.lp.A.rowPtr,
          current.lp.A.colVal,
          current.lp.A.rows);
      throw_if_cuda_error(cudaGetLastError(),
                          "_kernel_update_col_degrees_from_removed_rows");
      throw_if_cuda_error(cudaMemcpy(keep_row_before,
                                     plan.keep_row_mask,
                                     static_cast<std::size_t>(current.lp.A.rows),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy refresh previous row mask");
      throw_if_cuda_error(cudaMemcpy(keep_col_before,
                                     plan.keep_col_mask,
                                     static_cast<std::size_t>(current.lp.A.cols),
                                     cudaMemcpyDeviceToDevice),
                          "cudaMemcpy refresh previous col mask");
      _kernel_refresh_singleton_rows_from_degrees<<<row_blocks, threads>>>(
          stats.singleton_row_mask,
          stats.singleton_row_col,
          stats.singleton_row_val,
          stats.row_nnz,
          plan.keep_row_mask,
          plan.keep_col_mask,
          current.lp.A.rowPtr,
          current.lp.A.colVal,
          current.lp.A.nzVal,
          current.lp.A.rows);
      throw_if_cuda_error(cudaGetLastError(),
                          "_kernel_refresh_singleton_rows_from_degrees");
      _kernel_refresh_column_singletons_from_degrees<<<col_blocks, threads>>>(
          stats.column_singleton_mask,
          stats.column_singleton_row,
          stats.column_singleton_val,
          stats.col_nnz,
          plan.keep_row_mask,
          plan.keep_col_mask,
          current.lp.AT.rowPtr,
          current.lp.AT.colVal,
          current.lp.AT.nzVal,
          current.lp.A.cols);
      throw_if_cuda_error(cudaGetLastError(),
                          "_kernel_refresh_column_singletons_from_degrees");
    };

    bool changed_any = false;
    bool first_dual_sweep = true;
    std::int32_t rounds = 0;
    double eq_seconds = 0.0;
    double dual_seconds = 0.0;
    double refresh_seconds = 0.0;
    double row_seconds = 0.0;
    while (!terminal && !time_exceeded()) {
      const auto eq_start = std::chrono::steady_clock::now();
      plan.has_change = false;
      // Let the rule kernels detect the final empty round, avoiding a separate
      // column scan to count candidates.
      apply_rule_column_singletons_eq(plan, current.lp, stats, params, true);
      eq_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - eq_start).count();
      const bool changed_eq = plan.has_change;
      terminal = plan.has_infeasible || plan.has_unbounded;

      bool changed_dual = false;
      if (!terminal && !time_exceeded()) {
        const auto dual_start = std::chrono::steady_clock::now();
        if (first_dual_sweep) {
          throw_if_cuda_error(cudaMemset(dual_candidate_mask,
                                        1,
                                        static_cast<std::size_t>(current.lp.A.cols)),
                              "cudaMemset initial dual candidate sweep");
          first_dual_sweep = false;
        } else {
          mark_incremental_dual_candidates();
        }
        // Snapshot pre-commit locks so row removals and shifts from dual-fix
        // remain detectable in the next closure round.
        snapshot_dual_state();
        plan.has_change = false;
        apply_rule_dual_fix(
            plan, current.lp, stats, params, dual_candidate_mask);
        dual_seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - dual_start).count();
        changed_dual = plan.has_change;
        terminal = plan.has_infeasible || plan.has_unbounded;
      }

      bool changed_rows = false;
      if (!terminal && !time_exceeded()) {
        const auto refresh_start = std::chrono::steady_clock::now();
        refresh_incremental_stats();
        if (profile_rules) {
          throw_if_cuda_error(cudaDeviceSynchronize(),
                              "profile incremental stats refresh synchronize");
        }
        refresh_seconds += std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - refresh_start).count();
        const auto row_start = std::chrono::steady_clock::now();
        plan.has_change = false;
        apply_rule_singleton_rows(plan, current.lp, stats, params);
        row_seconds += std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - row_start).count();
        changed_rows = plan.has_change;
        terminal = plan.has_infeasible || plan.has_unbounded;
        if (!terminal && changed_rows) {
          const auto refresh_start_after_rows = std::chrono::steady_clock::now();
          refresh_incremental_stats();
          if (profile_rules) {
            throw_if_cuda_error(cudaDeviceSynchronize(),
                                "profile post-row stats refresh synchronize");
          }
          refresh_seconds += std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() -
                                 refresh_start_after_rows).count();
        }
      }

      const bool changed_round = changed_eq || changed_dual || changed_rows;
      changed_any = changed_any || changed_round;
      ++rounds;
      if (!changed_round) {
        break;
      }
    }

    summary.has_infeasible = plan.has_infeasible;
    summary.has_unbounded = plan.has_unbounded;
    plan.has_change = changed_any;
    if (changed_any && !terminal) {
      apply_plan_to_working_lp(current, plan, &summary.record);
      invalidate_column_singleton_clean();
    }
    if (profile_rules) {
      throw_if_cuda_error(cudaDeviceSynchronize(),
                          "profile batched singleton closure synchronize");
      const std::chrono::duration<double> phase_elapsed =
          std::chrono::steady_clock::now() - phase_start;
      std::cerr << ">>> [GPU Presolve C++ profile] phase=closure"
                << " rules=[column_singletons_eq,dual_fix,singleton_rows] changed="
                << (changed_any ? "true" : "false") << " rounds=" << rounds
                << " dims=(" << m_before << "," << n_before << ")->("
                << current.lp.A.rows << "," << current.lp.A.cols << ") nnz="
                << nnz_before << "->" << current.lp.A.nnz
                << " time=" << phase_elapsed.count() << "s"
                << " eq=" << eq_seconds << "s"
                << " dual=" << dual_seconds << "s"
                << " refresh=" << refresh_seconds << "s"
                << " rows=" << row_seconds << "s\n";
    }
    cudaFree(keep_row_before);
    cudaFree(keep_col_before);
    cudaFree(dual_candidate_mask);
    cudaFree(dual_keep_row_seen);
    cudaFree(dual_c_seen);
    cudaFree(dual_l_seen);
    cudaFree(dual_u_seen);
    cudaFree(dual_AL_seen);
    cudaFree(dual_AU_seen);
    free_stats(stats);
    free_plan(plan);
    return changed_any;
  };

  auto run_fast_phase = [&]() -> bool {
    bool changed_any = false;
    bool changed_singleton = false;
    changed_any = run_batched_singleton_cleanup_closure() || changed_any;
    while (!terminal && !time_exceeded()) {
      const bool changed_dual =
          column_singletons_dual_clean ? false
                                    : run_col_phase(false, false, true, false, false, false, false, false, false);
      const bool changed_eq =
          column_singletons_eq_clean ? false
                                  : run_col_phase(false, false, false, true, false, false, false, false, false);
      const bool changed = changed_dual || changed_eq;
      changed_any = changed_any || changed;
      changed_singleton = changed_singleton || changed;
      if (changed) {
        changed_any = run_cleanup() || changed_any;
      }
      if (!changed) {
        break;
      }
    }
    const bool changed_doubleton = run_repeated_doubleton();
    changed_any = changed_doubleton || changed_any;
    if (changed_doubleton || changed_singleton) {
      changed_any = run_cleanup() || changed_any;
    }
    return changed_any;
  };

  // Propagate from each round's entry bounds until stable or a limit is reached.
  // Commit row shifts immediately; batch pure bound tightenings in one plan
  // to preserve tape order without repeated copies or sparse-map scans.
  auto run_implied_variable_bounds_with_activity_closure = [&]() -> bool {
    if (!params.enable_implied_variable_bounds) {
      return false;
    }
    bool changed_any = false;
    const int max_rounds = std::max(1, params.implied_variable_bounds_max_rounds);
    const int configured_max_bound_only_rounds =
        std::max(1, params.implied_variable_bounds_max_bound_only_rounds);
    // For the wide, dense shape below, allow a second confirmation round
    // when the work cap would stop propagation after one changing round.
    // The configured round limits still apply.
    const std::int64_t propagation_rows =
        static_cast<std::int64_t>(current.lp.A.rows);
    const std::int64_t propagation_cols =
        static_cast<std::int64_t>(current.lp.A.cols);
    const std::int64_t propagation_nnz =
        static_cast<std::int64_t>(current.lp.A.nnz);
    const bool allow_wide_dense_confirmation_round =
        propagation_rows >= 512 && propagation_rows <= 2048 &&
        propagation_cols >= 128 * propagation_rows &&
        propagation_cols <= 256 * propagation_rows &&
        propagation_nnz >= 2048 * propagation_rows &&
        propagation_nnz <= 8192 * propagation_rows;
    auto effective_max_bound_only_rounds = [&]() -> int {
      int effective = configured_max_bound_only_rounds;
      const std::int64_t budget =
          params.implied_variable_bounds_bound_only_nnz_round_budget;
      if (budget <= 0 || current.lp.A.nnz <= 0) {
        return effective;
      }
      const std::int64_t nnz = static_cast<std::int64_t>(current.lp.A.nnz);
      const std::int64_t budget_rounds = std::max<std::int64_t>(1, budget / nnz);
      if (budget_rounds < static_cast<std::int64_t>(effective)) {
        effective = static_cast<int>(budget_rounds);
      }
      if (allow_wide_dense_confirmation_round) {
        effective = std::max(
            effective, std::min(2, configured_max_bound_only_rounds));
      }
      return effective;
    };
    int rounds = 0;
    int bound_only_rounds = 0;
    int bound_only_round_limit = configured_max_bound_only_rounds;
    bool stopped_on_bound_only_work_cap = false;
    const auto closure_start = std::chrono::steady_clock::now();
    while (rounds < max_rounds && !terminal && !time_exceeded()) {
      bound_only_round_limit = effective_max_bound_only_rounds();
      PresolvePlanGpu plan;
      PlanRequirements plan_requirements;
      plan_requirements.objective = true;
      plan_requirements.col_bounds = true;
      plan_requirements.row_bounds = true;
      allocate_plan(plan, current.lp, plan_requirements);

      PresolveStatsGpu stats;
      StatsRequirements stats_requirements;
      stats_requirements.row_nnz = true;
      allocate_stats(stats,
                     current.lp.A.rows,
                     current.lp.A.cols,
                     stats_requirements);
      recompute_row_stats_from_csr(stats, current.lp);

      bool group_changed = false;
      bool structural_change = false;
      while (rounds < max_rounds && !terminal && !time_exceeded()) {
        plan.has_change = false;
        apply_rule_implied_variable_bounds(plan, current.lp, stats, params);
        ++rounds;
        summary.has_infeasible = plan.has_infeasible;
        summary.has_unbounded = plan.has_unbounded;
        terminal = plan.has_infeasible || plan.has_unbounded;
        if (terminal || !plan.has_change) {
          break;
        }
        group_changed = true;
        changed_any = true;
        structural_change = plan.has_new_A || plan.has_row_action ||
                            plan.has_col_action;
        if (structural_change) {
          bound_only_rounds = 0;
          break;
        }
        ++bound_only_rounds;
        if (bound_only_rounds >= bound_only_round_limit) {
          stopped_on_bound_only_work_cap =
              bound_only_round_limit < configured_max_bound_only_rounds;
          break;
        }
      }

      if (group_changed && !terminal) {
        apply_plan_to_working_lp(current, plan, &summary.record);
        invalidate_column_singleton_clean();
      }
      free_stats(stats);
      free_plan(plan);
      if (terminal || !group_changed || !structural_change ||
          bound_only_rounds >= bound_only_round_limit) {
        break;
      }
    }
    if (profile_rules) {
      throw_if_cuda_error(cudaDeviceSynchronize(),
                          "profile implied variable bounds closure synchronize");
      const std::chrono::duration<double> elapsed =
          std::chrono::steady_clock::now() - closure_start;
      std::cerr << ">>> [GPU Presolve C++ profile] phase=primal_closure"
                << " changed=" << (changed_any ? "true" : "false")
                << " rounds=" << rounds
                << " bound_only_limit=" << bound_only_round_limit
                << " work_cap_hit="
                << (stopped_on_bound_only_work_cap ? "true" : "false")
                << " confirmation_round_guard="
                << (allow_wide_dense_confirmation_round ? "true" : "false")
                << " time=" << elapsed.count() << "s\n";
    }
    if (changed_any && !terminal && !time_exceeded()) {
      changed_any = run_row_phase(false, false, true, false, false) || changed_any;
    }
    return changed_any;
  };

  auto run_medium_phase = [&]() -> bool {
    bool changed_any = false;
    bool changed_prop = false;
    changed_prop = run_row_phase(false, false, true, false, false) || changed_prop;
    changed_prop = run_implied_variable_bounds_with_activity_closure() || changed_prop;
    changed_any = changed_prop || changed_any;
    if (changed_prop) {
      changed_any = run_cleanup() || changed_any;
    }
    const bool changed_duplicate_rows = run_row_phase(false, false, false, false, true);
    changed_any = changed_duplicate_rows || changed_any;
    const bool changed_duplicate_columns = run_col_phase(false, false, false, false, false, false, false, true, false);
    changed_any = changed_duplicate_columns || changed_any;
    if (changed_duplicate_rows || changed_duplicate_columns) {
      changed_any = run_cleanup() || changed_any;
    }
    return changed_any;
  };

  auto run_fixed_row_sequence = [&]() -> bool {
    bool changed_any = false;
    changed_any = run_row_phase(true, false, false, false, false) || changed_any;
    changed_any = run_row_phase(false, true, false, false, false) || changed_any;
    changed_any = run_row_phase(false, false, true, false, false) || changed_any;
    const bool changed_primal = run_implied_variable_bounds_with_activity_closure();
    changed_any = changed_primal || changed_any;
    if (changed_primal) {
      changed_any = run_trivial_cleanup_recirculation() || changed_any;
    }
    changed_any = run_row_phase(false, false, false, false, true) || changed_any;
    return changed_any;
  };

  auto run_cleanup_trigger_col_rule = [&](bool infeasible_fixed_variables,
                                          bool column_singletons_dual,
                                          bool column_singletons_eq,
                                          bool doubleton_equations,
                                          bool structural_l1,
                                          bool dual_fix,
                                          bool duplicate_columns) -> bool {
    const bool changed = run_col_phase(infeasible_fixed_variables,
                                       false,
                                       column_singletons_dual,
                                       column_singletons_eq,
                                       doubleton_equations,
                                       structural_l1,
                                       dual_fix,
                                       duplicate_columns,
                                       false);
    if (changed) {
      (void)run_trivial_cleanup_recirculation();
    }
    return changed;
  };

  auto run_fixed_col_sequence = [&]() -> bool {
    bool changed_any = false;
    changed_any = run_cleanup_trigger_col_rule(true, false, false, false, false, false, false) || changed_any;
    changed_any = run_cleanup_trigger_col_rule(false, false, false, false, true, false, false) || changed_any;
    changed_any = run_col_phase(false, true, false, false, false, false, false, false, false) || changed_any;

    bool changed_rule = false;
    do {
      changed_rule = run_cleanup_trigger_col_rule(false, true, false, false, false, false, false);
      changed_any = changed_rule || changed_any;
    } while (changed_rule && !terminal && !time_exceeded());

    do {
      changed_rule = run_cleanup_trigger_col_rule(false, false, true, false, false, false, false);
      changed_any = changed_rule || changed_any;
    } while (changed_rule && !terminal && !time_exceeded());

    do {
      changed_rule = run_cleanup_trigger_col_rule(false, false, false, true, false, false, false);
      changed_any = changed_rule || changed_any;
    } while (changed_rule && !params.doubleton_equations_single_batch_per_iter && !terminal && !time_exceeded());

    changed_any = run_cleanup_trigger_col_rule(false, false, false, false, false, true, false) || changed_any;
    changed_any = run_cleanup_trigger_col_rule(false, false, false, false, false, false, true) || changed_any;
    return changed_any;
  };

  if (!params.use_tiered_scheduler) {
    for (int iter = 0; iter < params.max_iters && !terminal && !time_exceeded(); ++iter) {
      bool changed_iter = false;
      changed_iter = run_fixed_row_sequence() || changed_iter;
      changed_iter = run_fixed_col_sequence() || changed_iter;
      ++summary.iterations;
      if (!changed_iter) {
        break;
      }
    }
  } else {
    if (params.enable_tiered_bootstrap && params.max_iters > 0 && !time_exceeded()) {
      // Run model-wide reductions before structural L1 and generic rewrites
      // to preserve their original-layout certificates.
      if (!terminal && params.enable_covering_cost_dominance) {
        CoveringCostDominanceAnalysisGpu analysis =
            analyze_covering_cost_dominance(current.lp, params);
        if (analysis.applicable) {
          PresolvePlanGpu plan;
          allocate_plan(plan, current.lp,
                        PlanRequirements{/*objective=*/false,
                                         /*col_bounds=*/true,
                                         /*row_bounds=*/false});
          build_covering_cost_dominance_plan(plan, current.lp, analysis,
                                             params);
          const bool changed_covering = plan.has_change;
          if (changed_covering) {
            apply_plan_to_working_lp(current, plan, &summary.record);
            invalidate_column_singleton_clean();
          }

          free_plan(plan);
        }
        free_covering_cost_dominance_analysis(analysis);
      }

      bool changed_linf_components = false;
      if (!terminal && !time_exceeded() && params.enable_linf_components) {
        const std::int32_t m_before = current.lp.A.rows;
        const std::int32_t n_before = current.lp.A.cols;
        const std::int32_t nnz_before = current.lp.A.nnz;
        const auto phase_start = std::chrono::steady_clock::now();
        LinfComponentAnalysisGpu analysis =
            analyze_linf_components(current.lp, params);
        if (analysis.applicable) {
          PresolvePlanGpu plan;
          allocate_plan(plan, current.lp, PlanRequirements{});
          build_linf_component_plan(plan, current.lp, analysis, params);
          summary.has_infeasible = plan.has_infeasible;
          summary.has_unbounded = plan.has_unbounded;
          terminal = plan.has_infeasible || plan.has_unbounded;
          changed_linf_components = !terminal && plan.has_change;
          if (changed_linf_components) {
            apply_plan_to_working_lp(current, plan, &summary.record);
            invalidate_column_singleton_clean();
          }
          free_plan(plan);
        }
        free_linf_component_analysis(analysis);

        if (profile_rules) {
          throw_if_cuda_error(cudaDeviceSynchronize(),
                              "profile bootstrap Linf component synchronize");
          const std::chrono::duration<double> elapsed =
              std::chrono::steady_clock::now() - phase_start;
          std::cerr << ">>> [GPU Presolve C++ profile]"
                    << " phase=bootstrap_linf_components"
                    << " changed="
                    << (changed_linf_components ? "true" : "false")
                    << " dims=(" << m_before << "," << n_before << ")->("
                    << current.lp.A.rows << "," << current.lp.A.cols << ")"
                    << " nnz=" << nnz_before << "->" << current.lp.A.nnz
                    << " time=" << elapsed.count() << "s\n";
        }
      }

      if (!terminal && !changed_linf_components && !time_exceeded() &&
          params.enable_antipodal_components) {
        AntipodalComponentAnalysisGpu analysis =
            analyze_antipodal_components(current.lp, params);
        if (analysis.applicable) {
          const std::int32_t m_before = current.lp.A.rows;
          const std::int32_t n_before = current.lp.A.cols;
          const std::int32_t nnz_before = current.lp.A.nnz;
          PresolvePlanGpu plan;
          allocate_plan(plan, current.lp, PlanRequirements{});
          build_antipodal_component_plan(plan, current.lp, analysis, params);
          summary.has_infeasible = plan.has_infeasible;
          summary.has_unbounded = plan.has_unbounded;
          terminal = plan.has_infeasible || plan.has_unbounded;
          const bool changed_antipodal = !terminal && plan.has_change;
          if (changed_antipodal) {
            apply_plan_to_working_lp(current, plan, &summary.record);
            invalidate_column_singleton_clean();
          }

          if (profile_rules) {
            throw_if_cuda_error(
                cudaDeviceSynchronize(),
                "profile bootstrap antipodal component synchronize");
            std::cerr << ">>> [GPU Presolve C++ profile]"
                      << " phase=bootstrap_antipodal_components"
                      << " changed="
                      << (changed_antipodal ? "true" : "false")
                      << " dims=(" << m_before << "," << n_before << ")->("
                      << current.lp.A.rows << "," << current.lp.A.cols
                      << ") nnz=" << nnz_before << "->" << current.lp.A.nnz
                      << "\n";
          }
          free_plan(plan);
        }
        free_antipodal_component_analysis(analysis);
      }

      // Commit the degree-two projection before detecting orphan McCormick
      // columns: rebuilding AT exposes their degree-three incidence.
      // Record both steps in order for postsolve, before structural L1.
      if (!terminal && !time_exceeded() &&
          params.enable_bounded_two_row_projection) {
        const std::int32_t m_before = current.lp.A.rows;
        const std::int32_t n_before = current.lp.A.cols;
        const std::int32_t nnz_before = current.lp.A.nnz;
        const auto phase_start = std::chrono::steady_clock::now();
        BoundedTwoRowProjectionAnalysisGpu analysis =
            analyze_bounded_two_row_projection(current.lp, params);
        bool changed_projection = false;
        if (analysis.applicable) {
          PresolvePlanGpu plan;
          allocate_plan(plan, current.lp,
                        PlanRequirements{/*objective=*/false,
                                         /*col_bounds=*/false,
                                         /*row_bounds=*/false});
          build_bounded_two_row_projection_plan(plan, current.lp, analysis,
                                                params);
          changed_projection = plan.has_change;
          if (changed_projection) {
            apply_plan_to_working_lp(current, plan, &summary.record);
            invalidate_column_singleton_clean();
          }
          free_plan(plan);
        }
        free_bounded_two_row_projection_analysis(analysis);

        if (profile_rules) {
          throw_if_cuda_error(
              cudaDeviceSynchronize(),
              "profile bootstrap bounded two-row projection synchronize");
          const std::chrono::duration<double> elapsed =
              std::chrono::steady_clock::now() - phase_start;
          std::cerr << ">>> [GPU Presolve C++ profile]"
                    << " phase=bootstrap_bounded_two_row_projection"
                    << " changed="
                    << (changed_projection ? "true" : "false")
                    << " dims=(" << m_before << "," << n_before << ")->("
                    << current.lp.A.rows << "," << current.lp.A.cols << ")"
                    << " nnz=" << nnz_before << "->" << current.lp.A.nnz
                    << " time=" << elapsed.count() << "s\n";
        }
      }

      if (!terminal && !time_exceeded() &&
          params.enable_orphan_mccormick_projection) {
        const std::int32_t m_before = current.lp.A.rows;
        const std::int32_t n_before = current.lp.A.cols;
        const std::int32_t nnz_before = current.lp.A.nnz;
        const auto phase_start = std::chrono::steady_clock::now();
        OrphanMcCormickProjectionAnalysisGpu analysis =
            analyze_orphan_mccormick_projection(current.lp, params);
        bool changed_projection = false;
        if (analysis.applicable) {
          PresolvePlanGpu plan;
          allocate_plan(plan, current.lp,
                        PlanRequirements{/*objective=*/false,
                                         /*col_bounds=*/false,
                                         /*row_bounds=*/false});
          build_orphan_mccormick_projection_plan(plan, current.lp, analysis,
                                                 params);
          changed_projection = plan.has_change;
          if (changed_projection) {
            apply_plan_to_working_lp(current, plan, &summary.record);
            invalidate_column_singleton_clean();
          }
          free_plan(plan);
        }
        free_orphan_mccormick_projection_analysis(analysis);

        if (profile_rules) {
          throw_if_cuda_error(
              cudaDeviceSynchronize(),
              "profile bootstrap orphan McCormick projection synchronize");
          const std::chrono::duration<double> elapsed =
              std::chrono::steady_clock::now() - phase_start;
          std::cerr << ">>> [GPU Presolve C++ profile]"
                    << " phase=bootstrap_orphan_mccormick_projection"
                    << " changed="
                    << (changed_projection ? "true" : "false")
                    << " dims=(" << m_before << "," << n_before << ")->("
                    << current.lp.A.rows << "," << current.lp.A.cols << ")"
                    << " nnz=" << nnz_before << "->" << current.lp.A.nnz
                    << " time=" << elapsed.count() << "s\n";
        }
      }

      const bool changed_structural_before =
          run_col_phase(false, false, false, false, false, true, false, false, false);

      if (changed_structural_before) {
        run_cleanup();
      }
      bool changed_generic_bootstrap = false;
      changed_generic_bootstrap = run_row_phase(true, false, false, false, false) || changed_generic_bootstrap;
      changed_generic_bootstrap = run_row_phase(false, true, false, false, false) || changed_generic_bootstrap;
      changed_generic_bootstrap = run_row_phase(false, false, true, false, false) || changed_generic_bootstrap;
      changed_generic_bootstrap = run_row_phase(false, false, false, false, true) || changed_generic_bootstrap;

      bool changed_rule = run_col_phase(true, true, false, false, false, false, false, false, false);
      changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
      if (changed_rule) {
        changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
      }

      changed_generic_bootstrap = run_col_phase(false, true, false, false, false, false, false, false, false) ||
                                  changed_generic_bootstrap;

      do {
        changed_rule = run_col_phase(false, false, true, false, false, false, false, false, false);
        changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
        if (changed_rule) {
          changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
        }
      } while (changed_rule && !terminal && !time_exceeded());

      do {
        changed_rule = run_col_phase(false, false, false, true, false, false, false, false, false);
        changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
        if (changed_rule) {
          changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
        }
      } while (changed_rule && !terminal && !time_exceeded());

      changed_rule = run_col_phase(false, false, false, false, false, false, true, false, false);
      changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
      if (changed_rule) {
        changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
      }

      changed_rule = run_col_phase(false, false, false, false, false, false, false, true, false);
      changed_generic_bootstrap = changed_rule || changed_generic_bootstrap;
      if (changed_rule) {
        changed_generic_bootstrap = run_trivial_cleanup_recirculation() || changed_generic_bootstrap;
      }

      if (changed_generic_bootstrap) {
        const bool changed_structural_after =
            run_col_phase(false, false, false, false, false, true, false, false, false);

        if (changed_structural_after) {
          run_cleanup();
        }
      }
      ++summary.iterations;
    }

    // Run tiered cycles after bootstrap whether or not it changed the model.
    // The iteration count already includes any completed bootstrap pass.
    bool fast_phase = true;
    std::int32_t cycle_nnz_before = current.lp.A.nnz;
    constexpr double progress_ratio = 0.95;
    for (int iter = summary.iterations;
         iter < params.max_iters && !terminal && !time_exceeded();
         ++iter) {
      bool changed_iter = false;
      changed_iter = run_cleanup() || changed_iter;
      const std::int32_t nnz_before_phase = current.lp.A.nnz;
      if (fast_phase) {
        changed_iter = run_fast_phase() || changed_iter;
        const bool productive = _has_good_nnz_progress(nnz_before_phase, current.lp.A.nnz, progress_ratio);
        fast_phase = productive;
      } else {
        changed_iter = run_medium_phase() || changed_iter;
        const bool productive = _has_good_nnz_progress(cycle_nnz_before, current.lp.A.nnz, progress_ratio);
        if (!changed_iter || !productive) {
          ++summary.iterations;
          break;
        }
        cycle_nnz_before = current.lp.A.nnz;
        fast_phase = true;
      }
      ++summary.iterations;
      if (!changed_iter && fast_phase) {
        break;
      }
    }
  }

  if (!terminal && !time_exceeded() &&
      (params.enable_redundant_bounds ||
       params.enable_zero_cost_redundant_box_bounds)) {
    (void)run_col_phase(false, false, false, false, false, false, false, false, true);
  }

  if (summary.record.tape_gpu.record_count != 0 ||
      summary.record.tape_gpu.index_count != 0 ||
      summary.record.tape_gpu.value_count != 0) {
    throw std::runtime_error(
        "GPU postsolve tape must remain deferred during presolve");
  }
  // Defer tape upload to postsolve_gpu, after the reduced solve.
  throw_if_cuda_error(cudaDeviceSynchronize(), "run_gpu_presolve_impl synchronize");
  summary.reduced_rows = current.lp.A.rows;
  summary.reduced_cols = current.lp.A.cols;
  summary.reduced_nnz = current.lp.A.nnz;
  summary.obj_constant_delta = current.lp.obj_constant - lp.obj_constant;
  summary.record.m1 = summary.reduced_rows;
  summary.record.n1 = summary.reduced_cols;
  summary.record.obj_constant_new = current.lp.obj_constant;
  rebuild_org2red(summary.record.row_org2red, summary.record.row_red2org, summary.record.m0);
  rebuild_org2red(summary.record.col_org2red, summary.record.col_red2org, summary.record.n0);
  const std::chrono::duration<double> presolve_elapsed =
      std::chrono::steady_clock::now() - presolve_start;
  summary.elapsed_seconds = presolve_elapsed.count();
  if (keep_reduced_lp) {
    summary.reduced_lp = current.lp;
    summary.owns_reduced_lp = current.owns;
    current.owns = false;
  } else {
    current.free_owned();
  }
  return summary;
}

}  // namespace

GpuPresolveSummary run_gpu_presolve_with_record(const LPInfoGpu& lp,
                                                const PresolveParams& params) {
  return run_gpu_presolve_impl(lp, params, false);
}

GpuPresolveSummary::GpuPresolveSummary(GpuPresolveSummary&& other) noexcept
    : original_rows(std::exchange(other.original_rows, 0)),
      original_cols(std::exchange(other.original_cols, 0)),
      reduced_rows(std::exchange(other.reduced_rows, 0)),
      reduced_cols(std::exchange(other.reduced_cols, 0)),
      reduced_nnz(std::exchange(other.reduced_nnz, 0)),
      iterations(std::exchange(other.iterations, 0)),
      elapsed_seconds(std::exchange(other.elapsed_seconds, 0.0)),
      obj_constant_delta(std::exchange(other.obj_constant_delta, 0.0)),
      has_infeasible(std::exchange(other.has_infeasible, false)),
      has_unbounded(std::exchange(other.has_unbounded, false)),
      record(std::move(other.record)),
      reduced_lp(std::exchange(other.reduced_lp, LPInfoGpu{})),
      owns_reduced_lp(std::exchange(other.owns_reduced_lp, false)) {}

GpuPresolveSummary& GpuPresolveSummary::operator=(GpuPresolveSummary&& other) noexcept {
  if (this != &other) {
    free_gpu_presolve_reduced_lp(*this);
    original_rows = std::exchange(other.original_rows, 0);
    original_cols = std::exchange(other.original_cols, 0);
    reduced_rows = std::exchange(other.reduced_rows, 0);
    reduced_cols = std::exchange(other.reduced_cols, 0);
    reduced_nnz = std::exchange(other.reduced_nnz, 0);
    iterations = std::exchange(other.iterations, 0);
    elapsed_seconds = std::exchange(other.elapsed_seconds, 0.0);
    obj_constant_delta = std::exchange(other.obj_constant_delta, 0.0);
    has_infeasible = std::exchange(other.has_infeasible, false);
    has_unbounded = std::exchange(other.has_unbounded, false);
    record = std::move(other.record);
    reduced_lp = std::exchange(other.reduced_lp, LPInfoGpu{});
    owns_reduced_lp = std::exchange(other.owns_reduced_lp, false);
  }
  return *this;
}

GpuPresolveSummary run_gpu_presolve_with_reduced_lp(const LPInfoGpu& lp,
                                                    const PresolveParams& params) {
  return run_gpu_presolve_impl(lp, params, true);
}

void free_gpu_presolve_reduced_lp(GpuPresolveSummary& summary) {
  if (!summary.owns_reduced_lp) {
    summary.reduced_lp = LPInfoGpu{};
    return;
  }
  cudaFree(summary.reduced_lp.A.rowPtr);
  cudaFree(summary.reduced_lp.A.colVal);
  cudaFree(summary.reduced_lp.A.nzVal);
  cudaFree(summary.reduced_lp.AT.rowPtr);
  cudaFree(summary.reduced_lp.AT.colVal);
  cudaFree(summary.reduced_lp.AT.nzVal);
  cudaFree(summary.reduced_lp.c);
  cudaFree(summary.reduced_lp.AL);
  cudaFree(summary.reduced_lp.AU);
  cudaFree(summary.reduced_lp.l);
  cudaFree(summary.reduced_lp.u);
  summary.reduced_lp = LPInfoGpu{};
  summary.owns_reduced_lp = false;
}

}  // namespace gpu_presolver::presolve
