#include "../presolve_helpers.hpp"
#include "gpu_presolver/presolve/rules/rule_duplicate_columns.hpp"

#include "gpu_presolver/presolve/fixed_col_tape.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

using detail::throw_if_cuda_error;
using detail::env_enabled;
using detail::append_postsolve_record;

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr unsigned long long FNV_OFFSET = 0xcbf29ce484222325ULL;
constexpr unsigned long long FNV_PRIME = 0x100000001b3ULL;

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

struct HashColLess {
  __host__ __device__ bool operator()(const thrust::tuple<unsigned long long, std::int32_t>& lhs,
                                      const thrust::tuple<unsigned long long, std::int32_t>& rhs) const {
    const unsigned long long lhs_hash = thrust::get<0>(lhs);
    const unsigned long long rhs_hash = thrust::get<0>(rhs);
    if (lhs_hash != rhs_hash) {
      return lhs_hash < rhs_hash;
    }
    return thrust::get<1>(lhs) < thrust::get<1>(rhs);
  }
};

struct DuplicateColumnTapeRecord {
  std::int32_t deleted_col;
  std::int32_t kept_col;
  double ratio;
  double deleted_l;
  double deleted_u;
  double kept_l;
  double kept_u;
};

__global__ void _kernel_pack_duplicate_column_tape(DuplicateColumnTapeRecord* records,
                                               std::int32_t* count,
                                               const std::uint8_t* col_delete,
                                               const std::int32_t* merge_to,
                                               const double* merge_ratio,
                                               const double* merge_from_l,
                                               const double* merge_from_u,
                                               const double* merge_to_l,
                                               const double* merge_to_u,
                                               std::int32_t n) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col < n && col_delete[col] != std::uint8_t{0}) {
    const std::int32_t slot = atomicAdd(count, 1);
    records[slot] = DuplicateColumnTapeRecord{col,
                                          merge_to[col],
                                          merge_ratio[col],
                                          merge_from_l[col],
                                          merge_from_u[col],
                                          merge_to_l[col],
                                          merge_to_u[col]};
  }
}

__global__ void _kernel_duplicate_column_sort_keys(unsigned long long* keys,
                                               const std::int32_t* row_ptr,
                                               const std::int32_t* col_val,
                                               std::int32_t rows) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < rows) {
    const unsigned long long row_key = static_cast<unsigned long long>(static_cast<std::uint32_t>(row)) << 32;
    for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      keys[p] = row_key | static_cast<unsigned long long>(static_cast<std::uint32_t>(col_val[p]));
    }
  }
}

__global__ void _kernel_duplicate_column_unpack_keys(std::int32_t* col_val,
                                                 const unsigned long long* keys,
                                                 std::int32_t nnz) {
  const std::int32_t p = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < nnz) {
    col_val[p] = static_cast<std::int32_t>(keys[p] & 0xffffffffULL);
  }
}

__device__ unsigned long long _duplicate_column_hash_mix(unsigned long long h,
                                                     unsigned long long x) {
  return (h ^ x) * FNV_PRIME;
}

__device__ unsigned long long _double_bits(double value) {
  return static_cast<unsigned long long>(__double_as_longlong(value));
}

__global__ void _kernel_duplicate_column_hashes(unsigned long long* col_hash,
                                            const std::uint8_t* keep_col,
                                            const std::uint8_t* keep_row,
                                            const std::int32_t* row_ptr,
                                            const std::int32_t* row_idx,
                                            const double* row_val,
                                            double coeff_tol,
                                            std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    if (keep_col[j] == std::uint8_t{0}) {
      col_hash[j] = static_cast<unsigned long long>(j);
      return;
    }
    const std::int32_t start_j = row_ptr[j];
    const std::int32_t stop_j = row_ptr[j + 1];
    std::int32_t live_len = 0;
    double pivot = 0.0;
    bool pivot_set = false;
    for (std::int32_t p = start_j; p < stop_j; ++p) {
      const std::int32_t row = row_idx[p];
      if (keep_row[row] != std::uint8_t{0}) {
        const double a = row_val[p];
        if (fabs(a) > coeff_tol) {
          ++live_len;
          if (!pivot_set) {
            pivot = a;
            pivot_set = true;
          }
        }
      }
    }
    if (!pivot_set || live_len == 0) {
      col_hash[j] = _duplicate_column_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(j));
      return;
    }
    unsigned long long h = _duplicate_column_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(live_len));
    for (std::int32_t p = start_j; p < stop_j; ++p) {
      const std::int32_t row = row_idx[p];
      if (keep_row[row] != std::uint8_t{0}) {
        const double a = row_val[p];
        if (fabs(a) > coeff_tol) {
          h = _duplicate_column_hash_mix(h, static_cast<unsigned long long>(row));
          h = _duplicate_column_hash_mix(h, _double_bits(a / pivot));
        }
      }
    }
    col_hash[j] = h;
  }
}

__device__ std::int32_t _next_live_duplicate_column_entry(std::int32_t ptr,
                                                      std::int32_t stop,
                                                      const std::int32_t* row_idx,
                                                      const double* row_val,
                                                      const std::uint8_t* keep_row,
                                                      double zero_tol) {
  std::int32_t cur = ptr;
  while (cur < stop) {
    const std::int32_t row = row_idx[cur];
    if (keep_row[row] != std::uint8_t{0}) {
      const double a = row_val[cur];
      if (fabs(a) > zero_tol) {
        return cur;
      }
    }
    ++cur;
  }
  return stop;
}

__device__ bool _duplicate_column_ratio(std::int32_t j,
                                    std::int32_t k,
                                    const std::int32_t* row_ptr,
                                    const std::int32_t* row_idx,
                                    const double* row_val,
                                    const std::uint8_t* keep_row,
                                    double zero_tol,
                                    double coeff_tol,
                                    double* ratio_out) {
  std::int32_t ptr_j = row_ptr[j];
  const std::int32_t stop_j = row_ptr[j + 1];
  std::int32_t ptr_k = row_ptr[k];
  const std::int32_t stop_k = row_ptr[k + 1];
  double ratio = 0.0;
  bool ratio_set = false;

  for (;;) {
    ptr_j = _next_live_duplicate_column_entry(ptr_j, stop_j, row_idx, row_val, keep_row, zero_tol);
    ptr_k = _next_live_duplicate_column_entry(ptr_k, stop_k, row_idx, row_val, keep_row, zero_tol);
    if (ptr_j >= stop_j || ptr_k >= stop_k) {
      break;
    }
    const std::int32_t row_j = row_idx[ptr_j];
    const std::int32_t row_k = row_idx[ptr_k];
    if (row_j != row_k) {
      return false;
    }
    const double a_j = row_val[ptr_j];
    const double a_k = row_val[ptr_k];
    if (!ratio_set) {
      if (fabs(a_j) <= zero_tol) {
        return false;
      }
      ratio = a_k / a_j;
      if (fabs(ratio) <= zero_tol || !isfinite(ratio)) {
        return false;
      }
      ratio_set = true;
    }
    if (fabs(a_k - ratio * a_j) > coeff_tol) {
      return false;
    }
    ++ptr_j;
    ++ptr_k;
  }

  ptr_j = _next_live_duplicate_column_entry(ptr_j, stop_j, row_idx, row_val, keep_row, zero_tol);
  ptr_k = _next_live_duplicate_column_entry(ptr_k, stop_k, row_idx, row_val, keep_row, zero_tol);
  if (ptr_j < stop_j || ptr_k < stop_k || !ratio_set) {
    return false;
  }
  *ratio_out = ratio;
  return true;
}

__device__ double _merged_lower_bound_duplicate_columns(double target_l,
                                                    double target_u,
                                                    double source_l,
                                                    double source_u,
                                                    double ratio) {
  (void)target_u;
  if (ratio > 0.0) {
    if (isfinite(target_l) && isfinite(source_l)) {
      return target_l + ratio * source_l;
    }
  } else if (isfinite(target_l) && isfinite(source_u)) {
    return target_l + ratio * source_u;
  }
  return -INFINITY;
}

__device__ double _merged_upper_bound_duplicate_columns(double target_l,
                                                    double target_u,
                                                    double source_l,
                                                    double source_u,
                                                    double ratio) {
  (void)target_l;
  if (ratio > 0.0) {
    if (isfinite(target_u) && isfinite(source_u)) {
      return target_u + ratio * source_u;
    }
  } else if (isfinite(target_u) && isfinite(source_l)) {
    return target_u + ratio * source_l;
  }
  return INFINITY;
}

__global__ void _kernel_count_small_high_ratio_parallel_groups(
    std::int32_t* count,
    const unsigned long long* sorted_hash,
    const std::int32_t* sorted_cols,
    const std::uint8_t* keep_col,
    const std::uint8_t* keep_row,
    const double* c,
    const std::int32_t* row_ptr,
    const std::int32_t* row_idx,
    const double* row_val,
    double zero_tol,
    double coeff_tol,
    double obj_tol,
    double max_abs_ratio,
    std::int32_t max_group_size,
    std::int32_t stop_after,
    std::int32_t n) {
  const std::int32_t s =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s >= n || atomicAdd(count, 0) >= stop_after) {
    return;
  }
  const unsigned long long hs = sorted_hash[s];
  if (s > 0 && sorted_hash[s - 1] == hs) {
    return;
  }
  std::int32_t e = s;
  while (e + 1 < n && sorted_hash[e + 1] == hs) {
    ++e;
  }
  const std::int32_t group_size = e - s + 1;
  if (group_size < 2 || group_size > max_group_size) {
    return;
  }
  for (std::int32_t a = s; a < e; ++a) {
    const std::int32_t j = sorted_cols[a];
    if (keep_col[j] == std::uint8_t{0}) {
      continue;
    }
    for (std::int32_t b = a + 1; b <= e; ++b) {
      const std::int32_t k = sorted_cols[b];
      if (keep_col[k] == std::uint8_t{0}) {
        continue;
      }
      double ratio = 0.0;
      if (!_duplicate_column_ratio(j, k, row_ptr, row_idx, row_val, keep_row,
                               zero_tol, coeff_tol, &ratio)) {
        continue;
      }
      if (fabs(c[k] - ratio * c[j]) > obj_tol) {
        continue;
      }
      const double abs_ratio = fabs(ratio);
      if (max_abs_ratio < 1.0 || abs_ratio > max_abs_ratio ||
          abs_ratio * max_abs_ratio < 1.0) {
        atomicAdd(count, 1);
        return;
      }
    }
  }
}

__global__ void _kernel_duplicate_column_groups(std::int32_t* status_flag,
                                            std::uint8_t* col_delete,
                                            std::uint8_t* fixed_mask,
                                            double* fixed_val,
                                            std::int32_t* merge_to,
                                            double* merge_ratio,
                                            double* merge_from_l,
                                            double* merge_from_u,
                                            double* merge_to_l,
                                            double* merge_to_u,
                                            const unsigned long long* sorted_hash,
                                            const std::int32_t* sorted_cols,
                                            const std::uint8_t* keep_col,
                                            const std::uint8_t* keep_row,
                                            const double* c,
                                            double* l,
                                            double* u,
                                            const std::int32_t* row_ptr,
                                            const std::int32_t* row_idx,
                                            const double* row_val,
                                            double zero_tol,
                                            double coeff_tol,
                                            double obj_tol,
                                            double max_abs_ratio,
                                            const std::int32_t* scale_guard_group_count,
                                            std::int32_t scale_guard_min_groups,
                                            std::int32_t n) {
  const std::int32_t s = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s < n) {
    const unsigned long long hs = sorted_hash[s];
    if (s > 0 && sorted_hash[s - 1] == hs) {
      return;
    }
    std::int32_t e = s;
    while (e + 1 < n && sorted_hash[e + 1] == hs) {
      ++e;
    }
    if (e <= s) {
      return;
    }

    for (std::int32_t a = s; a < e; ++a) {
      const std::int32_t j = sorted_cols[a];
      if (keep_col[j] == std::uint8_t{0} || col_delete[j] != std::uint8_t{0} || fixed_mask[j] != std::uint8_t{0}) {
        continue;
      }
      for (std::int32_t b = a + 1; b <= e; ++b) {
        const std::int32_t k = sorted_cols[b];
        if (keep_col[k] == std::uint8_t{0} || col_delete[k] != std::uint8_t{0} || fixed_mask[k] != std::uint8_t{0}) {
          continue;
        }
        double ratio = 0.0;
        if (!_duplicate_column_ratio(j, k, row_ptr, row_idx, row_val, keep_row, zero_tol, coeff_tol, &ratio)) {
          continue;
        }
        double active_max_abs_ratio = max_abs_ratio;
        if (scale_guard_group_count != nullptr &&
            *scale_guard_group_count < scale_guard_min_groups) {
          active_max_abs_ratio = INFINITY;
        }
        const double abs_ratio = fabs(ratio);
        if (isfinite(active_max_abs_ratio) &&
            (active_max_abs_ratio < 1.0 || abs_ratio > active_max_abs_ratio ||
             abs_ratio * active_max_abs_ratio < 1.0)) {
          continue;
        }

        const double obj_gap = c[k] - ratio * c[j];
        const double target_l = l[j];
        const double target_u = u[j];
        const double source_l = l[k];
        const double source_u = u[k];

        if (fabs(obj_gap) <= obj_tol) {
          merge_to[k] = j;
          merge_ratio[k] = ratio;
          merge_from_l[k] = source_l;
          merge_from_u[k] = source_u;
          merge_to_l[k] = target_l;
          merge_to_u[k] = target_u;
          l[j] = _merged_lower_bound_duplicate_columns(target_l, target_u, source_l, source_u, ratio);
          u[j] = _merged_upper_bound_duplicate_columns(target_l, target_u, source_l, source_u, ratio);
          col_delete[k] = std::uint8_t{1};
          atomicMax(&status_flag[1], 1);
          continue;
        }

        bool fix_xk_to_lower = false;
        bool fix_xk_to_upper = false;
        bool fix_xj_to_lower = false;
        bool fix_xj_to_upper = false;
        if (obj_gap > obj_tol) {
          if (ratio > 0.0) {
            fix_xk_to_lower = !isfinite(target_u);
            fix_xj_to_upper = !isfinite(source_l);
          } else {
            fix_xk_to_lower = !isfinite(target_l);
            fix_xj_to_lower = !isfinite(source_l);
          }
        } else {
          if (ratio > 0.0) {
            fix_xk_to_upper = !isfinite(target_l);
            fix_xj_to_lower = !isfinite(source_u);
          } else {
            fix_xk_to_upper = !isfinite(target_u);
            fix_xj_to_upper = !isfinite(source_u);
          }
        }

        if (fix_xk_to_lower) {
          if (!isfinite(source_l)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[k] = std::uint8_t{1};
          fixed_val[k] = source_l;
          atomicMax(&status_flag[2], 1);
          continue;
        } else if (fix_xk_to_upper) {
          if (!isfinite(source_u)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[k] = std::uint8_t{1};
          fixed_val[k] = source_u;
          atomicMax(&status_flag[2], 1);
          continue;
        }

        if (fix_xj_to_lower) {
          if (!isfinite(target_l)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[j] = std::uint8_t{1};
          fixed_val[j] = target_l;
          atomicMax(&status_flag[2], 1);
          break;
        } else if (fix_xj_to_upper) {
          if (!isfinite(target_u)) {
            atomicMax(&status_flag[0], 1);
            return;
          }
          fixed_mask[j] = std::uint8_t{1};
          fixed_val[j] = target_u;
          atomicMax(&status_flag[2], 1);
          break;
        }
      }
    }
  }
}

__global__ void _kernel_duplicate_column_fixed_row_shift(double* row_shift,
                                                     const std::uint8_t* fixed_mask,
                                                     const double* fixed_val,
                                                     const std::uint8_t* keep_row,
                                                     const std::int32_t* row_ptr,
                                                     const std::int32_t* row_idx,
                                                     const double* row_val,
                                                     std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && fixed_mask[j] != std::uint8_t{0}) {
    const double vj = fixed_val[j];
    const std::int32_t start_j = row_ptr[j];
    const std::int32_t stop_j = row_ptr[j + 1];
    for (std::int32_t p = start_j; p < stop_j; ++p) {
      const std::int32_t row = row_idx[p];
      if (keep_row[row] != std::uint8_t{0}) {
        atomicAdd(&row_shift[row], row_val[p] * vj);
      }
    }
  }
}

__global__ void _kernel_apply_duplicate_columns(std::uint8_t* keep_col,
                                            double* AL,
                                            double* AU,
                                            double* new_l,
                                            double* new_u,
                                            double* obj_delta,
                                            const double* c,
                                            const std::uint8_t* col_delete,
                                            const std::uint8_t* fixed_mask,
                                            const double* fixed_val,
                                            const double* row_shift,
                                            std::int32_t m,
                                            std::int32_t n) {
  const std::int32_t q = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q < n && (col_delete[q] != std::uint8_t{0} || fixed_mask[q] != std::uint8_t{0})) {
    keep_col[q] = std::uint8_t{0};
    if (fixed_mask[q] != std::uint8_t{0}) {
      new_l[q] = fixed_val[q];
      new_u[q] = fixed_val[q];
      atomicAdd(obj_delta, c[q] * fixed_val[q]);
    }
  }
  if (q < m) {
    AL[q] -= row_shift[q];
    AU[q] -= row_shift[q];
  }
}

}  // namespace

void apply_rule_duplicate_columns(PresolvePlanGpu& plan,
                              const LPInfoGpu& lp,
                              const PresolveStatsGpu& stats,
                              const PresolveParams& pparams) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (n <= 1) {
    return;
  }
  const auto rule_start = std::chrono::steady_clock::now();
  const bool profile = env_enabled("GPUPRESOLVER_DUPLICATE_COLUMNS_PROFILE");
  auto profile_stage = [&](const char* stage, const std::chrono::steady_clock::time_point& stage_start) {
    if (!profile) {
      return;
    }
    throw_if_cuda_error(cudaDeviceSynchronize(), "duplicate_columns profile synchronize");
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - stage_start;
    std::cerr << ">>> [duplicate_columns C++] " << stage << " = " << elapsed.count() << "s\n";
  };

  if (lp.AT.nnz > 0) {
    const auto sort_at_start = std::chrono::steady_clock::now();
    unsigned long long* sort_keys = nullptr;
    throw_if_cuda_error(cudaMalloc(&sort_keys, sizeof(unsigned long long) * static_cast<std::size_t>(lp.AT.nnz)),
                        "cudaMalloc duplicate_columns sort keys");
    const int blocks_at_rows = (lp.AT.rows + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_duplicate_column_sort_keys<<<blocks_at_rows, GPU_PRESOLVE_THREADS>>>(
        sort_keys, lp.AT.rowPtr, lp.AT.colVal, lp.AT.rows);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_duplicate_column_sort_keys");
    thrust::sort_by_key(thrust::device_pointer_cast(sort_keys),
                        thrust::device_pointer_cast(sort_keys + lp.AT.nnz),
                        thrust::device_pointer_cast(lp.AT.nzVal));
    throw_if_cuda_error(cudaGetLastError(), "thrust duplicate_columns sort AT rows");
    const int blocks_at_nnz = (lp.AT.nnz + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_duplicate_column_unpack_keys<<<blocks_at_nnz, GPU_PRESOLVE_THREADS>>>(
        lp.AT.colVal, sort_keys, lp.AT.nnz);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_duplicate_column_unpack_keys");
    throw_if_cuda_error(cudaDeviceSynchronize(), "duplicate_columns sort AT synchronize");
    cudaFree(sort_keys);
    profile_stage("sort_AT_rows", sort_at_start);
  }

  const std::size_t n_size = static_cast<std::size_t>(n);
  const std::size_t m_size = static_cast<std::size_t>(m);
  std::size_t scratch_bytes = 0;
  const auto col_hash_offset = reserve_scratch<unsigned long long>(scratch_bytes, n_size);
  const auto sorted_cols_offset = reserve_scratch<std::int32_t>(scratch_bytes, n_size);
  const auto status_flag_offset = reserve_scratch<std::int32_t>(scratch_bytes, 3);
  const auto scale_guard_group_count_offset =
      reserve_scratch<std::int32_t>(scratch_bytes, 1);
  const auto col_delete_offset = reserve_scratch<std::uint8_t>(scratch_bytes, n_size);
  const auto fixed_mask_offset = reserve_scratch<std::uint8_t>(scratch_bytes, n_size);
  const auto fixed_val_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_to_offset = reserve_scratch<std::int32_t>(scratch_bytes, n_size);
  const auto merge_ratio_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_from_l_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_from_u_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_to_l_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto merge_to_u_offset = reserve_scratch<double>(scratch_bytes, n_size);
  const auto row_shift_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto obj_delta_offset = reserve_scratch<double>(scratch_bytes, 1);
  void* scratch_storage = nullptr;
  throw_if_cuda_error(cudaMalloc(&scratch_storage, scratch_bytes),
                      "cudaMalloc duplicate_columns scratch storage");
  auto* col_hash = scratch_at<unsigned long long>(scratch_storage, col_hash_offset);
  auto* sorted_cols = scratch_at<std::int32_t>(scratch_storage, sorted_cols_offset);
  auto* status_flag = scratch_at<std::int32_t>(scratch_storage, status_flag_offset);
  auto* scale_guard_group_count =
      scratch_at<std::int32_t>(scratch_storage, scale_guard_group_count_offset);
  auto* col_delete = scratch_at<std::uint8_t>(scratch_storage, col_delete_offset);
  auto* fixed_mask = scratch_at<std::uint8_t>(scratch_storage, fixed_mask_offset);
  auto* fixed_val = scratch_at<double>(scratch_storage, fixed_val_offset);
  auto* merge_to = scratch_at<std::int32_t>(scratch_storage, merge_to_offset);
  auto* merge_ratio = scratch_at<double>(scratch_storage, merge_ratio_offset);
  auto* merge_from_l = scratch_at<double>(scratch_storage, merge_from_l_offset);
  auto* merge_from_u = scratch_at<double>(scratch_storage, merge_from_u_offset);
  auto* merge_to_l = scratch_at<double>(scratch_storage, merge_to_l_offset);
  auto* merge_to_u = scratch_at<double>(scratch_storage, merge_to_u_offset);
  auto* row_shift = scratch_at<double>(scratch_storage, row_shift_offset);
  auto* obj_delta_device = scratch_at<double>(scratch_storage, obj_delta_offset);
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t) * 3), "cudaMemset duplicate_columns status_flag");
  throw_if_cuda_error(
      cudaMemset(scale_guard_group_count, 0, sizeof(std::int32_t)),
      "cudaMemset duplicate_columns scale_guard_group_count");
  throw_if_cuda_error(cudaMemset(col_delete, 0, static_cast<std::size_t>(n)), "cudaMemset duplicate_columns col_delete");
  throw_if_cuda_error(cudaMemset(fixed_mask, 0, static_cast<std::size_t>(n)), "cudaMemset duplicate_columns fixed_mask");
  throw_if_cuda_error(cudaMemset(merge_to, 0, sizeof(std::int32_t) * static_cast<std::size_t>(n)),
                      "cudaMemset duplicate_columns merge_to");
  throw_if_cuda_error(cudaMemset(row_shift, 0, sizeof(double) * static_cast<std::size_t>(m)), "cudaMemset duplicate_columns row_shift");
  throw_if_cuda_error(cudaMemset(obj_delta_device, 0, sizeof(double)), "cudaMemset duplicate_columns obj_delta");

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_duplicate_column_hashes<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      col_hash,
      plan.keep_col_mask,
      plan.keep_row_mask,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.zero_tol,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_duplicate_column_hashes");
  throw_if_cuda_error(cudaDeviceSynchronize(), "duplicate_columns hash synchronize");
  profile_stage("hash", rule_start);

  const auto sort_start = std::chrono::steady_clock::now();
  auto col_hash_begin = thrust::device_pointer_cast(col_hash);
  auto sorted_cols_begin = thrust::device_pointer_cast(sorted_cols);
  thrust::sequence(sorted_cols_begin, sorted_cols_begin + n, std::int32_t{0});
  auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(col_hash_begin, sorted_cols_begin));
  thrust::sort(zipped_begin, zipped_begin + n, HashColLess{});
  throw_if_cuda_error(cudaGetLastError(), "thrust duplicate_columns sort hashes");
  profile_stage("device_sort", sort_start);

  const double coeff_tol = fmax(pparams.zero_tol, pparams.bound_tol);
  const double obj_tol = fmax(pparams.zero_tol, pparams.bound_tol);
  const bool use_scale_guard_gate =
      isfinite(pparams.duplicate_columns_max_abs_ratio) &&
      pparams.duplicate_columns_scale_guard_min_groups > 1;
  if (use_scale_guard_gate) {
    _kernel_count_small_high_ratio_parallel_groups<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        scale_guard_group_count,
        col_hash,
        sorted_cols,
        plan.keep_col_mask,
        plan.keep_row_mask,
        plan.new_c,
        lp.AT.rowPtr,
        lp.AT.colVal,
        lp.AT.nzVal,
        pparams.zero_tol,
        coeff_tol,
        obj_tol,
        pparams.duplicate_columns_max_abs_ratio,
        pparams.duplicate_columns_scale_guard_max_group_size,
        pparams.duplicate_columns_scale_guard_min_groups,
        n);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_count_small_high_ratio_parallel_groups");
  }
  const auto groups_start = std::chrono::steady_clock::now();
  _kernel_duplicate_column_groups<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      col_delete,
      fixed_mask,
      fixed_val,
      merge_to,
      merge_ratio,
      merge_from_l,
      merge_from_u,
      merge_to_l,
      merge_to_u,
      col_hash,
      sorted_cols,
      plan.keep_col_mask,
      plan.keep_row_mask,
      plan.new_c,
      plan.new_l,
      plan.new_u,
      lp.AT.rowPtr,
      lp.AT.colVal,
      lp.AT.nzVal,
      pparams.zero_tol,
      coeff_tol,
      obj_tol,
      pparams.duplicate_columns_max_abs_ratio,
      use_scale_guard_gate ? scale_guard_group_count : nullptr,
      pparams.duplicate_columns_scale_guard_min_groups,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_duplicate_column_groups");
  profile_stage("groups", groups_start);

  const auto status_start = std::chrono::steady_clock::now();
  std::int32_t status[3] = {0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                      "cudaMemcpy duplicate_columns status");
  profile_stage("status", status_start);
  if (status[0] != 0) {
    plan.has_unbounded = true;
  } else if (status[1] != 0 || status[2] != 0) {
    const auto tape_start = std::chrono::steady_clock::now();
    if (status[1] != 0 && pparams.record_postsolve_tape) {
      const auto col_delete_begin = thrust::device_pointer_cast(col_delete);
      const std::int32_t merge_count = static_cast<std::int32_t>(
          thrust::count(col_delete_begin, col_delete_begin + n, std::uint8_t{1}));
      std::vector<DuplicateColumnTapeRecord> records(static_cast<std::size_t>(merge_count));
      DuplicateColumnTapeRecord* packed_records = nullptr;
      if (merge_count > 0) {
        throw_if_cuda_error(
            cudaMalloc(&packed_records,
                       sizeof(DuplicateColumnTapeRecord) * static_cast<std::size_t>(merge_count)),
            "cudaMalloc packed duplicate-column tape");
        throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)),
                            "cudaMemset packed duplicate-column count");
        _kernel_pack_duplicate_column_tape<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
            packed_records, status_flag, col_delete, merge_to, merge_ratio,
            merge_from_l, merge_from_u, merge_to_l, merge_to_u, n);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_duplicate_column_tape");
        throw_if_cuda_error(
            cudaMemcpy(records.data(), packed_records,
                       sizeof(DuplicateColumnTapeRecord) * records.size(),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy packed duplicate-column tape");
        cudaFree(packed_records);
      }
      std::sort(records.begin(), records.end(),
                [](const DuplicateColumnTapeRecord& lhs, const DuplicateColumnTapeRecord& rhs) {
                  return lhs.deleted_col < rhs.deleted_col;
                });
      plan.tape.types.reserve(plan.tape.types.size() + records.size());
      plan.tape.indices.reserve(plan.tape.indices.size() + 2 * records.size());
      plan.tape.vals.reserve(plan.tape.vals.size() + 5 * records.size());
      plan.tape.index_starts.reserve(plan.tape.index_starts.size() + records.size());
      plan.tape.value_starts.reserve(plan.tape.value_starts.size() + records.size());
      plan.tape.dual_modes.reserve(plan.tape.dual_modes.size() + records.size());
      for (const DuplicateColumnTapeRecord& record : records) {
        const std::int32_t indices[] = {record.deleted_col, record.kept_col};
        const double vals[] = {record.ratio,
                               record.deleted_l,
                               record.deleted_u,
                               record.kept_l,
                               record.kept_u};
        append_postsolve_record(
            plan.tape,
            PostsolveReductionType::DuplicateColumn,
            indices,
            2,
            vals,
            5,
            PostsolveDualMode::Minimal);
      }
    }
    if (status[2] != 0) {
      if (pparams.record_postsolve_tape) {
        append_compact_fixed_col_tape_from_device(&plan.tape,
                                                  fixed_mask,
                                                  nullptr,
                                                  fixed_val,
                                                  plan.keep_row_mask,
                                                  plan.new_c,
                                                  lp.AT,
                                                  n,
                                                  "duplicate_columns fixed-col tape");
      }
      _kernel_duplicate_column_fixed_row_shift<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
          row_shift,
          fixed_mask,
          fixed_val,
          plan.keep_row_mask,
          lp.AT.rowPtr,
          lp.AT.colVal,
          lp.AT.nzVal,
          n);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_duplicate_column_fixed_row_shift");
      plan.has_row_action = true;
    }
    profile_stage("tape", tape_start);
    const auto apply_start = std::chrono::steady_clock::now();
    const std::int32_t max_mn = m > n ? m : n;
    const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    _kernel_apply_duplicate_columns<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
        plan.keep_col_mask,
        plan.new_AL,
        plan.new_AU,
        plan.new_l,
        plan.new_u,
        obj_delta_device,
        plan.new_c,
        col_delete,
        fixed_mask,
        fixed_val,
        row_shift,
        m,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_duplicate_columns");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_duplicate_columns synchronize");
    double obj_delta = 0.0;
    throw_if_cuda_error(cudaMemcpy(&obj_delta, obj_delta_device, sizeof(double), cudaMemcpyDeviceToHost),
                        "cudaMemcpy duplicate_columns obj_delta");
    profile_stage("apply_delete", apply_start);
    plan.obj_constant_delta += obj_delta;
    plan.has_col_action = true;
    plan.has_change = true;
  }

  cudaFree(scratch_storage);
  profile_stage("total", rule_start);
}

}  // namespace gpu_presolver::presolve
