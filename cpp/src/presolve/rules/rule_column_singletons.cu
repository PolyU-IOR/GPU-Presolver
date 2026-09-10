#include "../presolve_helpers.hpp"
#include "gpu_presolver/presolve/rules/rule_column_singletons.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {
namespace {

using detail::throw_if_cuda_error;
using detail::append_postsolve_record;

constexpr int GPU_PRESOLVE_THREADS = 256;
constexpr std::uint8_t SINGLETON_COL_INEQ_NO_ACTION = 0;
constexpr std::uint8_t SINGLETON_COL_INEQ_ELIMINATE = 1;
constexpr std::uint8_t SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS = 2;
constexpr std::uint8_t SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS = 3;
constexpr std::uint8_t SINGLETON_COL_RULE_EQ = 1;
constexpr std::uint8_t SINGLETON_COL_RULE_DUAL_INFER = 2;
constexpr std::int32_t SINGLETON_ACTIVITY_LONG_ROW_THRESHOLD = 1024;

template <class T>
cudaError_t column_singletons_cuda_malloc(T** pointer, std::size_t bytes) {
  return cudaMallocAsync(reinterpret_cast<void**>(pointer), bytes, nullptr);
}

cudaError_t column_singletons_cuda_free(void* pointer) {
  return pointer == nullptr ? cudaSuccess : cudaFreeAsync(pointer, nullptr);
}

template <typename T>
void free_device_pointer(T*& pointer) noexcept {
  T* allocation = pointer;
  pointer = nullptr;
  if (allocation != nullptr) {
    column_singletons_cuda_free(allocation);
  }
}

template <typename T>
void allocate_device_pointer(T*& pointer,
                             std::size_t count,
                             const char* context) {
  pointer = nullptr;
  void* allocation = nullptr;
  const cudaError_t status = column_singletons_cuda_malloc(&allocation, sizeof(T) * count);
  if (status != cudaSuccess) {
    if (allocation != nullptr) {
      column_singletons_cuda_free(allocation);
    }
    throw_if_cuda_error(status, context);
  }
  pointer = static_cast<T*>(allocation);
}

struct ColumnSingletonsWorkspace {
  std::int32_t* status_flag = nullptr;
  std::int32_t* row_owner = nullptr;
  std::uint8_t* col_delete = nullptr;
  std::uint8_t* row_delete = nullptr;
  std::uint8_t* row_lhs_change = nullptr;
  std::uint8_t* row_rhs_change = nullptr;
  std::int32_t* pair_row = nullptr;
  double* chosen_side = nullptr;
  double* c_new = nullptr;
  double* AL_new = nullptr;
  double* AU_new = nullptr;
  double* obj_contrib = nullptr;
  double* activity_finite_min = nullptr;
  double* activity_finite_max = nullptr;
  std::int32_t* activity_min_neg_inf = nullptr;
  std::int32_t* activity_min_pos_inf = nullptr;
  std::int32_t* activity_max_neg_inf = nullptr;
  std::int32_t* activity_max_pos_inf = nullptr;
  std::uint8_t* activity_invalid = nullptr;
  std::uint8_t* activity_row_needed = nullptr;
  std::int32_t* activity_long_rows = nullptr;
  std::int32_t* activity_long_row_count = nullptr;
  std::size_t row_capacity = 0;
  std::size_t col_capacity = 0;

  ColumnSingletonsWorkspace() = default;
  ColumnSingletonsWorkspace(const ColumnSingletonsWorkspace&) = delete;
  ColumnSingletonsWorkspace& operator=(const ColumnSingletonsWorkspace&) = delete;

  void release_fixed() noexcept {
    free_device_pointer(status_flag);
    free_device_pointer(activity_long_row_count);
  }

  void release_rows() noexcept {
    row_capacity = 0;
    free_device_pointer(row_owner);
    free_device_pointer(row_delete);
    free_device_pointer(row_lhs_change);
    free_device_pointer(row_rhs_change);
    free_device_pointer(AL_new);
    free_device_pointer(AU_new);
    free_device_pointer(activity_finite_min);
    free_device_pointer(activity_finite_max);
    free_device_pointer(activity_min_neg_inf);
    free_device_pointer(activity_min_pos_inf);
    free_device_pointer(activity_max_neg_inf);
    free_device_pointer(activity_max_pos_inf);
    free_device_pointer(activity_invalid);
    free_device_pointer(activity_row_needed);
    free_device_pointer(activity_long_rows);
  }

  void release_cols() noexcept {
    col_capacity = 0;
    free_device_pointer(col_delete);
    free_device_pointer(pair_row);
    free_device_pointer(chosen_side);
    free_device_pointer(c_new);
    free_device_pointer(obj_contrib);
  }

  void release() noexcept {
    release_fixed();
    release_rows();
    release_cols();
  }

  void ensure(std::int32_t m, std::int32_t n) {
    if (status_flag == nullptr || activity_long_row_count == nullptr) {
      release_fixed();
      try {
        allocate_device_pointer(status_flag, 3, "cudaMalloc singleton workspace status");
        allocate_device_pointer(activity_long_row_count,
                                1,
                                "cudaMalloc singleton workspace long-row count");
      } catch (...) {
        release_fixed();
        throw;
      }
    }
    const std::size_t required_rows = static_cast<std::size_t>(m);
    if (required_rows > row_capacity) {
      release_rows();
      try {
        allocate_device_pointer(row_owner,
                                required_rows,
                                "cudaMalloc singleton workspace row_owner");
        allocate_device_pointer(row_delete,
                                required_rows,
                                "cudaMalloc singleton workspace row_delete");
        allocate_device_pointer(row_lhs_change,
                                required_rows,
                                "cudaMalloc singleton workspace row_lhs_change");
        allocate_device_pointer(row_rhs_change,
                                required_rows,
                                "cudaMalloc singleton workspace row_rhs_change");
        allocate_device_pointer(AL_new,
                                required_rows,
                                "cudaMalloc singleton workspace AL_new");
        allocate_device_pointer(AU_new,
                                required_rows,
                                "cudaMalloc singleton workspace AU_new");
        allocate_device_pointer(activity_finite_min,
                                required_rows,
                                "cudaMalloc singleton workspace activity finite min");
        allocate_device_pointer(activity_finite_max,
                                required_rows,
                                "cudaMalloc singleton workspace activity finite max");
        allocate_device_pointer(activity_min_neg_inf,
                                required_rows,
                                "cudaMalloc singleton workspace activity min -inf");
        allocate_device_pointer(activity_min_pos_inf,
                                required_rows,
                                "cudaMalloc singleton workspace activity min +inf");
        allocate_device_pointer(activity_max_neg_inf,
                                required_rows,
                                "cudaMalloc singleton workspace activity max -inf");
        allocate_device_pointer(activity_max_pos_inf,
                                required_rows,
                                "cudaMalloc singleton workspace activity max +inf");
        allocate_device_pointer(activity_invalid,
                                required_rows,
                                "cudaMalloc singleton workspace activity invalid");
        allocate_device_pointer(activity_row_needed,
                                required_rows,
                                "cudaMalloc singleton workspace activity rows");
        allocate_device_pointer(activity_long_rows,
                                required_rows,
                                "cudaMalloc singleton workspace long rows");
        row_capacity = required_rows;
      } catch (...) {
        release_rows();
        throw;
      }
    }
    const std::size_t required_cols = static_cast<std::size_t>(n);
    if (required_cols > col_capacity) {
      release_cols();
      try {
        allocate_device_pointer(col_delete,
                                required_cols,
                                "cudaMalloc singleton workspace col_delete");
        allocate_device_pointer(pair_row,
                                required_cols,
                                "cudaMalloc singleton workspace pair_row");
        allocate_device_pointer(chosen_side,
                                required_cols,
                                "cudaMalloc singleton workspace chosen_side");
        allocate_device_pointer(c_new,
                                required_cols,
                                "cudaMalloc singleton workspace c_new");
        allocate_device_pointer(obj_contrib,
                                required_cols,
                                "cudaMalloc singleton workspace objective contribution");
        col_capacity = required_cols;
      } catch (...) {
        release_cols();
        throw;
      }
    }
  }

  ~ColumnSingletonsWorkspace() { release(); }
};

struct DeviceScratchBuffer {
  void* data = nullptr;
  std::size_t capacity = 0;

  DeviceScratchBuffer() = default;
  DeviceScratchBuffer(const DeviceScratchBuffer&) = delete;
  DeviceScratchBuffer& operator=(const DeviceScratchBuffer&) = delete;

  void release() noexcept {
    capacity = 0;
    void* allocation = data;
    data = nullptr;
    if (allocation != nullptr) {
      column_singletons_cuda_free(allocation);
    }
  }

  void ensure(std::size_t bytes, const char* context) {
    if (bytes <= capacity) {
      return;
    }
    release();
    void* allocation = nullptr;
    const cudaError_t status = column_singletons_cuda_malloc(&allocation, bytes);
    if (status != cudaSuccess) {
      if (allocation != nullptr) {
        column_singletons_cuda_free(allocation);
      }
      data = nullptr;
      capacity = 0;
      throw_if_cuda_error(status, context);
    }
    data = allocation;
    capacity = bytes;
  }

  ~DeviceScratchBuffer() { release(); }
};

struct ColumnSingletonsTlsCache {
  int owner_device = -1;
  ColumnSingletonsWorkspace workspace;
  DeviceScratchBuffer inclusive_scratch;
  DeviceScratchBuffer reduce_scratch;
  DeviceScratchBuffer reduce_result_scratch;

  ColumnSingletonsTlsCache() = default;
  ColumnSingletonsTlsCache(const ColumnSingletonsTlsCache&) = delete;
  ColumnSingletonsTlsCache& operator=(const ColumnSingletonsTlsCache&) = delete;

  void release_allocations() noexcept {
    workspace.release();
    inclusive_scratch.release();
    reduce_scratch.release();
    reduce_result_scratch.release();
  }

  void release() noexcept {
    const int allocation_device = owner_device;
    owner_device = -1;

    int current_device = -1;
    const bool have_current_device = cudaGetDevice(&current_device) == cudaSuccess;
    bool restore_current_device = false;
    if (allocation_device >= 0 &&
        (!have_current_device || current_device != allocation_device)) {
      restore_current_device = cudaSetDevice(allocation_device) == cudaSuccess &&
                               have_current_device;
    }

    release_allocations();

    if (restore_current_device) {
      cudaSetDevice(current_device);
    }
  }

  void bind_to_device(int device) noexcept {
    if (owner_device == device) {
      return;
    }
    if (owner_device >= 0) {
      release();
    }
    owner_device = device;
  }

  ~ColumnSingletonsTlsCache() { release(); }
};

ColumnSingletonsTlsCache& column_singletons_tls_cache_storage() noexcept {
  thread_local ColumnSingletonsTlsCache cache;
  return cache;
}

ColumnSingletonsTlsCache& column_singletons_tls_cache() {
  int current_device = -1;
  throw_if_cuda_error(cudaGetDevice(&current_device),
                      "cudaGetDevice singleton workspace");
  ColumnSingletonsTlsCache& cache = column_singletons_tls_cache_storage();
  cache.bind_to_device(current_device);
  return cache;
}

ColumnSingletonsWorkspace& column_singletons_workspace() {
  return column_singletons_tls_cache().workspace;
}

DeviceScratchBuffer& column_singletons_inclusive_scratch() {
  return column_singletons_tls_cache().inclusive_scratch;
}

DeviceScratchBuffer& column_singletons_reduce_scratch() {
  return column_singletons_tls_cache().reduce_scratch;
}

DeviceScratchBuffer& column_singletons_reduce_result_scratch() {
  return column_singletons_tls_cache().reduce_result_scratch;
}

void inclusive_scan_i32(std::int32_t* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return;
  }
  DeviceScratchBuffer& scratch = column_singletons_inclusive_scratch();
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(temp_storage, temp_bytes, values, values, n),
                      context);
  scratch.ensure(temp_bytes, context);
  throw_if_cuda_error(cub::DeviceScan::InclusiveSum(scratch.data, temp_bytes, values, values, n),
                      context);
}

double sum_device_double(const double* values, std::int32_t n, const char* context) {
  if (n <= 0) {
    return 0.0;
  }
  DeviceScratchBuffer& scratch = column_singletons_reduce_scratch();
  DeviceScratchBuffer& result_scratch = column_singletons_reduce_result_scratch();
  void* temp_storage = nullptr;
  std::size_t temp_bytes = 0;
  double result = 0.0;
  result_scratch.ensure(sizeof(double), context);
  double* result_device = static_cast<double*>(result_scratch.data);
  throw_if_cuda_error(cub::DeviceReduce::Sum(temp_storage, temp_bytes, values, result_device, n),
                      context);
  scratch.ensure(temp_bytes, context);
  throw_if_cuda_error(cub::DeviceReduce::Sum(scratch.data, temp_bytes, values, result_device, n),
                      context);
  throw_if_cuda_error(cudaMemcpy(&result, result_device, sizeof(double), cudaMemcpyDeviceToHost),
                      context);
  return result;
}

void append_eq_to_ineq_tape_from_device(PostsolveTape& tape,
                                        const std::uint8_t* row_lhs_change,
                                        const std::uint8_t* row_rhs_change,
                                        std::int32_t m,
                                        const char* context) {
  std::vector<std::uint8_t> host_lhs(static_cast<std::size_t>(m));
  std::vector<std::uint8_t> host_rhs(static_cast<std::size_t>(m));
  throw_if_cuda_error(cudaMemcpy(host_lhs.data(), row_lhs_change, static_cast<std::size_t>(m),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_rhs.data(), row_rhs_change, static_cast<std::size_t>(m),
                                 cudaMemcpyDeviceToHost),
                      context);
  for (std::int32_t row = 0; row < m; ++row) {
    if (host_lhs[static_cast<std::size_t>(row)] == std::uint8_t{0} &&
        host_rhs[static_cast<std::size_t>(row)] == std::uint8_t{0}) {
      continue;
    }
    append_postsolve_record(tape,
                            PostsolveReductionType::EqToIneq,
                            {row},
                            {0.0},
                            PostsolveDualMode::Minimal);
  }
}

__global__ void _kernel_fill_i32(std::int32_t* values, std::int32_t value, std::int32_t n) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    values[i] = value;
  }
}

__device__ bool _column_singleton_direct_unbounded(double cj,
                                                double a,
                                                double lhs,
                                                double rhs,
                                                double lb,
                                                double ub,
                                                double zero_tol) {
  return ((cj > zero_tol && a > zero_tol && !isfinite(lhs) && !isfinite(lb)) ||
          (cj > zero_tol && a < -zero_tol && !isfinite(rhs) && !isfinite(lb)) ||
          (cj < -zero_tol && a < -zero_tol && !isfinite(lhs) && !isfinite(ub)) ||
          (cj < -zero_tol && a > zero_tol && !isfinite(rhs) && !isfinite(ub)));
}

__device__ bool _column_singleton_eq_free_from_above(double implied_ub, double ub, double tol) {
  return !isfinite(ub) || implied_ub <= ub + tol;
}

__device__ bool _column_singleton_eq_free_from_below(double implied_lb, double lb, double tol) {
  return !isfinite(lb) || implied_lb >= lb - tol;
}

__device__ void _apply_column_singleton_eq_one_sided_row_update(double* AL,
                                                             double* AU,
                                                             std::int32_t row,
                                                             double a,
                                                             double bound_val,
                                                             bool keep_lower_part) {
  const double shifted_rhs = AU[row] - a * bound_val;
  if (keep_lower_part) {
    AL[row] = shifted_rhs;
    AU[row] = INFINITY;
  } else {
    AL[row] = -INFINITY;
    AU[row] = shifted_rhs;
  }
}

__global__ void _kernel_singleton_row_activity_cache(
    double* finite_min,
    double* finite_max,
    std::int32_t* min_neg_inf,
    std::int32_t* min_pos_inf,
    std::int32_t* max_neg_inf,
    std::int32_t* max_pos_inf,
    std::uint8_t* invalid,
    const std::uint8_t* activity_row_needed,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const double* l,
    const double* u,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    std::int32_t* long_rows,
    std::int32_t* long_row_count,
    std::int32_t m) {
  const std::int32_t row =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  if (activity_row_needed[row] == std::uint8_t{0}) {
    return;
  }

  double row_finite_min = 0.0;
  double row_finite_max = 0.0;
  std::int32_t row_min_neg_inf = 0;
  std::int32_t row_min_pos_inf = 0;
  std::int32_t row_max_neg_inf = 0;
  std::int32_t row_max_pos_inf = 0;
  bool row_invalid = false;
  if (row_ptr[row + 1] - row_ptr[row] > SINGLETON_ACTIVITY_LONG_ROW_THRESHOLD) {
    const std::int32_t slot = atomicAdd(long_row_count, 1);
    long_rows[slot] = row;
    return;
  }
  if (keep_row[row] != std::uint8_t{0}) {
    for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
      const std::int32_t col = col_val[p];
      if (keep_col[col] == std::uint8_t{0}) {
        continue;
      }
      const double a = nz_val[p];
      const double term_min = a >= 0.0 ? a * l[col] : a * u[col];
      const double term_max = a >= 0.0 ? a * u[col] : a * l[col];
      if (isnan(term_min) || isnan(term_max)) {
        row_invalid = true;
        continue;
      }
      if (isinf(term_min)) {
        if (term_min < 0.0) {
          ++row_min_neg_inf;
        } else {
          ++row_min_pos_inf;
        }
      } else {
        row_finite_min += term_min;
      }
      if (isinf(term_max)) {
        if (term_max < 0.0) {
          ++row_max_neg_inf;
        } else {
          ++row_max_pos_inf;
        }
      } else {
        row_finite_max += term_max;
      }
    }
  }
  finite_min[row] = row_finite_min;
  finite_max[row] = row_finite_max;
  min_neg_inf[row] = row_min_neg_inf;
  min_pos_inf[row] = row_min_pos_inf;
  max_neg_inf[row] = row_max_neg_inf;
  max_pos_inf[row] = row_max_pos_inf;
  invalid[row] = row_invalid ? std::uint8_t{1} : std::uint8_t{0};
}

__global__ void _kernel_mark_singleton_activity_rows(
    std::uint8_t* activity_row_needed,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const std::uint8_t* column_singleton_mask,
    const std::int32_t* column_singleton_row,
    std::int32_t m,
    std::int32_t n) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || keep_col[col] == std::uint8_t{0} ||
      column_singleton_mask[col] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t row = column_singleton_row[col];
  if (row >= 0 && row < m && keep_row[row] != std::uint8_t{0}) {
    activity_row_needed[row] = std::uint8_t{1};
  }
}

__global__ void _kernel_singleton_long_row_activity_cache(
    double* finite_min,
    double* finite_max,
    std::int32_t* min_neg_inf,
    std::int32_t* min_pos_inf,
    std::int32_t* max_neg_inf,
    std::int32_t* max_pos_inf,
    std::uint8_t* invalid,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const double* l,
    const double* u,
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
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  __shared__ double shared_finite_min[GPU_PRESOLVE_THREADS];
  __shared__ double shared_finite_max[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_min_neg_inf[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_min_pos_inf[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_max_neg_inf[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_max_pos_inf[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t shared_invalid[GPU_PRESOLVE_THREADS];

  double thread_finite_min = 0.0;
  double thread_finite_max = 0.0;
  std::int32_t thread_min_neg_inf = 0;
  std::int32_t thread_min_pos_inf = 0;
  std::int32_t thread_max_neg_inf = 0;
  std::int32_t thread_max_pos_inf = 0;
  std::int32_t thread_invalid = 0;
  if (keep_row[row] != std::uint8_t{0}) {
    for (std::int32_t p = row_ptr[row] + tid; p < row_ptr[row + 1];
         p += static_cast<std::int32_t>(blockDim.x)) {
      const std::int32_t col = col_val[p];
      if (keep_col[col] == std::uint8_t{0}) {
        continue;
      }
      const double a = nz_val[p];
      const double term_min = a >= 0.0 ? a * l[col] : a * u[col];
      const double term_max = a >= 0.0 ? a * u[col] : a * l[col];
      if (isnan(term_min) || isnan(term_max)) {
        thread_invalid = 1;
        continue;
      }
      if (isinf(term_min)) {
        term_min < 0.0 ? ++thread_min_neg_inf : ++thread_min_pos_inf;
      } else {
        thread_finite_min += term_min;
      }
      if (isinf(term_max)) {
        term_max < 0.0 ? ++thread_max_neg_inf : ++thread_max_pos_inf;
      } else {
        thread_finite_max += term_max;
      }
    }
  }
  shared_finite_min[tid] = thread_finite_min;
  shared_finite_max[tid] = thread_finite_max;
  shared_min_neg_inf[tid] = thread_min_neg_inf;
  shared_min_pos_inf[tid] = thread_min_pos_inf;
  shared_max_neg_inf[tid] = thread_max_neg_inf;
  shared_max_pos_inf[tid] = thread_max_pos_inf;
  shared_invalid[tid] = thread_invalid;
  __syncthreads();

  for (std::int32_t stride = GPU_PRESOLVE_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_finite_min[tid] += shared_finite_min[tid + stride];
      shared_finite_max[tid] += shared_finite_max[tid + stride];
      shared_min_neg_inf[tid] += shared_min_neg_inf[tid + stride];
      shared_min_pos_inf[tid] += shared_min_pos_inf[tid + stride];
      shared_max_neg_inf[tid] += shared_max_neg_inf[tid + stride];
      shared_max_pos_inf[tid] += shared_max_pos_inf[tid + stride];
      shared_invalid[tid] |= shared_invalid[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    finite_min[row] = shared_finite_min[0];
    finite_max[row] = shared_finite_max[0];
    min_neg_inf[row] = shared_min_neg_inf[0];
    min_pos_inf[row] = shared_min_pos_inf[0];
    max_neg_inf[row] = shared_max_neg_inf[0];
    max_pos_inf[row] = shared_max_pos_inf[0];
    invalid[row] = shared_invalid[0] != 0 ? std::uint8_t{1} : std::uint8_t{0};
  }
}

__device__ bool _column_singleton_activity_bounds_from_cache(
    std::int32_t row,
    std::int32_t excluded_col,
    double excluded_val,
    const double* l,
    const double* u,
    const double* finite_min,
    const double* finite_max,
    const std::int32_t* min_neg_inf,
    const std::int32_t* min_pos_inf,
    const std::int32_t* max_neg_inf,
    const std::int32_t* max_pos_inf,
    const std::uint8_t* invalid,
    double* rest_min,
    double* rest_max) {
  if (invalid[row] != std::uint8_t{0}) {
    return false;
  }

  const double term_min =
      excluded_val >= 0.0 ? excluded_val * l[excluded_col]
                          : excluded_val * u[excluded_col];
  const double term_max =
      excluded_val >= 0.0 ? excluded_val * u[excluded_col]
                          : excluded_val * l[excluded_col];
  if (isnan(term_min) || isnan(term_max)) {
    return false;
  }

  std::int32_t min_neg = min_neg_inf[row];
  std::int32_t min_pos = min_pos_inf[row];
  std::int32_t max_neg = max_neg_inf[row];
  std::int32_t max_pos = max_pos_inf[row];
  double min_value = finite_min[row];
  double max_value = finite_max[row];
  if (isinf(term_min)) {
    if (term_min < 0.0) {
      --min_neg;
    } else {
      --min_pos;
    }
  } else {
    min_value -= term_min;
  }
  if (isinf(term_max)) {
    if (term_max < 0.0) {
      --max_neg;
    } else {
      --max_pos;
    }
  } else {
    max_value -= term_max;
  }
  if (min_neg < 0 || min_pos < 0 || max_neg < 0 || max_pos < 0 ||
      (min_neg > 0 && min_pos > 0) || (max_neg > 0 && max_pos > 0)) {
    return false;
  }
  *rest_min = min_neg > 0 ? -INFINITY : (min_pos > 0 ? INFINITY : min_value);
  *rest_max = max_neg > 0 ? -INFINITY : (max_pos > 0 ? INFINITY : max_value);
  return !isnan(*rest_min) && !isnan(*rest_max);
}

__device__ bool _column_singleton_implied_free_from_above(double a,
                                                       double lhs,
                                                       double rhs,
                                                       double ub,
                                                       double rest_min,
                                                       double rest_max,
                                                       double tol) {
  if (!isfinite(ub)) {
    return true;
  }
  double implied_ub = INFINITY;
  if (a > 0.0 && isfinite(rhs)) {
    implied_ub = (rhs - rest_min) / a;
  } else if (a < 0.0 && isfinite(lhs)) {
    implied_ub = (lhs - rest_max) / a;
  }
  return implied_ub <= ub + tol;
}

__device__ bool _column_singleton_implied_free_from_below(double a,
                                                       double lhs,
                                                       double rhs,
                                                       double lb,
                                                       double rest_min,
                                                       double rest_max,
                                                       double tol) {
  if (!isfinite(lb)) {
    return true;
  }
  double implied_lb = -INFINITY;
  if (a > 0.0 && isfinite(lhs)) {
    implied_lb = (lhs - rest_max) / a;
  } else if (a < 0.0 && isfinite(rhs)) {
    implied_lb = (rhs - rest_min) / a;
  }
  return implied_lb >= lb - tol;
}

__device__ double _column_singleton_active_side(double cj,
                                             double a,
                                             double lhs,
                                             double rhs,
                                             double zero_tol) {
  if ((cj > zero_tol && a > 0.0) || (cj < -zero_tol && a < 0.0)) {
    return lhs;
  }
  if ((cj > zero_tol && a < 0.0) || (cj < -zero_tol && a > 0.0)) {
    return rhs;
  }
  return isfinite(lhs) ? lhs : rhs;
}

__device__ std::uint8_t _column_singleton_ineq_action(double cj,
                                                   double a,
                                                   double lhs,
                                                   double rhs,
                                                   bool impl_free_from_above,
                                                   bool impl_free_from_below,
                                                   double zero_tol,
                                                   double* action_side) {
  if (impl_free_from_above && impl_free_from_below) {
    *action_side = _column_singleton_active_side(cj, a, lhs, rhs, zero_tol);
    if (isfinite(*action_side)) {
      return SINGLETON_COL_INEQ_ELIMINATE;
    }
    return SINGLETON_COL_INEQ_NO_ACTION;
  }
  const bool tighten_lhs_to_rhs =
      ((cj < -zero_tol && a > 0.0 && impl_free_from_above) ||
       (cj > zero_tol && a < 0.0 && impl_free_from_below)) &&
      isfinite(rhs);
  if (tighten_lhs_to_rhs) {
    *action_side = rhs;
    return SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS;
  }
  const bool tighten_rhs_to_lhs =
      ((cj > zero_tol && a > 0.0 && impl_free_from_below) ||
       (cj < -zero_tol && a < 0.0 && impl_free_from_above)) &&
      isfinite(lhs);
  if (tighten_rhs_to_lhs) {
    *action_side = lhs;
    return SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS;
  }
  *action_side = 0.0;
  return SINGLETON_COL_INEQ_NO_ACTION;
}

__global__ void _kernel_column_singleton_row_owner(std::int32_t* status_flag,
                                                std::int32_t* row_owner,
                                                const std::uint8_t* keep_row,
                                                const std::uint8_t* keep_col,
                                                const std::uint8_t* singleton_mask,
                                                const std::int32_t* support_row,
                                                const double* support_val,
                                                const double* c_cur,
                                                const double* l_cur,
                                                const double* u_cur,
                                                const double* AL_cur,
                                                const double* AU_cur,
                                                const double* activity_finite_min,
                                                const double* activity_finite_max,
                                                const std::int32_t* activity_min_neg_inf,
                                                const std::int32_t* activity_min_pos_inf,
                                                const std::int32_t* activity_max_neg_inf,
                                                const std::int32_t* activity_max_pos_inf,
                                                const std::uint8_t* activity_invalid,
                                                const std::int32_t* row_ptr,
                                                const std::int32_t* col_val,
                                                const double* nz_val,
                                                double tol,
                                                double zero_tol,
                                                std::uint8_t rule_mode,
                                                std::int32_t m,
                                                std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    if (keep_col[j] == std::uint8_t{0} || singleton_mask[j] == std::uint8_t{0}) {
      return;
    }
    const std::int32_t row = support_row[j];
    if (row < 0 || row >= m || keep_row[row] == std::uint8_t{0}) {
      return;
    }
    const double a = support_val[j];
    if (fabs(a) <= zero_tol) {
      return;
    }

    const std::int32_t row_start = row_ptr[row];
    const std::int32_t row_stop = row_ptr[row + 1];
    std::int32_t live_row_nnz = 0;
    for (std::int32_t p = row_start; p < row_stop; ++p) {
      const std::int32_t col = col_val[p];
      if (keep_col[col] != std::uint8_t{0}) {
        ++live_row_nnz;
        if (live_row_nnz > 1) {
          break;
        }
      }
    }
    if (live_row_nnz <= 1) {
      return;
    }

    const double lhs = AL_cur[row];
    const double rhs = AU_cur[row];
    const double cj = c_cur[j];
    if (rule_mode != SINGLETON_COL_RULE_EQ &&
        _column_singleton_direct_unbounded(cj, a, lhs, rhs, l_cur[j], u_cur[j], zero_tol)) {
      atomicMax(&status_flag[0], 1);
      return;
    }

    double rest_min = 0.0;
    double rest_max = 0.0;
    if (!_column_singleton_activity_bounds_from_cache(
            row, j, a, l_cur, u_cur, activity_finite_min, activity_finite_max,
            activity_min_neg_inf, activity_min_pos_inf, activity_max_neg_inf,
            activity_max_pos_inf, activity_invalid, &rest_min, &rest_max)) {
      return;
    }

    const bool is_eq_row = isfinite(lhs) && isfinite(rhs) && fabs(lhs - rhs) <= tol;
    if (is_eq_row) {
      if (rule_mode == SINGLETON_COL_RULE_DUAL_INFER) {
        return;
      }
      const double x1 = (rhs - rest_min) / a;
      const double x2 = (rhs - rest_max) / a;
      const double implied_lb = fmin(x1, x2);
      const double implied_ub = fmax(x1, x2);
      const bool impl_free_from_above = _column_singleton_eq_free_from_above(implied_ub, u_cur[j], tol);
      const bool impl_free_from_below = _column_singleton_eq_free_from_below(implied_lb, l_cur[j], tol);
      if (!(impl_free_from_above || impl_free_from_below)) {
        return;
      }
    } else {
      if (rule_mode == SINGLETON_COL_RULE_EQ) {
        return;
      }
      const bool impl_free_from_above =
          _column_singleton_implied_free_from_above(a, lhs, rhs, u_cur[j], rest_min, rest_max, tol);
      const bool impl_free_from_below =
          _column_singleton_implied_free_from_below(a, lhs, rhs, l_cur[j], rest_min, rest_max, tol);
      if (rule_mode == SINGLETON_COL_RULE_DUAL_INFER) {
        double action_side = 0.0;
        if (_column_singleton_ineq_action(cj,
                                       a,
                                       lhs,
                                       rhs,
                                       impl_free_from_above,
                                       impl_free_from_below,
                                       zero_tol,
                                       &action_side) == SINGLETON_COL_INEQ_NO_ACTION) {
          return;
        }
      } else {
        double action_side = 0.0;
        if (_column_singleton_ineq_action(cj, a, lhs, rhs, impl_free_from_above, impl_free_from_below, zero_tol, &action_side) ==
            SINGLETON_COL_INEQ_NO_ACTION) {
          return;
        }
      }
    }
    atomicMin(&row_owner[row], j);
  }
}

__global__ void _kernel_process_column_singletons(std::int32_t* status_flag,
                                               std::uint8_t* row_delete,
                                               std::uint8_t* row_lhs_change,
                                               std::uint8_t* row_rhs_change,
                                               std::uint8_t* col_delete,
                                               std::int32_t* pair_row,
                                               double* chosen_side,
                                               double* AL_new,
                                               double* AU_new,
                                               double* obj_contrib,
                                               const std::int32_t* row_owner,
                                               const std::uint8_t* keep_row,
                                               const std::uint8_t* keep_col,
                                               const std::uint8_t* singleton_mask,
                                               const std::int32_t* support_row,
                                               const double* support_val,
                                               const double* c_cur,
                                               const double* l_cur,
                                               const double* u_cur,
                                               const double* activity_finite_min,
                                               const double* activity_finite_max,
                                               const std::int32_t* activity_min_neg_inf,
                                               const std::int32_t* activity_min_pos_inf,
                                               const std::int32_t* activity_max_neg_inf,
                                               const std::int32_t* activity_max_pos_inf,
                                               const std::uint8_t* activity_invalid,
                                               const std::int32_t* row_ptr,
                                               const std::int32_t* col_val,
                                               const double* nz_val,
                                               double tol,
                                               double zero_tol,
                                               std::uint8_t rule_mode,
                                               std::int32_t m,
                                               std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n) {
    if (keep_col[j] == std::uint8_t{0} || singleton_mask[j] == std::uint8_t{0}) {
      return;
    }
    const std::int32_t row = support_row[j];
    if (row < 0 || row >= m || keep_row[row] == std::uint8_t{0}) {
      return;
    }
    if (row_owner[row] != j) {
      return;
    }
    const double a = support_val[j];
    if (fabs(a) <= zero_tol) {
      return;
    }
    const double lhs = AL_new[row];
    const double rhs = AU_new[row];
    const double cj = c_cur[j];
    if (rule_mode != SINGLETON_COL_RULE_EQ &&
        _column_singleton_direct_unbounded(cj, a, lhs, rhs, l_cur[j], u_cur[j], zero_tol)) {
      atomicMax(&status_flag[0], 1);
      return;
    }

    double rest_min = 0.0;
    double rest_max = 0.0;
    if (!_column_singleton_activity_bounds_from_cache(
            row, j, a, l_cur, u_cur, activity_finite_min, activity_finite_max,
            activity_min_neg_inf, activity_min_pos_inf, activity_max_neg_inf,
            activity_max_pos_inf, activity_invalid, &rest_min, &rest_max)) {
      return;
    }

    const bool is_eq_row = isfinite(lhs) && isfinite(rhs) && fabs(lhs - rhs) <= tol;
    if (is_eq_row) {
      if (rule_mode == SINGLETON_COL_RULE_DUAL_INFER) {
        return;
      }
      const double x1 = (rhs - rest_min) / a;
      const double x2 = (rhs - rest_max) / a;
      const double implied_lb = fmin(x1, x2);
      const double implied_ub = fmax(x1, x2);
      const bool impl_free_from_above = _column_singleton_eq_free_from_above(implied_ub, u_cur[j], tol);
      const bool impl_free_from_below = _column_singleton_eq_free_from_below(implied_lb, l_cur[j], tol);
      if (!(impl_free_from_above || impl_free_from_below)) {
        return;
      }
      chosen_side[j] = rhs;
      obj_contrib[j] = cj * rhs / a;
      if (impl_free_from_above && impl_free_from_below) {
        row_delete[row] = std::uint8_t{1};
      } else if (impl_free_from_above) {
        _apply_column_singleton_eq_one_sided_row_update(AL_new, AU_new, row, a, l_cur[j], a < 0.0);
      } else {
        _apply_column_singleton_eq_one_sided_row_update(AL_new, AU_new, row, a, u_cur[j], a > 0.0);
      }
      col_delete[j] = std::uint8_t{1};
      pair_row[j] = row;
      atomicMax(&status_flag[1], 1);
      return;
    }

    if (rule_mode == SINGLETON_COL_RULE_EQ) {
      return;
    }

    const bool impl_free_from_above =
        _column_singleton_implied_free_from_above(a, lhs, rhs, u_cur[j], rest_min, rest_max, tol);
    const bool impl_free_from_below =
        _column_singleton_implied_free_from_below(a, lhs, rhs, l_cur[j], rest_min, rest_max, tol);
    if (rule_mode == SINGLETON_COL_RULE_DUAL_INFER) {
      double action_side = 0.0;
      const std::uint8_t action =
          _column_singleton_ineq_action(cj,
                                     a,
                                     lhs,
                                     rhs,
                                     impl_free_from_above,
                                     impl_free_from_below,
                                     zero_tol,
                                     &action_side);
      if (action == SINGLETON_COL_INEQ_NO_ACTION) {
        return;
      }
      if (action == SINGLETON_COL_INEQ_ELIMINATE) {
        if (!isfinite(lhs) || fabs(lhs - action_side) > tol) {
          row_lhs_change[row] = std::uint8_t{1};
        }
        if (!isfinite(rhs) || fabs(rhs - action_side) > tol) {
          row_rhs_change[row] = std::uint8_t{1};
        }
        AL_new[row] = action_side;
        AU_new[row] = action_side;
      } else if (action == SINGLETON_COL_INEQ_TIGHTEN_LHS_TO_RHS) {
        row_lhs_change[row] = std::uint8_t{1};
        AL_new[row] = rhs;
      } else if (action == SINGLETON_COL_INEQ_TIGHTEN_RHS_TO_LHS) {
        row_rhs_change[row] = std::uint8_t{1};
        AU_new[row] = lhs;
      }
      atomicMax(&status_flag[2], 1);
    }
  }
}

__global__ void _kernel_apply_singleton_objective_updates_deterministic(
    double* c_new,
    const std::uint8_t* keep_col,
    const std::uint8_t* col_delete,
    const std::int32_t* row_owner,
    const double* support_val,
    const double* c_cur,
    const std::int32_t* at_row_ptr,
    const std::int32_t* at_col_val,
    const double* at_nz_val,
    std::int32_t n) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= n || keep_col[col] == std::uint8_t{0} ||
      col_delete[col] != std::uint8_t{0}) {
    return;
  }

  double updated = c_new[col];
  for (std::int32_t p = at_row_ptr[col]; p < at_row_ptr[col + 1]; ++p) {
    const std::int32_t row = at_col_val[p];
    const std::int32_t owner = row_owner[row];
    if (owner < 0 || owner >= n || col_delete[owner] == std::uint8_t{0}) {
      continue;
    }
    updated += -(c_cur[owner] * at_nz_val[p] / support_val[owner]);
  }
  c_new[col] = updated;
}

__global__ void _kernel_count_column_singleton_support_eq(
    std::int32_t* counts,
    const std::uint8_t* keep_row,
    const std::uint8_t* keep_col,
    const std::uint8_t* singleton_mask,
    const std::int32_t* support_row,
    const double* AL,
    const double* AU,
    double tol,
    std::int32_t n) {
  const std::int32_t j = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j < n && keep_col[j] != std::uint8_t{0} && singleton_mask[j] != std::uint8_t{0}) {
    const std::int32_t row = support_row[j];
    if (row >= 0 && keep_row[row] != std::uint8_t{0}) {
      atomicAdd(&counts[0], 1);
      const double lhs = AL[row];
      const double rhs = AU[row];
      if (isfinite(lhs) && isfinite(rhs) && fabs(rhs - lhs) <= tol) {
        atomicAdd(&counts[1], 1);
      }
    }
  }
}

__global__ void _kernel_apply_column_singletons(std::uint8_t* keep_row,
                                             std::uint8_t* keep_col,
                                             double* c_cur,
                                             double* AL_cur,
                                             double* AU_cur,
                                             const std::uint8_t* row_delete,
                                             const std::uint8_t* col_delete,
                                             const double* c_new,
                                             const double* AL_new,
                                             const double* AU_new,
                                             std::int32_t m,
                                             std::int32_t n) {
  const std::int32_t q = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (q < m) {
    if (row_delete[q] != std::uint8_t{0}) {
      keep_row[q] = std::uint8_t{0};
    }
    AL_cur[q] = AL_new[q];
    AU_cur[q] = AU_new[q];
  }
  if (q < n) {
    if (col_delete[q] != std::uint8_t{0}) {
      keep_col[q] = std::uint8_t{0};
    }
    c_cur[q] = c_new[q];
  }
}

__global__ void _kernel_singleton_subcol_tape_counts(std::int32_t* selected_scan,
                                                     std::int32_t* support_scan,
                                                     const std::uint8_t* col_delete,
                                                     const std::int32_t* row_owner,
                                                     const std::uint8_t* keep_col,
                                                     const std::int32_t* row_ptr,
                                                     const std::int32_t* col_val,
                                                     std::int32_t* long_rows,
                                                     std::int32_t* long_row_count,
                                                     std::int32_t m,
                                                     std::int32_t n) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m) {
    const std::int32_t col = row_owner[row];
    const bool selected = col >= 0 && col < n &&
                          col_delete[col] != std::uint8_t{0};
    selected_scan[row] = selected ? 1 : 0;
    std::int32_t support_count = 0;
    if (selected) {
      if (row_ptr[row + 1] - row_ptr[row] > SINGLETON_ACTIVITY_LONG_ROW_THRESHOLD) {
        const std::int32_t slot = atomicAdd(long_row_count, 1);
        long_rows[slot] = row;
        support_scan[row] = 0;
        return;
      }
      for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
        const std::int32_t row_col = col_val[p];
        if (row_col != col && keep_col[row_col] != std::uint8_t{0}) {
          ++support_count;
        }
      }
    }
    support_scan[row] = support_count;
  }
}

__global__ void _kernel_singleton_subcol_long_tape_counts(
    std::int32_t* support_scan,
    const std::uint8_t* keep_col,
    const std::int32_t* row_owner,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const std::int32_t* long_rows,
    std::int32_t long_row_count) {
  const std::int32_t long_index = static_cast<std::int32_t>(blockIdx.x);
  if (long_index >= long_row_count) return;
  const std::int32_t row = long_rows[long_index];
  const std::int32_t col = row_owner[row];
  std::int32_t count = 0;
  for (std::int32_t p = row_ptr[row] + static_cast<std::int32_t>(threadIdx.x);
       p < row_ptr[row + 1]; p += static_cast<std::int32_t>(blockDim.x)) {
    const std::int32_t row_col = col_val[p];
    count += row_col != col && keep_col[row_col] != std::uint8_t{0} ? 1 : 0;
  }
  __shared__ std::int32_t counts[GPU_PRESOLVE_THREADS];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  counts[tid] = count;
  __syncthreads();
  for (std::int32_t stride = GPU_PRESOLVE_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) counts[tid] += counts[tid + stride];
    __syncthreads();
  }
  if (tid == 0) support_scan[row] = counts[0];
}

__global__ void _kernel_singleton_pack_subcol_tape(std::int32_t* packed_cols,
                                                   std::int32_t* packed_rows,
                                                   std::int32_t* packed_support_starts,
                                                   std::int32_t* packed_support_cols,
                                                   double* packed_support_coeffs,
                                                   double* packed_pivot,
                                                   double* packed_chosen_side,
                                                   double* packed_l,
                                                   double* packed_u,
                                                   double* packed_c,
                                                   double* packed_row_deleted,
                                                   const std::int32_t* selected_scan,
                                                   const std::int32_t* support_scan,
                                                   const std::uint8_t* col_delete,
                                                   const std::uint8_t* row_delete,
                                                   const std::uint8_t* keep_col,
                                                   const std::int32_t* row_owner,
                                                   const double* chosen_side,
                                                   const double* c,
                                                   const double* l,
                                                   const double* u,
                                                   const std::int32_t* row_ptr,
                                                   const std::int32_t* col_val,
                                                   const double* nz_val,
                                                   std::int32_t long_row_threshold,
                                                   std::int32_t m,
                                                   std::int32_t n) {
  const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) {
    return;
  }
  const std::int32_t col = row_owner[row];
  if (col < 0 || col >= n || col_delete[col] == std::uint8_t{0}) {
    return;
  }
  if (row_ptr[row + 1] - row_ptr[row] > long_row_threshold) {
    return;
  }
  const std::int32_t prev_selected = row == 0 ? 0 : selected_scan[row - 1];
  const std::int32_t record = selected_scan[row] - 1;
  if (selected_scan[row] == prev_selected || record < 0) {
    return;
  }

  const std::int32_t support_start = row == 0 ? 0 : support_scan[row - 1];
  const std::int32_t support_stop = support_scan[row];
  packed_cols[record] = col;
  packed_rows[record] = row;
  packed_support_starts[record + 1] = support_stop;
  packed_chosen_side[record] = chosen_side[col];
  packed_l[record] = l[col];
  packed_u[record] = u[col];
  packed_c[record] = c[col];
  packed_row_deleted[record] = row_delete[row] != std::uint8_t{0} ? 1.0 : 0.0;

  double pivot = 0.0;
  std::int32_t out = support_start;
  for (std::int32_t p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
    const std::int32_t row_col = col_val[p];
    const double coeff = nz_val[p];
    if (row_col == col) {
      pivot = coeff;
    } else if (keep_col[row_col] != std::uint8_t{0} && out < support_stop) {
      packed_support_cols[out] = row_col;
      packed_support_coeffs[out] = coeff;
      ++out;
    }
  }
  packed_pivot[record] = pivot;
}

__global__ void _kernel_singleton_pack_long_subcol_tape(
    std::int32_t* packed_cols,
    std::int32_t* packed_rows,
    std::int32_t* packed_support_starts,
    std::int32_t* packed_support_cols,
    double* packed_support_coeffs,
    double* packed_pivot,
    double* packed_chosen_side,
    double* packed_l,
    double* packed_u,
    double* packed_c,
    double* packed_row_deleted,
    const std::int32_t* selected_scan,
    const std::int32_t* support_scan,
    const std::uint8_t* row_delete,
    const std::uint8_t* keep_col,
    const std::int32_t* row_owner,
    const double* chosen_side,
    const double* c,
    const double* l,
    const double* u,
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const std::int32_t* long_rows,
    std::int32_t long_row_count) {
  const std::int32_t long_index = static_cast<std::int32_t>(blockIdx.x);
  if (long_index >= long_row_count) return;
  const std::int32_t row = long_rows[long_index];
  const std::int32_t col = row_owner[row];
  const std::int32_t record = selected_scan[row] - 1;
  const std::int32_t support_start = row == 0 ? 0 : support_scan[row - 1];
  const std::int32_t support_stop = support_scan[row];
  const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
  if (tid == 0) {
    packed_cols[record] = col;
    packed_rows[record] = row;
    packed_support_starts[record + 1] = support_stop;
    packed_chosen_side[record] = chosen_side[col];
    packed_l[record] = l[col];
    packed_u[record] = u[col];
    packed_c[record] = c[col];
    packed_row_deleted[record] = row_delete[row] != std::uint8_t{0} ? 1.0 : 0.0;
  }
  __shared__ std::int32_t scan[GPU_PRESOLVE_THREADS];
  __shared__ std::int32_t base;
  __shared__ double pivot;
  if (tid == 0) {
    base = 0;
    pivot = 0.0;
  }
  __syncthreads();
  for (std::int32_t chunk = row_ptr[row]; chunk < row_ptr[row + 1];
       chunk += GPU_PRESOLVE_THREADS) {
    const std::int32_t p = chunk + tid;
    std::int32_t row_col = -1;
    double coeff = 0.0;
    if (p < row_ptr[row + 1]) {
      row_col = col_val[p];
      coeff = nz_val[p];
      if (row_col == col) pivot = coeff;
    }
    const bool copy = row_col >= 0 && row_col != col &&
                      keep_col[row_col] != std::uint8_t{0};
    scan[tid] = copy ? 1 : 0;
    __syncthreads();
    for (std::int32_t offset = 1; offset < GPU_PRESOLVE_THREADS; offset <<= 1) {
      const std::int32_t value = scan[tid] + (tid >= offset ? scan[tid - offset] : 0);
      __syncthreads();
      scan[tid] = value;
      __syncthreads();
    }
    if (copy) {
      const std::int32_t out = support_start + base + scan[tid] - 1;
      if (out < support_stop) {
        packed_support_cols[out] = row_col;
        packed_support_coeffs[out] = coeff;
      }
    }
    const std::int32_t chunk_count = scan[GPU_PRESOLVE_THREADS - 1];
    __syncthreads();
    if (tid == 0) base += chunk_count;
    __syncthreads();
  }
  if (tid == 0) packed_pivot[record] = pivot;
}

void append_subcol_tape_compacted_from_device(PostsolveTape& tape,
                                              const std::uint8_t* col_delete,
                                              const std::uint8_t* row_delete,
                                              const std::uint8_t* keep_col,
                                              const std::int32_t* row_owner,
                                              const double* chosen_side,
                                              const double* c,
                                              const double* l,
                                              const double* u,
                                              const DeviceCsrMatrix& A,
                                              double zero_tol,
                                              const char* context) {
  const std::int32_t m = A.rows;
  const std::int32_t n = A.cols;
  if (m <= 0 || n <= 0) {
    return;
  }

  std::int32_t* selected_scan = nullptr;
  std::int32_t* support_scan = nullptr;
  std::int32_t* long_rows = nullptr;
  std::int32_t* long_row_count = nullptr;
  throw_if_cuda_error(column_singletons_cuda_malloc(&selected_scan, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&support_scan, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&long_rows, sizeof(std::int32_t) * static_cast<std::size_t>(m)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&long_row_count, sizeof(std::int32_t)), context);
  throw_if_cuda_error(cudaMemset(long_row_count, 0, sizeof(std::int32_t)), context);

  const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_singleton_subcol_tape_counts<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
      selected_scan,
      support_scan,
      col_delete,
      row_owner,
      keep_col,
      A.rowPtr,
      A.colVal,
      long_rows,
      long_row_count,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_subcol_tape_counts");
  std::int32_t long_row_count_host = 0;
  throw_if_cuda_error(cudaMemcpy(&long_row_count_host, long_row_count,
                                 sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      context);
  if (long_row_count_host > 0) {
    _kernel_singleton_subcol_long_tape_counts
        <<<long_row_count_host, GPU_PRESOLVE_THREADS>>>(
            support_scan, keep_col, row_owner, A.rowPtr, A.colVal,
            long_rows, long_row_count_host);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_singleton_subcol_long_tape_counts");
  }
  inclusive_scan_i32(selected_scan, m, context);
  inclusive_scan_i32(support_scan, m, context);

  std::int32_t counts[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpy(&counts[0], selected_scan + m - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(&counts[1], support_scan + m - 1, sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      context);
  const std::int32_t record_count = counts[0];
  const std::int32_t support_nnz = counts[1];
  if (record_count <= 0) {
    column_singletons_cuda_free(selected_scan);
    column_singletons_cuda_free(support_scan);
    column_singletons_cuda_free(long_rows);
    column_singletons_cuda_free(long_row_count);
    return;
  }

  std::int32_t* packed_cols = nullptr;
  std::int32_t* packed_rows = nullptr;
  std::int32_t* packed_support_starts = nullptr;
  std::int32_t* packed_support_cols = nullptr;
  double* packed_support_coeffs = nullptr;
  double* packed_pivot = nullptr;
  double* packed_chosen_side = nullptr;
  double* packed_l = nullptr;
  double* packed_u = nullptr;
  double* packed_c = nullptr;
  double* packed_row_deleted = nullptr;
  throw_if_cuda_error(column_singletons_cuda_malloc(&packed_cols, sizeof(std::int32_t) * static_cast<std::size_t>(record_count)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&packed_rows, sizeof(std::int32_t) * static_cast<std::size_t>(record_count)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&packed_support_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(record_count + 1)),
                      context);
  throw_if_cuda_error(cudaMemset(packed_support_starts, 0, sizeof(std::int32_t)), context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&packed_pivot, sizeof(double) * static_cast<std::size_t>(record_count)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&packed_chosen_side, sizeof(double) * static_cast<std::size_t>(record_count)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&packed_l, sizeof(double) * static_cast<std::size_t>(record_count)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&packed_u, sizeof(double) * static_cast<std::size_t>(record_count)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&packed_c, sizeof(double) * static_cast<std::size_t>(record_count)),
                      context);
  throw_if_cuda_error(column_singletons_cuda_malloc(&packed_row_deleted, sizeof(double) * static_cast<std::size_t>(record_count)),
                      context);
  if (support_nnz > 0) {
    throw_if_cuda_error(column_singletons_cuda_malloc(&packed_support_cols,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(support_nnz)),
                        context);
    throw_if_cuda_error(column_singletons_cuda_malloc(&packed_support_coeffs,
                                   sizeof(double) * static_cast<std::size_t>(support_nnz)),
                        context);
  }

  _kernel_singleton_pack_subcol_tape<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
      packed_cols,
      packed_rows,
      packed_support_starts,
      packed_support_cols,
      packed_support_coeffs,
      packed_pivot,
      packed_chosen_side,
      packed_l,
      packed_u,
      packed_c,
      packed_row_deleted,
      selected_scan,
      support_scan,
      col_delete,
      row_delete,
      keep_col,
      row_owner,
      chosen_side,
      c,
      l,
      u,
      A.rowPtr,
      A.colVal,
      A.nzVal,
      SINGLETON_ACTIVITY_LONG_ROW_THRESHOLD,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_pack_subcol_tape");
  if (long_row_count_host > 0) {
    _kernel_singleton_pack_long_subcol_tape
        <<<long_row_count_host, GPU_PRESOLVE_THREADS>>>(
            packed_cols, packed_rows, packed_support_starts,
            packed_support_cols, packed_support_coeffs, packed_pivot,
            packed_chosen_side, packed_l, packed_u, packed_c,
            packed_row_deleted, selected_scan, support_scan, row_delete,
            keep_col, row_owner, chosen_side, c, l, u, A.rowPtr, A.colVal,
            A.nzVal, long_rows, long_row_count_host);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_singleton_pack_long_subcol_tape");
  }

  std::vector<std::int32_t> host_cols(static_cast<std::size_t>(record_count));
  std::vector<std::int32_t> host_rows(static_cast<std::size_t>(record_count));
  std::vector<std::int32_t> host_support_starts(static_cast<std::size_t>(record_count + 1));
  std::vector<std::int32_t> host_support_cols(static_cast<std::size_t>(support_nnz));
  std::vector<double> host_support_coeffs(static_cast<std::size_t>(support_nnz));
  std::vector<double> host_pivot(static_cast<std::size_t>(record_count));
  std::vector<double> host_chosen_side(static_cast<std::size_t>(record_count));
  std::vector<double> host_l(static_cast<std::size_t>(record_count));
  std::vector<double> host_u(static_cast<std::size_t>(record_count));
  std::vector<double> host_c(static_cast<std::size_t>(record_count));
  std::vector<double> host_row_deleted(static_cast<std::size_t>(record_count));
  throw_if_cuda_error(cudaMemcpy(host_cols.data(), packed_cols,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_rows.data(), packed_rows,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_support_starts.data(), packed_support_starts,
                                 sizeof(std::int32_t) * static_cast<std::size_t>(record_count + 1),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_pivot.data(), packed_pivot,
                                 sizeof(double) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_chosen_side.data(), packed_chosen_side,
                                 sizeof(double) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_l.data(), packed_l,
                                 sizeof(double) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_u.data(), packed_u,
                                 sizeof(double) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_c.data(), packed_c,
                                 sizeof(double) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  throw_if_cuda_error(cudaMemcpy(host_row_deleted.data(), packed_row_deleted,
                                 sizeof(double) * static_cast<std::size_t>(record_count),
                                 cudaMemcpyDeviceToHost),
                      context);
  if (support_nnz > 0) {
    throw_if_cuda_error(cudaMemcpy(host_support_cols.data(), packed_support_cols,
                                   sizeof(std::int32_t) * static_cast<std::size_t>(support_nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
    throw_if_cuda_error(cudaMemcpy(host_support_coeffs.data(), packed_support_coeffs,
                                   sizeof(double) * static_cast<std::size_t>(support_nnz),
                                   cudaMemcpyDeviceToHost),
                        context);
  }

  for (std::int32_t rec = 0; rec < record_count; ++rec) {
    if (fabs(host_pivot[static_cast<std::size_t>(rec)]) <= zero_tol) {
      continue;
    }
    const std::int32_t start = host_support_starts[static_cast<std::size_t>(rec)];
    const std::int32_t stop = host_support_starts[static_cast<std::size_t>(rec + 1)];
    std::vector<std::int32_t> indices{
        host_cols[static_cast<std::size_t>(rec)],
        host_rows[static_cast<std::size_t>(rec)],
        stop - start};
    std::vector<double> vals{
        host_pivot[static_cast<std::size_t>(rec)],
        host_chosen_side[static_cast<std::size_t>(rec)],
        host_l[static_cast<std::size_t>(rec)],
        host_u[static_cast<std::size_t>(rec)],
        host_c[static_cast<std::size_t>(rec)],
        host_row_deleted[static_cast<std::size_t>(rec)]};
    for (std::int32_t p = start; p < stop; ++p) {
      indices.push_back(host_support_cols[static_cast<std::size_t>(p)]);
      vals.push_back(host_support_coeffs[static_cast<std::size_t>(p)]);
    }
    append_postsolve_record(
        tape, PostsolveReductionType::SubCol, indices, vals, PostsolveDualMode::Minimal);
  }

  column_singletons_cuda_free(selected_scan);
  column_singletons_cuda_free(support_scan);
  column_singletons_cuda_free(long_rows);
  column_singletons_cuda_free(long_row_count);
  column_singletons_cuda_free(packed_cols);
  column_singletons_cuda_free(packed_rows);
  column_singletons_cuda_free(packed_support_starts);
  column_singletons_cuda_free(packed_pivot);
  column_singletons_cuda_free(packed_chosen_side);
  column_singletons_cuda_free(packed_l);
  column_singletons_cuda_free(packed_u);
  column_singletons_cuda_free(packed_c);
  column_singletons_cuda_free(packed_row_deleted);
  if (packed_support_cols != nullptr) {
    column_singletons_cuda_free(packed_support_cols);
  }
  if (packed_support_coeffs != nullptr) {
    column_singletons_cuda_free(packed_support_coeffs);
  }
}

void apply_rule_column_singletons_mode(PresolvePlanGpu& plan,
                                    const LPInfoGpu& lp,
                                    const PresolveStatsGpu& stats,
                                    const PresolveParams& pparams,
                                    std::uint8_t rule_mode,
                                    bool skip_precheck) {
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }
  const std::int32_t m = lp.A.rows;
  const std::int32_t n = lp.A.cols;
  if (m == 0 || n == 0) {
    return;
  }

  const int blocks_n = (n + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  if (!skip_precheck) {
    std::int32_t* singleton_counts_device = nullptr;
    throw_if_cuda_error(column_singletons_cuda_malloc(&singleton_counts_device, sizeof(std::int32_t) * 2),
                        "cudaMalloc column_singletons counts");
    throw_if_cuda_error(cudaMemset(singleton_counts_device, 0, sizeof(std::int32_t) * 2),
                        "cudaMemset column_singletons counts");
    _kernel_count_column_singleton_support_eq<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        singleton_counts_device,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.column_singleton_mask,
        stats.column_singleton_row,
        plan.new_AL,
        plan.new_AU,
        pparams.bound_tol,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_count_column_singleton_support_eq");
    std::int32_t singleton_counts[2] = {0, 0};
    throw_if_cuda_error(cudaMemcpy(singleton_counts,
                                   singleton_counts_device,
                                   sizeof(singleton_counts),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy column_singletons counts");
    column_singletons_cuda_free(singleton_counts_device);
    const std::int32_t singleton_count = singleton_counts[0];
    const std::int32_t eq_support_count = singleton_counts[1];
    if (singleton_count == 0 ||
        (rule_mode == SINGLETON_COL_RULE_EQ && eq_support_count == 0) ||
        (rule_mode == SINGLETON_COL_RULE_DUAL_INFER && singleton_count == eq_support_count)) {
      return;
    }
  }

  ColumnSingletonsWorkspace& workspace = column_singletons_workspace();
  workspace.ensure(m, n);
  std::int32_t* status_flag = workspace.status_flag;
  std::int32_t* row_owner = workspace.row_owner;
  std::uint8_t* col_delete = workspace.col_delete;
  std::uint8_t* row_delete = workspace.row_delete;
  std::uint8_t* row_lhs_change = workspace.row_lhs_change;
  std::uint8_t* row_rhs_change = workspace.row_rhs_change;
  std::int32_t* pair_row = workspace.pair_row;
  double* chosen_side = workspace.chosen_side;
  double* c_new = workspace.c_new;
  double* AL_new = workspace.AL_new;
  double* AU_new = workspace.AU_new;
  double* obj_contrib = workspace.obj_contrib;
  double* activity_finite_min = workspace.activity_finite_min;
  double* activity_finite_max = workspace.activity_finite_max;
  std::int32_t* activity_min_neg_inf = workspace.activity_min_neg_inf;
  std::int32_t* activity_min_pos_inf = workspace.activity_min_pos_inf;
  std::int32_t* activity_max_neg_inf = workspace.activity_max_neg_inf;
  std::int32_t* activity_max_pos_inf = workspace.activity_max_pos_inf;
  std::uint8_t* activity_invalid = workspace.activity_invalid;
  std::uint8_t* activity_row_needed = workspace.activity_row_needed;
  std::int32_t* activity_long_rows = workspace.activity_long_rows;
  std::int32_t* activity_long_row_count = workspace.activity_long_row_count;
  throw_if_cuda_error(cudaMemset(activity_long_row_count, 0, sizeof(std::int32_t)),
                      "cudaMemset column_singletons activity_long_row_count");
  throw_if_cuda_error(cudaMemset(activity_row_needed, 0, static_cast<std::size_t>(m)),
                      "cudaMemset column_singletons activity_row_needed");
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t) * 3), "cudaMemset column_singletons status_flag");
  throw_if_cuda_error(cudaMemset(col_delete, 0, static_cast<std::size_t>(n)), "cudaMemset column_singletons col_delete");
  throw_if_cuda_error(cudaMemset(row_delete, 0, static_cast<std::size_t>(m)), "cudaMemset column_singletons row_delete");
  throw_if_cuda_error(cudaMemset(row_lhs_change, 0, static_cast<std::size_t>(m)), "cudaMemset column_singletons row_lhs_change");
  throw_if_cuda_error(cudaMemset(row_rhs_change, 0, static_cast<std::size_t>(m)), "cudaMemset column_singletons row_rhs_change");
  throw_if_cuda_error(cudaMemset(chosen_side, 0, sizeof(double) * static_cast<std::size_t>(n)), "cudaMemset column_singletons chosen_side");
  throw_if_cuda_error(cudaMemset(obj_contrib, 0, sizeof(double) * static_cast<std::size_t>(n)), "cudaMemset column_singletons obj_contrib");
  throw_if_cuda_error(cudaMemcpy(c_new, plan.new_c, sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy column_singletons c_new");
  throw_if_cuda_error(cudaMemcpy(AL_new, plan.new_AL, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy column_singletons AL_new");
  throw_if_cuda_error(cudaMemcpy(AU_new, plan.new_AU, sizeof(double) * static_cast<std::size_t>(m), cudaMemcpyDeviceToDevice),
                      "cudaMemcpy column_singletons AU_new");

  const int blocks_m = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  _kernel_fill_i32<<<blocks_m, GPU_PRESOLVE_THREADS>>>(row_owner, INT_MAX, m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_fill_i32 singleton row_owner");

  _kernel_mark_singleton_activity_rows<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      activity_row_needed,
      plan.keep_row_mask,
      plan.keep_col_mask,
      stats.column_singleton_mask,
      stats.column_singleton_row,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_mark_singleton_activity_rows");

  // Compute each row's activity once.  A column singleton can then exclude its
  // own contribution in O(1), avoiding repeated full scans when many singleton
  // columns share one very long row.
  _kernel_singleton_row_activity_cache<<<blocks_m, GPU_PRESOLVE_THREADS>>>(
      activity_finite_min,
      activity_finite_max,
      activity_min_neg_inf,
      activity_min_pos_inf,
      activity_max_neg_inf,
      activity_max_pos_inf,
      activity_invalid,
      activity_row_needed,
      plan.keep_row_mask,
      plan.keep_col_mask,
      plan.new_l,
      plan.new_u,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      activity_long_rows,
      activity_long_row_count,
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_singleton_row_activity_cache");
  std::int32_t activity_long_row_count_host = 0;
  throw_if_cuda_error(cudaMemcpy(&activity_long_row_count_host,
                                 activity_long_row_count,
                                 sizeof(std::int32_t),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy column_singletons activity_long_row_count");
  if (activity_long_row_count_host > 0) {
    _kernel_singleton_long_row_activity_cache
        <<<activity_long_row_count_host, GPU_PRESOLVE_THREADS>>>(
            activity_finite_min,
            activity_finite_max,
            activity_min_neg_inf,
            activity_min_pos_inf,
            activity_max_neg_inf,
            activity_max_pos_inf,
            activity_invalid,
            plan.keep_row_mask,
            plan.keep_col_mask,
            plan.new_l,
            plan.new_u,
            lp.A.rowPtr,
            lp.A.colVal,
            lp.A.nzVal,
            activity_long_rows,
            activity_long_row_count_host);
    throw_if_cuda_error(cudaGetLastError(),
                        "_kernel_singleton_long_row_activity_cache");
  }

  _kernel_column_singleton_row_owner<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      row_owner,
      plan.keep_row_mask,
      plan.keep_col_mask,
      stats.column_singleton_mask,
      stats.column_singleton_row,
      stats.column_singleton_val,
      plan.new_c,
      plan.new_l,
      plan.new_u,
      plan.new_AL,
      plan.new_AU,
      activity_finite_min,
      activity_finite_max,
      activity_min_neg_inf,
      activity_min_pos_inf,
      activity_max_neg_inf,
      activity_max_pos_inf,
      activity_invalid,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      pparams.bound_tol,
      pparams.zero_tol,
      rule_mode,
      m,
      n);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_column_singleton_row_owner");

  std::int32_t status[3] = {0, 0, 0};
  throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                      "cudaMemcpy column_singletons status after owner");
  if (status[0] != 0) {
    plan.has_unbounded = true;
  } else {
    _kernel_process_column_singletons<<<blocks_n, GPU_PRESOLVE_THREADS>>>(
        status_flag,
        row_delete,
        row_lhs_change,
        row_rhs_change,
        col_delete,
        pair_row,
        chosen_side,
        AL_new,
        AU_new,
        obj_contrib,
        row_owner,
        plan.keep_row_mask,
        plan.keep_col_mask,
        stats.column_singleton_mask,
        stats.column_singleton_row,
        stats.column_singleton_val,
        plan.new_c,
        plan.new_l,
        plan.new_u,
        activity_finite_min,
        activity_finite_max,
        activity_min_neg_inf,
        activity_min_pos_inf,
        activity_max_neg_inf,
        activity_max_pos_inf,
        activity_invalid,
        lp.A.rowPtr,
        lp.A.colVal,
        lp.A.nzVal,
        pparams.bound_tol,
        pparams.zero_tol,
        rule_mode,
        m,
        n);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_process_column_singletons");
    throw_if_cuda_error(cudaMemcpy(status, status_flag, sizeof(status), cudaMemcpyDeviceToHost),
                        "cudaMemcpy column_singletons status after process");
    if (status[0] != 0) {
      plan.has_unbounded = true;
    } else if (status[1] != 0 || status[2] != 0) {
      if (status[1] != 0 && rule_mode == SINGLETON_COL_RULE_EQ) {
        _kernel_apply_singleton_objective_updates_deterministic
            <<<blocks_n, GPU_PRESOLVE_THREADS>>>(
                c_new,
                plan.keep_col_mask,
                col_delete,
                row_owner,
                stats.column_singleton_val,
                plan.new_c,
                lp.AT.rowPtr,
                lp.AT.colVal,
                lp.AT.nzVal,
                n);
        throw_if_cuda_error(cudaGetLastError(),
                            "_kernel_apply_singleton_objective_updates_deterministic");
      }
      const std::int32_t max_mn = m > n ? m : n;
      const int blocks_mn = (max_mn + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
      if (status[1] != 0 && rule_mode == SINGLETON_COL_RULE_EQ && pparams.record_postsolve_tape) {
        append_subcol_tape_compacted_from_device(plan.tape,
                                                 col_delete,
                                                 row_delete,
                                                 plan.keep_col_mask,
                                                 row_owner,
                                                 chosen_side,
                                                 plan.new_c,
                                                 plan.new_l,
                                                 plan.new_u,
                                                 lp.A,
                                                 pparams.zero_tol,
                                                 "cudaMemcpy singleton compact SubCol tape");
      }
      if (status[2] != 0 && rule_mode == SINGLETON_COL_RULE_DUAL_INFER &&
          pparams.record_postsolve_tape) {
        append_eq_to_ineq_tape_from_device(plan.tape,
                                           row_lhs_change,
                                           row_rhs_change,
                                           m,
                                           "cudaMemcpy singleton eq-to-ineq tape");
      }
      _kernel_apply_column_singletons<<<blocks_mn, GPU_PRESOLVE_THREADS>>>(
          plan.keep_row_mask,
          plan.keep_col_mask,
          plan.new_c,
          plan.new_AL,
          plan.new_AU,
          row_delete,
          col_delete,
          c_new,
          AL_new,
          AU_new,
          m,
          n);
      throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_column_singletons");
      throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_column_singletons synchronize");
      const double obj_delta = sum_device_double(obj_contrib, n, "cudaMemcpy column_singletons obj_contrib");
      plan.obj_constant_delta += obj_delta;
      plan.has_row_action = true;
      plan.has_col_action = plan.has_col_action || status[1] != 0;
      plan.has_change = true;
    }
  }

}

}  // namespace

void apply_rule_column_singletons_eq(PresolvePlanGpu& plan,
                                  const LPInfoGpu& lp,
                                  const PresolveStatsGpu& stats,
                                  const PresolveParams& pparams,
                                  bool skip_precheck) {
  apply_rule_column_singletons_mode(
      plan, lp, stats, pparams, SINGLETON_COL_RULE_EQ, skip_precheck);
}

void apply_rule_column_singletons_dual_infer(PresolvePlanGpu& plan,
                                          const LPInfoGpu& lp,
                                          const PresolveStatsGpu& stats,
                                          const PresolveParams& pparams) {
  apply_rule_column_singletons_mode(
      plan, lp, stats, pparams, SINGLETON_COL_RULE_DUAL_INFER, false);
}

void release_column_singletons_workspace() noexcept {
  column_singletons_tls_cache_storage().release();
}

}  // namespace gpu_presolver::presolve
