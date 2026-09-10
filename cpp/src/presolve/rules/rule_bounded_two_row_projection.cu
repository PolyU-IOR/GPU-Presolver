#include "../presolve_helpers.hpp"
#include "gpu_presolver/presolve/rules/rule_bounded_two_row_projection.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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
constexpr int MAX_CERTIFIED_ROW_NNZ = 8;
constexpr int MAX_AFFINE_TERMS = MAX_CERTIFIED_ROW_NNZ - 1;
constexpr int MAX_DIFFERENCE_TERMS = 2 * MAX_AFFINE_TERMS;

template <class T>
std::vector<T> copy_device_vector(const T* src,
                                  std::int32_t count,
                                  const char* context) {
  std::vector<T> out(static_cast<std::size_t>(std::max(0, count)));
  if (count > 0) {
    throw_if_cuda_error(cudaMemcpy(out.data(), src,
                                   sizeof(T) * static_cast<std::size_t>(count),
                                   cudaMemcpyDeviceToHost),
                        context);
  }
  return out;
}

struct DeviceAffine {
  double constant = 0.0;
  std::int32_t length = 0;
  std::int32_t cols[MAX_AFFINE_TERMS];
  double coeffs[MAX_AFFINE_TERMS];
};

__device__ bool exact_negative_infinity(double value) {
  return isinf(value) && value < 0.0;
}

__device__ bool exact_positive_infinity(double value) {
  return isinf(value) && value > 0.0;
}

__device__ bool classify_one_sided_row(double AL,
                                       double AU,
                                       double pivot,
                                       double zero_tol,
                                       bool* gives_lower,
                                       double* finite_bound) {
  if (!isfinite(pivot) || fabs(pivot) <= zero_tol) {
    return false;
  }
  if (isfinite(AL) && exact_positive_infinity(AU)) {
    *gives_lower = pivot > 0.0;
    *finite_bound = AL;
    return true;
  }
  if (exact_negative_infinity(AL) && isfinite(AU)) {
    *gives_lower = pivot < 0.0;
    *finite_bound = AU;
    return true;
  }
  return false;
}

__device__ bool add_affine_term(std::int32_t col,
                                double coeff,
                                std::int32_t* cols,
                                double* coeffs,
                                std::int32_t* length,
                                std::int32_t capacity) {
  if (!isfinite(coeff)) {
    return false;
  }
  for (std::int32_t k = 0; k < *length; ++k) {
    if (cols[k] == col) {
      coeffs[k] += coeff;
      return isfinite(coeffs[k]);
    }
  }
  if (*length >= capacity) {
    return false;
  }
  cols[*length] = col;
  coeffs[*length] = coeff;
  ++(*length);
  return true;
}

__device__ bool build_row_affine(DeviceAffine* affine,
                                 bool* gives_lower,
                                 const std::int32_t* row_ptr,
                                 const std::int32_t* col_val,
                                 const double* nz_val,
                                 const double* AL,
                                 const double* AU,
                                 std::int32_t row,
                                 std::int32_t elim_col,
                                 std::int32_t max_row_nnz,
                                 double zero_tol) {
  const std::int32_t first = row_ptr[row];
  const std::int32_t last = row_ptr[row + 1];
  const std::int32_t row_nnz = last - first;
  if (row_nnz < 2 || row_nnz > max_row_nnz ||
      row_nnz > MAX_CERTIFIED_ROW_NNZ) {
    return false;
  }
  std::int32_t pivot_count = 0;
  double pivot = 0.0;
  for (std::int32_t p = first; p < last; ++p) {
    if (col_val[p] == elim_col) {
      ++pivot_count;
      pivot = nz_val[p];
    }
  }
  if (pivot_count != 1) {
    return false;
  }
  double finite_bound = 0.0;
  if (!classify_one_sided_row(AL[row], AU[row], pivot, zero_tol,
                              gives_lower, &finite_bound)) {
    return false;
  }
  affine->constant = finite_bound / pivot;
  affine->length = 0;
  if (!isfinite(affine->constant)) {
    return false;
  }
  for (std::int32_t p = first; p < last; ++p) {
    if (col_val[p] == elim_col) {
      continue;
    }
    if (!add_affine_term(col_val[p], -nz_val[p] / pivot, affine->cols,
                         affine->coeffs, &affine->length,
                         MAX_AFFINE_TERMS)) {
      return false;
    }
  }
  return true;
}

__device__ bool affine_interval(const DeviceAffine& affine,
                                const double* lower,
                                const double* upper,
                                double* min_value,
                                double* max_value) {
  double lo = affine.constant;
  double hi = affine.constant;
  for (std::int32_t k = 0; k < affine.length; ++k) {
    const std::int32_t col = affine.cols[k];
    const double l = lower[col];
    const double u = upper[col];
    const double q = affine.coeffs[k];
    if (!isfinite(l) || !isfinite(u) || l > u) {
      return false;
    }
    lo += q >= 0.0 ? q * l : q * u;
    hi += q >= 0.0 ? q * u : q * l;
  }
  if (!isfinite(lo) || !isfinite(hi)) {
    return false;
  }
  *min_value = lo;
  *max_value = hi;
  return true;
}

__device__ bool strictly_le_with_guard(double lhs,
                                       double rhs,
                                       double safety_tol) {
  if (!isfinite(lhs) || !isfinite(rhs)) {
    return false;
  }
  const double scale = 1.0 + fmax(fabs(lhs), fabs(rhs));
  return rhs - lhs >= safety_tol * scale;
}

__device__ bool difference_max(const DeviceAffine& lower_affine,
                               const DeviceAffine& upper_affine,
                               const double* lower,
                               const double* upper,
                               double* result) {
  std::int32_t cols[MAX_DIFFERENCE_TERMS];
  double coeffs[MAX_DIFFERENCE_TERMS];
  std::int32_t length = 0;
  for (std::int32_t k = 0; k < lower_affine.length; ++k) {
    if (!add_affine_term(lower_affine.cols[k], lower_affine.coeffs[k],
                         cols, coeffs, &length, MAX_DIFFERENCE_TERMS)) {
      return false;
    }
  }
  for (std::int32_t k = 0; k < upper_affine.length; ++k) {
    if (!add_affine_term(upper_affine.cols[k], -upper_affine.coeffs[k],
                         cols, coeffs, &length, MAX_DIFFERENCE_TERMS)) {
      return false;
    }
  }
  double value = lower_affine.constant - upper_affine.constant;
  for (std::int32_t k = 0; k < length; ++k) {
    const double l = lower[cols[k]];
    const double u = upper[cols[k]];
    if (!isfinite(l) || !isfinite(u) || l > u) {
      return false;
    }
    value += coeffs[k] >= 0.0 ? coeffs[k] * u : coeffs[k] * l;
  }
  if (!isfinite(value)) {
    return false;
  }
  *result = value;
  return true;
}

__global__ void detect_bounded_two_row_candidates(
    std::uint8_t* candidate,
    const std::int32_t* A_row_ptr,
    const std::int32_t* A_col_val,
    const double* A_nz_val,
    const std::int32_t* AT_row_ptr,
    const std::int32_t* AT_col_val,
    const double* c,
    const double* AL,
    const double* AU,
    const double* lower,
    const double* upper,
    std::int32_t cols,
    std::int32_t max_row_nnz,
    double zero_tol,
    double safety_tol) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= cols) {
    return;
  }
  candidate[col] = std::uint8_t{0};
  if (c[col] != 0.0 || !isfinite(lower[col]) || !isfinite(upper[col]) ||
      !strictly_le_with_guard(lower[col], upper[col], safety_tol) ||
      AT_row_ptr[col + 1] - AT_row_ptr[col] != 2) {
    return;
  }
  const std::int32_t at_first = AT_row_ptr[col];
  const std::int32_t row0 = AT_col_val[at_first];
  const std::int32_t row1 = AT_col_val[at_first + 1];
  if (row0 == row1) {
    return;
  }

  DeviceAffine affine0;
  DeviceAffine affine1;
  bool lower0 = false;
  bool lower1 = false;
  if (!build_row_affine(&affine0, &lower0, A_row_ptr, A_col_val,
                        A_nz_val, AL, AU, row0, col, max_row_nnz,
                        zero_tol) ||
      !build_row_affine(&affine1, &lower1, A_row_ptr, A_col_val,
                        A_nz_val, AL, AU, row1, col, max_row_nnz,
                        zero_tol) ||
      lower0 == lower1) {
    return;
  }

  const DeviceAffine& lower_affine = lower0 ? affine0 : affine1;
  const DeviceAffine& upper_affine = lower0 ? affine1 : affine0;
  double lower_min = 0.0;
  double lower_max = 0.0;
  double upper_min = 0.0;
  double upper_max = 0.0;
  if (!affine_interval(lower_affine, lower, upper, &lower_min,
                       &lower_max) ||
      !affine_interval(upper_affine, lower, upper, &upper_min,
                       &upper_max)) {
    return;
  }
  (void)lower_min;
  (void)upper_max;
  if (!strictly_le_with_guard(lower_max, upper[col], safety_tol) ||
      !strictly_le_with_guard(lower[col], upper_min, safety_tol)) {
    return;
  }
  double max_difference = 0.0;
  if (!difference_max(lower_affine, upper_affine, lower, upper,
                      &max_difference) ||
      !strictly_le_with_guard(max_difference, 0.0, safety_tol)) {
    return;
  }
  candidate[col] = std::uint8_t{1};
}

__global__ void count_candidate_rows(std::int32_t* row_candidate_count,
                                     const std::uint8_t* candidate,
                                     const std::int32_t* AT_row_ptr,
                                     const std::int32_t* AT_col_val,
                                     std::int32_t cols) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= cols || candidate[col] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t first = AT_row_ptr[col];
  atomicAdd(&row_candidate_count[AT_col_val[first]], 1);
  atomicAdd(&row_candidate_count[AT_col_val[first + 1]], 1);
}

__global__ void select_nonconflicting_candidates(
    std::uint8_t* candidate,
    std::int32_t* selected_count,
    const std::int32_t* row_candidate_count,
    const std::int32_t* AT_row_ptr,
    const std::int32_t* AT_col_val,
    std::int32_t cols) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= cols || candidate[col] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t first = AT_row_ptr[col];
  const bool selected =
      row_candidate_count[AT_col_val[first]] == 1 &&
      row_candidate_count[AT_col_val[first + 1]] == 1;
  candidate[col] = selected ? std::uint8_t{1} : std::uint8_t{0};
  if (selected) {
    atomicAdd(selected_count, 1);
  }
}

__global__ void apply_bounded_two_row_projection(
    std::uint8_t* keep_row,
    std::uint8_t* keep_col,
    const std::uint8_t* selected_col,
    const std::int32_t* AT_row_ptr,
    const std::int32_t* AT_col_val,
    std::int32_t cols) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= cols || selected_col[col] == std::uint8_t{0}) {
    return;
  }
  keep_col[col] = std::uint8_t{0};
  const std::int32_t first = AT_row_ptr[col];
  keep_row[AT_col_val[first]] = std::uint8_t{0};
  keep_row[AT_col_val[first + 1]] = std::uint8_t{0};
}

struct HostRowAffine {
  bool valid = false;
  bool gives_lower = false;
  double constant = 0.0;
  std::vector<std::int32_t> cols;
  std::vector<double> coeffs;
};

HostRowAffine build_host_row_affine(
    std::int32_t row,
    std::int32_t elim_col,
    const std::vector<std::int32_t>& row_ptr,
    const std::vector<std::int32_t>& col_val,
    const std::vector<double>& nz_val,
    const std::vector<double>& AL,
    const std::vector<double>& AU) {
  HostRowAffine out;
  double pivot = 0.0;
  std::int32_t pivot_count = 0;
  for (std::int32_t p = row_ptr[static_cast<std::size_t>(row)];
       p < row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
    if (col_val[static_cast<std::size_t>(p)] == elim_col) {
      pivot = nz_val[static_cast<std::size_t>(p)];
      ++pivot_count;
    }
  }
  if (pivot_count != 1 || !std::isfinite(pivot) || pivot == 0.0) {
    return out;
  }
  double bound = 0.0;
  if (std::isfinite(AL[static_cast<std::size_t>(row)]) &&
      std::isinf(AU[static_cast<std::size_t>(row)]) &&
      AU[static_cast<std::size_t>(row)] > 0.0) {
    out.gives_lower = pivot > 0.0;
    bound = AL[static_cast<std::size_t>(row)];
  } else if (std::isinf(AL[static_cast<std::size_t>(row)]) &&
             AL[static_cast<std::size_t>(row)] < 0.0 &&
             std::isfinite(AU[static_cast<std::size_t>(row)])) {
    out.gives_lower = pivot < 0.0;
    bound = AU[static_cast<std::size_t>(row)];
  } else {
    return out;
  }
  out.constant = bound / pivot;
  for (std::int32_t p = row_ptr[static_cast<std::size_t>(row)];
       p < row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
    const std::int32_t col = col_val[static_cast<std::size_t>(p)];
    if (col == elim_col) {
      continue;
    }
    out.cols.push_back(col);
    out.coeffs.push_back(-nz_val[static_cast<std::size_t>(p)] / pivot);
  }
  out.valid = true;
  return out;
}

void append_projection_tape_from_device(
    PostsolveTape& tape,
    const LPInfoGpu& lp,
    const std::uint8_t* selected_col) {
  const std::vector<std::uint8_t> selected = copy_device_vector(
      selected_col, lp.A.cols, "cudaMemcpy bounded projection selected mask");
  const std::vector<std::int32_t> A_row_ptr = copy_device_vector(
      lp.A.rowPtr, lp.A.rows + 1, "cudaMemcpy bounded projection A rowPtr");
  const std::vector<std::int32_t> A_col_val = copy_device_vector(
      lp.A.colVal, lp.A.nnz, "cudaMemcpy bounded projection A colVal");
  const std::vector<double> A_nz_val = copy_device_vector(
      lp.A.nzVal, lp.A.nnz, "cudaMemcpy bounded projection A values");
  const std::vector<std::int32_t> AT_row_ptr = copy_device_vector(
      lp.AT.rowPtr, lp.A.cols + 1, "cudaMemcpy bounded projection AT rowPtr");
  const std::vector<std::int32_t> AT_col_val = copy_device_vector(
      lp.AT.colVal, lp.AT.nnz, "cudaMemcpy bounded projection AT colVal");
  const std::vector<double> AL = copy_device_vector(
      lp.AL, lp.A.rows, "cudaMemcpy bounded projection AL");
  const std::vector<double> AU = copy_device_vector(
      lp.AU, lp.A.rows, "cudaMemcpy bounded projection AU");
  const std::vector<double> lower = copy_device_vector(
      lp.l, lp.A.cols, "cudaMemcpy bounded projection lower bounds");

  for (std::int32_t col = 0; col < lp.A.cols; ++col) {
    if (selected[static_cast<std::size_t>(col)] == std::uint8_t{0}) {
      continue;
    }
    const std::int32_t at_first = AT_row_ptr[static_cast<std::size_t>(col)];
    const std::int32_t row0 = AT_col_val[static_cast<std::size_t>(at_first)];
    const std::int32_t row1 =
        AT_col_val[static_cast<std::size_t>(at_first + 1)];
    const HostRowAffine affine0 = build_host_row_affine(
        row0, col, A_row_ptr, A_col_val, A_nz_val, AL, AU);
    const HostRowAffine affine1 = build_host_row_affine(
        row1, col, A_row_ptr, A_col_val, A_nz_val, AL, AU);
    if (!affine0.valid || !affine1.valid ||
        affine0.gives_lower == affine1.gives_lower) {
      throw std::runtime_error(
          "bounded two-row projection tape certificate changed before commit");
    }
    const HostRowAffine& lower_affine =
        affine0.gives_lower ? affine0 : affine1;
    std::vector<std::int32_t> indices{
        2, col, row0, row1,
        static_cast<std::int32_t>(lower_affine.cols.size())};
    indices.insert(indices.end(), lower_affine.cols.begin(),
                   lower_affine.cols.end());
    std::vector<double> vals{
        2.0,
        1.0, 0.0, lower[static_cast<std::size_t>(col)],
        1.0, static_cast<double>(lower_affine.cols.size()),
        lower_affine.constant};
    vals.insert(vals.end(), lower_affine.coeffs.begin(),
                lower_affine.coeffs.end());
    append_postsolve_record(tape, indices, vals);
  }
}

}  // namespace

BoundedTwoRowProjectionAnalysisGpu analyze_bounded_two_row_projection(
    const LPInfoGpu& lp,
    const PresolveParams& params) {
  BoundedTwoRowProjectionAnalysisGpu analysis;
  if (!params.enable_bounded_two_row_projection || lp.A.rows <= 0 ||
      lp.A.cols <= 0 || lp.A.nnz <= 0 || lp.AT.rows != lp.A.cols ||
      lp.AT.cols != lp.A.rows || lp.AT.nnz != lp.A.nnz ||
      lp.A.rowPtr == nullptr || lp.A.colVal == nullptr ||
      lp.A.nzVal == nullptr || lp.AT.rowPtr == nullptr ||
      lp.AT.colVal == nullptr || lp.c == nullptr || lp.AL == nullptr ||
      lp.AU == nullptr || lp.l == nullptr || lp.u == nullptr) {
    return analysis;
  }

  const std::int32_t max_row_nnz = std::max(
      2, std::min(MAX_CERTIFIED_ROW_NNZ,
                  params.bounded_two_row_projection_max_row_nnz));
  const double safety_tol = std::max(
      params.bounded_two_row_projection_safety_tol,
      std::max(10.0 * params.feasibility_tol, 100.0 * params.zero_tol));
  std::int32_t* row_candidate_count = nullptr;
  std::int32_t* selected_count = nullptr;
  try {
    throw_if_cuda_error(
        cudaMalloc(&analysis.selected_col,
                   static_cast<std::size_t>(lp.A.cols)),
        "cudaMalloc bounded projection candidate mask");
    throw_if_cuda_error(
        cudaMalloc(&row_candidate_count,
                   sizeof(std::int32_t) * static_cast<std::size_t>(lp.A.rows)),
        "cudaMalloc bounded projection row counts");
    throw_if_cuda_error(
        cudaMemset(row_candidate_count, 0,
                   sizeof(std::int32_t) * static_cast<std::size_t>(lp.A.rows)),
        "cudaMemset bounded projection row counts");
    throw_if_cuda_error(cudaMalloc(&selected_count, sizeof(std::int32_t)),
                        "cudaMalloc bounded projection selected count");
    throw_if_cuda_error(cudaMemset(selected_count, 0, sizeof(std::int32_t)),
                        "cudaMemset bounded projection selected count");

    const int blocks =
        (lp.A.cols + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    detect_bounded_two_row_candidates<<<blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.selected_col, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal,
        lp.AT.rowPtr, lp.AT.colVal, lp.c, lp.AL, lp.AU, lp.l, lp.u,
        lp.A.cols, max_row_nnz, params.zero_tol, safety_tol);
    throw_if_cuda_error(cudaGetLastError(),
                        "detect_bounded_two_row_candidates");
    count_candidate_rows<<<blocks, GPU_PRESOLVE_THREADS>>>(
        row_candidate_count, analysis.selected_col, lp.AT.rowPtr,
        lp.AT.colVal, lp.A.cols);
    throw_if_cuda_error(cudaGetLastError(),
                        "count bounded projection candidate rows");
    select_nonconflicting_candidates<<<blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.selected_col, selected_count, row_candidate_count,
        lp.AT.rowPtr, lp.AT.colVal, lp.A.cols);
    throw_if_cuda_error(cudaGetLastError(),
                        "select bounded projection candidates");
    throw_if_cuda_error(
        cudaMemcpy(&analysis.candidate_count, selected_count,
                   sizeof(std::int32_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy bounded projection selected count");
    const std::int32_t min_candidates =
        std::max(1, params.bounded_two_row_projection_min_candidates);
    const double candidate_ratio =
        static_cast<double>(analysis.candidate_count) /
        static_cast<double>(lp.A.cols);
    analysis.applicable =
        analysis.candidate_count >= min_candidates &&
        candidate_ratio + 1.0e-15 >=
            params.bounded_two_row_projection_min_candidate_ratio;
    if (!analysis.applicable) {
      cudaFree(analysis.selected_col);
      analysis.selected_col = nullptr;
      analysis.candidate_count = 0;
    }
    cudaFree(row_candidate_count);
    cudaFree(selected_count);
    return analysis;
  } catch (...) {
    cudaFree(row_candidate_count);
    cudaFree(selected_count);
    cudaFree(analysis.selected_col);
    analysis = BoundedTwoRowProjectionAnalysisGpu{};
    throw;
  }
}

void build_bounded_two_row_projection_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const BoundedTwoRowProjectionAnalysisGpu& analysis,
    const PresolveParams& params) {
  if (!analysis.applicable || analysis.candidate_count <= 0 ||
      analysis.selected_col == nullptr || plan.keep_row_mask == nullptr ||
      plan.keep_col_mask == nullptr) {
    return;
  }
  if (params.record_postsolve_tape) {
    append_projection_tape_from_device(plan.tape, lp,
                                       analysis.selected_col);
  }
  const int blocks =
      (lp.A.cols + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  apply_bounded_two_row_projection<<<blocks, GPU_PRESOLVE_THREADS>>>(
      plan.keep_row_mask, plan.keep_col_mask, analysis.selected_col,
      lp.AT.rowPtr, lp.AT.colVal, lp.A.cols);
  throw_if_cuda_error(cudaGetLastError(),
                      "apply_bounded_two_row_projection");
  throw_if_cuda_error(cudaDeviceSynchronize(),
                      "bounded two-row projection synchronize");
  plan.has_change = true;
  plan.has_row_action = true;
  plan.has_col_action = true;
  plan.has_projected_auxiliary_reduction = true;
}

void free_bounded_two_row_projection_analysis(
    BoundedTwoRowProjectionAnalysisGpu& analysis) {
  cudaFree(analysis.selected_col);
  analysis = BoundedTwoRowProjectionAnalysisGpu{};
}

}  // namespace gpu_presolver::presolve
