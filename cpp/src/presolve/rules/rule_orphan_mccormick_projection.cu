#include "../presolve_helpers.hpp"
#include "gpu_presolver/presolve/rules/rule_orphan_mccormick_projection.hpp"

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

__device__ bool exact_negative_infinity(double value) {
  return isinf(value) && value < 0.0;
}

__device__ bool exact_positive_infinity(double value) {
  return isinf(value) && value > 0.0;
}

struct McCormickRowPattern {
  bool valid = false;
  bool gives_lower = false;
  std::int32_t endpoint0 = -1;
  std::int32_t endpoint1 = -1;
};

__device__ McCormickRowPattern match_mccormick_row(
    const std::int32_t* row_ptr,
    const std::int32_t* col_val,
    const double* nz_val,
    const double* AL,
    const double* AU,
    std::int32_t row,
    std::int32_t z_col) {
  McCormickRowPattern result;
  const std::int32_t first = row_ptr[row];
  const std::int32_t last = row_ptr[row + 1];
  const std::int32_t row_nnz = last - first;
  if (row_nnz != 2 && row_nnz != 3) {
    return result;
  }
  double pivot = 0.0;
  std::int32_t pivot_count = 0;
  for (std::int32_t p = first; p < last; ++p) {
    if (col_val[p] == z_col) {
      pivot = nz_val[p];
      ++pivot_count;
    }
  }
  if (pivot_count != 1 || !isfinite(pivot) || pivot == 0.0) {
    return result;
  }

  double finite_bound = 0.0;
  if (isfinite(AL[row]) && exact_positive_infinity(AU[row])) {
    result.gives_lower = pivot > 0.0;
    finite_bound = AL[row];
  } else if (exact_negative_infinity(AL[row]) && isfinite(AU[row])) {
    result.gives_lower = pivot < 0.0;
    finite_bound = AU[row];
  } else {
    return result;
  }

  const std::int32_t expected_nnz = result.gives_lower ? 3 : 2;
  const double expected_bound = result.gives_lower ? -pivot : 0.0;
  if (row_nnz != expected_nnz || finite_bound != expected_bound) {
    return result;
  }
  std::int32_t endpoint_count = 0;
  for (std::int32_t p = first; p < last; ++p) {
    const std::int32_t col = col_val[p];
    if (col == z_col) {
      continue;
    }
    if (nz_val[p] != -pivot || endpoint_count >= 2) {
      return result;
    }
    if (endpoint_count == 0) {
      result.endpoint0 = col;
    } else {
      result.endpoint1 = col;
    }
    ++endpoint_count;
  }
  if (endpoint_count != (result.gives_lower ? 2 : 1) ||
      (result.gives_lower && result.endpoint0 == result.endpoint1)) {
    return result;
  }
  result.valid = true;
  return result;
}

__global__ void detect_orphan_mccormick_candidates(
    std::uint8_t* candidate,
    std::int32_t* endpoint0,
    std::int32_t* endpoint1,
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
    std::int32_t cols) {
  const std::int32_t z_col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (z_col >= cols) {
    return;
  }
  candidate[z_col] = std::uint8_t{0};
  endpoint0[z_col] = -1;
  endpoint1[z_col] = -1;
  if (c[z_col] != 0.0 || lower[z_col] != 0.0 || upper[z_col] != 1.0 ||
      AT_row_ptr[z_col + 1] - AT_row_ptr[z_col] != 3) {
    return;
  }

  const std::int32_t at_first = AT_row_ptr[z_col];
  const std::int32_t row0 = AT_col_val[at_first];
  const std::int32_t row1 = AT_col_val[at_first + 1];
  const std::int32_t row2 = AT_col_val[at_first + 2];
  if (row0 == row1 || row0 == row2 || row1 == row2) {
    return;
  }
  const McCormickRowPattern patterns[3] = {
      match_mccormick_row(A_row_ptr, A_col_val, A_nz_val, AL, AU,
                          row0, z_col),
      match_mccormick_row(A_row_ptr, A_col_val, A_nz_val, AL, AU,
                          row1, z_col),
      match_mccormick_row(A_row_ptr, A_col_val, A_nz_val, AL, AU,
                          row2, z_col)};
  std::int32_t lower_index = -1;
  std::int32_t upper_indices[2] = {-1, -1};
  std::int32_t upper_count = 0;
  for (std::int32_t k = 0; k < 3; ++k) {
    if (!patterns[k].valid) {
      return;
    }
    if (patterns[k].gives_lower) {
      if (lower_index >= 0) {
        return;
      }
      lower_index = k;
    } else {
      if (upper_count >= 2) {
        return;
      }
      upper_indices[upper_count++] = k;
    }
  }
  if (lower_index < 0 || upper_count != 2) {
    return;
  }
  const std::int32_t x = patterns[lower_index].endpoint0;
  const std::int32_t y = patterns[lower_index].endpoint1;
  const std::int32_t upper_x = patterns[upper_indices[0]].endpoint0;
  const std::int32_t upper_y = patterns[upper_indices[1]].endpoint0;
  if (x < 0 || y < 0 || x == y ||
      !((upper_x == x && upper_y == y) ||
        (upper_x == y && upper_y == x)) ||
      lower[x] != 0.0 || upper[x] != 1.0 ||
      lower[y] != 0.0 || upper[y] != 1.0) {
    return;
  }
  endpoint0[z_col] = x;
  endpoint1[z_col] = y;
  candidate[z_col] = std::uint8_t{1};
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
  atomicAdd(&row_candidate_count[AT_col_val[first + 2]], 1);
}

__global__ void select_orphan_mccormick_candidates(
    std::uint8_t* selected,
    std::int32_t* selected_count,
    const std::uint8_t* candidate,
    const std::int32_t* endpoint0,
    const std::int32_t* endpoint1,
    const std::int32_t* row_candidate_count,
    const std::int32_t* AT_row_ptr,
    const std::int32_t* AT_col_val,
    std::int32_t cols) {
  const std::int32_t col =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= cols) {
    return;
  }
  selected[col] = std::uint8_t{0};
  if (candidate[col] == std::uint8_t{0}) {
    return;
  }
  const std::int32_t first = AT_row_ptr[col];
  const bool row_disjoint =
      row_candidate_count[AT_col_val[first]] == 1 &&
      row_candidate_count[AT_col_val[first + 1]] == 1 &&
      row_candidate_count[AT_col_val[first + 2]] == 1;
  const bool endpoint_disjoint =
      candidate[endpoint0[col]] == std::uint8_t{0} &&
      candidate[endpoint1[col]] == std::uint8_t{0};
  if (row_disjoint && endpoint_disjoint) {
    selected[col] = std::uint8_t{1};
    atomicAdd(selected_count, 1);
  }
}

__global__ void apply_orphan_mccormick_projection(
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
  keep_row[AT_col_val[first + 2]] = std::uint8_t{0};
}

struct HostMcCormickPattern {
  bool valid = false;
  bool gives_lower = false;
  double constant = 0.0;
  std::vector<std::int32_t> endpoints;
  std::vector<double> coeffs;
};

HostMcCormickPattern build_host_pattern(
    std::int32_t row,
    std::int32_t z_col,
    const std::vector<std::int32_t>& row_ptr,
    const std::vector<std::int32_t>& col_val,
    const std::vector<double>& nz_val,
    const std::vector<double>& AL,
    const std::vector<double>& AU) {
  HostMcCormickPattern out;
  double pivot = 0.0;
  std::int32_t pivot_count = 0;
  for (std::int32_t p = row_ptr[static_cast<std::size_t>(row)];
       p < row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
    if (col_val[static_cast<std::size_t>(p)] == z_col) {
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
    if (col == z_col) {
      continue;
    }
    out.endpoints.push_back(col);
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
      selected_col, lp.A.cols, "cudaMemcpy McCormick selected mask");
  const std::vector<std::int32_t> A_row_ptr = copy_device_vector(
      lp.A.rowPtr, lp.A.rows + 1, "cudaMemcpy McCormick A rowPtr");
  const std::vector<std::int32_t> A_col_val = copy_device_vector(
      lp.A.colVal, lp.A.nnz, "cudaMemcpy McCormick A colVal");
  const std::vector<double> A_nz_val = copy_device_vector(
      lp.A.nzVal, lp.A.nnz, "cudaMemcpy McCormick A values");
  const std::vector<std::int32_t> AT_row_ptr = copy_device_vector(
      lp.AT.rowPtr, lp.A.cols + 1, "cudaMemcpy McCormick AT rowPtr");
  const std::vector<std::int32_t> AT_col_val = copy_device_vector(
      lp.AT.colVal, lp.AT.nnz, "cudaMemcpy McCormick AT colVal");
  const std::vector<double> AL = copy_device_vector(
      lp.AL, lp.A.rows, "cudaMemcpy McCormick AL");
  const std::vector<double> AU = copy_device_vector(
      lp.AU, lp.A.rows, "cudaMemcpy McCormick AU");

  for (std::int32_t z_col = 0; z_col < lp.A.cols; ++z_col) {
    if (selected[static_cast<std::size_t>(z_col)] == std::uint8_t{0}) {
      continue;
    }
    const std::int32_t at_first =
        AT_row_ptr[static_cast<std::size_t>(z_col)];
    const std::int32_t rows[3] = {
        AT_col_val[static_cast<std::size_t>(at_first)],
        AT_col_val[static_cast<std::size_t>(at_first + 1)],
        AT_col_val[static_cast<std::size_t>(at_first + 2)]};
    HostMcCormickPattern lower_pattern;
    for (const std::int32_t row : rows) {
      const HostMcCormickPattern pattern = build_host_pattern(
          row, z_col, A_row_ptr, A_col_val, A_nz_val, AL, AU);
      if (pattern.valid && pattern.gives_lower) {
        lower_pattern = pattern;
        break;
      }
    }
    if (!lower_pattern.valid || lower_pattern.endpoints.size() != 2) {
      throw std::runtime_error(
          "orphan McCormick tape certificate changed before commit");
    }
    std::vector<std::int32_t> indices{
        3, z_col, rows[0], rows[1], rows[2],
        static_cast<std::int32_t>(lower_pattern.endpoints.size())};
    indices.insert(indices.end(), lower_pattern.endpoints.begin(),
                   lower_pattern.endpoints.end());
    std::vector<double> vals{
        2.0,
        1.0, 0.0, 0.0,
        1.0, static_cast<double>(lower_pattern.endpoints.size()),
        lower_pattern.constant};
    vals.insert(vals.end(), lower_pattern.coeffs.begin(),
                lower_pattern.coeffs.end());
    append_postsolve_record(tape, indices, vals);
  }
}

}  // namespace

OrphanMcCormickProjectionAnalysisGpu analyze_orphan_mccormick_projection(
    const LPInfoGpu& lp,
    const PresolveParams& params) {
  OrphanMcCormickProjectionAnalysisGpu analysis;
  if (!params.enable_orphan_mccormick_projection || lp.A.rows <= 0 ||
      lp.A.cols <= 0 || lp.A.nnz <= 0 || lp.AT.rows != lp.A.cols ||
      lp.AT.cols != lp.A.rows || lp.AT.nnz != lp.A.nnz ||
      lp.A.rowPtr == nullptr || lp.A.colVal == nullptr ||
      lp.A.nzVal == nullptr || lp.AT.rowPtr == nullptr ||
      lp.AT.colVal == nullptr || lp.c == nullptr || lp.AL == nullptr ||
      lp.AU == nullptr || lp.l == nullptr || lp.u == nullptr) {
    return analysis;
  }

  std::uint8_t* candidate = nullptr;
  std::int32_t* endpoint0 = nullptr;
  std::int32_t* endpoint1 = nullptr;
  std::int32_t* row_candidate_count = nullptr;
  std::int32_t* selected_count = nullptr;
  try {
    throw_if_cuda_error(
        cudaMalloc(&candidate, static_cast<std::size_t>(lp.A.cols)),
        "cudaMalloc McCormick candidate mask");
    throw_if_cuda_error(
        cudaMalloc(&analysis.selected_col,
                   static_cast<std::size_t>(lp.A.cols)),
        "cudaMalloc McCormick selected mask");
    throw_if_cuda_error(
        cudaMalloc(&endpoint0,
                   sizeof(std::int32_t) * static_cast<std::size_t>(lp.A.cols)),
        "cudaMalloc McCormick endpoint0");
    throw_if_cuda_error(
        cudaMalloc(&endpoint1,
                   sizeof(std::int32_t) * static_cast<std::size_t>(lp.A.cols)),
        "cudaMalloc McCormick endpoint1");
    throw_if_cuda_error(
        cudaMalloc(&row_candidate_count,
                   sizeof(std::int32_t) * static_cast<std::size_t>(lp.A.rows)),
        "cudaMalloc McCormick row counts");
    throw_if_cuda_error(
        cudaMemset(row_candidate_count, 0,
                   sizeof(std::int32_t) * static_cast<std::size_t>(lp.A.rows)),
        "cudaMemset McCormick row counts");
    throw_if_cuda_error(cudaMalloc(&selected_count, sizeof(std::int32_t)),
                        "cudaMalloc McCormick selected count");
    throw_if_cuda_error(cudaMemset(selected_count, 0, sizeof(std::int32_t)),
                        "cudaMemset McCormick selected count");

    const int blocks =
        (lp.A.cols + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
    detect_orphan_mccormick_candidates<<<blocks, GPU_PRESOLVE_THREADS>>>(
        candidate, endpoint0, endpoint1, lp.A.rowPtr, lp.A.colVal,
        lp.A.nzVal, lp.AT.rowPtr, lp.AT.colVal, lp.c, lp.AL, lp.AU,
        lp.l, lp.u, lp.A.cols);
    throw_if_cuda_error(cudaGetLastError(),
                        "detect_orphan_mccormick_candidates");
    count_candidate_rows<<<blocks, GPU_PRESOLVE_THREADS>>>(
        row_candidate_count, candidate, lp.AT.rowPtr, lp.AT.colVal,
        lp.A.cols);
    throw_if_cuda_error(cudaGetLastError(),
                        "count McCormick candidate rows");
    select_orphan_mccormick_candidates<<<blocks, GPU_PRESOLVE_THREADS>>>(
        analysis.selected_col, selected_count, candidate, endpoint0,
        endpoint1, row_candidate_count, lp.AT.rowPtr, lp.AT.colVal,
        lp.A.cols);
    throw_if_cuda_error(cudaGetLastError(),
                        "select orphan McCormick candidates");
    throw_if_cuda_error(
        cudaMemcpy(&analysis.candidate_count, selected_count,
                   sizeof(std::int32_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy McCormick selected count");
    const std::int32_t min_candidates =
        std::max(1, params.orphan_mccormick_projection_min_candidates);
    const double candidate_ratio =
        static_cast<double>(analysis.candidate_count) /
        static_cast<double>(lp.A.cols);
    analysis.applicable =
        analysis.candidate_count >= min_candidates &&
        candidate_ratio + 1.0e-15 >=
            params.orphan_mccormick_projection_min_candidate_ratio;
    if (!analysis.applicable) {
      cudaFree(analysis.selected_col);
      analysis.selected_col = nullptr;
      analysis.candidate_count = 0;
    }
    cudaFree(candidate);
    cudaFree(endpoint0);
    cudaFree(endpoint1);
    cudaFree(row_candidate_count);
    cudaFree(selected_count);
    return analysis;
  } catch (...) {
    cudaFree(candidate);
    cudaFree(endpoint0);
    cudaFree(endpoint1);
    cudaFree(row_candidate_count);
    cudaFree(selected_count);
    cudaFree(analysis.selected_col);
    analysis = OrphanMcCormickProjectionAnalysisGpu{};
    throw;
  }
}

void build_orphan_mccormick_projection_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const OrphanMcCormickProjectionAnalysisGpu& analysis,
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
  apply_orphan_mccormick_projection<<<blocks, GPU_PRESOLVE_THREADS>>>(
      plan.keep_row_mask, plan.keep_col_mask, analysis.selected_col,
      lp.AT.rowPtr, lp.AT.colVal, lp.A.cols);
  throw_if_cuda_error(cudaGetLastError(),
                      "apply_orphan_mccormick_projection");
  throw_if_cuda_error(cudaDeviceSynchronize(),
                      "orphan McCormick projection synchronize");
  plan.has_change = true;
  plan.has_row_action = true;
  plan.has_col_action = true;
  plan.has_projected_auxiliary_reduction = true;
}

void free_orphan_mccormick_projection_analysis(
    OrphanMcCormickProjectionAnalysisGpu& analysis) {
  cudaFree(analysis.selected_col);
  analysis = OrphanMcCormickProjectionAnalysisGpu{};
}

}  // namespace gpu_presolver::presolve
