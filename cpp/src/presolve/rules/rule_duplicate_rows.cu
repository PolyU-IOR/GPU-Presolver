#include "../presolve_helpers.hpp"
#include "gpu_presolver/presolve/rules/rule_duplicate_rows.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
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

struct HashRowLess {
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

struct DuplicateRowTapeRecord {
  std::int32_t step;
  std::int32_t kept_row;
  std::int32_t deleted_row;
  double ratio;
  double kept_old_AL;
  double kept_old_AU;
  double deleted_old_AL;
  double deleted_old_AU;
};

__device__ __host__ unsigned long long _duplicate_row_hash_mix(unsigned long long h,
                                                              unsigned long long x) {
  return (h ^ x) * FNV_PRIME;
}

__device__ unsigned long long _double_bits(double value) {
  return static_cast<unsigned long long>(__double_as_longlong(value));
}

__global__ void _kernel_duplicate_row_hashes(unsigned long long* row_hash,
                                            const std::uint8_t* keep_row,
                                            const std::int32_t* row_ptr,
                                            const std::int32_t* col_val,
                                            const double* nz_val,
                                            double coeff_tol,
                                            std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m) {
    if (keep_row[i] == std::uint8_t{0}) {
      row_hash[i] = static_cast<unsigned long long>(i);
      return;
    }

    const std::int32_t start_i = row_ptr[i];
    const std::int32_t stop_i = row_ptr[i + 1];
    const std::int32_t len_i = stop_i - start_i;
    if (len_i <= 0) {
      row_hash[i] = _duplicate_row_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(i));
      return;
    }

    const double pivot = nz_val[start_i];
    if (fabs(pivot) <= coeff_tol) {
      row_hash[i] = _duplicate_row_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(i));
      return;
    }

    unsigned long long h = _duplicate_row_hash_mix(FNV_OFFSET, static_cast<unsigned long long>(len_i));
    for (std::int32_t p = start_i; p < stop_i; ++p) {
      h = _duplicate_row_hash_mix(h, static_cast<unsigned long long>(col_val[p]));
      const double norm = nz_val[p] / pivot;
      h = _duplicate_row_hash_mix(h, _double_bits(norm));
    }
    row_hash[i] = h;
  }
}

__global__ void _kernel_duplicate_row_hashes_warp(unsigned long long* row_hash,
                                                 const std::uint8_t* keep_row,
                                                 const std::int32_t* row_ptr,
                                                 const std::int32_t* col_val,
                                                 const double* nz_val,
                                                 double coeff_tol,
                                                 std::int32_t m) {
  constexpr std::int32_t warp_size = 32;
  const std::int32_t lane =
      static_cast<std::int32_t>(threadIdx.x) & (warp_size - 1);
  const std::int32_t warps_per_block =
      static_cast<std::int32_t>(blockDim.x) / warp_size;
  const std::int32_t warp =
      static_cast<std::int32_t>(threadIdx.x) / warp_size;
  const std::int32_t i =
      static_cast<std::int32_t>(blockIdx.x) * warps_per_block + warp;
  if (i >= m) {
    return;
  }

  if (keep_row[i] == std::uint8_t{0}) {
    if (lane == 0) {
      row_hash[i] = static_cast<unsigned long long>(i);
    }
    return;
  }

  const std::int32_t start_i = row_ptr[i];
  const std::int32_t stop_i = row_ptr[i + 1];
  const std::int32_t len_i = stop_i - start_i;
  if (len_i <= 0) {
    if (lane == 0) {
      row_hash[i] = _duplicate_row_hash_mix(
          FNV_OFFSET, static_cast<unsigned long long>(i));
    }
    return;
  }

  const double pivot = nz_val[start_i];
  if (fabs(pivot) <= coeff_tol) {
    if (lane == 0) {
      row_hash[i] = _duplicate_row_hash_mix(
          FNV_OFFSET, static_cast<unsigned long long>(i));
    }
    return;
  }

  unsigned long long aggregate = 0ULL;
  for (std::int32_t p = start_i + lane; p < stop_i; p += warp_size) {
    unsigned long long entry = _duplicate_row_hash_mix(
        FNV_OFFSET, static_cast<unsigned long long>(col_val[p]));
    const double norm = nz_val[p] / pivot;
    entry = _duplicate_row_hash_mix(entry, _double_bits(norm));
    aggregate ^= entry;
  }
  for (std::int32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    aggregate ^= __shfl_down_sync(0xffffffffU, aggregate, offset);
  }
  if (lane == 0) {
    const unsigned long long seed = _duplicate_row_hash_mix(
        FNV_OFFSET, static_cast<unsigned long long>(len_i));
    row_hash[i] = _duplicate_row_hash_mix(seed, aggregate);
  }
}

__device__ bool _duplicate_row_ratio(std::int32_t i,
                                    std::int32_t k,
                                    const std::int32_t* row_ptr,
                                    const std::int32_t* col_val,
                                    const double* nz_val,
                                    double coeff_tol,
                                    double* ratio_out) {
  const std::int32_t start_i = row_ptr[i];
  const std::int32_t stop_i = row_ptr[i + 1];
  const std::int32_t start_k = row_ptr[k];
  const std::int32_t stop_k = row_ptr[k + 1];
  const std::int32_t len_i = stop_i - start_i;
  const std::int32_t len_k = stop_k - start_k;
  if (len_i != len_k) {
    return false;
  }
  if (len_i <= 0) {
    *ratio_out = 1.0;
    return true;
  }

  double ratio = 0.0;
  bool ratio_set = false;
  for (std::int32_t offset = 0; offset < len_i; ++offset) {
    const std::int32_t p_i = start_i + offset;
    const std::int32_t p_k = start_k + offset;
    if (col_val[p_i] != col_val[p_k]) {
      return false;
    }
    const double a_i = nz_val[p_i];
    const double a_k = nz_val[p_k];
    if (!ratio_set) {
      if (fabs(a_k) <= coeff_tol) {
        return false;
      }
      ratio = a_i / a_k;
      ratio_set = true;
    }
    if (fabs(a_i - ratio * a_k) > coeff_tol) {
      return false;
    }
  }
  *ratio_out = ratio;
  return ratio_set;
}

__device__ void _scaled_interval_for_duplicate_row(double lower,
                                                  double upper,
                                                  double ratio,
                                                  double* scaled_l,
                                                  double* scaled_u) {
  const double raw_l = lower * ratio;
  const double raw_u = upper * ratio;
  if (ratio >= 0.0) {
    *scaled_l = raw_l;
    *scaled_u = raw_u;
  } else {
    *scaled_l = raw_u;
    *scaled_u = raw_l;
  }
}

__device__ bool _intervals_disjoint(double l1, double u1, double l2, double u2, double tol) {
  return fmax(l1, l2) > fmin(u1, u2) + tol;
}

__device__ bool _interval_contains(double outer_l,
                                   double outer_u,
                                   double inner_l,
                                   double inner_u,
                                   double tol) {
  return outer_l <= inner_l + tol && inner_u <= outer_u + tol;
}

__device__ bool _row_interval_is_equality(double lower, double upper, double tol) {
  return isfinite(lower) && isfinite(upper) && fabs(lower - upper) <= tol;
}

__global__ void _kernel_duplicate_row_groups(std::int32_t* status_flag,
                                            std::uint8_t* row_delete,
                                            std::int32_t* merge_to,
                                            double* merge_ratio,
                                            double* kept_old_AL,
                                            double* kept_old_AU,
                                            double* deleted_old_AL,
                                            double* deleted_old_AU,
                                            std::int32_t* merge_step,
                                            const unsigned long long* sorted_hash,
                                            const std::int32_t* sorted_rows,
                                            const std::uint8_t* keep_row,
                                            double* AL,
                                            double* AU,
                                            const std::int32_t* row_ptr,
                                            const std::int32_t* col_val,
                                            const double* nz_val,
                                            double coeff_tol,
                                            double interval_tol,
                                            std::int32_t m) {
  const std::int32_t s = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s < m) {
    const unsigned long long hs = sorted_hash[s];
    if (s > 0 && sorted_hash[s - 1] == hs) {
      return;
    }

    std::int32_t e = s;
    while (e + 1 < m && sorted_hash[e + 1] == hs) {
      ++e;
    }
    if (e <= s) {
      return;
    }

    std::int32_t rep_seed = -1;
    bool rep_seed_is_eq = false;
    for (std::int32_t a = s; a <= e; ++a) {
      const std::int32_t row = sorted_rows[a];
      if (keep_row[row] == std::uint8_t{0}) {
        continue;
      }
      const double row_l = AL[row];
      const double row_u = AU[row];
      const bool row_is_eq = _row_interval_is_equality(row_l, row_u, interval_tol);
      if (rep_seed < 0 || (row_is_eq && !rep_seed_is_eq)) {
        rep_seed = row;
        rep_seed_is_eq = row_is_eq;
      }
    }
    if (rep_seed < 0) {
      return;
    }

    std::int32_t rep = rep_seed;
    const double rep_seed_l = AL[rep_seed];
    const double rep_seed_u = AU[rep_seed];
    double rep_scaled_l = rep_seed_l;
    double rep_scaled_u = rep_seed_u;
    for (std::int32_t a = s; a <= e; ++a) {
      const std::int32_t row = sorted_rows[a];
      if (row == rep_seed || keep_row[row] == std::uint8_t{0}) {
        continue;
      }

      double ratio_seed = 0.0;
      if (!_duplicate_row_ratio(rep_seed, row, row_ptr, col_val, nz_val, coeff_tol, &ratio_seed)) {
        continue;
      }

      double row_scaled_l = 0.0;
      double row_scaled_u = 0.0;
      _scaled_interval_for_duplicate_row(AL[row], AU[row], ratio_seed, &row_scaled_l, &row_scaled_u);
      if (_interval_contains(rep_scaled_l, rep_scaled_u, row_scaled_l, row_scaled_u, interval_tol)) {
        rep = row;
        rep_scaled_l = row_scaled_l;
        rep_scaled_u = row_scaled_u;
      }
    }

    double rep_l = AL[rep];
    double rep_u = AU[rep];
    bool found_parallel_peer = false;
    std::int32_t step = 0;
    for (std::int32_t a = s; a <= e; ++a) {
      const std::int32_t row = sorted_rows[a];
      if (row == rep || keep_row[row] == std::uint8_t{0}) {
        continue;
      }

      double ratio = 0.0;
      if (!_duplicate_row_ratio(rep, row, row_ptr, col_val, nz_val, coeff_tol, &ratio)) {
        continue;
      }
      found_parallel_peer = true;

      double row_l = 0.0;
      double row_u = 0.0;
      _scaled_interval_for_duplicate_row(AL[row], AU[row], ratio, &row_l, &row_u);
      if (_intervals_disjoint(rep_l, rep_u, row_l, row_u, interval_tol)) {
        atomicMax(&status_flag[0], 1);
        return;
      }

      ++step;
      merge_to[row] = rep;
      merge_ratio[row] = ratio;
      kept_old_AL[row] = rep_l;
      kept_old_AU[row] = rep_u;
      deleted_old_AL[row] = AL[row];
      deleted_old_AU[row] = AU[row];
      merge_step[row] = step;

      rep_l = fmax(rep_l, row_l);
      rep_u = fmin(rep_u, row_u);
      row_delete[row] = std::uint8_t{1};
    }

    if (found_parallel_peer) {
      AL[rep] = rep_l;
      AU[rep] = rep_u;
    }
  }
}

__global__ void _kernel_apply_duplicate_row_deletions(std::uint8_t* keep_row,
                                                     const std::uint8_t* row_delete,
                                                     std::int32_t* changed,
                                                     std::int32_t m) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < m && row_delete[i] != std::uint8_t{0}) {
    keep_row[i] = std::uint8_t{0};
    atomicAdd(changed, 1);
  }
}

__global__ void _kernel_pack_duplicate_row_tape(
    DuplicateRowTapeRecord* records,
    std::int32_t* count,
    const std::uint8_t* row_delete,
    const std::int32_t* merge_to,
    const double* merge_ratio,
    const double* kept_old_AL,
    const double* kept_old_AU,
    const double* deleted_old_AL,
    const double* deleted_old_AU,
    const std::int32_t* merge_step,
    std::int32_t m) {
  const std::int32_t row =
      static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row < m && row_delete[row] != std::uint8_t{0}) {
    const std::int32_t slot = atomicAdd(count, 1);
    records[slot] = DuplicateRowTapeRecord{merge_step[row],
                                          merge_to[row],
                                          row,
                                          merge_ratio[row],
                                          kept_old_AL[row],
                                          kept_old_AU[row],
                                          deleted_old_AL[row],
                                          deleted_old_AU[row]};
  }
}

}  // namespace

void apply_rule_duplicate_rows(PresolvePlanGpu& plan,
                              const LPInfoGpu& lp,
                              const PresolveStatsGpu& stats,
                              const PresolveParams& pparams) {
  (void)stats;
  if (plan.has_infeasible || plan.has_unbounded) {
    return;
  }

  const std::int32_t m = lp.A.rows;
  if (m <= 1) {
    return;
  }
  const bool profile = env_enabled("GPUPRESOLVER_DUPLICATE_ROWS_PROFILE");
  const auto rule_start = std::chrono::steady_clock::now();
  auto profile_stage = [&](const char* stage, const std::chrono::steady_clock::time_point& stage_start) {
    if (!profile) {
      return;
    }
    throw_if_cuda_error(cudaDeviceSynchronize(), "duplicate_rows profile synchronize");
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - stage_start;
    std::cerr << ">>> [duplicate_rows C++] " << stage << " = " << elapsed.count() << "s\n";
  };

  const std::size_t m_size = static_cast<std::size_t>(m);
  std::size_t scratch_bytes = 0;
  const auto row_hash_offset = reserve_scratch<unsigned long long>(scratch_bytes, m_size);
  const auto sorted_rows_offset = reserve_scratch<std::int32_t>(scratch_bytes, m_size);
  const auto status_flag_offset = reserve_scratch<std::int32_t>(scratch_bytes, 1);
  const auto row_delete_offset = reserve_scratch<std::uint8_t>(scratch_bytes, m_size);
  const auto merge_to_offset = reserve_scratch<std::int32_t>(scratch_bytes, m_size);
  const auto merge_ratio_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto kept_old_AL_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto kept_old_AU_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto deleted_old_AL_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto deleted_old_AU_offset = reserve_scratch<double>(scratch_bytes, m_size);
  const auto merge_step_offset = reserve_scratch<std::int32_t>(scratch_bytes, m_size);
  const auto changed_offset = reserve_scratch<std::int32_t>(scratch_bytes, 1);
  void* scratch_storage = nullptr;
  throw_if_cuda_error(cudaMalloc(&scratch_storage, scratch_bytes),
                      "cudaMalloc duplicate_rows scratch storage");
  auto* row_hash = scratch_at<unsigned long long>(scratch_storage, row_hash_offset);
  auto* sorted_rows = scratch_at<std::int32_t>(scratch_storage, sorted_rows_offset);
  auto* status_flag = scratch_at<std::int32_t>(scratch_storage, status_flag_offset);
  auto* row_delete = scratch_at<std::uint8_t>(scratch_storage, row_delete_offset);
  auto* merge_to = scratch_at<std::int32_t>(scratch_storage, merge_to_offset);
  auto* merge_ratio = scratch_at<double>(scratch_storage, merge_ratio_offset);
  auto* kept_old_AL = scratch_at<double>(scratch_storage, kept_old_AL_offset);
  auto* kept_old_AU = scratch_at<double>(scratch_storage, kept_old_AU_offset);
  auto* deleted_old_AL = scratch_at<double>(scratch_storage, deleted_old_AL_offset);
  auto* deleted_old_AU = scratch_at<double>(scratch_storage, deleted_old_AU_offset);
  auto* merge_step = scratch_at<std::int32_t>(scratch_storage, merge_step_offset);
  auto* changed = scratch_at<std::int32_t>(scratch_storage, changed_offset);
  throw_if_cuda_error(cudaMemset(status_flag, 0, sizeof(std::int32_t)), "cudaMemset duplicate_rows status_flag");
  throw_if_cuda_error(cudaMemset(row_delete, 0, static_cast<std::size_t>(m)), "cudaMemset duplicate_rows row_delete");
  throw_if_cuda_error(cudaMemset(changed, 0, sizeof(std::int32_t)), "cudaMemset duplicate_rows changed");

  const int blocks = (m + GPU_PRESOLVE_THREADS - 1) / GPU_PRESOLVE_THREADS;
  const bool use_warp_hash =
      static_cast<std::int64_t>(lp.A.nnz) >
      32LL * static_cast<std::int64_t>(m);
  if (use_warp_hash) {
    constexpr int warps_per_block = GPU_PRESOLVE_THREADS / 32;
    const int warp_blocks = (m + warps_per_block - 1) / warps_per_block;
    _kernel_duplicate_row_hashes_warp<<<warp_blocks, GPU_PRESOLVE_THREADS>>>(
        row_hash, plan.keep_row_mask, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal,
        pparams.zero_tol, m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_duplicate_row_hashes_warp");
  } else {
    _kernel_duplicate_row_hashes<<<blocks, GPU_PRESOLVE_THREADS>>>(
        row_hash, plan.keep_row_mask, lp.A.rowPtr, lp.A.colVal, lp.A.nzVal,
        pparams.zero_tol, m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_duplicate_row_hashes");
  }
  throw_if_cuda_error(cudaDeviceSynchronize(), "duplicate_rows hash synchronize");
  profile_stage("hash", rule_start);

  const auto sort_start = std::chrono::steady_clock::now();
  auto row_hash_begin = thrust::device_pointer_cast(row_hash);
  auto sorted_rows_begin = thrust::device_pointer_cast(sorted_rows);
  thrust::sequence(sorted_rows_begin, sorted_rows_begin + m, std::int32_t{0});
  auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(row_hash_begin, sorted_rows_begin));
  thrust::sort(zipped_begin, zipped_begin + m, HashRowLess{});
  throw_if_cuda_error(cudaGetLastError(), "thrust duplicate_rows sort hashes");
  profile_stage("device_sort", sort_start);

  const auto groups_start = std::chrono::steady_clock::now();
  _kernel_duplicate_row_groups<<<blocks, GPU_PRESOLVE_THREADS>>>(
      status_flag,
      row_delete,
      merge_to,
      merge_ratio,
      kept_old_AL,
      kept_old_AU,
      deleted_old_AL,
      deleted_old_AU,
      merge_step,
      row_hash,
      sorted_rows,
      plan.keep_row_mask,
      plan.new_AL,
      plan.new_AU,
      lp.A.rowPtr,
      lp.A.colVal,
      lp.A.nzVal,
      pparams.zero_tol,
      pparams.feasibility_tol,
      m);
  throw_if_cuda_error(cudaGetLastError(), "_kernel_duplicate_row_groups");
  profile_stage("groups", groups_start);

  const auto apply_start = std::chrono::steady_clock::now();
  std::int32_t status = 0;
  throw_if_cuda_error(cudaMemcpy(&status, status_flag, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                      "cudaMemcpy duplicate_rows status");
  if (status != 0) {
    plan.has_infeasible = true;
  } else {
    _kernel_apply_duplicate_row_deletions<<<blocks, GPU_PRESOLVE_THREADS>>>(
        plan.keep_row_mask,
        row_delete,
        changed,
        m);
    throw_if_cuda_error(cudaGetLastError(), "_kernel_apply_duplicate_row_deletions");
    throw_if_cuda_error(cudaDeviceSynchronize(), "apply_rule_duplicate_rows synchronize");
    std::int32_t changed_host = 0;
    throw_if_cuda_error(cudaMemcpy(&changed_host, changed, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                        "cudaMemcpy duplicate_rows changed");
    profile_stage("apply_delete", apply_start);
    if (changed_host != 0) {
      if (pparams.record_postsolve_tape) {
        const auto tape_start = std::chrono::steady_clock::now();
        DuplicateRowTapeRecord* packed_records = nullptr;
        throw_if_cuda_error(
            cudaMalloc(&packed_records,
                       sizeof(DuplicateRowTapeRecord) * static_cast<std::size_t>(changed_host)),
            "cudaMalloc packed duplicate-row tape");
        throw_if_cuda_error(cudaMemset(changed, 0, sizeof(std::int32_t)),
                            "cudaMemset duplicate-row packed count");
        _kernel_pack_duplicate_row_tape<<<blocks, GPU_PRESOLVE_THREADS>>>(
            packed_records, changed, row_delete, merge_to, merge_ratio, kept_old_AL,
            kept_old_AU, deleted_old_AL, deleted_old_AU, merge_step, m);
        throw_if_cuda_error(cudaGetLastError(), "_kernel_pack_duplicate_row_tape");
        std::vector<DuplicateRowTapeRecord> tape_records(
            static_cast<std::size_t>(changed_host));
        throw_if_cuda_error(
            cudaMemcpy(tape_records.data(), packed_records,
                       sizeof(DuplicateRowTapeRecord) * tape_records.size(),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy packed duplicate-row tape");
        cudaFree(packed_records);
        std::sort(tape_records.begin(), tape_records.end(),
                  [](const DuplicateRowTapeRecord& lhs, const DuplicateRowTapeRecord& rhs) {
                    if (lhs.step != rhs.step) {
                      return lhs.step < rhs.step;
                    }
                    return lhs.deleted_row < rhs.deleted_row;
                  });
        plan.tape.types.reserve(plan.tape.types.size() + tape_records.size());
        plan.tape.indices.reserve(plan.tape.indices.size() + 2 * tape_records.size());
        plan.tape.vals.reserve(plan.tape.vals.size() + 5 * tape_records.size());
        plan.tape.index_starts.reserve(plan.tape.index_starts.size() + tape_records.size());
        plan.tape.value_starts.reserve(plan.tape.value_starts.size() + tape_records.size());
        plan.tape.dual_modes.reserve(plan.tape.dual_modes.size() + tape_records.size());
        for (const DuplicateRowTapeRecord& record : tape_records) {
          const std::int32_t indices[] = {record.kept_row, record.deleted_row};
          const double vals[] = {record.ratio,
                                 record.kept_old_AL,
                                 record.kept_old_AU,
                                 record.deleted_old_AL,
                                 record.deleted_old_AU};
          append_postsolve_record(
              plan.tape,
              PostsolveReductionType::DuplicateRow,
              indices,
              2,
              vals,
              5,
              PostsolveDualMode::Exact);
        }
        profile_stage("tape", tape_start);
      }
      plan.has_row_action = true;
      plan.has_change = true;
    }
  }

  cudaFree(scratch_storage);
  profile_stage("total", rule_start);
}

}  // namespace gpu_presolver::presolve
