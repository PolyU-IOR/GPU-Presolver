#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

struct GpuPresolveSummary {
  GpuPresolveSummary() = default;
  GpuPresolveSummary(const GpuPresolveSummary&) = delete;
  GpuPresolveSummary& operator=(const GpuPresolveSummary&) = delete;
  GpuPresolveSummary(GpuPresolveSummary&& other) noexcept;
  GpuPresolveSummary& operator=(GpuPresolveSummary&& other) noexcept;

  std::int32_t original_rows = 0;
  std::int32_t original_cols = 0;
  std::int32_t reduced_rows = 0;
  std::int32_t reduced_cols = 0;
  std::int32_t reduced_nnz = 0;
  std::int32_t iterations = 0;
  double elapsed_seconds = 0.0;
  double obj_constant_delta = 0.0;
  bool has_infeasible = false;
  bool has_unbounded = false;
  PresolveRecordGpu record;
  LPInfoGpu reduced_lp;
  bool owns_reduced_lp = false;
};

// Tests whether the nonzero count decreased beyond the supplied ratio.
bool _has_good_nnz_progress(std::int32_t nnz_before, std::int32_t nnz_after, double ratio);

GpuPresolveSummary run_gpu_presolve_with_record(const LPInfoGpu& lp,
                                                const PresolveParams& params);

GpuPresolveSummary run_gpu_presolve_with_reduced_lp(const LPInfoGpu& lp,
                                                    const PresolveParams& params);

void free_gpu_presolve_reduced_lp(GpuPresolveSummary& summary);

}  // namespace gpu_presolver::presolve
