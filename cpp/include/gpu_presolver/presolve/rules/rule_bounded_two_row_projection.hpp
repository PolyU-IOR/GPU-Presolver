#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

// A selected column is certified zero-cost, finite-bounded, degree two, and
// incident to two distinct short one-sided rows that give opposite affine
// bounds.  All three projected lower/upper compatibility conditions are
// strictly redundant over the current variable box.
struct BoundedTwoRowProjectionAnalysisGpu {
  bool applicable = false;
  std::int32_t candidate_count = 0;
  std::uint8_t* selected_col = nullptr;
};

BoundedTwoRowProjectionAnalysisGpu analyze_bounded_two_row_projection(
    const LPInfoGpu& lp,
    const PresolveParams& params);

void build_bounded_two_row_projection_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const BoundedTwoRowProjectionAnalysisGpu& analysis,
    const PresolveParams& params);

void free_bounded_two_row_projection_analysis(
    BoundedTwoRowProjectionAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve
