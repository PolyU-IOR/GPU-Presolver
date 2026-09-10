#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

// Exact detector for a zero-cost unit-box auxiliary z whose complete current
// incidence is z >= x+y-1, z <= x, z <= y.  Conflicting gadgets and gadgets
// whose endpoint is itself selected are rejected conservatively.
struct OrphanMcCormickProjectionAnalysisGpu {
  bool applicable = false;
  std::int32_t candidate_count = 0;
  std::uint8_t* selected_col = nullptr;
};

OrphanMcCormickProjectionAnalysisGpu analyze_orphan_mccormick_projection(
    const LPInfoGpu& lp,
    const PresolveParams& params);

void build_orphan_mccormick_projection_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const OrphanMcCormickProjectionAnalysisGpu& analysis,
    const PresolveParams& params);

void free_orphan_mccormick_projection_analysis(
    OrphanMcCormickProjectionAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve
