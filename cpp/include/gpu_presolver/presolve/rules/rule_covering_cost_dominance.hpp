#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

// Device-owned deletion certificate for the strict unit set-covering rule.
// `applicable` is set only after every active row, column, coefficient, and
// row witness has passed the full detector.
struct CoveringCostDominanceAnalysisGpu {
  bool applicable = false;
  std::int32_t candidate_count = 0;
  std::uint8_t* delete_col = nullptr;
};

// Sampled rejection filter only.  A true result never authorizes a model
// change; analyze_covering_cost_dominance still performs a full certificate.
bool quick_probe_covering_cost_dominance(const LPInfoGpu& lp,
                                         const PresolveParams& params);

CoveringCostDominanceAnalysisGpu analyze_covering_cost_dominance(
    const LPInfoGpu& lp,
    const PresolveParams& params);

void build_covering_cost_dominance_plan(
    PresolvePlanGpu& plan,
    const LPInfoGpu& lp,
    const CoveringCostDominanceAnalysisGpu& analysis,
    const PresolveParams& params);

void free_covering_cost_dominance_analysis(
    CoveringCostDominanceAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve
