#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

void apply_rule_column_singletons_eq(PresolvePlanGpu &plan, const LPInfoGpu &lp,
                                     const PresolveStatsGpu &stats,
                                     const PresolveParams &pparams,
                                     bool skip_precheck = false);

void apply_rule_column_singletons_dual_infer(PresolvePlanGpu &plan,
                                             const LPInfoGpu &lp,
                                             const PresolveStatsGpu &stats,
                                             const PresolveParams &pparams);

void release_column_singletons_workspace() noexcept;

} // namespace gpu_presolver::presolve
