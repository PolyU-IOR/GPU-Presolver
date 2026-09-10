#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

void apply_rule_implied_variable_bounds(PresolvePlanGpu &plan,
                                        const LPInfoGpu &lp,
                                        const PresolveStatsGpu &stats,
                                        const PresolveParams &pparams);

// Releases thread-local CUDA scratch retained for reuse within one presolve.
void release_implied_variable_bounds_workspace() noexcept;

} // namespace gpu_presolver::presolve
