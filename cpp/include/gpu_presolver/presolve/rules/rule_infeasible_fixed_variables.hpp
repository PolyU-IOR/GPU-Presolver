#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

namespace gpu_presolver::presolve {

// Detect inconsistent bounds and apply fixed-variable reductions to the plan.
void apply_rule_infeasible_fixed_variables(PresolvePlanGpu &plan,
                                           const LPInfoGpu &lp,
                                           const PresolveParams &pparams);

} // namespace gpu_presolver::presolve
