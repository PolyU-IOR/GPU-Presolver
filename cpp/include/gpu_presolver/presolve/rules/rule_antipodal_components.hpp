#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

// Device analysis shared by detection and CSR rebuilding to avoid
// repeating union-find or transferring component maps to the host.
struct AntipodalComponentAnalysisGpu {
  bool applicable = false;
  std::int32_t pair_count = 0;
  std::int32_t edge_count = 0;
  std::int32_t component_count = 0;
  std::int32_t zero_rows = 0;

  unsigned long long* edge_by_row = nullptr;
  std::int32_t* parent = nullptr;
  double* pair_shift = nullptr;
  double* component_plus_cap = nullptr;
  double* component_minus_cap = nullptr;
  double* component_rho = nullptr;
  double objective_constant_delta = 0.0;
};

// Sample column pairs, degrees, and merge-row density for early rejection.
// A true result requires full validation before any transformation.
bool quick_probe_antipodal_components(const LPInfoGpu& lp,
                                      const PresolveParams& params);

AntipodalComponentAnalysisGpu analyze_antipodal_components(
    const LPInfoGpu& lp,
    const PresolveParams& params);

void build_antipodal_component_plan(PresolvePlanGpu& plan,
                                    const LPInfoGpu& lp,
                                    AntipodalComponentAnalysisGpu& analysis,
                                    const PresolveParams& params);

void free_antipodal_component_analysis(AntipodalComponentAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve
