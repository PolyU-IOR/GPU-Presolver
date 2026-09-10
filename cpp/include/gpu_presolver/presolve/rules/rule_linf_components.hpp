#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

// Device-owned state shared by the fail-closed detector and the one-pass CSR
// quotient builder.  Linf models are laid out as
//
//   p_0 ... p_{k-1}, n_0 ... n_{k-1}, T_0 ... T_{g-1},
//
// where the short objective suffix contains only the T columns.  Each pair
// has one three-entry epigraph row; all remaining rows use p_i/n_i with
// exactly opposite coefficients.
struct LinfComponentAnalysisGpu {
  bool applicable = false;
  std::int32_t pair_count = 0;
  std::int32_t tail_count = 0;
  std::int32_t edge_count = 0;
  std::int32_t component_count = 0;
  std::int32_t removed_rows = 0;

  // Per-row classification.  NO_EDGE / -1 denote an ordinary quotient row.
  unsigned long long* edge_by_row = nullptr;
  std::int32_t* g_pair_by_row = nullptr;

  // Per-pair epigraph metadata and shifted-difference offset.
  std::int32_t* pair_g_row = nullptr;
  std::int32_t* pair_tail_col = nullptr;
  double* pair_g_coefficient = nullptr;
  double* pair_g_lower = nullptr;
  double* pair_shift = nullptr;

  // DSU and root-only original-coordinate upper-bound intersections.
  std::int32_t* parent = nullptr;
  std::int32_t* component_size = nullptr;
  double* component_plus_upper = nullptr;
  double* component_minus_upper = nullptr;
};

// Sample objective, pair, and row data for early rejection.
// A true result requires full validation before any transformation.
bool quick_probe_linf_components(const LPInfoGpu& lp,
                                 const PresolveParams& params);

LinfComponentAnalysisGpu analyze_linf_components(
    const LPInfoGpu& lp,
    const PresolveParams& params);

void build_linf_component_plan(PresolvePlanGpu& plan,
                               const LPInfoGpu& lp,
                               LinfComponentAnalysisGpu& analysis,
                               const PresolveParams& params);

void free_linf_component_analysis(LinfComponentAnalysisGpu& analysis);

}  // namespace gpu_presolver::presolve
