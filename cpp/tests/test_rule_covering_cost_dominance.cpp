#include "test_cuda_helpers.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/rules/rule_covering_cost_dominance.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using gpu_presolver_test::check;

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  T* out = nullptr;
  check(cudaMalloc(&out, sizeof(T) * values.size()), "cudaMalloc test data");
  check(cudaMemcpy(out, values.data(), sizeof(T) * values.size(),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy test data");
  return out;
}

void free_lp(gpu_presolver::presolve::LPInfoGpu& lp) {
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
  cudaFree(lp.l);
  cudaFree(lp.u);
  lp = gpu_presolver::presolve::LPInfoGpu{};
}

enum class Variant {
  Valid,
  NonUnitCoefficient,
  NonCoveringRow,
  MissingWitness,
  NonBinaryBound,
};

gpu_presolver::presolve::LPInfoGpu make_covering_lp(Variant variant) {
  using gpu_presolver::presolve::LPInfoGpu;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> a_values(11, 1.0);
  std::vector<double> at_values(11, 1.0);
  std::vector<double> costs{1.0, 1.0, 1.0, 3.0, 3.0, 4.0};
  std::vector<double> row_lower{1.0, 1.0, 1.0};
  std::vector<double> row_upper{inf, inf, inf};
  std::vector<double> upper(6, 1.0);
  if (variant == Variant::NonUnitCoefficient) {
    a_values[1] = 0.5;
    at_values[3] = 0.5;
  } else if (variant == Variant::NonCoveringRow) {
    row_upper[0] = 1.0;
  } else if (variant == Variant::MissingWitness) {
    costs[2] = 2.0;
  } else if (variant == Variant::NonBinaryBound) {
    upper[5] = 2.0;
  }

  LPInfoGpu lp;
  lp.A = {3,
          6,
          11,
          copy_to_device<std::int32_t>({0, 4, 8, 11}),
          copy_to_device<std::int32_t>({0, 3, 4, 5, 1, 3, 4, 5, 2, 4, 5}),
          copy_to_device<double>(a_values)};
  lp.AT = {6,
           3,
           11,
           copy_to_device<std::int32_t>({0, 1, 2, 3, 5, 8, 11}),
           copy_to_device<std::int32_t>({0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2}),
           copy_to_device<double>(at_values)};
  lp.c = copy_to_device<double>(costs);
  lp.AL = copy_to_device<double>(row_lower);
  lp.AU = copy_to_device<double>(row_upper);
  lp.l = copy_to_device<double>(std::vector<double>(6, 0.0));
  lp.u = copy_to_device<double>(upper);
  return lp;
}

gpu_presolver::presolve::PresolveParams test_params() {
  gpu_presolver::presolve::PresolveParams params;
  params.max_iters = 1;
  params.enable_antipodal_components = false;
  params.enable_infeasible_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_redundant_rows = false;
  params.enable_implied_variable_bounds = false;
  params.enable_duplicate_rows = false;
  params.enable_empty_cols = false;
  params.enable_column_singletons_eq = false;
  params.enable_column_singletons_dual_infer = false;
  params.enable_doubleton_equations = false;
  params.enable_structural_l1_substitution = false;
  params.enable_dual_fix = false;
  params.enable_duplicate_columns = false;
  params.enable_redundant_bounds = false;
  params.covering_cost_dominance_min_fixed_cols = 1;
  params.covering_cost_dominance_min_candidate_ratio = 0.20;
  params.covering_cost_dominance_probe_rows = 3;
  params.covering_cost_dominance_probe_cols = 6;
  params.covering_cost_dominance_probe_nnz = 11;
  return params;
}

void test_exact_reduction_and_primal_recovery() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_covering_lp(Variant::Valid);
  PresolveParams params = test_params();
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 3);
  assert(summary.reduced_cols == 4);
  assert(summary.reduced_nnz == 6);
  assert(summary.record.has_primal_only_covering_cost_dominance_reduction);
  assert(summary.record.fixed_idx.size() == 2);
  assert(std::find(summary.record.fixed_idx.begin(), summary.record.fixed_idx.end(), 3) !=
         summary.record.fixed_idx.end());
  assert(std::find(summary.record.fixed_idx.begin(), summary.record.fixed_idx.end(), 5) !=
         summary.record.fixed_idx.end());

  double* x_red = copy_to_device<double>({1.0, 1.0, 1.0, 0.0});
  double* y_red = copy_to_device<double>({0.0, 0.0, 0.0});
  double* z_red = copy_to_device<double>({0.0, 0.0, 0.0, 0.0});
  GpuPostsolveResult post =
      postsolve_gpu(x_red, y_red, z_red, summary.record);
  std::vector<double> x(6);
  check(cudaMemcpy(x.data(), post.x_org, sizeof(double) * x.size(),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy recovered covering primal");
  assert(std::abs(x[0] - 1.0) < 1.0e-12);
  assert(std::abs(x[1] - 1.0) < 1.0e-12);
  assert(std::abs(x[2] - 1.0) < 1.0e-12);
  assert(std::abs(x[3]) < 1.0e-12);
  assert(std::abs(x[4]) < 1.0e-12);
  assert(std::abs(x[5]) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(post.x_org);
  cudaFree(post.y_org);
  cudaFree(post.z_org);
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

void test_rule_is_owned_by_tiered_bootstrap() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_covering_lp(Variant::Valid);
  PresolveParams params = test_params();
  params.enable_tiered_bootstrap = false;
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  assert(summary.reduced_rows == 3);
  assert(summary.reduced_cols == 6);
  assert(summary.reduced_nnz == 11);
  assert(!summary.record.has_primal_only_covering_cost_dominance_reduction);
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

void assert_fail_closed(Variant variant) {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_covering_lp(variant);
  PresolveParams params = test_params();
  CoveringCostDominanceAnalysisGpu analysis =
      analyze_covering_cost_dominance(lp, params);
  assert(!analysis.applicable);
  assert(analysis.delete_col == nullptr);
  free_covering_cost_dominance_analysis(analysis);
  free_lp(lp);
}

}  // namespace

int main() {
  test_exact_reduction_and_primal_recovery();
  test_rule_is_owned_by_tiered_bootstrap();
  assert_fail_closed(Variant::NonUnitCoefficient);
  assert_fail_closed(Variant::NonCoveringRow);
  assert_fail_closed(Variant::MissingWitness);
  assert_fail_closed(Variant::NonBinaryBound);
  std::cout << "test_rule_covering_cost_dominance passed\n";
  return 0;
}
