#include "test_cuda_helpers.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/rules/rule_antipodal_components.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
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

gpu_presolver::presolve::LPInfoGpu make_three_pair_lp(bool conflicting_shift) {
  using gpu_presolver::presolve::LPInfoGpu;
  LPInfoGpu lp;
  lp.A = {2,
          6,
          10,
          copy_to_device<std::int32_t>({0, 4, 10}),
          copy_to_device<std::int32_t>({0, 1, 3, 4, 0, 1, 2, 3, 4, 5}),
          copy_to_device<double>(
              {1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0})};
  lp.AT = {6,
           2,
           10,
           copy_to_device<std::int32_t>({0, 2, 4, 5, 7, 9, 10}),
           copy_to_device<std::int32_t>({0, 1, 0, 1, 1, 0, 1, 0, 1, 1}),
           copy_to_device<double>(
               {1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0})};
  lp.c = copy_to_device<double>({1.0, 2.0, 3.0, 1.0, 2.0, 3.0});
  lp.AL = copy_to_device<double>({0.0, 5.0});
  lp.AU = copy_to_device<double>({0.0, 5.0});
  lp.l = conflicting_shift
             ? copy_to_device<double>({2.0, 4.0, 0.0, 1.0, 2.0, 0.0})
             : copy_to_device<double>({2.0, 4.0, 0.0, 1.0, 3.0, 0.0});
  lp.u = copy_to_device<double>({10.0, 9.0, 6.0, 8.0, 8.0, 7.0});
  return lp;
}

gpu_presolver::presolve::PresolveParams test_params() {
  gpu_presolver::presolve::PresolveParams params;
  params.max_iters = 1;
  params.enable_covering_cost_dominance = false;
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
  params.antipodal_min_pairs = 1;
  params.antipodal_probe_pairs = 3;
  params.antipodal_min_potential_gain = 0.20;
  params.antipodal_min_actual_gain = 0.20;
  return params;
}

void test_exact_shifted_quotient_and_primal_recovery() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_three_pair_lp(false);
  PresolveParams params = test_params();
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 1);
  assert(summary.reduced_cols == 4);
  assert(summary.reduced_nnz == 4);
  assert(std::abs(summary.obj_constant_delta - 17.0) < 1.0e-12);
  assert(summary.record.has_primal_only_antipodal_reduction);

  // Compact column order is root P0, singleton P2, root N0, singleton N2.
  double* x_red = copy_to_device<double>({1.7, 0.5, 0.2, 0.5});
  double* y_red = copy_to_device<double>({0.0});
  double* z_red = copy_to_device<double>({0.0, 0.0, 0.0, 0.0});
  GpuPostsolveResult post =
      postsolve_gpu(x_red, y_red, z_red, summary.record);
  std::vector<double> x(6);
  check(cudaMemcpy(x.data(), post.x_org, sizeof(double) * x.size(),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy recovered antipodal primal");
  assert(std::abs(x[0] - 3.7) < 1.0e-12);
  assert(std::abs(x[1] - 5.7) < 1.0e-12);
  assert(std::abs(x[2] - 0.5) < 1.0e-12);
  assert(std::abs(x[3] - 1.2) < 1.0e-12);
  assert(std::abs(x[4] - 3.2) < 1.0e-12);
  assert(std::abs(x[5] - 0.5) < 1.0e-12);
  const double merge_activity = (x[0] - x[3]) - (x[1] - x[4]);
  const double retained_activity =
      (x[0] - x[3]) + (x[1] - x[4]) - (x[2] - x[5]);
  assert(std::abs(merge_activity) < 1.0e-12);
  assert(std::abs(retained_activity - 5.0) < 1.0e-12);
  double objective = 0.0;
  const double c[6] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
  for (int j = 0; j < 6; ++j) {
    objective += c[j] * x[static_cast<std::size_t>(j)];
  }
  assert(std::abs(objective - 25.7) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(post.x_org);
  cudaFree(post.y_org);
  cudaFree(post.z_org);
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

void test_component_shift_conflict_is_fail_closed() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_three_pair_lp(true);
  PresolveParams params = test_params();
  AntipodalComponentAnalysisGpu analysis =
      analyze_antipodal_components(lp, params);
  assert(!analysis.applicable);
  assert(analysis.parent == nullptr);
  assert(analysis.edge_by_row == nullptr);
  free_antipodal_component_analysis(analysis);
  free_lp(lp);
}

void test_rule_is_owned_by_tiered_bootstrap() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_three_pair_lp(false);
  PresolveParams params = test_params();
  params.enable_tiered_bootstrap = false;
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  assert(summary.reduced_rows == 2);
  assert(summary.reduced_cols == 6);
  assert(summary.reduced_nnz == 10);
  assert(!summary.record.has_primal_only_antipodal_reduction);
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

}  // namespace

int main() {
  test_exact_shifted_quotient_and_primal_recovery();
  test_component_shift_conflict_is_fail_closed();
  test_rule_is_owned_by_tiered_bootstrap();
  std::cout << "test_rule_antipodal_components passed\n";
  return 0;
}
