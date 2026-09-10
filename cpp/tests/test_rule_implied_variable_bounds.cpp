#include "test_cuda_helpers.hpp"
#include "gpu_presolver/presolve/rules/rule_implied_variable_bounds.hpp"

#include <cuda_runtime.h>

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
using gpu_presolver_test::copy_to_device;
using gpu_presolver_test::copy_to_host;

void free_lp(gpu_presolver::presolve::LPInfoGpu& lp) {
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
}

}  // namespace

void test_tightens_bounds_from_row_activity() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {2, 3, 4,
          copy_to_device<std::int32_t>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 2}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.AT = {3, 2, 4,
           copy_to_device<std::int32_t>({0, 2, 3, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0, 0.0});
  plan.new_AL = copy_to_device<double>({5.0, -inf});
  plan.new_AU = copy_to_device<double>({inf, 6.0});
  plan.new_l = copy_to_device<double>({0.0, 2.0, 0.0});
  plan.new_u = copy_to_device<double>({10.0, 2.0, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2, 2});

  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_implied_variable_bounds(plan, lp, stats, params);
  const std::vector<double> l = copy_to_host(plan.new_l, 3);
  const std::vector<double> u = copy_to_host(plan.new_u, 3);
  assert(std::fabs(l[0] - 3.0) < 1.0e-12);
  assert(std::fabs(u[0] - 6.0) < 1.0e-12);
  assert(!plan.has_row_action);
  assert(!plan.has_col_action);
  assert(plan.has_change);
  assert(!plan.has_infeasible);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow),
                                 static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 2, 4}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 4, 8}));
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal),
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0, 0, 1}));
  assert(std::fabs(plan.tape.vals[0] - 0.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[4] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[5] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[6] - 3.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[7] - 6.0) < 1.0e-12);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_records_fixed_column_from_row_activity() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<std::int32_t>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 1.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<std::int32_t>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({4.0, 0.0});
  plan.new_AL = copy_to_device<double>({6.0});
  plan.new_AU = copy_to_device<double>({inf});
  plan.new_l = copy_to_device<double>({0.0, 5.0});
  plan.new_u = copy_to_device<double>({1.0, 5.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2});
  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_implied_variable_bounds(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> l = copy_to_host(plan.new_l, 2);
  const std::vector<double> u = copy_to_host(plan.new_u, 2);
  assert(keep_col[0] == 0);
  assert(std::fabs(l[0] - 1.0) < 1.0e-12);
  assert(std::fabs(u[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 4.0) < 1.0e-12);
  assert(plan.tape.types.size() == 1);
  assert(plan.tape.types[0] == static_cast<std::int32_t>(PostsolveReductionType::FixedCol));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0}));
  assert(std::fabs(plan.tape.vals[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 4.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 1.0) < 1.0e-12);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_detects_infeasible_tightening() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<std::int32_t>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({1.0, 1.0})};
  lp.AT = {2, 1, 2,
           copy_to_device<std::int32_t>({0, 1, 2}),
           copy_to_device<std::int32_t>({0, 0}),
           copy_to_device<double>({1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0});
  plan.new_AL = copy_to_device<double>({10.0});
  plan.new_AU = copy_to_device<double>({inf});
  plan.new_l = copy_to_device<double>({0.0, 0.0});
  plan.new_u = copy_to_device<double>({4.0, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_implied_variable_bounds(plan, lp, stats, params);
  assert(plan.has_infeasible);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_preserves_wide_box_on_free_zero_cost_column() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  LPInfoGpu lp;
  lp.A = {2, 2, 4,
          copy_to_device<std::int32_t>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 0, 1}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};
  lp.AT = {2, 2, 4,
           copy_to_device<std::int32_t>({0, 2, 4}),
           copy_to_device<std::int32_t>({0, 1, 0, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0})};

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0});
  plan.new_AL = copy_to_device<double>({-5.0, -inf});
  plan.new_AU = copy_to_device<double>({inf, 5.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0});
  plan.new_u = copy_to_device<double>({inf, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2, 2});
  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_implied_variable_bounds(plan, lp, stats, params);
  std::vector<double> l = copy_to_host(plan.new_l, 2);
  std::vector<double> u = copy_to_host(plan.new_u, 2);
  assert(std::isinf(l[0]) && l[0] < 0.0);
  assert(std::isinf(u[0]) && u[0] > 0.0);
  assert(!plan.has_change);
  assert(plan.tape.types.empty());

  params.implied_variable_bounds_preserve_free_zero_cost = false;
  gpu_presolver::presolve::apply_rule_implied_variable_bounds(plan, lp, stats, params);
  l = copy_to_host(plan.new_l, 2);
  u = copy_to_host(plan.new_u, 2);
  assert(std::fabs(l[0] + 5.0) < 1.0e-12);
  assert(std::fabs(u[0] - 5.0) < 1.0e-12);
  assert(plan.has_change);

  free_lp(lp);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

void test_preserves_free_zero_cost_state_across_propagation_rounds() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  // Round one derives x >= 0 from 0 <= x-z and z <= 1 from z+w <= 1.
  // Only round two can derive x <= 11 from x-z <= 10.  x was originally
  // free and has zero cost, so the second side must not materialize a box.
  LPInfoGpu lp;
  lp.A = {2, 3, 4,
          copy_to_device<std::int32_t>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 1, 2}),
          copy_to_device<double>({1.0, -1.0, 1.0, 1.0})};
  lp.AT = {3, 2, 4,
           copy_to_device<std::int32_t>({0, 1, 3, 4}),
           copy_to_device<std::int32_t>({0, 0, 1, 1}),
           copy_to_device<double>({1.0, -1.0, 1.0, 1.0})};
  lp.l = copy_to_device<double>({-inf, 0.0, 0.0});
  lp.u = copy_to_device<double>({inf, inf, 0.0});

  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
  plan.new_c = copy_to_device<double>({0.0, 0.0, 0.0});
  plan.new_AL = copy_to_device<double>({0.0, -inf});
  plan.new_AU = copy_to_device<double>({10.0, 1.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0, 0.0});
  plan.new_u = copy_to_device<double>({inf, inf, 0.0});

  PresolveStatsGpu stats;
  stats.row_nnz = copy_to_device<std::int32_t>({2, 2});
  PresolveParams params;
  params.feasibility_tol = 1.0e-9;
  params.zero_tol = 1.0e-12;

  gpu_presolver::presolve::apply_rule_implied_variable_bounds(plan, lp, stats, params);
  std::vector<double> l = copy_to_host(plan.new_l, 3);
  std::vector<double> u = copy_to_host(plan.new_u, 3);
  assert(std::fabs(l[0]) < 1.0e-12);
  assert(std::isinf(u[0]) && u[0] > 0.0);
  assert(std::fabs(u[1] - 1.0) < 1.0e-12);

  plan.has_change = false;
  gpu_presolver::presolve::apply_rule_implied_variable_bounds(plan, lp, stats, params);
  l = copy_to_host(plan.new_l, 3);
  u = copy_to_host(plan.new_u, 3);
  assert(std::fabs(l[0]) < 1.0e-12);
  assert(std::isinf(u[0]) && u[0] > 0.0);
  assert(!plan.has_change);

  free_lp(lp);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(stats.row_nnz);
}

int main() {
  test_tightens_bounds_from_row_activity();
  test_records_fixed_column_from_row_activity();
  test_detects_infeasible_tightening();
  test_preserves_wide_box_on_free_zero_cost_column();
  test_preserves_free_zero_cost_state_across_propagation_rounds();
  std::cout << "test_rule_implied_variable_bounds passed\n";
  return 0;
}
