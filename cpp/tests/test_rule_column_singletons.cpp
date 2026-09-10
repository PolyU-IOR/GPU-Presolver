#include "test_cuda_helpers.hpp"
#include "gpu_presolver/presolve/rules/rule_column_singletons.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using gpu_presolver_test::check;
using gpu_presolver_test::copy_to_device;
using gpu_presolver_test::copy_to_host;

}  // namespace

void test_eliminates_eq_column_singleton() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

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
  plan.new_c = copy_to_device<double>({4.0, 1.0});
  plan.new_l = copy_to_device<double>({0.0, 2.0});
  plan.new_u = copy_to_device<double>({10.0, 2.0});
  plan.new_AL = copy_to_device<double>({5.0});
  plan.new_AU = copy_to_device<double>({5.0});

  PresolveStatsGpu stats;
  stats.column_singleton_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.column_singleton_row = copy_to_device<std::int32_t>({0, -1});
  stats.column_singleton_val = copy_to_device<double>({1.0, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_column_singletons_eq(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_row = copy_to_host(plan.keep_row_mask, 1);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> c = copy_to_host(plan.new_c, 2);
  assert(keep_row[0] == 0);
  assert(keep_col[0] == 0);
  assert(keep_col[1] == 1);
  assert(std::fabs(c[1] + 3.0) < 1.0e-12);
  assert(std::fabs(plan.obj_constant_delta - 20.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::SubCol)}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0, 0, 1, 1}));
  assert(plan.tape.vals.size() == 7);
  assert(std::fabs(plan.tape.vals[0] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[1] - 5.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[2] - 0.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[3] - 10.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[4] - 4.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[5] - 1.0) < 1.0e-12);
  assert(std::fabs(plan.tape.vals[6] - 1.0) < 1.0e-12);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.column_singleton_mask);
  cudaFree(stats.column_singleton_row);
  cudaFree(stats.column_singleton_val);
}

void test_eq_objective_updates_are_bitwise_deterministic() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;
  const double inf = std::numeric_limits<double>::infinity();

  std::uint64_t reference_bits = 0;
  for (int repetition = 0; repetition < 32; ++repetition) {
    LPInfoGpu lp;
    lp.A = {2, 3, 4,
            copy_to_device<std::int32_t>({0, 2, 4}),
            copy_to_device<std::int32_t>({0, 2, 1, 2}),
            copy_to_device<double>({3.0, 0.1, 7.0, 0.2})};
    lp.AT = {3, 2, 4,
             copy_to_device<std::int32_t>({0, 1, 2, 4}),
             copy_to_device<std::int32_t>({0, 1, 0, 1}),
             copy_to_device<double>({3.0, 7.0, 0.1, 0.2})};

    PresolvePlanGpu plan;
    plan.keep_row_mask = copy_to_device<std::uint8_t>({1, 1});
    plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1, 1});
    plan.new_c = copy_to_device<double>({100000000.1, -99999999.9, 0.3});
    plan.new_l = copy_to_device<double>({-inf, -inf, 0.0});
    plan.new_u = copy_to_device<double>({inf, inf, 0.0});
    plan.new_AL = copy_to_device<double>({5.0, 6.0});
    plan.new_AU = copy_to_device<double>({5.0, 6.0});

    PresolveStatsGpu stats;
    stats.column_singleton_mask = copy_to_device<std::uint8_t>({1, 1, 0});
    stats.column_singleton_row = copy_to_device<std::int32_t>({0, 1, -1});
    stats.column_singleton_val = copy_to_device<double>({3.0, 7.0, 0.0});
    PresolveParams params;

    gpu_presolver::presolve::apply_rule_column_singletons_eq(plan, lp, stats, params);
    const std::vector<double> c = copy_to_host(plan.new_c, 3);
    std::uint64_t current_bits = 0;
    std::memcpy(&current_bits, &c[2], sizeof(current_bits));
    if (repetition == 0) {
      reference_bits = current_bits;
    } else {
      assert(current_bits == reference_bits);
    }

    double expected = 0.3;
    expected += -(100000000.1 * 0.1 / 3.0);
    expected += -(-99999999.9 * 0.2 / 7.0);
    std::uint64_t expected_bits = 0;
    std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
    assert(current_bits == expected_bits);

    cudaFree(lp.A.rowPtr);
    cudaFree(lp.A.colVal);
    cudaFree(lp.A.nzVal);
    cudaFree(lp.AT.rowPtr);
    cudaFree(lp.AT.colVal);
    cudaFree(lp.AT.nzVal);
    cudaFree(plan.keep_row_mask);
    cudaFree(plan.keep_col_mask);
    cudaFree(plan.new_c);
    cudaFree(plan.new_l);
    cudaFree(plan.new_u);
    cudaFree(plan.new_AL);
    cudaFree(plan.new_AU);
    cudaFree(stats.column_singleton_mask);
    cudaFree(stats.column_singleton_row);
    cudaFree(stats.column_singleton_val);
  }
}

void test_dual_infer_marks_direct_unbounded() {
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
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 0.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0});
  plan.new_u = copy_to_device<double>({10.0, 1.0});
  plan.new_AL = copy_to_device<double>({-inf});
  plan.new_AU = copy_to_device<double>({5.0});
  PresolveStatsGpu stats;
  stats.column_singleton_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.column_singleton_row = copy_to_device<std::int32_t>({0, -1});
  stats.column_singleton_val = copy_to_device<double>({1.0, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_column_singletons_dual_infer(plan, lp, stats, params);
  assert(plan.has_unbounded);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.column_singleton_mask);
  cudaFree(stats.column_singleton_row);
  cudaFree(stats.column_singleton_val);
}

void test_dual_infer_records_eq_to_ineq_tape() {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
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
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({1.0, 0.0});
  plan.new_l = copy_to_device<double>({-inf, 0.0});
  plan.new_u = copy_to_device<double>({inf, 0.0});
  plan.new_AL = copy_to_device<double>({1.0});
  plan.new_AU = copy_to_device<double>({5.0});
  PresolveStatsGpu stats;
  stats.column_singleton_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.column_singleton_row = copy_to_device<std::int32_t>({0, -1});
  stats.column_singleton_val = copy_to_device<double>({1.0, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_column_singletons_dual_infer(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 1);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 1);
  assert(keep_col[0] == 1);
  assert(std::fabs(AL[0] - 1.0) < 1.0e-12);
  assert(std::fabs(AU[0] - 1.0) < 1.0e-12);
  assert(plan.has_row_action);
  assert(!plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::EqToIneq)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 1}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 1}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0}));
  assert(plan.tape.vals.size() == 1);
  assert(std::fabs(plan.tape.vals[0]) < 1.0e-12);
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.column_singleton_mask);
  cudaFree(stats.column_singleton_row);
  cudaFree(stats.column_singleton_val);
}

struct OneSidedSingletonCase {
  double objective;
  double coefficient;
  double lower;
  double upper;
  double lhs;
  double rhs;
  double expected_side;
};

void check_dual_infer_tightens_one_sided_implied_free_row(
    const OneSidedSingletonCase& test_case) {
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveDualMode;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;
  using gpu_presolver::presolve::PresolvePlanGpu;
  using gpu_presolver::presolve::PresolveStatsGpu;

  LPInfoGpu lp;
  lp.A = {1, 2, 2,
          copy_to_device<std::int32_t>({0, 2}),
          copy_to_device<std::int32_t>({0, 1}),
          copy_to_device<double>({test_case.coefficient, 1.0})};
  PresolvePlanGpu plan;
  plan.keep_row_mask = copy_to_device<std::uint8_t>({1});
  plan.keep_col_mask = copy_to_device<std::uint8_t>({1, 1});
  plan.new_c = copy_to_device<double>({test_case.objective, 0.0});
  plan.new_l = copy_to_device<double>({test_case.lower, 0.0});
  plan.new_u = copy_to_device<double>({test_case.upper, 0.0});
  plan.new_AL = copy_to_device<double>({test_case.lhs});
  plan.new_AU = copy_to_device<double>({test_case.rhs});
  PresolveStatsGpu stats;
  stats.column_singleton_mask = copy_to_device<std::uint8_t>({1, 0});
  stats.column_singleton_row = copy_to_device<std::int32_t>({0, -1});
  stats.column_singleton_val = copy_to_device<double>({test_case.coefficient, 0.0});
  PresolveParams params;

  gpu_presolver::presolve::apply_rule_column_singletons_dual_infer(plan, lp, stats, params);
  const std::vector<std::uint8_t> keep_col = copy_to_host(plan.keep_col_mask, 2);
  const std::vector<double> AL = copy_to_host(plan.new_AL, 1);
  const std::vector<double> AU = copy_to_host(plan.new_AU, 1);
  assert(keep_col[0] == 1);
  assert(std::fabs(AL[0] - test_case.expected_side) < 1.0e-12);
  assert(std::fabs(AU[0] - test_case.expected_side) < 1.0e-12);
  assert(plan.has_row_action);
  assert(!plan.has_col_action);
  assert(plan.has_change);
  assert((plan.tape.types == std::vector<std::int32_t>{
                                 static_cast<std::int32_t>(PostsolveReductionType::EqToIneq)}));
  assert((plan.tape.index_starts == std::vector<std::int32_t>{0, 1}));
  assert((plan.tape.value_starts == std::vector<std::int32_t>{0, 1}));
  assert((plan.tape.indices == std::vector<std::int32_t>{0}));
  assert(plan.tape.vals.size() == 1);
  assert(std::fabs(plan.tape.vals[0]) < 1.0e-12);
  assert((plan.tape.dual_modes == std::vector<std::uint8_t>{
                                      static_cast<std::uint8_t>(PostsolveDualMode::Minimal)}));

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(plan.keep_row_mask);
  cudaFree(plan.keep_col_mask);
  cudaFree(plan.new_c);
  cudaFree(plan.new_l);
  cudaFree(plan.new_u);
  cudaFree(plan.new_AL);
  cudaFree(plan.new_AU);
  cudaFree(stats.column_singleton_mask);
  cudaFree(stats.column_singleton_row);
  cudaFree(stats.column_singleton_val);
}

void test_dual_infer_tightens_one_sided_implied_free_row() {
  // c < 0, a > 0, implied-free from above: the upper row side is active.
  check_dual_infer_tightens_one_sided_implied_free_row(
      OneSidedSingletonCase{-1.0, 1.0, 10.0, 30.0, 5.0, 20.0, 20.0});
  // c > 0, a < 0, implied-free from below: the upper row side is active.
  check_dual_infer_tightens_one_sided_implied_free_row(
      OneSidedSingletonCase{1.0, -1.0, 0.0, 10.0, -20.0, -5.0, -5.0});
  // c > 0, a > 0, implied-free from below: the lower row side is active.
  check_dual_infer_tightens_one_sided_implied_free_row(
      OneSidedSingletonCase{1.0, 1.0, 0.0, 10.0, 5.0, 20.0, 5.0});
  // c < 0, a < 0, implied-free from above: the lower row side is active.
  check_dual_infer_tightens_one_sided_implied_free_row(
      OneSidedSingletonCase{-1.0, -1.0, 10.0, 30.0, -20.0, -5.0, -20.0});
}

int main() {
  test_eliminates_eq_column_singleton();
  test_eq_objective_updates_are_bitwise_deterministic();
  test_dual_infer_marks_direct_unbounded();
  test_dual_infer_records_eq_to_ineq_tape();
  test_dual_infer_tightens_one_sided_implied_free_row();
  std::cout << "test_rule_column_singletons passed\n";
  return 0;
}
