#include "test_cuda_helpers.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"

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

void test_post_propagation_activity_closure(bool tiered_scheduler,
                                            bool record_activity_tape) {
  using gpu_presolver::presolve::GpuPostsolveResult;
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PostsolveReductionType;
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu lp;
  lp.A = {2, 3, 5,
          copy_to_device<std::int32_t>({0, 2, 5}),
          copy_to_device<std::int32_t>({0, 1, 0, 1, 2}),
          copy_to_device<double>({1.0, 1.0, 1.0, 1.0, -1.0})};
  lp.AT = {3, 2, 5,
           copy_to_device<std::int32_t>({0, 2, 4, 5}),
           copy_to_device<std::int32_t>({0, 1, 0, 1, 1}),
           copy_to_device<double>({1.0, 1.0, 1.0, 1.0, -1.0})};
  lp.c = copy_to_device<double>({0.0, 0.0, 0.0});
  lp.l = copy_to_device<double>({0.0, 0.0, 0.0});
  lp.u = copy_to_device<double>({10.0, 1.0, inf});
  lp.AL = copy_to_device<double>({-inf, -inf});
  lp.AU = copy_to_device<double>({2.0, 3.0});

  PresolveParams params;
  params.max_iters = tiered_scheduler ? 2 : 1;
  params.use_tiered_scheduler = tiered_scheduler;
  params.enable_tiered_bootstrap = false;
  params.record_postsolve_tape = true;
  params.record_postsolve_tape_cpu = record_activity_tape;
  params.enable_infeasible_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_duplicate_rows = false;
  params.enable_empty_cols = false;
  params.enable_column_singletons_eq = false;
  params.enable_column_singletons_dual_infer = false;
  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_duplicate_columns = false;
  params.enable_structural_l1_substitution = false;
  params.enable_redundant_bounds = false;

  const GpuPresolveSummary summary =
      gpu_presolver::presolve::run_gpu_presolve_with_record(lp, params);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 1);
  assert(summary.reduced_cols == 3);
  assert(summary.reduced_nnz == 2);
  assert(summary.record.row_red2org == std::vector<std::int32_t>{0});
  assert(summary.record.row_org2red == std::vector<std::int32_t>({0, -1}));

  assert(summary.record.tape.types.size() == (record_activity_tape ? 2 : 1));
  assert(summary.record.tape_gpu.record_count == 0);
  assert(summary.record.tape_gpu.index_count == 0);
  assert(summary.record.tape_gpu.value_count == 0);
  const std::int32_t bound_type = summary.record.tape.types[0];
  assert(bound_type == static_cast<std::int32_t>(PostsolveReductionType::BoundChangeTheRow) ||
         bound_type == static_cast<std::int32_t>(PostsolveReductionType::BoundChangeNoRow));
  if (record_activity_tape) {
    assert(summary.record.tape.types[1] ==
           static_cast<std::int32_t>(PostsolveReductionType::DeletedRow));
    const std::int32_t deleted_row_index_start = summary.record.tape.index_starts[1];
    assert(summary.record.tape.indices[static_cast<std::size_t>(deleted_row_index_start)] == 1);
  }

  double* x_red = copy_to_device<double>({1.0, 1.0, 0.0});
  double* y_red = copy_to_device<double>({0.0});
  double* z_red = copy_to_device<double>({0.0, 0.0, 0.0});
  GpuPostsolveResult post = gpu_presolver::presolve::postsolve_gpu(
      x_red, y_red, z_red, summary.record, &lp);
  const std::vector<double> x = copy_to_host(post.x_org, 3);
  const std::vector<double> y = copy_to_host(post.y_org, 2);
  const std::vector<double> z = copy_to_host(post.z_org, 3);
  assert(std::fabs(x[0] - 1.0) < 1.0e-12);
  assert(std::fabs(x[1] - 1.0) < 1.0e-12);
  assert(std::fabs(x[2]) < 1.0e-12);
  assert(std::fabs(y[0]) < 1.0e-12);
  assert(std::fabs(y[1]) < 1.0e-12);
  assert(std::fabs(z[0]) < 1.0e-12);
  assert(std::fabs(z[1]) < 1.0e-12);
  assert(std::fabs(z[2]) < 1.0e-12);
  assert(x[0] + x[1] <= 2.0 + 1.0e-12);
  assert(x[0] + x[1] - x[2] <= 3.0 + 1.0e-12);

  cudaFree(x_red);
  cudaFree(y_red);
  cudaFree(z_red);
  cudaFree(post.x_org);
  cudaFree(post.y_org);
  cudaFree(post.z_org);
  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
}

void test_implied_variable_bounds_reaches_dependency_chain_fixed_point() {
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu lp;
  lp.A = {2, 4, 5,
          copy_to_device<std::int32_t>({0, 2, 5}),
          copy_to_device<std::int32_t>({0, 1, 1, 2, 3}),
          copy_to_device<double>({1.0, 1.0, -1.0, 1.0, 1.0})};
  lp.AT = {4, 2, 5,
           copy_to_device<std::int32_t>({0, 1, 3, 4, 5}),
           copy_to_device<std::int32_t>({0, 0, 1, 1, 1}),
           copy_to_device<double>({1.0, 1.0, -1.0, 1.0, 1.0})};
  lp.c = copy_to_device<double>({0.0, 0.0, 0.0, 0.0});
  lp.l = copy_to_device<double>({1.0, 0.0, 0.0, 0.0});
  lp.u = copy_to_device<double>({1.0, inf, inf, 0.0});
  lp.AL = copy_to_device<double>({-inf, -inf});
  lp.AU = copy_to_device<double>({1.0, 0.0});

  PresolveParams params;
  params.max_iters = 1;
  params.use_tiered_scheduler = false;
  params.enable_infeasible_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_redundant_rows = false;
  params.enable_duplicate_rows = false;
  params.enable_empty_cols = false;
  params.enable_column_singletons_eq = false;
  params.enable_column_singletons_dual_infer = false;
  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_duplicate_columns = false;
  params.enable_structural_l1_substitution = false;
  params.enable_redundant_bounds = false;
  // Even the smallest pure-bound work budget must not stop a chain that fixes
  // columns: every structural change commits the plan and resets the budget.
  params.implied_variable_bounds_bound_only_nnz_round_budget = 1;

  const GpuPresolveSummary summary =
      gpu_presolver::presolve::run_gpu_presolve_with_record(lp, params);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 2);
  assert(summary.reduced_cols == 2);
  assert(summary.reduced_nnz == 2);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
}

std::vector<double> run_pure_bound_dependency_chain(std::int64_t nnz_round_budget) {
  using gpu_presolver::presolve::GpuPresolveSummary;
  using gpu_presolver::presolve::LPInfoGpu;
  using gpu_presolver::presolve::PresolveParams;

  const double inf = std::numeric_limits<double>::infinity();
  LPInfoGpu lp;
  // x0 + x1 <= 2 and -x1 + x2 <= 0 with x0 fixed at one.  Synchronous
  // propagation discovers u1=1 in the first round and u2=1 in the second;
  // neither tightening is structural.
  lp.A = {2, 3, 4,
          copy_to_device<std::int32_t>({0, 2, 4}),
          copy_to_device<std::int32_t>({0, 1, 1, 2}),
          copy_to_device<double>({1.0, 1.0, -1.0, 1.0})};
  lp.AT = {3, 2, 4,
           copy_to_device<std::int32_t>({0, 1, 3, 4}),
           copy_to_device<std::int32_t>({0, 0, 1, 1}),
           copy_to_device<double>({1.0, 1.0, -1.0, 1.0})};
  lp.c = copy_to_device<double>({0.0, 0.0, 0.0});
  lp.l = copy_to_device<double>({1.0, 0.0, 0.0});
  lp.u = copy_to_device<double>({1.0, inf, inf});
  lp.AL = copy_to_device<double>({-inf, -inf});
  lp.AU = copy_to_device<double>({2.0, 0.0});

  PresolveParams params;
  params.max_iters = 1;
  params.use_tiered_scheduler = false;
  params.enable_tiered_bootstrap = false;
  params.record_postsolve_tape = false;
  params.enable_infeasible_fixed_variables = false;
  params.enable_empty_rows = false;
  params.enable_singleton_rows = false;
  params.enable_infeasible_redundant_rows = false;
  params.enable_duplicate_rows = false;
  params.enable_empty_cols = false;
  params.enable_column_singletons_eq = false;
  params.enable_column_singletons_dual_infer = false;
  params.enable_doubleton_equations = false;
  params.enable_dual_fix = false;
  params.enable_duplicate_columns = false;
  params.enable_structural_l1_substitution = false;
  params.enable_redundant_bounds = false;
  params.implied_variable_bounds_bound_only_nnz_round_budget = nnz_round_budget;

  GpuPresolveSummary summary =
      gpu_presolver::presolve::run_gpu_presolve_with_reduced_lp(lp, params);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 2);
  assert(summary.reduced_cols == 3);
  assert(summary.reduced_nnz == 4);
  const std::vector<double> upper = copy_to_host(summary.reduced_lp.u, 3);
  gpu_presolver::presolve::free_gpu_presolve_reduced_lp(summary);

  cudaFree(lp.A.rowPtr);
  cudaFree(lp.A.colVal);
  cudaFree(lp.A.nzVal);
  cudaFree(lp.AT.rowPtr);
  cudaFree(lp.AT.colVal);
  cudaFree(lp.AT.nzVal);
  cudaFree(lp.c);
  cudaFree(lp.l);
  cudaFree(lp.u);
  cudaFree(lp.AL);
  cudaFree(lp.AU);
  return upper;
}

void test_implied_variable_bounds_bound_only_work_budget() {
  // Four nnz and a four-nnz-round budget allow exactly one bound-only round.
  const std::vector<double> capped = run_pure_bound_dependency_chain(4);
  assert(std::fabs(capped[0] - 1.0) < 1.0e-12);
  assert(std::fabs(capped[1] - 1.0) < 1.0e-12);
  assert(std::isinf(capped[2]));

  // A nonpositive budget disables the work cap; the round limits still apply.
  const std::vector<double> uncapped = run_pure_bound_dependency_chain(0);
  assert(std::fabs(uncapped[0] - 1.0) < 1.0e-12);
  assert(std::fabs(uncapped[1] - 1.0) < 1.0e-12);
  assert(std::fabs(uncapped[2] - 1.0) < 1.0e-12);
}

}  // namespace

int main() {
  assert(!gpu_presolver::presolve::_has_good_nnz_progress(0, 0, 0.95));
  assert(gpu_presolver::presolve::_has_good_nnz_progress(100, 94, 0.95));
  assert(!gpu_presolver::presolve::_has_good_nnz_progress(100, 95, 0.95));

  // The fixed-order case has exactly one scheduler iteration, so row 1 cannot
  // be removed by a second propagation.  It also exercises the default
  // activity-tape setting, where CPU-side recording stays disabled.
  test_post_propagation_activity_closure(false, false);
  // The tiered case records activity actions explicitly and verifies that the
  // DeletedRow record follows the propagation BoundChange record.
  test_post_propagation_activity_closure(true, true);
  test_implied_variable_bounds_reaches_dependency_chain_fixed_point();
  test_implied_variable_bounds_bound_only_work_budget();

  const double inf = std::numeric_limits<double>::infinity();
  gpu_presolver::presolve::LPInfoGpu sched_lp;
  sched_lp.A = {3, 3, 2,
                copy_to_device<std::int32_t>({0, 1, 2, 2}),
                copy_to_device<std::int32_t>({0, 1}),
                copy_to_device<double>({1.0, 1.0})};
  sched_lp.AT = {3, 3, 2,
                 copy_to_device<std::int32_t>({0, 1, 2, 2}),
                 copy_to_device<std::int32_t>({0, 1}),
                 copy_to_device<double>({1.0, 1.0})};
  sched_lp.c = copy_to_device<double>({0.0, 0.0, 2.0});
  sched_lp.l = copy_to_device<double>({5.0, 0.0, 1.0});
  sched_lp.u = copy_to_device<double>({5.0, 10.0, 3.0});
  sched_lp.AL = copy_to_device<double>({5.0, -inf, -1.0});
  sched_lp.AU = copy_to_device<double>({5.0, inf, 1.0});

  gpu_presolver::presolve::PresolveParams params;
  params.max_iters = 4;
  const gpu_presolver::presolve::GpuPresolveSummary summary =
      gpu_presolver::presolve::run_gpu_presolve_with_record(sched_lp, params);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 0);
  assert(summary.reduced_cols == 0);

  gpu_presolver::presolve::LPInfoGpu fixed_sched_lp;
  fixed_sched_lp.A = {3, 3, 2,
                      copy_to_device<std::int32_t>({0, 1, 2, 2}),
                      copy_to_device<std::int32_t>({0, 1}),
                      copy_to_device<double>({1.0, 1.0})};
  fixed_sched_lp.AT = {3, 3, 2,
                       copy_to_device<std::int32_t>({0, 1, 2, 2}),
                       copy_to_device<std::int32_t>({0, 1}),
                       copy_to_device<double>({1.0, 1.0})};
  fixed_sched_lp.c = copy_to_device<double>({0.0, 0.0, 2.0});
  fixed_sched_lp.l = copy_to_device<double>({5.0, 0.0, 1.0});
  fixed_sched_lp.u = copy_to_device<double>({5.0, 10.0, 3.0});
  fixed_sched_lp.AL = copy_to_device<double>({5.0, -inf, -1.0});
  fixed_sched_lp.AU = copy_to_device<double>({5.0, inf, 1.0});

  gpu_presolver::presolve::PresolveParams fixed_params;
  fixed_params.max_iters = 4;
  fixed_params.use_tiered_scheduler = false;
  const gpu_presolver::presolve::GpuPresolveSummary fixed_summary =
      gpu_presolver::presolve::run_gpu_presolve_with_record(fixed_sched_lp, fixed_params);
  assert(!fixed_summary.has_infeasible);
  assert(!fixed_summary.has_unbounded);
  assert(fixed_summary.reduced_rows == 0);
  assert(fixed_summary.reduced_cols == 0);

  cudaFree(fixed_sched_lp.A.rowPtr);
  cudaFree(fixed_sched_lp.A.colVal);
  cudaFree(fixed_sched_lp.A.nzVal);
  cudaFree(fixed_sched_lp.AT.rowPtr);
  cudaFree(fixed_sched_lp.AT.colVal);
  cudaFree(fixed_sched_lp.AT.nzVal);
  cudaFree(fixed_sched_lp.c);
  cudaFree(fixed_sched_lp.l);
  cudaFree(fixed_sched_lp.u);
  cudaFree(fixed_sched_lp.AL);
  cudaFree(fixed_sched_lp.AU);

  cudaFree(sched_lp.A.rowPtr);
  cudaFree(sched_lp.A.colVal);
  cudaFree(sched_lp.A.nzVal);
  cudaFree(sched_lp.AT.rowPtr);
  cudaFree(sched_lp.AT.colVal);
  cudaFree(sched_lp.AT.nzVal);
  cudaFree(sched_lp.c);
  cudaFree(sched_lp.l);
  cudaFree(sched_lp.u);
  cudaFree(sched_lp.AL);
  cudaFree(sched_lp.AU);

  constexpr std::int32_t doubleton_rows = 300;
  constexpr std::int32_t doubleton_cols = doubleton_rows * 2;
  std::vector<std::int32_t> doubleton_row_ptr(static_cast<std::size_t>(doubleton_rows + 1));
  std::vector<std::int32_t> doubleton_col_val(static_cast<std::size_t>(doubleton_rows * 2));
  std::vector<double> doubleton_a(static_cast<std::size_t>(doubleton_rows * 2), 1.0);
  for (std::int32_t row = 0; row < doubleton_rows; ++row) {
    doubleton_row_ptr[static_cast<std::size_t>(row)] = 2 * row;
    doubleton_col_val[static_cast<std::size_t>(2 * row)] = 2 * row;
    doubleton_col_val[static_cast<std::size_t>(2 * row + 1)] = 2 * row + 1;
  }
  doubleton_row_ptr[static_cast<std::size_t>(doubleton_rows)] = doubleton_rows * 2;
  std::vector<std::int32_t> doubleton_at_row_ptr(static_cast<std::size_t>(doubleton_cols + 1));
  std::vector<std::int32_t> doubleton_at_col_val(static_cast<std::size_t>(doubleton_rows * 2));
  for (std::int32_t col = 0; col < doubleton_cols; ++col) {
    doubleton_at_row_ptr[static_cast<std::size_t>(col)] = col;
    doubleton_at_col_val[static_cast<std::size_t>(col)] = col / 2;
  }
  doubleton_at_row_ptr[static_cast<std::size_t>(doubleton_cols)] = doubleton_cols;

  gpu_presolver::presolve::LPInfoGpu doubleton_only_lp;
  doubleton_only_lp.A = {doubleton_rows, doubleton_cols, doubleton_rows * 2,
                         copy_to_device<std::int32_t>(doubleton_row_ptr),
                         copy_to_device<std::int32_t>(doubleton_col_val),
                         copy_to_device<double>(doubleton_a)};
  doubleton_only_lp.AT = {doubleton_cols, doubleton_rows, doubleton_rows * 2,
                          copy_to_device<std::int32_t>(doubleton_at_row_ptr),
                          copy_to_device<std::int32_t>(doubleton_at_col_val),
                          copy_to_device<double>(doubleton_a)};
  doubleton_only_lp.c = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_cols), 0.0));
  doubleton_only_lp.l = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_cols), 0.0));
  doubleton_only_lp.u = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_cols), 10.0));
  doubleton_only_lp.AL = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_rows), 1.0));
  doubleton_only_lp.AU = copy_to_device<double>(std::vector<double>(static_cast<std::size_t>(doubleton_rows), 1.0));

  gpu_presolver::presolve::PresolveParams doubleton_only_params;
  doubleton_only_params.max_iters = 4;
  doubleton_only_params.enable_infeasible_fixed_variables = false;
  doubleton_only_params.enable_empty_rows = false;
  doubleton_only_params.enable_singleton_rows = false;
  doubleton_only_params.enable_infeasible_redundant_rows = false;
  doubleton_only_params.enable_implied_variable_bounds = false;
  doubleton_only_params.enable_duplicate_rows = false;
  doubleton_only_params.enable_empty_cols = false;
  doubleton_only_params.enable_column_singletons_eq = false;
  doubleton_only_params.enable_column_singletons_dual_infer = false;
  doubleton_only_params.enable_dual_fix = false;
  doubleton_only_params.enable_duplicate_columns = false;
  doubleton_only_params.enable_structural_l1_substitution = false;
  doubleton_only_params.enable_redundant_bounds = false;
  const gpu_presolver::presolve::GpuPresolveSummary doubleton_only_summary =
      gpu_presolver::presolve::run_gpu_presolve_with_record(doubleton_only_lp, doubleton_only_params);
  assert(!doubleton_only_summary.has_infeasible);
  assert(!doubleton_only_summary.has_unbounded);
  assert(doubleton_only_summary.reduced_rows == 0);
  assert(doubleton_only_summary.reduced_cols == doubleton_rows);
  assert(doubleton_only_summary.reduced_nnz == 0);
  assert(doubleton_only_summary.iterations > 1);

  cudaFree(doubleton_only_lp.A.rowPtr);
  cudaFree(doubleton_only_lp.A.colVal);
  cudaFree(doubleton_only_lp.A.nzVal);
  cudaFree(doubleton_only_lp.AT.rowPtr);
  cudaFree(doubleton_only_lp.AT.colVal);
  cudaFree(doubleton_only_lp.AT.nzVal);
  cudaFree(doubleton_only_lp.c);
  cudaFree(doubleton_only_lp.l);
  cudaFree(doubleton_only_lp.u);
  cudaFree(doubleton_only_lp.AL);
  cudaFree(doubleton_only_lp.AU);

  std::cout << "test_gpu_direct_port passed\n";
  return 0;
}
