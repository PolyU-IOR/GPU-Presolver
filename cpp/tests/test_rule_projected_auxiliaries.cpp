#include "test_cuda_helpers.hpp"
#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver/presolve/rules/rule_bounded_two_row_projection.hpp"
#include "gpu_presolver/presolve/rules/rule_orphan_mccormick_projection.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using gpu_presolver_test::check;

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  if (values.empty()) {
    return nullptr;
  }
  T* out = nullptr;
  check(cudaMalloc(&out, sizeof(T) * values.size()), "cudaMalloc test data");
  check(cudaMemcpy(out, values.data(), sizeof(T) * values.size(),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy test data");
  return out;
}

template <class T>
std::vector<T> copy_to_host(const T* values, std::size_t count) {
  std::vector<T> out(count);
  if (count > 0) {
    check(cudaMemcpy(out.data(), values, sizeof(T) * count,
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy test result");
  }
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

struct HostCsr {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::vector<std::int32_t> row_ptr;
  std::vector<std::int32_t> col_val;
  std::vector<double> nz_val;
};

HostCsr transpose(const HostCsr& A) {
  HostCsr AT;
  AT.rows = A.cols;
  AT.cols = A.rows;
  AT.row_ptr.assign(static_cast<std::size_t>(AT.rows + 1), 0);
  for (const std::int32_t col : A.col_val) {
    ++AT.row_ptr[static_cast<std::size_t>(col + 1)];
  }
  for (std::int32_t i = 0; i < AT.rows; ++i) {
    AT.row_ptr[static_cast<std::size_t>(i + 1)] +=
        AT.row_ptr[static_cast<std::size_t>(i)];
  }
  AT.col_val.resize(A.col_val.size());
  AT.nz_val.resize(A.nz_val.size());
  std::vector<std::int32_t> cursor = AT.row_ptr;
  for (std::int32_t row = 0; row < A.rows; ++row) {
    for (std::int32_t p = A.row_ptr[static_cast<std::size_t>(row)];
         p < A.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
      const std::int32_t col = A.col_val[static_cast<std::size_t>(p)];
      const std::int32_t dst = cursor[static_cast<std::size_t>(col)]++;
      AT.col_val[static_cast<std::size_t>(dst)] = row;
      AT.nz_val[static_cast<std::size_t>(dst)] =
          A.nz_val[static_cast<std::size_t>(p)];
    }
  }
  return AT;
}

gpu_presolver::presolve::LPInfoGpu upload_lp(
    const HostCsr& A,
    const std::vector<double>& c,
    const std::vector<double>& AL,
    const std::vector<double>& AU,
    const std::vector<double>& lower,
    const std::vector<double>& upper) {
  using gpu_presolver::presolve::LPInfoGpu;
  const HostCsr AT = transpose(A);
  LPInfoGpu lp;
  lp.A = {A.rows, A.cols, static_cast<std::int32_t>(A.nz_val.size()),
          copy_to_device(A.row_ptr), copy_to_device(A.col_val),
          copy_to_device(A.nz_val)};
  lp.AT = {AT.rows, AT.cols, static_cast<std::int32_t>(AT.nz_val.size()),
           copy_to_device(AT.row_ptr), copy_to_device(AT.col_val),
           copy_to_device(AT.nz_val)};
  lp.c = copy_to_device(c);
  lp.AL = copy_to_device(AL);
  lp.AU = copy_to_device(AU);
  lp.l = copy_to_device(lower);
  lp.u = copy_to_device(upper);
  return lp;
}

gpu_presolver::presolve::LPInfoGpu make_pair_block_lp(
    double second_schedule_rhs = 4.0,
    double recourse_cost = 0.0) {
  const double inf = std::numeric_limits<double>::infinity();
  HostCsr A;
  A.rows = 5;
  A.cols = 6;
  A.row_ptr = {0, 3, 5, 7, 11, 15};
  A.col_val = {0, 1, 4,
               0, 4,
               1, 4,
               2, 3, 4, 5,
               2, 3, 4, 5};
  A.nz_val = {-1.0, -1.0, 1.0,
              -1.0, 1.0,
              -1.0, 1.0,
              1.0, -1.0, 2.0, -2.0,
              -1.0, 1.0, 2.0, 2.0};
  return upload_lp(A,
                   {-1.0, -2.0, 0.0, 0.0, 0.0, recourse_cost},
                   {-1.0, -inf, -inf, -inf, -inf},
                   {inf, 0.0, 0.0, 2.0, second_schedule_rhs},
                   std::vector<double>(6, 0.0),
                   std::vector<double>(6, 1.0));
}

gpu_presolver::presolve::LPInfoGpu make_mccormick_lp(
    bool corrupt_upper_row,
    bool nonunit_endpoint) {
  const double inf = std::numeric_limits<double>::infinity();
  HostCsr A;
  A.rows = 3;
  A.cols = 3;
  A.row_ptr = {0, 3, 5, 7};
  A.col_val = {0, 1, 2, 0, 2, 1, 2};
  A.nz_val = {-1.0, -1.0, 1.0,
              corrupt_upper_row ? -0.5 : -1.0, 1.0,
              -1.0, 1.0};
  std::vector<double> upper{1.0, nonunit_endpoint ? 2.0 : 1.0, 1.0};
  return upload_lp(A,
                   {0.0, 0.0, 0.0},
                   {-1.0, -inf, -inf},
                   {inf, 0.0, 0.0},
                   {0.0, 0.0, 0.0},
                   upper);
}

gpu_presolver::presolve::PresolveParams projected_only_params() {
  using gpu_presolver::presolve::PresolveParams;
  PresolveParams params;
  params.max_iters = 1;
  params.enable_covering_cost_dominance = false;
  params.enable_antipodal_components = false;
  params.enable_structural_l1_substitution = false;
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
  params.enable_dual_fix = false;
  params.enable_duplicate_columns = false;
  params.enable_redundant_bounds = false;
  params.enable_zero_cost_redundant_box_bounds = false;
  params.bounded_two_row_projection_min_candidates = 1;
  params.bounded_two_row_projection_min_candidate_ratio = 0.0;
  params.orphan_mccormick_projection_min_candidates = 1;
  params.orphan_mccormick_projection_min_candidate_ratio = 0.0;
  return params;
}

void test_chained_projection_and_primal_recovery() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_pair_block_lp();
  PresolveParams params = projected_only_params();
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  assert(!summary.has_infeasible);
  assert(!summary.has_unbounded);
  assert(summary.reduced_rows == 0);
  assert(summary.reduced_cols == 4);
  assert(summary.reduced_nnz == 0);
  assert(summary.record.has_primal_only_projected_auxiliary_reduction);
  assert(summary.record.tape.types.size() == 2);
  assert(summary.record.tape.types[0] ==
         static_cast<std::int32_t>(PostsolveReductionType::FmeCol));
  assert(summary.record.tape.types[1] ==
         static_cast<std::int32_t>(PostsolveReductionType::FmeCol));

  double* x_red = copy_to_device<double>({1.0, 1.0, 0.4, 0.2});
  double* z_red = copy_to_device<double>({0.0, 0.0, 0.0, 0.0});
  GpuPostsolveResult post =
      postsolve_gpu(x_red, nullptr, z_red, summary.record, &lp);
  const std::vector<double> x = copy_to_host(post.x_org, 6);
  assert(std::fabs(x[0] - 1.0) < 1.0e-12);
  assert(std::fabs(x[1] - 1.0) < 1.0e-12);
  assert(std::fabs(x[2] - 0.4) < 1.0e-12);
  assert(std::fabs(x[3] - 0.2) < 1.0e-12);
  assert(std::fabs(x[4] - 1.0) < 1.0e-12);
  assert(std::fabs(x[5] - 0.1) < 1.0e-12);

  cudaFree(x_red);
  cudaFree(z_red);
  cudaFree(post.x_org);
  cudaFree(post.y_org);
  cudaFree(post.z_org);
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

void test_bounded_projection_fails_closed() {
  using namespace gpu_presolver::presolve;
  PresolveParams params = projected_only_params();

  LPInfoGpu boundary = make_pair_block_lp(3.0, 0.0);
  BoundedTwoRowProjectionAnalysisGpu analysis =
      analyze_bounded_two_row_projection(boundary, params);
  assert(!analysis.applicable);
  assert(analysis.candidate_count == 0);
  free_bounded_two_row_projection_analysis(analysis);
  free_lp(boundary);

  LPInfoGpu nonzero_cost = make_pair_block_lp(4.0, 0.25);
  analysis = analyze_bounded_two_row_projection(nonzero_cost, params);
  assert(!analysis.applicable);
  assert(analysis.candidate_count == 0);
  free_bounded_two_row_projection_analysis(analysis);
  free_lp(nonzero_cost);
}

void test_mccormick_detector_is_exact_and_fail_closed() {
  using namespace gpu_presolver::presolve;
  PresolveParams params = projected_only_params();

  LPInfoGpu exact = make_mccormick_lp(false, false);
  OrphanMcCormickProjectionAnalysisGpu analysis =
      analyze_orphan_mccormick_projection(exact, params);
  assert(analysis.applicable);
  assert(analysis.candidate_count == 1);
  free_orphan_mccormick_projection_analysis(analysis);
  free_lp(exact);

  LPInfoGpu corrupted = make_mccormick_lp(true, false);
  analysis = analyze_orphan_mccormick_projection(corrupted, params);
  assert(!analysis.applicable);
  free_orphan_mccormick_projection_analysis(analysis);
  free_lp(corrupted);

  LPInfoGpu nonunit = make_mccormick_lp(false, true);
  analysis = analyze_orphan_mccormick_projection(nonunit, params);
  assert(!analysis.applicable);
  free_orphan_mccormick_projection_analysis(analysis);
  free_lp(nonunit);
}

void test_rules_are_tiered_bootstrap_only() {
  using namespace gpu_presolver::presolve;
  LPInfoGpu lp = make_pair_block_lp();
  PresolveParams params = projected_only_params();
  params.enable_tiered_bootstrap = false;
  GpuPresolveSummary summary = run_gpu_presolve_with_reduced_lp(lp, params);
  assert(summary.reduced_rows == 5);
  assert(summary.reduced_cols == 6);
  assert(summary.reduced_nnz == 15);
  assert(summary.record.tape.types.empty());
  free_gpu_presolve_reduced_lp(summary);
  free_lp(lp);
}

}  // namespace

int main() {
  test_chained_projection_and_primal_recovery();
  test_bounded_projection_fails_closed();
  test_mccormick_detector_is_exact_and_fail_closed();
  test_rules_are_tiered_bootstrap_only();
  std::cout << "test_rule_projected_auxiliaries passed\n";
  return 0;
}
