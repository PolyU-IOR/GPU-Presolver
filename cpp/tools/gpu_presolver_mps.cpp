#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver_tool_common.hpp"

#include <chrono>
#include <iostream>
#include <string>

namespace {
double seconds_since(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: gpu_presolver_mps <model.mps|->\n";
    return 2;
  }

  try {
    const std::string path = argv[1];
    const auto parse_start = std::chrono::steady_clock::now();
    mps_reader::MpsModel model = gpu_presolver::tools::read_mps_file(path);
    const auto parse_stop = std::chrono::steady_clock::now();

    auto device_lp = gpu_presolver::tools::upload_lp(model.lp);
    gpu_presolver::presolve::PresolveParams params;
    gpu_presolver::tools::apply_presolve_env_overrides(params);
    const auto presolve_start = std::chrono::steady_clock::now();
    const gpu_presolver::presolve::GpuPresolveSummary summary =
        gpu_presolver::presolve::run_gpu_presolve_with_record(device_lp.lp, params);
    const auto presolve_stop = std::chrono::steady_clock::now();

    std::cout << "model " << model.name << '\n';
    std::cout << "status " << (summary.has_infeasible ? "infeasible" : (summary.has_unbounded ? "unbounded" : "ok")) << '\n';
    std::cout << "changed " << ((summary.reduced_rows != summary.original_rows ||
                                  summary.reduced_cols != summary.original_cols) ? 1 : 0) << '\n';
    std::cout << "original_rows " << summary.original_rows << '\n';
    std::cout << "original_cols " << summary.original_cols << '\n';
    std::cout << "original_nnz " << device_lp.lp.A.nnz << '\n';
    std::cout << "reduced_rows " << summary.reduced_rows << '\n';
    std::cout << "reduced_cols " << summary.reduced_cols << '\n';
    std::cout << "reduced_nnz " << summary.reduced_nnz << '\n';
    std::cout << "objective_shift " << summary.obj_constant_delta << '\n';
    std::cout << "iterations " << summary.iterations << '\n';
    std::cout << "parse_seconds " << seconds_since(parse_start, parse_stop) << '\n';
    std::cout << "presolve_seconds " << seconds_since(presolve_start, presolve_stop) << '\n';
    return summary.has_infeasible || summary.has_unbounded ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
