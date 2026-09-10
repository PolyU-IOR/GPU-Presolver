#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver_tool_common.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdlib>

namespace {

void write_lp_export(const std::filesystem::path& out_dir,
                     const std::string& model_name,
                     const gpu_presolver::presolve::GpuPresolveSummary& summary) {
  namespace tools = gpu_presolver::tools;
  const gpu_presolver::presolve::LPInfoGpu& lp = summary.reduced_lp;
  std::filesystem::create_directories(out_dir);

  std::ofstream meta(out_dir / "meta.txt");
  if (!meta.is_open()) {
    throw std::runtime_error("failed to open export meta");
  }
  meta << std::setprecision(std::numeric_limits<double>::max_digits10);
  meta << "format gpu_presolver_reduced_lp_v1\n";
  meta << "model " << model_name << "\n";
  meta << "status " << (summary.has_infeasible ? "infeasible" : (summary.has_unbounded ? "unbounded" : "ok")) << "\n";
  meta << "original_rows " << summary.original_rows << "\n";
  meta << "original_cols " << summary.original_cols << "\n";
  meta << "reduced_rows " << summary.reduced_rows << "\n";
  meta << "reduced_cols " << summary.reduced_cols << "\n";
  meta << "reduced_nnz " << summary.reduced_nnz << "\n";
  meta << "obj_constant " << lp.obj_constant << "\n";
  meta << "objective_shift " << summary.obj_constant_delta << "\n";
  meta << "iterations " << summary.iterations << "\n";
  tools::close_output_file(meta, out_dir / "meta.txt");

  tools::write_binary_vector(out_dir / "A_rowPtr_i32.bin",
                             tools::copy_to_host(lp.A.rowPtr, static_cast<std::size_t>(lp.A.rows) + 1));
  tools::write_binary_vector(out_dir / "A_colVal_i32.bin",
                             tools::copy_to_host(lp.A.colVal, static_cast<std::size_t>(lp.A.nnz)));
  tools::write_binary_vector(out_dir / "A_nzVal_f64.bin",
                             tools::copy_to_host(lp.A.nzVal, static_cast<std::size_t>(lp.A.nnz)));
  tools::write_binary_vector(out_dir / "c_f64.bin",
                             tools::copy_to_host(lp.c, static_cast<std::size_t>(lp.A.cols)));
  tools::write_binary_vector(out_dir / "l_f64.bin",
                             tools::copy_to_host(lp.l, static_cast<std::size_t>(lp.A.cols)));
  tools::write_binary_vector(out_dir / "u_f64.bin",
                             tools::copy_to_host(lp.u, static_cast<std::size_t>(lp.A.cols)));
  tools::write_binary_vector(out_dir / "AL_f64.bin",
                             tools::copy_to_host(lp.AL, static_cast<std::size_t>(lp.A.rows)));
  tools::write_binary_vector(out_dir / "AU_f64.bin",
                             tools::copy_to_host(lp.AU, static_cast<std::size_t>(lp.A.rows)));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: gpu_presolver_export_reduced <model.mps|-> <out_dir>\n";
    return 2;
  }

  try {
    const std::string input_path = argv[1];
    const std::filesystem::path out_dir = argv[2];
    mps_reader::MpsModel model = gpu_presolver::tools::read_mps_file(input_path);
    gpu_presolver::tools::DeviceLpOwner device_lp = gpu_presolver::tools::upload_lp(model.lp);

    gpu_presolver::presolve::PresolveParams params;
    gpu_presolver::tools::apply_presolve_env_overrides(params);
    gpu_presolver::presolve::GpuPresolveSummary summary =
        gpu_presolver::presolve::run_gpu_presolve_with_reduced_lp(device_lp.lp, params);

    write_lp_export(out_dir, model.name, summary);
    std::cout << "status " << (summary.has_infeasible ? "infeasible" : (summary.has_unbounded ? "unbounded" : "ok")) << "\n";
    std::cout << "reduced_rows " << summary.reduced_rows << "\n";
    std::cout << "reduced_cols " << summary.reduced_cols << "\n";
    std::cout << "reduced_nnz " << summary.reduced_nnz << "\n";
    std::cout << "out_dir " << out_dir.string() << "\n";
    gpu_presolver::presolve::free_gpu_presolve_reduced_lp(summary);
    return summary.has_infeasible || summary.has_unbounded ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
