#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "mps_reader/mpsreader.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

struct CudaFree {
  void operator()(void* pointer) const noexcept { cudaFree(pointer); }
};

// Own the input arrays until presolve (and any subsequent postsolve) finishes.
struct DeviceMemory {
  std::vector<std::unique_ptr<void, CudaFree>> buffers;

  template <class T>
  T* upload(const std::vector<T>& values) {
    if (values.empty()) return nullptr;
    T* pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, values.size() * sizeof(T)));
    std::unique_ptr<void, CudaFree> owner(pointer);
    check_cuda(cudaMemcpy(pointer, values.data(), values.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    buffers.push_back(std::move(owner));
    return pointer;
  }
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: presolve_example <model.mps>\n";
    return 2;
  }

  try {
    std::cout << "Start reading file...." << std::endl;
    const auto read_start = Clock::now();
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("Cannot open " + std::string(argv[1]));
    const auto model = mps_reader::read_mps(input);
    const auto& csr = model.lp.csr();
    const auto& csc = model.lp.csc();
    const double read_seconds = seconds_since(read_start);
    std::cout << "File reading time: " << read_seconds << " seconds\n";
    std::cout << "problem information: nRow = " << csr.rows()
              << ", nCol = " << csr.cols() << ", nnz A = " << csr.nnz() << '\n';

    // Initialize the selected CUDA device before timing input allocation/copy.
    check_cuda(cudaSetDevice(0));
    check_cuda(cudaDeviceSynchronize());
    const auto copy_start = Clock::now();
    DeviceMemory memory;
    gpu_presolver::presolve::LPInfoGpu lp{};
    lp.A.rows = csr.rows();
    lp.A.cols = csr.cols();
    lp.A.nnz = csr.nnz();
    lp.A.rowPtr = memory.upload(std::vector<std::int32_t>(csr.row_ptr().begin(), csr.row_ptr().end()));
    lp.A.colVal = memory.upload(std::vector<std::int32_t>(csr.col_idx().begin(), csr.col_idx().end()));
    lp.A.nzVal = memory.upload(csr.values());

    // The CSC arrays of A are the CSR arrays of its transpose.
    lp.AT.rows = csc.cols();
    lp.AT.cols = csc.rows();
    lp.AT.nnz = csc.nnz();
    lp.AT.rowPtr = memory.upload(std::vector<std::int32_t>(csc.col_ptr().begin(), csc.col_ptr().end()));
    lp.AT.colVal = memory.upload(std::vector<std::int32_t>(csc.row_idx().begin(), csc.row_idx().end()));
    lp.AT.nzVal = memory.upload(csc.values());
    lp.c = memory.upload(model.lp.objective());
    lp.AL = memory.upload(model.lp.row_lower());
    lp.AU = memory.upload(model.lp.row_upper());
    lp.l = memory.upload(model.lp.col_lower());
    lp.u = memory.upload(model.lp.col_upper());
    lp.obj_constant = model.lp.obj_constant();
    check_cuda(cudaDeviceSynchronize());
    const double copy_seconds = seconds_since(copy_start);
    std::cout << "GPU-Presolver input copy time (excluded): "
              << copy_seconds << " seconds\n";

    using namespace gpu_presolver::presolve;
    const auto wall_start = Clock::now();
    PresolveParams params;
    params.max_iters = 10;
    std::cout << "GPU structure-specific rules: "
              << (params.enable_structure_specific_rules ? "Enabled" : "Disabled")
              << " (PresolveParams)\n";
    std::cout << "Doing presolve (GPU-Presolver)..." << std::endl;
    auto summary = run_gpu_presolve_with_reduced_lp(lp, params);
    // Wall time covers parameter setup and the complete API call, ending when
    // the reduced device LP is ready.
    const auto sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
      free_gpu_presolve_reduced_lp(summary);
      check_cuda(sync_status);
    }
    const double wall_seconds = seconds_since(wall_start);
    std::cout << "GPU-Presolver presolve wall time: "
              << wall_seconds << " seconds\n";
    const bool stopped = summary.has_infeasible || summary.has_unbounded;
    std::cout << "Status: " << (summary.has_infeasible ? "infeasible" :
                                summary.has_unbounded ? "unbounded" : "ok") << '\n';
    if (!stopped) {
      // Pass this device LP to your solver; this example only runs presolve.
      const auto& reduced = summary.reduced_lp;
      std::cout << "GPU-Presolver reduced problem: (" << lp.A.rows << ", "
                << lp.A.cols << ") -> (" << reduced.A.rows << ", "
                << reduced.A.cols << ")\n";
      std::cout << "GPU-Presolver reduced nnz: " << reduced.A.nnz << '\n';
    }

    free_gpu_presolve_reduced_lp(summary);
    // The summary releases its recovery record; memory releases the input arrays.
    return stopped ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
