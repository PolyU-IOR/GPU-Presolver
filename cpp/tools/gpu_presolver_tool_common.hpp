#pragma once

#include "mps_reader/mpsreader.hpp"
#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_presolver::tools {

inline bool parse_bool_env_value(const char* value, bool fallback) {
  if (value == nullptr) {
    return fallback;
  }
  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (text == "1" || text == "true" || text == "yes" || text == "on") {
    return true;
  }
  if (text == "0" || text == "false" || text == "no" || text == "off") {
    return false;
  }
  return fallback;
}

inline std::int64_t parse_int64_env_value(const char* value, std::int64_t fallback) {
  if (value == nullptr) {
    return fallback;
  }
  char* end = nullptr;
  errno = 0;
  const long long parsed = std::strtoll(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') {
    return fallback;
  }
  return static_cast<std::int64_t>(parsed);
}

inline double parse_double_env_value(const char* value, double fallback) {
  if (value == nullptr) {
    return fallback;
  }
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(value, &end);
  if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed)) {
    return fallback;
  }
  return parsed;
}

inline void apply_presolve_env_overrides(gpu_presolver::presolve::PresolveParams& params) {
  if (const char* value = std::getenv("GPUPRESOLVER_MAX_ITERS")) {
    params.max_iters = std::max(0, std::atoi(value));
  }
  if (const char* value = std::getenv("GPUPRESOLVER_SCHEDULER")) {
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    if (text == "fixed" || text == "legacy" || text == "old") {
      params.use_tiered_scheduler = false;
    } else if (text == "tiered" || text == "default") {
      params.use_tiered_scheduler = true;
    }
  }
  params.use_tiered_scheduler = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_USE_TIERED_SCHEDULER"), params.use_tiered_scheduler);
  params.enable_doubleton_equations = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_DOUBLETON_EQUATIONS"),
      params.enable_doubleton_equations);
  params.enable_dual_fix = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_DUAL_FIX"), params.enable_dual_fix);
  params.enable_duplicate_columns = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_DUPLICATE_COLUMNS"),
      params.enable_duplicate_columns);
  params.dual_fix_max_nonzero_cost_col_degree = static_cast<int>(
      parse_int64_env_value(
          std::getenv("GPUPRESOLVER_DUAL_FIX_MAX_NONZERO_COST_COL_DEGREE"),
          params.dual_fix_max_nonzero_cost_col_degree));
  params.duplicate_columns_max_abs_ratio = parse_double_env_value(
      std::getenv("GPUPRESOLVER_DUPLICATE_COLUMNS_MAX_ABS_RATIO"),
      params.duplicate_columns_max_abs_ratio);
  params.implied_variable_bounds_preserve_free_zero_cost = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_IMPLIED_VARIABLE_BOUNDS_PRESERVE_FREE_ZERO_COST"),
      params.implied_variable_bounds_preserve_free_zero_cost);
  params.implied_variable_bounds_bound_only_nnz_round_budget = parse_int64_env_value(
      std::getenv("GPUPRESOLVER_IMPLIED_VARIABLE_BOUNDS_BOUND_ONLY_NNZ_ROUND_BUDGET"),
      params.implied_variable_bounds_bound_only_nnz_round_budget);
  params.enable_redundant_bounds = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_REDUNDANT_BOUNDS"), params.enable_redundant_bounds);
  params.enable_zero_cost_redundant_box_bounds = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_ZERO_COST_REDUNDANT_BOX_BOUNDS"),
      params.enable_zero_cost_redundant_box_bounds);
  params.enable_structure_specific_rules = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_STRUCTURE_SPECIFIC_RULES"),
      params.enable_structure_specific_rules);
  params.enable_structural_l1_substitution = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_STRUCTURAL_L1_SUBSTITUTION"),
      params.enable_structural_l1_substitution);
  params.enable_linf_components = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_LINF_COMPONENTS"),
      params.enable_linf_components);
  params.enable_antipodal_components = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_ANTIPODAL_COMPONENTS"),
      params.enable_antipodal_components);
  params.enable_covering_cost_dominance = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_COVERING_COST_DOMINANCE"),
      params.enable_covering_cost_dominance);
  params.enable_bounded_two_row_projection = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_BOUNDED_TWO_ROW_PROJECTION"),
      params.enable_bounded_two_row_projection);
  params.enable_orphan_mccormick_projection = parse_bool_env_value(
      std::getenv("GPUPRESOLVER_ENABLE_ORPHAN_MCCORMICK_PROJECTION"),
      params.enable_orphan_mccormick_projection);
}

inline void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

template <class T>
T* copy_to_device(const std::vector<T>& values) {
  if (values.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  check_cuda(cudaMalloc(&device, sizeof(T) * values.size()), "cudaMalloc");
  const cudaError_t status =
      cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cudaFree(device);
    check_cuda(status, "cudaMemcpy H2D");
  }
  return device;
}

template <class T>
std::vector<T> copy_to_host(const T* device, std::size_t size) {
  std::vector<T> values(size);
  if (size == 0) {
    return values;
  }
  check_cuda(cudaMemcpy(values.data(), device, sizeof(T) * size, cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H");
  return values;
}

inline std::vector<std::int32_t> to_i32(const std::vector<int>& input) {
  std::vector<std::int32_t> out(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    out[i] = static_cast<std::int32_t>(input[i]);
  }
  return out;
}

struct DeviceLpOwner {
  gpu_presolver::presolve::LPInfoGpu lp{};

  DeviceLpOwner() = default;
  DeviceLpOwner(const DeviceLpOwner&) = delete;
  DeviceLpOwner& operator=(const DeviceLpOwner&) = delete;
  DeviceLpOwner(DeviceLpOwner&& other) noexcept : lp(std::exchange(other.lp, {})) {}
  DeviceLpOwner& operator=(DeviceLpOwner&& other) noexcept {
    if (this != &other) {
      release();
      lp = std::exchange(other.lp, {});
    }
    return *this;
  }

  ~DeviceLpOwner() { release(); }

 private:
  void release() noexcept {
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
    lp = {};
  }
};

inline DeviceLpOwner upload_lp(const mps_reader::LpModel& model) {
  DeviceLpOwner owner;
  const mps_reader::CsrMatrix& csr = model.csr();
  const mps_reader::CscMatrix& csc = model.csc();

  const std::vector<std::int32_t> row_ptr = to_i32(csr.row_ptr());
  const std::vector<std::int32_t> col_idx = to_i32(csr.col_idx());
  const std::vector<std::int32_t> at_row_ptr = to_i32(csc.col_ptr());
  const std::vector<std::int32_t> at_col_idx = to_i32(csc.row_idx());

  owner.lp.A.rows = static_cast<std::int32_t>(csr.rows());
  owner.lp.A.cols = static_cast<std::int32_t>(csr.cols());
  owner.lp.A.nnz = static_cast<std::int32_t>(csr.nnz());
  owner.lp.A.rowPtr = copy_to_device(row_ptr);
  owner.lp.A.colVal = copy_to_device(col_idx);
  owner.lp.A.nzVal = copy_to_device(csr.values());

  owner.lp.AT.rows = static_cast<std::int32_t>(csc.cols());
  owner.lp.AT.cols = static_cast<std::int32_t>(csc.rows());
  owner.lp.AT.nnz = static_cast<std::int32_t>(csc.nnz());
  owner.lp.AT.rowPtr = copy_to_device(at_row_ptr);
  owner.lp.AT.colVal = copy_to_device(at_col_idx);
  owner.lp.AT.nzVal = copy_to_device(csc.values());

  owner.lp.c = copy_to_device(model.objective());
  owner.lp.AL = copy_to_device(model.row_lower());
  owner.lp.AU = copy_to_device(model.row_upper());
  owner.lp.l = copy_to_device(model.col_lower());
  owner.lp.u = copy_to_device(model.col_upper());
  owner.lp.obj_constant = model.obj_constant();
  return owner;
}

inline mps_reader::MpsModel read_mps_file(const std::string& path) {
  std::unique_ptr<std::ifstream> file;
  std::istream* input = &std::cin;
  if (path != "-") {
    file = std::make_unique<std::ifstream>(path);
    if (!file->is_open()) {
      throw std::runtime_error("failed to open " + path);
    }
    input = file.get();
  }
  return mps_reader::read_mps(*input);
}

inline void close_output_file(std::ofstream& out, const std::filesystem::path& path) {
  out.flush();
  const bool flushed = static_cast<bool>(out);
  out.close();
  if (!flushed || !out) {
    throw std::runtime_error("failed to write " + path.string());
  }
}

template <class T>
std::streamsize binary_vector_bytes(std::size_t count) {
  if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T)) {
    throw std::length_error("binary vector is too large");
  }
  return static_cast<std::streamsize>(count * sizeof(T));
}

template <class T>
void write_binary_vector(const std::filesystem::path& path, const std::vector<T>& values) {
  const auto bytes = binary_vector_bytes<T>(values.size());
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open " + path.string());
  }
  if (!values.empty()) {
    out.write(reinterpret_cast<const char*>(values.data()),
              bytes);
  }
  close_output_file(out, path);
}

template <class T>
std::vector<T> read_binary_vector(const std::filesystem::path& path, std::size_t count) {
  const auto bytes = binary_vector_bytes<T>(count);
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open " + path.string());
  }
  if (in.tellg() != std::streampos(bytes)) {
    throw std::runtime_error("unexpected binary file size: " + path.string());
  }
  in.seekg(0);
  std::vector<T> values(count);
  if (count > 0) {
    in.read(reinterpret_cast<char*>(values.data()),
            bytes);
    if (!in) {
      throw std::runtime_error("failed to read " + path.string());
    }
  }
  if (!in || in.peek() != std::char_traits<char>::eof() || in.bad()) {
    throw std::runtime_error("failed to read complete file " + path.string());
  }
  return values;
}

}  // namespace gpu_presolver::tools
