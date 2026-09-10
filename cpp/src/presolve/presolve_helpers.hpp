#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace gpu_presolver::presolve::detail {

static inline void throw_if_cuda_error(cudaError_t status, const char* context) {
  if (status == cudaSuccess) {
    return;
  }
  throw std::runtime_error(
      std::string(context) + ": " + cudaGetErrorString(status));
}

static inline bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static inline void append_postsolve_record(PostsolveTape& tape,
                             PostsolveReductionType type,
                             const std::vector<std::int32_t>& indices,
                             const std::vector<double>& vals,
                             PostsolveDualMode dual_mode) {
  tape.types.push_back(static_cast<std::int32_t>(type));
  tape.indices.insert(tape.indices.end(), indices.begin(), indices.end());
  tape.vals.insert(tape.vals.end(), vals.begin(), vals.end());
  tape.index_starts.push_back(static_cast<std::int32_t>(tape.indices.size()));
  tape.value_starts.push_back(static_cast<std::int32_t>(tape.vals.size()));
  tape.dual_modes.push_back(static_cast<std::uint8_t>(dual_mode));
}

static inline void append_postsolve_record(PostsolveTape& tape,
                             PostsolveReductionType type,
                             const std::int32_t* indices,
                             std::size_t index_count,
                             const double* vals,
                             std::size_t value_count,
                             PostsolveDualMode dual_mode) {
  tape.types.push_back(static_cast<std::int32_t>(type));
  tape.indices.insert(tape.indices.end(), indices, indices + index_count);
  tape.vals.insert(tape.vals.end(), vals, vals + value_count);
  tape.index_starts.push_back(static_cast<std::int32_t>(tape.indices.size()));
  tape.value_starts.push_back(static_cast<std::int32_t>(tape.vals.size()));
  tape.dual_modes.push_back(static_cast<std::uint8_t>(dual_mode));
}

static inline void append_postsolve_record(PostsolveTape& tape,
                             const std::vector<std::int32_t>& indices,
                             const std::vector<double>& vals) {
  tape.types.push_back(
      static_cast<std::int32_t>(PostsolveReductionType::FmeCol));
  tape.indices.insert(tape.indices.end(), indices.begin(), indices.end());
  tape.vals.insert(tape.vals.end(), vals.begin(), vals.end());
  tape.index_starts.push_back(
      static_cast<std::int32_t>(tape.indices.size()));
  tape.value_starts.push_back(
      static_cast<std::int32_t>(tape.vals.size()));
  tape.dual_modes.push_back(
      static_cast<std::uint8_t>(PostsolveDualMode::Minimal));
}

}  // namespace gpu_presolver::presolve::detail
