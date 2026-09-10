#pragma once

#include "gpu_presolver/presolve/presolve_structs.hpp"

#include <cstdint>

namespace gpu_presolver::presolve {

// Packs FixedCol postsolve records on the GPU and transfers only selected
// columns and coefficients belonging to rows that remain in the model.
// `tape` may be null when only the stable fixed-objective delta is needed.
// `excluded_mask` may be null; otherwise nonzero entries are skipped.
double append_compact_fixed_col_tape_from_device(PostsolveTape* tape,
                                               const std::uint8_t* fixed_mask,
                                               const std::uint8_t* excluded_mask,
                                               const double* fixed_val,
                                               const std::uint8_t* keep_row,
                                               const double* c,
                                               const DeviceCsrMatrix& AT,
                                               std::int32_t n,
                                               const char* context);

}  // namespace gpu_presolver::presolve
