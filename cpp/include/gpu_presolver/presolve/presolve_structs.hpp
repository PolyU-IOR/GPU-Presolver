#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gpu_presolver::presolve {

enum class StructuralL1ResidualBoundMode : std::uint8_t {
  // Preserve every finite side of the original residual-variable box by
  // adding the exact ranged row l_e <= p - n <= u_e after the split.
  PreserveFinite = 0,
  // Apply the split only to residual variables with no finite bounds.
  StrictFreeOnly = 1,
  // Opt-in heuristic: treat sufficiently wide finite boxes as free.
  // This reduction is not exact.
  LegacyWideAsFree = 2,
};

// Presolve limits, tolerances, and reduction controls.
struct PresolveParams {
  int max_iters = 10;
  double max_time = __builtin_huge_val();
  bool verbose = false;
  bool record_postsolve_tape = true;
  bool record_postsolve_tape_cpu = false;

  double feasibility_tol = 1.0e-6;
  double bound_tol = 1.0e-6;
  double zero_tol = 1.0e-10;
  // One GPU kernel round sees only its entry bounds; bound dependency chains
  // therefore need repeated rounds, subject to the propagation limits.
  int implied_variable_bounds_max_rounds = 64;
  // Batch bound-only rounds in one plan to avoid rebuilding the LP each round.
  int implied_variable_bounds_max_bound_only_rounds = 64;
  // Avoid adding wide redundant boxes to free zero-cost auxiliaries.
  // Keep them free unless propagation nearly fixes them; one-sided and
  // nonzero-cost tightening is unchanged.
  bool implied_variable_bounds_preserve_free_zero_cost = true;
  bool doubleton_equations_single_batch_per_iter = false;
  // Cap the estimated fill-in from doubleton substitutions.
  // Sparse storage is rebuilt after every accepted batch.
  int doubleton_equations_max_fill_in_proxy = 96;
  bool doubleton_equations_scan = true;
  int doubleton_equations_min_selected_per_batch = 256;
  double doubleton_equations_min_selected_ratio = 0.005;
  int doubleton_equations_max_batch_rounds = 0;
  double doubleton_equations_max_time = 0.0;

  bool enable_infeasible_fixed_variables = true;
  bool enable_empty_rows = true;
  bool enable_singleton_rows = true;
  bool enable_infeasible_redundant_rows = true;
  bool enable_implied_variable_bounds = true;
  bool enable_duplicate_rows = true;
  bool enable_empty_cols = true;
  bool enable_column_singletons_eq = true;
  bool enable_column_singletons_dual_infer = true;
  bool enable_doubleton_equations = true;
  bool enable_dual_fix = true;
  // Limit nonzero-cost lock fixing to columns within this degree cap.
  // This retains dense slack directions; zero-cost candidates are unrestricted.
  int dual_fix_max_nonzero_cost_col_degree = 128;
  bool enable_duplicate_columns = true;
  // For A_k = ratio * A_j, require 1/R <= |ratio| <= R to limit scaling.
  double duplicate_columns_max_abs_ratio = 2.0;
  // Apply the ratio guard only to widespread small groups of scaled columns.
  int duplicate_columns_scale_guard_max_group_size = 4;
  int duplicate_columns_scale_guard_min_groups = 128;
  // Master switch for the five block reductions. Paired-variable component
  // quotienting has separate Linf and shifted-antipodal implementation paths.
  // When false, it overrides the six implementation-path flags below.
  bool enable_structure_specific_rules = true;
  // No-fill projection of zero-cost, finite-bounded columns in two short
  // one-sided rows. Require positive interval-certificate margins.
  bool enable_bounded_two_row_projection = true;
  int bounded_two_row_projection_max_row_nnz = 8;
  double bounded_two_row_projection_safety_tol = 1.0e-8;
  int bounded_two_row_projection_min_candidates = 128;
  double bounded_two_row_projection_min_candidate_ratio = 0.01;
  // Strict Fortet/McCormick auxiliary detector.  It accepts only the exact
  // binary-box convex envelope z >= x+y-1, z <= x, z <= y with c_z = 0.
  bool enable_orphan_mccormick_projection = true;
  int orphan_mccormick_projection_min_candidates = 128;
  double orphan_mccormick_projection_min_candidate_ratio = 0.01;
  // Strict Linf epigraph/component detector for the layout
  // [p_0...p_{k-1}, n_0...n_{k-1}, T_0...T_{g-1}].  Accepted models are
  // fully verified before equality-connected p_i-n_i components are merged.
  bool enable_linf_components = true;
  int linf_probe_pairs = 256;
  int linf_min_pairs = 4096;
  double linf_min_potential_gain = 0.20;
  double linf_min_actual_gain = 0.20;
  int linf_max_tail_cols = 8;
  int linf_max_quotient_terms_per_row = 5;
  bool enable_structural_l1_substitution = true;
  // Detect strict half-offset (+/-) column pairs, collapse equality-connected
  // pair components, and rebuild the LP in shifted quotient coordinates. The
  // sampled probe only rejects; accepted models are fully verified on GPU.
  bool enable_antipodal_components = true;
  int antipodal_probe_pairs = 256;
  int antipodal_min_pairs = 4096;
  double antipodal_min_potential_gain = 0.20;
  double antipodal_min_actual_gain = 0.20;
  int antipodal_max_quotient_terms_per_row = 6;
  // Strict unit set-covering detector.  A model is changed only after a full
  // GPU certificate proves rows Ax>=1, A in {0,1}, x in [0,1], positive
  // integer costs, and a cost-one witness for every row.  Columns with
  // c_j > nnz(A[:,j]) are then fixed to zero.
  bool enable_covering_cost_dominance = true;
  int covering_cost_dominance_probe_rows = 64;
  int covering_cost_dominance_probe_cols = 256;
  int covering_cost_dominance_probe_nnz = 256;
  int covering_cost_dominance_min_fixed_cols = 4096;
  double covering_cost_dominance_min_candidate_ratio = 0.20;
  StructuralL1ResidualBoundMode structural_l1_residual_bound_mode =
      StructuralL1ResidualBoundMode::PreserveFinite;
  // Used only when structural_l1_residual_bound_mode is LegacyWideAsFree.
  double structural_l1_residual_bound_as_free_min = 1.0e8;
  // At the end of presolve, remove redundant finite boxes from zero-cost
  // auxiliaries only when both bounds are independently implied by retained rows.
  bool enable_zero_cost_redundant_box_bounds = false;
  // Broader pass for redundant one-sided bounds.  Keep opt-in because
  // it can be substantially more expensive on large sparse models.
  bool enable_redundant_bounds = false;

  bool use_tiered_scheduler = true;
  bool enable_tiered_bootstrap = true;

  // Work cap for consecutive bound-only rounds, measured in nnz * rounds.
  // Structural changes reset the budget; nonpositive values disable this cap.
  // Round limits still apply; see the scheduler for the dense-model exception.
  std::int64_t implied_variable_bounds_bound_only_nnz_round_budget = 8000000;
};

// CSR matrix stored on the device with zero-based indices.
struct DeviceCsrMatrix {
  std::int32_t rows = 0;
  std::int32_t cols = 0;
  std::int32_t nnz = 0;
  std::int32_t* rowPtr = nullptr;
  std::int32_t* colVal = nullptr;
  double* nzVal = nullptr;
};

struct LPInfoGpu {
  DeviceCsrMatrix A;
  DeviceCsrMatrix AT;
  double* c = nullptr;
  double* AL = nullptr;
  double* AU = nullptr;
  double* l = nullptr;
  double* u = nullptr;
  double obj_constant = 0.0;
};

// Device buffers for row and column statistics used by reduction rules.
struct PresolveStatsGpu {
  void* contiguous_storage = nullptr;
  std::int32_t* row_nnz = nullptr;
  std::uint8_t* singleton_row_mask = nullptr;
  std::int32_t* singleton_row_col = nullptr;
  double* singleton_row_val = nullptr;

  std::int32_t* col_nnz = nullptr;
  std::uint8_t* empty_col_mask = nullptr;
  std::uint8_t* column_singleton_mask = nullptr;
  std::int32_t* column_singleton_row = nullptr;
  double* column_singleton_val = nullptr;

};

struct StructuralL1SplitRecovery {
  std::int32_t t_col = -1;
  std::int32_t e_col = -1;
  double rho = 1.0;
};

struct StructuralOuterPairRecovery {
  std::int32_t bound_col = -1;
  std::int32_t free_col = -1;
};

struct StructuralLinkedSlackRecovery {
  std::int32_t slack_col = -1;
  std::int32_t t_col = -1;
  double factor = 0.0;
};

struct StructuralMaxSlackRecovery {
  std::int32_t slack_col = -1;
  std::vector<std::int32_t> t_cols;
  std::vector<double> factors;
};

struct StructuralL1PrimalRecoveryStep {
  std::string pattern;
  std::vector<StructuralL1SplitRecovery> splits;
  std::vector<StructuralOuterPairRecovery> outer_pairs;
  std::vector<StructuralLinkedSlackRecovery> linked_slacks;
  std::vector<StructuralMaxSlackRecovery> max_slacks;
};

// Exact primal recovery for the model-wide antipodal quotient. The retained
// root slots contain shifted nonnegative variables P_C/N_C; recovery expands
// p_i = lower_p_i + P_C and n_i = lower_n_i + N_C. Deleted equality-row duals
// are intentionally not advertised as recoverable.
struct AntipodalComponentPrimalRecoveryStep {
  std::vector<std::int32_t> plus_cols;
  std::vector<std::int32_t> minus_cols;
  std::vector<std::int32_t> root_plus_cols;
  std::vector<std::int32_t> root_minus_cols;
  std::vector<double> plus_lower;
  std::vector<double> minus_lower;
};

// Each primal-recovery checkpoint stores the preceding tape-record count.
// Postsolve traverses checkpoints backwards, replaying each intervening
// tape range before expanding the checkpoint payload.
enum class PrimalRecoveryKind : std::uint8_t {
  StructuralL1 = 0,
  AntipodalComponent = 1,
};

struct PrimalRecoveryCheckpoint {
  PrimalRecoveryKind kind = PrimalRecoveryKind::StructuralL1;
  std::int32_t payload_index = -1;
  std::int32_t tape_position = 0;
};

enum class PostsolveReductionType : std::int32_t {
  FixedCol = 0,
  FixedColInf = 1,
  SubCol = 2,
  DuplicateColumn = 3,
  DuplicateRow = 4,
  DeletedRow = 5,
  AddedRow = 6,
  AddedRows = 7,
  LhsChange = 8,
  RhsChange = 9,
  EqToIneq = 10,
  BoundChangeNoRow = 11,
  BoundChangeTheRow = 12,
  DoubletonEquation = 13,
  FmeCol = 14,
};

enum class PostsolveDualMode : std::uint8_t {
  None = 0,
  Exact = 1,
  Minimal = 2,
};

struct PostsolveTape {
  std::vector<std::int32_t> types;
  std::vector<std::int32_t> index_starts{0};
  std::vector<std::int32_t> value_starts{0};
  std::vector<std::uint8_t> dual_modes;
  std::vector<std::int32_t> indices;
  std::vector<double> vals;
};

// Optional device-resident copy of the chronological postsolve tape.
// The host tape remains canonical.  postsolve_gpu reuses this copy when
// populated and otherwise uploads the host tape on demand.
struct PostsolveTapeGpu {
  PostsolveTapeGpu() = default;
  ~PostsolveTapeGpu();

  PostsolveTapeGpu(const PostsolveTapeGpu&) = delete;
  PostsolveTapeGpu& operator=(const PostsolveTapeGpu&) = delete;
  PostsolveTapeGpu(PostsolveTapeGpu&& other) noexcept;
  PostsolveTapeGpu& operator=(PostsolveTapeGpu&& other) noexcept;

  void reset() noexcept;

  std::int32_t* types = nullptr;
  std::int32_t* index_starts = nullptr;
  std::int32_t* value_starts = nullptr;
  std::uint8_t* dual_modes = nullptr;
  std::int32_t* indices = nullptr;
  double* vals = nullptr;
  std::int32_t record_count = 0;
  std::int32_t index_count = 0;
  std::int32_t value_count = 0;
  std::int32_t record_capacity = 0;
  std::int32_t index_capacity = 0;
  std::int32_t value_capacity = 0;
};

struct PresolveRecordGpu {
  std::int32_t m0 = 0;
  std::int32_t n0 = 0;
  std::int32_t m1 = 0;
  std::int32_t n1 = 0;

  std::vector<std::int32_t> row_org2red;
  std::vector<std::int32_t> row_red2org;
  std::vector<std::int32_t> col_org2red;
  std::vector<std::int32_t> col_red2org;

  std::vector<std::int32_t> fixed_idx;
  std::vector<double> fixed_val;
  std::vector<std::int32_t> removed_row_idx;
  std::vector<std::int32_t> removed_col_idx;

  double obj_constant_old = 0.0;
  double obj_constant_new = 0.0;

  std::vector<StructuralL1PrimalRecoveryStep> structural_primal_recoveries;
  std::vector<AntipodalComponentPrimalRecoveryStep> antipodal_component_recoveries;
  std::vector<PrimalRecoveryCheckpoint> primal_recovery_timeline;
  bool has_primal_only_antipodal_reduction = false;
  bool has_primal_only_covering_cost_dominance_reduction = false;
  bool has_primal_only_projected_auxiliary_reduction = false;
  PostsolveTape tape;
  PostsolveTapeGpu tape_gpu;
};

struct GpuPostsolveResult {
  double* x_org = nullptr;
  double* y_org = nullptr;
  double* z_org = nullptr;
  std::int32_t n0 = 0;
  std::int32_t m0 = 0;
};

// Pending row, column, bound, and objective changes for a reduction batch.
struct PresolvePlanGpu {
  void* contiguous_storage = nullptr;
  std::uint8_t* keep_row_mask = nullptr;
  std::uint8_t* keep_col_mask = nullptr;

  DeviceCsrMatrix new_A;
  bool has_new_A = false;

  double* new_c = nullptr;
  double* new_AL = nullptr;
  double* new_AU = nullptr;
  double* new_l = nullptr;
  double* new_u = nullptr;
  double obj_constant_delta = 0.0;

  bool has_change = false;
  bool has_row_action = false;
  bool has_col_action = false;
  bool has_infeasible = false;
  bool has_unbounded = false;

  bool has_structural_primal_recovery = false;
  StructuralL1PrimalRecoveryStep structural_primal_recovery;
  bool has_antipodal_component_recovery = false;
  AntipodalComponentPrimalRecoveryStep antipodal_component_recovery;
  bool has_covering_cost_dominance_reduction = false;
  bool has_projected_auxiliary_reduction = false;
  PostsolveTape tape;
  PostsolveTapeGpu tape_gpu;
};

}  // namespace gpu_presolver::presolve
