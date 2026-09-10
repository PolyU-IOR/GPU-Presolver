#include "gpu_presolver/presolve/gpu_postsolve.hpp"
#include "gpu_presolver/presolve/gpu_presolve.hpp"
#include "gpu_presolver_tool_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

class Fnv1a64 {
 public:
  template <typename T>
  void add_scalar(const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    add_bytes(&value, sizeof(value));
  }

  template <typename T>
  void add_vector(const std::vector<T>& values) {
    const std::uint64_t size = static_cast<std::uint64_t>(values.size());
    add_scalar(size);
    if (!values.empty()) {
      add_bytes(values.data(), values.size() * sizeof(T));
    }
  }

  void add_string(const std::string& value) {
    const std::uint64_t size = static_cast<std::uint64_t>(value.size());
    add_scalar(size);
    add_bytes(value.data(), value.size());
  }

  std::uint64_t value() const { return hash_; }

 private:
  void add_bytes(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
      hash_ ^= static_cast<std::uint64_t>(bytes[i]);
      hash_ *= UINT64_C(1099511628211);
    }
  }

  std::uint64_t hash_ = UINT64_C(14695981039346656037);
};

template <typename T>
std::uint64_t raw_vector_hash(const std::vector<T>& values) {
  Fnv1a64 hash;
  hash.add_vector(values);
  return hash.value();
}

std::uint64_t record_hash(const gpu_presolver::presolve::PresolveRecordGpu& record) {
  Fnv1a64 hash;
  hash.add_scalar(record.m0);
  hash.add_scalar(record.n0);
  hash.add_scalar(record.m1);
  hash.add_scalar(record.n1);
  hash.add_vector(record.row_org2red);
  hash.add_vector(record.row_red2org);
  hash.add_vector(record.col_org2red);
  hash.add_vector(record.col_red2org);
  hash.add_vector(record.fixed_idx);
  hash.add_vector(record.fixed_val);
  hash.add_vector(record.removed_row_idx);
  hash.add_vector(record.removed_col_idx);
  hash.add_scalar(record.obj_constant_old);
  hash.add_scalar(record.obj_constant_new);
  hash.add_scalar(static_cast<std::uint8_t>(
      record.has_primal_only_antipodal_reduction));
  hash.add_scalar(static_cast<std::uint8_t>(
      record.has_primal_only_covering_cost_dominance_reduction));
  hash.add_scalar(static_cast<std::uint8_t>(
      record.has_primal_only_projected_auxiliary_reduction));

  const std::uint64_t recovery_count =
      static_cast<std::uint64_t>(record.structural_primal_recoveries.size());
  hash.add_scalar(recovery_count);
  for (const auto& recovery : record.structural_primal_recoveries) {
    hash.add_string(recovery.pattern);
    const std::uint64_t split_count = static_cast<std::uint64_t>(recovery.splits.size());
    hash.add_scalar(split_count);
    for (const auto& split : recovery.splits) {
      hash.add_scalar(split.t_col);
      hash.add_scalar(split.e_col);
      hash.add_scalar(split.rho);
    }
    const std::uint64_t outer_count = static_cast<std::uint64_t>(recovery.outer_pairs.size());
    hash.add_scalar(outer_count);
    for (const auto& outer : recovery.outer_pairs) {
      hash.add_scalar(outer.bound_col);
      hash.add_scalar(outer.free_col);
    }
    const std::uint64_t linked_count =
        static_cast<std::uint64_t>(recovery.linked_slacks.size());
    hash.add_scalar(linked_count);
    for (const auto& linked : recovery.linked_slacks) {
      hash.add_scalar(linked.slack_col);
      hash.add_scalar(linked.t_col);
      hash.add_scalar(linked.factor);
    }
    const std::uint64_t max_count = static_cast<std::uint64_t>(recovery.max_slacks.size());
    hash.add_scalar(max_count);
    for (const auto& max_slack : recovery.max_slacks) {
      hash.add_scalar(max_slack.slack_col);
      hash.add_vector(max_slack.t_cols);
      hash.add_vector(max_slack.factors);
    }
  }

  const std::uint64_t antipodal_recovery_count =
      static_cast<std::uint64_t>(
          record.antipodal_component_recoveries.size());
  hash.add_scalar(antipodal_recovery_count);
  for (const auto& recovery : record.antipodal_component_recoveries) {
    hash.add_vector(recovery.plus_cols);
    hash.add_vector(recovery.minus_cols);
    hash.add_vector(recovery.root_plus_cols);
    hash.add_vector(recovery.root_minus_cols);
    hash.add_vector(recovery.plus_lower);
    hash.add_vector(recovery.minus_lower);
  }

  const std::uint64_t checkpoint_count = static_cast<std::uint64_t>(
      record.primal_recovery_timeline.size());
  hash.add_scalar(checkpoint_count);
  for (const auto& checkpoint : record.primal_recovery_timeline) {
    hash.add_scalar(static_cast<std::uint8_t>(checkpoint.kind));
    hash.add_scalar(checkpoint.payload_index);
    hash.add_scalar(checkpoint.tape_position);
  }

  hash.add_vector(record.tape.types);
  hash.add_vector(record.tape.index_starts);
  hash.add_vector(record.tape.value_starts);
  hash.add_vector(record.tape.dual_modes);
  hash.add_vector(record.tape.indices);
  hash.add_vector(record.tape.vals);
  return hash.value();
}

double finite_bound_norm_l2(const std::vector<double>& lower,
                            const std::vector<double>& upper) {
  double norm = 0.0;
  for (const double v : lower) {
    if (std::isfinite(v)) {
      norm = std::hypot(norm, v);
    }
  }
  for (const double v : upper) {
    if (std::isfinite(v)) {
      norm = std::hypot(norm, v);
    }
  }
  if (!std::isfinite(norm)) {
    throw std::runtime_error("non-finite constraint bound norm");
  }
  return 1.0 + norm;
}

void require_finite_vector(const std::vector<double>& values, const char* name) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error(std::string("non-finite ") + name + " at index " + std::to_string(i));
    }
  }
}

struct PrimalFeasibility {
  double normalized = 0.0;
  double row_l2 = 0.0;
  double row_max = 0.0;
  double col_l2 = 0.0;
  double col_max = 0.0;
};

struct ViolationDetail {
  int index = -1;
  double violation = 0.0;
  double value = 0.0;
  double lower = 0.0;
  double upper = 0.0;
};

void keep_top_violation(std::vector<ViolationDetail>& top, ViolationDetail item, std::size_t limit) {
  if (item.violation <= 0.0 || limit == 0) {
    return;
  }
  top.push_back(item);
  std::sort(top.begin(), top.end(), [](const ViolationDetail& lhs, const ViolationDetail& rhs) {
    if (lhs.violation != rhs.violation) {
      return lhs.violation > rhs.violation;
    }
    return lhs.index < rhs.index;
  });
  if (top.size() > limit) {
    top.resize(limit);
  }
}

PrimalFeasibility compute_primal_feasibility(const mps_reader::LpModel& model,
                                             const std::vector<double>& x) {
  const mps_reader::CsrMatrix& A = model.csr();
  const std::vector<double>& AL = model.row_lower();
  const std::vector<double>& AU = model.row_upper();
  const std::vector<double>& l = model.col_lower();
  const std::vector<double>& u = model.col_upper();

  require_finite_vector(x, "original primal solution");
  PrimalFeasibility out;
  for (int i = 0; i < A.rows(); ++i) {
    double activity = 0.0;
    for (int p = A.row_ptr()[static_cast<std::size_t>(i)];
         p < A.row_ptr()[static_cast<std::size_t>(i + 1)];
         ++p) {
      const int col = A.col_idx()[static_cast<std::size_t>(p)];
      activity += A.values()[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(col)];
    }
    if (!std::isfinite(activity)) {
      throw std::runtime_error("non-finite row activity at index " + std::to_string(i));
    }
    const double violation = std::max({0.0, AL[static_cast<std::size_t>(i)] - activity,
                                       activity - AU[static_cast<std::size_t>(i)]});
    out.row_l2 = std::hypot(out.row_l2, violation);
    out.row_max = std::max(out.row_max, violation);
  }

  for (std::size_t j = 0; j < x.size(); ++j) {
    const double violation = std::max({0.0, l[j] - x[j], x[j] - u[j]});
    out.col_l2 = std::hypot(out.col_l2, violation);
    out.col_max = std::max(out.col_max, violation);
  }
  out.normalized = std::max(out.row_l2, out.col_l2) / finite_bound_norm_l2(AL, AU);
  return out;
}

void print_top_violations(const mps_reader::LpModel& model,
                          const std::vector<double>& x,
                          std::size_t limit) {
  const mps_reader::CsrMatrix& A = model.csr();
  const std::vector<double>& AL = model.row_lower();
  const std::vector<double>& AU = model.row_upper();
  const std::vector<double>& l = model.col_lower();
  const std::vector<double>& u = model.col_upper();

  std::vector<ViolationDetail> top_rows;
  std::vector<ViolationDetail> top_cols;
  for (int i = 0; i < A.rows(); ++i) {
    double activity = 0.0;
    for (int p = A.row_ptr()[static_cast<std::size_t>(i)];
         p < A.row_ptr()[static_cast<std::size_t>(i + 1)];
         ++p) {
      const int col = A.col_idx()[static_cast<std::size_t>(p)];
      activity += A.values()[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(col)];
    }
    const double violation = std::max({0.0, AL[static_cast<std::size_t>(i)] - activity,
                                       activity - AU[static_cast<std::size_t>(i)]});
    keep_top_violation(top_rows,
                       ViolationDetail{i, violation, activity, AL[static_cast<std::size_t>(i)],
                                       AU[static_cast<std::size_t>(i)]},
                       limit);
  }

  for (std::size_t j = 0; j < x.size(); ++j) {
    const double violation = std::max({0.0, l[j] - x[j], x[j] - u[j]});
    keep_top_violation(top_cols,
                       ViolationDetail{static_cast<int>(j), violation, x[j], l[j], u[j]},
                       limit);
  }

  for (std::size_t k = 0; k < top_rows.size(); ++k) {
    const ViolationDetail& item = top_rows[k];
    std::cout << "top_row_violation_" << (k + 1) << " "
              << item.index << "," << item.violation << "," << item.value << ","
              << item.lower << "," << item.upper << "\n";
  }
  for (std::size_t k = 0; k < top_cols.size(); ++k) {
    const ViolationDetail& item = top_cols[k];
    std::cout << "top_col_violation_" << (k + 1) << " "
              << item.index << "," << item.violation << "," << item.value << ","
              << item.lower << "," << item.upper << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: gpu_presolver_check_postsolve_solution <model.mps|-> <solution_dir>\n";
    return 2;
  }

  try {
    const std::string input_path = argv[1];
    const std::filesystem::path solution_dir = argv[2];
    mps_reader::MpsModel model = gpu_presolver::tools::read_mps_file(input_path);
    gpu_presolver::tools::DeviceLpOwner device_lp = gpu_presolver::tools::upload_lp(model.lp);

    gpu_presolver::presolve::PresolveParams params;
    gpu_presolver::tools::apply_presolve_env_overrides(params);
    gpu_presolver::presolve::GpuPresolveSummary summary =
        gpu_presolver::presolve::run_gpu_presolve_with_record(device_lp.lp, params);

    if (summary.has_infeasible || summary.has_unbounded) {
      std::cerr << "cannot check a solution: presolve reports "
                << (summary.has_infeasible ? "infeasible" : "unbounded") << "\n";
      return 1;
    }

    const std::vector<double> x_red_h =
        gpu_presolver::tools::read_binary_vector<double>(solution_dir / "x_f64.bin",
                                                         static_cast<std::size_t>(summary.reduced_cols));
    const std::vector<double> y_red_h =
        gpu_presolver::tools::read_binary_vector<double>(solution_dir / "y_f64.bin",
                                                         static_cast<std::size_t>(summary.reduced_rows));
    const std::vector<double> z_red_h =
        gpu_presolver::tools::read_binary_vector<double>(solution_dir / "z_f64.bin",
                                                         static_cast<std::size_t>(summary.reduced_cols));
    require_finite_vector(x_red_h, "reduced primal solution");
    require_finite_vector(y_red_h, "reduced row dual solution");
    require_finite_vector(z_red_h, "reduced bound dual solution");

    double* x_red = gpu_presolver::tools::copy_to_device(x_red_h);
    double* y_red = gpu_presolver::tools::copy_to_device(y_red_h);
    double* z_red = gpu_presolver::tools::copy_to_device(z_red_h);

    gpu_presolver::presolve::GpuPostsolveResult post =
        gpu_presolver::presolve::postsolve_gpu(x_red, y_red, z_red, summary.record, &device_lp.lp);
    const std::vector<double> x_org =
        gpu_presolver::tools::copy_to_host(post.x_org, static_cast<std::size_t>(summary.original_cols));
    const std::vector<double> y_org =
        gpu_presolver::tools::copy_to_host(post.y_org, static_cast<std::size_t>(summary.original_rows));
    const std::vector<double> z_org =
        gpu_presolver::tools::copy_to_host(post.z_org, static_cast<std::size_t>(summary.original_cols));
    const PrimalFeasibility feas = compute_primal_feasibility(model.lp, x_org);

    std::cout << "status " << (summary.has_infeasible ? "infeasible" : (summary.has_unbounded ? "unbounded" : "ok")) << "\n";
    std::cout << "reduced_rows " << summary.reduced_rows << "\n";
    std::cout << "reduced_cols " << summary.reduced_cols << "\n";
    std::cout << "primal_feas_l2_normalized " << feas.normalized << "\n";
    std::cout << "row_violation_l2 " << feas.row_l2 << "\n";
    std::cout << "row_violation_max " << feas.row_max << "\n";
    std::cout << "col_violation_l2 " << feas.col_l2 << "\n";
    std::cout << "col_violation_max " << feas.col_max << "\n";
    std::cout << "record_hash " << record_hash(summary.record) << "\n";
    std::cout << "x_org_hash " << raw_vector_hash(x_org) << "\n";
    std::cout << "y_org_hash " << raw_vector_hash(y_org) << "\n";
    std::cout << "z_org_hash " << raw_vector_hash(z_org) << "\n";
    if (std::getenv("GPUPRESOLVER_PRINT_TOP_VIOLATIONS") != nullptr) {
      print_top_violations(model.lp, x_org, 10);
    }

    cudaFree(x_red);
    cudaFree(y_red);
    cudaFree(z_red);
    cudaFree(post.x_org);
    cudaFree(post.y_org);
    cudaFree(post.z_org);
    return std::isfinite(feas.normalized) && feas.normalized <= 1.0e-6 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
