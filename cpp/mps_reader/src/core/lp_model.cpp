#include "mps_reader/lp_model.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace mps_reader {
namespace {

void require_size(const char* name, std::size_t actual, std::size_t expected) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(name) + " has size " +
                                std::to_string(actual) + ", expected " +
                                std::to_string(expected));
  }
}

void require_bounds(const char* lower_name,
                    const char* upper_name,
                    const std::vector<double>& lower,
                    const std::vector<double>& upper) {
  for (std::size_t k = 0; k < lower.size(); ++k) {
    if (lower[k] > upper[k]) {
      throw std::invalid_argument(std::string(lower_name) + "[" + std::to_string(k) +
                                  "] exceeds " + upper_name + "[" + std::to_string(k) + "]");
    }
  }
}

}  // namespace

LpModel::LpModel(CscMatrix matrix,
                 std::vector<double> objective,
                 std::vector<double> row_lower,
                 std::vector<double> row_upper,
                 std::vector<double> col_lower,
                 std::vector<double> col_upper,
                 double obj_constant)
    : csc_(std::move(matrix)),
      csr_(csc_.to_csr()),
      objective_(std::move(objective)),
      row_lower_(std::move(row_lower)),
      row_upper_(std::move(row_upper)),
      col_lower_(std::move(col_lower)),
      col_upper_(std::move(col_upper)),
      obj_constant_(obj_constant) {
  require_size("objective", objective_.size(), static_cast<std::size_t>(num_cols()));
  require_size("row_lower", row_lower_.size(), static_cast<std::size_t>(num_rows()));
  require_size("row_upper", row_upper_.size(), static_cast<std::size_t>(num_rows()));
  require_size("col_lower", col_lower_.size(), static_cast<std::size_t>(num_cols()));
  require_size("col_upper", col_upper_.size(), static_cast<std::size_t>(num_cols()));
  require_bounds("row_lower", "row_upper", row_lower_, row_upper_);
  require_bounds("col_lower", "col_upper", col_lower_, col_upper_);
}

int LpModel::num_rows() const noexcept { return csc_.rows(); }
int LpModel::num_cols() const noexcept { return csc_.cols(); }
const CscMatrix& LpModel::csc() const { return csc_; }
const CsrMatrix& LpModel::csr() const { return csr_; }

}  // namespace mps_reader
