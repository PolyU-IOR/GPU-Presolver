#pragma once

#include "mps_reader/sparse_matrix.hpp"

#include <vector>

namespace mps_reader {

class LpModel {
public:
  LpModel(CscMatrix matrix,
          std::vector<double> objective,
          std::vector<double> row_lower,
          std::vector<double> row_upper,
          std::vector<double> col_lower,
          std::vector<double> col_upper,
          double obj_constant);
  int num_rows() const noexcept;
  int num_cols() const noexcept;

  const CscMatrix& csc() const;
  const CsrMatrix& csr() const;

  const std::vector<double>& objective() const noexcept { return objective_; }
  const std::vector<double>& row_lower() const noexcept { return row_lower_; }
  const std::vector<double>& row_upper() const noexcept { return row_upper_; }
  const std::vector<double>& col_lower() const noexcept { return col_lower_; }
  const std::vector<double>& col_upper() const noexcept { return col_upper_; }
  double obj_constant() const noexcept { return obj_constant_; }

private:
  CscMatrix csc_;
  CsrMatrix csr_;
  std::vector<double> objective_;
  std::vector<double> row_lower_;
  std::vector<double> row_upper_;
  std::vector<double> col_lower_;
  std::vector<double> col_upper_;
  double obj_constant_;
};

}  // namespace mps_reader
