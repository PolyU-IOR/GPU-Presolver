#include "test_common.hpp"

#include "mps_reader/lp_model.hpp"

using mps_reader::CscMatrix;
using mps_reader::LpModel;

MPS_READER_TEST(lp_model_rejects_dimension_mismatch) {
  CscMatrix csc(2, 2, {0, 1, 2}, {0, 1}, {1.0, 2.0});

  bool threw = false;
  try {
    LpModel model(csc, {1.0}, {0.0, 0.0}, {1.0, 1.0}, {0.0, 0.0}, {10.0, 10.0}, 0.0);
    (void)model;
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  MPS_READER_REQUIRE(threw);
}
MPS_READER_TEST(lp_model_builds_both_sparse_views) {
  CscMatrix csc(
      2,
      3,
      {0, 1, 2, 3},
      {0, 1, 0},
      {1.0, 2.0, 3.0});

  LpModel model(
      csc,
      {4.0, 5.0, 6.0},
      {0.0, -1.0},
      {10.0, 12.0},
      {0.0, 0.0, 0.0},
      {100.0, 100.0, 100.0},
      7.0);

  MPS_READER_REQUIRE(model.num_rows() == 2);
  MPS_READER_REQUIRE(model.num_cols() == 3);
  MPS_READER_REQUIRE(model.csr().row_ptr() == std::vector<int>({0, 2, 3}));
  MPS_READER_REQUIRE_NEAR(model.obj_constant(), 7.0, 1.0e-12);
}
