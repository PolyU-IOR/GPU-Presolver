#include "test_common.hpp"

#include "mps_reader/mpsreader.hpp"

#include <limits>
#include <sstream>

using mps_reader::read_mps;

MPS_READER_TEST(mpsreader_parses_rows_columns_rhs_and_bounds) {
  const double inf = std::numeric_limits<double>::infinity();
  std::istringstream input(
      "NAME          toy\n"
      "ROWS\n"
      " N  COST\n"
      " E  EQ1\n"
      " G  GE1\n"
      " L  LE1\n"
      "COLUMNS\n"
      "    X1        COST       3.0   EQ1        2.0\n"
      "    X1        GE1        1.0\n"
      "    X2        COST       5.0   LE1       -4.0\n"
      "    X2        EQ1        1.0\n"
      "RHS\n"
      "    RHS1      EQ1       10.0   GE1        7.0\n"
      "    RHS1      LE1       20.0\n"
      "BOUNDS\n"
      " LO BND1      X1        -1.0\n"
      " UP BND1      X1         6.0\n"
      " FR BND1      X2\n"
      "ENDATA\n");

  const auto model = read_mps(input);
  const auto& lp = model.lp;

  MPS_READER_REQUIRE(model.name == "toy");
  MPS_READER_REQUIRE(lp.num_rows() == 3);
  MPS_READER_REQUIRE(lp.num_cols() == 2);
  MPS_READER_REQUIRE(lp.objective() == std::vector<double>({3.0, 5.0}));
  MPS_READER_REQUIRE(lp.row_lower() == std::vector<double>({10.0, 7.0, -inf}));
  MPS_READER_REQUIRE(lp.row_upper() == std::vector<double>({10.0, inf, 20.0}));
  MPS_READER_REQUIRE(lp.col_lower() == std::vector<double>({-1.0, -inf}));
  MPS_READER_REQUIRE(lp.col_upper() == std::vector<double>({6.0, inf}));
  MPS_READER_REQUIRE(lp.csc().col_ptr() == std::vector<int>({0, 2, 4}));
  MPS_READER_REQUIRE(lp.csc().row_idx() == std::vector<int>({0, 1, 0, 2}));
  MPS_READER_REQUIRE(lp.csc().values() == std::vector<double>({2.0, 1.0, 1.0, -4.0}));
}

MPS_READER_TEST(mpsreader_supports_ranges_and_duplicate_entries) {
  std::istringstream input(
      "NAME ranged\n"
      "ROWS\n"
      " N OBJ\n"
      " E EQ1\n"
      " L LE1\n"
      " G GE1\n"
      "COLUMNS\n"
      "    X1        OBJ        1.0   EQ1        1.0\n"
      "    X1        LE1        2.0   LE1        3.0\n"
      "    X2        OBJ        2.0   GE1       -1.0\n"
      "RHS\n"
      "    RHS1      EQ1       10.0   LE1       20.0\n"
      "    RHS1      GE1        5.0\n"
      "RANGES\n"
      "    RNG1      EQ1        4.0   LE1        6.0\n"
      "    RNG1      GE1        7.0\n"
      "BOUNDS\n"
      " UP BND1      X1         1.0\n"
      " UP BND1      X2         9.0\n"
      "ENDATA\n");

  const auto model = read_mps(input);
  const auto& lp = model.lp;

  MPS_READER_REQUIRE(model.name == "ranged");
  MPS_READER_REQUIRE(lp.num_rows() == 3);
  MPS_READER_REQUIRE(lp.num_cols() == 2);
  MPS_READER_REQUIRE(lp.objective() == std::vector<double>({1.0, 2.0}));
  MPS_READER_REQUIRE(lp.row_lower() == std::vector<double>({10.0, 14.0, 5.0}));
  MPS_READER_REQUIRE(lp.row_upper() == std::vector<double>({14.0, 20.0, 12.0}));
  MPS_READER_REQUIRE(lp.col_lower() == std::vector<double>({0.0, 0.0}));
  MPS_READER_REQUIRE(lp.col_upper() == std::vector<double>({1.0, 9.0}));
  MPS_READER_REQUIRE(lp.csc().col_ptr() == std::vector<int>({0, 2, 3}));
  MPS_READER_REQUIRE(lp.csc().row_idx() == std::vector<int>({0, 1, 2}));
  MPS_READER_REQUIRE(lp.csc().values() == std::vector<double>({1.0, 5.0, -1.0}));
}

MPS_READER_TEST(mpsreader_matches_julia_decimal_rounding_for_micro_bounds) {
  std::istringstream input(
      "NAME micro\n"
      "ROWS\n"
      " N OBJ\n"
      " L ROW1\n"
      "COLUMNS\n"
      "    X1        OBJ        1.0   ROW1       1.0\n"
      "RHS\n"
      "    RHS1      ROW1       1.0\n"
      "BOUNDS\n"
      " UP BND1      X1         0.000001\n"
      "ENDATA\n");

  const auto model = read_mps(input);
  const auto& upper = model.lp.col_upper();

  MPS_READER_REQUIRE(upper.size() == 1);
  MPS_READER_REQUIRE(upper[0] > 1.0e-6);
}

MPS_READER_TEST(mpsreader_auto_falls_back_to_fixed_format) {
  std::istringstream input(
      "NAME          fixed\n"
      "ROWS\n"
      " N  COST\n"
      " L  ROW00001\n"
      "COLUMNS\n"
      "    COL00001  COST       1.0   ROW00001   2.0\n"
      "RHS\n"
      "    RHS00001  ROW00001   3.0\n"
      "ENDATA\n");

  const auto model = read_mps(input);
  const auto& lp = model.lp;

  MPS_READER_REQUIRE(model.name == "fixed");
  MPS_READER_REQUIRE(lp.num_rows() == 1);
  MPS_READER_REQUIRE(lp.num_cols() == 1);
  MPS_READER_REQUIRE(lp.objective() == std::vector<double>({1.0}));
  MPS_READER_REQUIRE(lp.row_upper() == std::vector<double>({3.0}));
  MPS_READER_REQUIRE(lp.csc().values() == std::vector<double>({2.0}));
}

namespace {
std::string numeric_model(const std::string& columns = "    X OBJ 1 R 1\n",
                          const std::string& rhs = "    RHS1 R 1\n",
                          const std::string& bounds = " UP BND1 X 2\n") {
  return "NAME validation\nROWS\n N OBJ\n L R\nCOLUMNS\n" + columns +
         "RHS\n" + rhs + "BOUNDS\n" + bounds + "ENDATA\n";
}

void require_rejected(const std::string& text) {
  bool rejected = false;
  try {
    std::istringstream input(text);
    read_mps(input);
  } catch (const std::exception&) {
    rejected = true;
  }
  MPS_READER_REQUIRE(rejected);
}
}

MPS_READER_TEST(mpsreader_validates_objective_sense) {
  for (const std::string sense : {"OBJSENSE MIN\n", "OBJSENSE\n MIN\n"}) {
    auto text = numeric_model();
    text.insert(text.find("ROWS"), sense);
    std::istringstream input(text);
    MPS_READER_REQUIRE(read_mps(input).lp.objective()[0] == 1.0);
  }
  for (const std::string sense : {"OBJSENSE MAX\n", "OBJSENSE\n MAX\n",
                                 "OBJSENSE INVALID\n", "OBJSENSE\n INVALID\n",
                                 "OBJSENSE\n", "OBJSENSE MIN MAX\n"}) {
    auto text = numeric_model();
    text.insert(text.find("ROWS"), sense);
    require_rejected(text);
  }
}

MPS_READER_TEST(mpsreader_rejects_nonfinite_numeric_data) {
  for (const std::string value : {"nan", "NaN", "inf", "-inf", "1e999", "1e99999999999999"}) {
    require_rejected(numeric_model("    X OBJ " + value + " R 1\n"));
    require_rejected(numeric_model("    X OBJ 1 R " + value + "\n"));
    require_rejected(numeric_model("    X OBJ 1 R 1\n", "    RHS1 OBJ " + value + " R 1\n"));
    require_rejected(numeric_model("    X OBJ 1 R 1\n", "    RHS1 R " + value + "\n"));
    auto ranged = numeric_model();
    ranged.insert(ranged.find("BOUNDS"), "RANGES\n    RNG1 R " + value + "\n");
    require_rejected(ranged);
  }
  for (const std::string bound : {"LO BND1 X nan", "UP BND1 X nan", "FX BND1 X nan",
                                 "LO BND1 X inf", "UP BND1 X -inf", "FX BND1 X inf",
                                 "UP BND1 X 1e999"}) {
    require_rejected(numeric_model("    X OBJ 1 R 1\n", "    RHS1 R 1\n", " " + bound + "\n"));
  }
  require_rejected(numeric_model("    X OBJ 1e308 OBJ 1e308\n    X R 1\n"));
  require_rejected(numeric_model("    X OBJ 1 R 1e308\n    X R 1e308\n"));
  auto ranged = numeric_model("    X OBJ 1 R 1\n", "    RHS1 R -1e308\n");
  ranged.insert(ranged.find("BOUNDS"), "RANGES\n    RNG1 R 1e308\n");
  require_rejected(ranged);
}

MPS_READER_TEST(mpsreader_preserves_unbounded_variables) {
  const double inf = std::numeric_limits<double>::infinity();
  for (const std::string bounds : {" FR BND1 X\n", " FR BND1 X 0\n", " MI BND1 X\n PL BND1 X\n",
                                   " LO BND1 X -inf\n UP BND1 X +Infinity\n"}) {
    std::istringstream input(numeric_model("    X OBJ 1 R 1\n", "    RHS1 R 1\n", bounds));
    const auto model = read_mps(input);
    MPS_READER_REQUIRE(model.lp.col_lower()[0] == -inf);
    MPS_READER_REQUIRE(model.lp.col_upper()[0] == inf);
  }
}

MPS_READER_TEST(mpsreader_rejects_incomplete_records) {
  require_rejected(numeric_model("    X OBJ 1 R\n"));
  require_rejected(numeric_model("    X OBJ 1 R 1\n", "    RHS1 R 1 OBJ\n"));
  auto text = numeric_model();
  text.insert(text.find("BOUNDS"), "RANGES\n    RNG1 R 1 R\n");
  require_rejected(text);
  for (const std::string bounds : {" UP BND1 X\n", " UP BND1 X 2 EXTRA\n", " FR BND1 X 0 EXTRA\n"}) {
    require_rejected(numeric_model("    X OBJ 1 R 1\n", "    RHS1 R 1\n", bounds));
  }
}

MPS_READER_TEST(mpsreader_rejects_integer_models) {
  for (const std::string bound : {"BV BND1 X", "LI BND1 X 0", "UI BND1 X 1", "SI BND1 X 1"}) {
    require_rejected(numeric_model("    X OBJ 1 R 1\n", "    RHS1 R 0.5\n", " " + bound + "\n"));
  }
  for (const std::string marker : {"    MARK0000 'MARKER' 'INTORG'\n", "    MARK0000 MARKER INTORG\n"}) {
    require_rejected(numeric_model(marker + "    X OBJ 1 R 1\n"));
  }
}

MPS_READER_TEST(mpsreader_preserves_extreme_finite_numbers) {
  const std::pair<const char*, double> cases[] = {
      {"1e-309", 1e-309}, {"-1D-309", -1e-309}, {"1e-320", 1e-320},
      {"4.9406564584124654e-324", std::numeric_limits<double>::denorm_min()},
      {"0.000000001e309", 1e300}, {"0.1e-308", 1e-309}, {"0e999", 0.0}};
  for (const auto& [token, expected] : cases) {
    std::istringstream input(numeric_model(std::string("    X OBJ ") + token + " R 1\n"));
    const double actual = read_mps(input).lp.objective()[0];
    MPS_READER_REQUIRE(std::isfinite(actual));
    MPS_READER_REQUIRE(actual == expected);
  }
  for (const std::string token : {"1e309", "1e-999", "1e99999999999", "1e-99999999999"}) {
    require_rejected(numeric_model("    X OBJ " + token + " R 1\n"));
  }
}
