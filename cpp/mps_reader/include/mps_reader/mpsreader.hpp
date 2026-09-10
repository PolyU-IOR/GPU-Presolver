#pragma once

#include "mps_reader/lp_model.hpp"

#include <iosfwd>
#include <string>

namespace mps_reader {

struct MpsModel {
  std::string name;
  LpModel lp;
};

MpsModel read_mps(std::istream& input);

}  // namespace mps_reader
