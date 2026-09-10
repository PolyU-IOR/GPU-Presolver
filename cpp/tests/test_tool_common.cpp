#include "gpu_presolver_tool_common.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

namespace tools = gpu_presolver::tools;
namespace fs = std::filesystem;

static_assert(!std::is_copy_constructible_v<tools::DeviceLpOwner>);
static_assert(!std::is_copy_assignable_v<tools::DeviceLpOwner>);
static_assert(std::is_nothrow_move_constructible_v<tools::DeviceLpOwner>);
static_assert(std::is_nothrow_move_assignable_v<tools::DeviceLpOwner>);

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class F>
void require_error(F fn) {
  bool caught = false;
  try { fn(); } catch (const std::exception&) { caught = true; }
  require(caught, "expected file operation to fail");
}

void write_text(const fs::path& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
  tools::close_output_file(out, path);
}

void write_solution(const fs::path& dir, const std::vector<double>& x,
                    const std::vector<double>& y = {0.0}) {
  fs::create_directories(dir);
  tools::write_binary_vector(dir / "x_f64.bin", x);
  tools::write_binary_vector(dir / "y_f64.bin", y);
  tools::write_binary_vector(dir / "z_f64.bin", std::vector<double>(x.size(), 0.0));
}

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "verify-export") {
      const auto original = tools::read_mps_file(argv[3]);
      const double expected = original.lp.col_lower()[0];
      std::ifstream meta(fs::path(argv[2]) / "meta.txt");
      std::string line;
      int verified = 0;
      while (std::getline(meta, line)) {
        std::istringstream in(line);
        std::string key;
        in >> key;
        if (key == "obj_constant" || key == "objective_shift") {
          double value;
          require(static_cast<bool>(in >> value), "missing export value");
          require(value == expected, "export lost double precision");
          ++verified;
        }
      }
      require(verified == 2, "missing export constants");
      return 0;
    }
    require(argc == 2, "expected a test output directory");
    const fs::path dir = argv[1];
    fs::create_directories(dir);
    const fs::path binary = dir / "vector.bin";
    const std::vector<double> values = {1.0, -2.0};
    tools::write_binary_vector(binary, values);
    require(tools::read_binary_vector<double>(binary, 2) == values, "binary round trip");
    for (std::size_t count : {0, 1, 3}) {
      require_error([&] { tools::read_binary_vector<double>(binary, count); });
    }
    tools::write_binary_vector<double>(binary, {});
    require(tools::read_binary_vector<double>(binary, 0).empty(), "empty vector round trip");
    write_text(binary, "x");
    require_error([&] { tools::read_binary_vector<double>(binary, 0); });
    require_error([&] { tools::read_binary_vector<double>(binary, 1); });
    require_error([&] { tools::read_binary_vector<double>(dir / "missing", 0); });
    if (fs::exists("/dev/full")) {
      require_error([&] { tools::write_binary_vector<double>("/dev/full", {1.0}); });
      require_error([&] { write_text("/dev/full", "metadata"); });
    }

    const std::string toy =
        "NAME toy\nROWS\n N OBJ\n G DEMAND\nCOLUMNS\n"
        "    X OBJ 1 DEMAND 1\n    Y OBJ 2 DEMAND 1\nRHS\n"
        "    RHS1 DEMAND 3\nBOUNDS\n UP BND1 X 2\n UP BND1 Y 4\nENDATA\n";
    write_text(dir / "toy.mps", toy);
    const auto model = tools::read_mps_file((dir / "toy.mps").string());
    {
      auto owner = tools::upload_lp(model.lp);
      tools::DeviceLpOwner moved(std::move(owner));
      require(owner.lp.c == nullptr && owner.lp.A.rowPtr == nullptr, "move construction retained ownership");
      auto assigned = tools::upload_lp(model.lp);
      assigned = std::move(moved);
      require(moved.lp.c == nullptr && moved.lp.AT.nzVal == nullptr, "move assignment retained ownership");
      require(tools::copy_to_host(assigned.lp.c, 2) == model.lp.objective(), "invalid moved device storage");
    }
    tools::check_cuda(cudaDeviceSynchronize(), "owner lifetime test");
    write_text(dir / "precision.mps",
        "NAME precision\nROWS\n N OBJ\n G R\nCOLUMNS\n    X OBJ 1 R 1\n"
        "RHS\n    RHS1 R 0\nBOUNDS\n FX BND1 X 1.2345678901234567\nENDATA\n");
    write_text(dir / "activity-overflow.mps",
        "NAME overflow\nROWS\n N OBJ\n L R\nCOLUMNS\n    X OBJ 1 R 1e308\n"
        "RHS\n    RHS1 R 1\nBOUNDS\n FR BND1 X\nENDATA\n");
    write_text(dir / "large-bounds.mps",
        "NAME large\nROWS\n N OBJ\n L R\nCOLUMNS\n    X OBJ 1 R 1\n"
        "RHS\n    RHS1 R 1e200\nBOUNDS\n FR BND1 X\nENDATA\n");
    write_solution(dir / "valid", {2.0, 1.0});
    write_solution(dir / "infeasible", {0.0, 0.0});
    write_solution(dir / "nan", {std::numeric_limits<double>::quiet_NaN(), 1.0});
    write_solution(dir / "inf", {std::numeric_limits<double>::infinity(), 1.0});
    write_solution(dir / "long", {2.0, 1.0, 0.0});
    write_solution(dir / "short", {2.0});
    write_solution(dir / "empty", {}, {});
    write_solution(dir / "overflow", {2.0});
    write_solution(dir / "large-infeasible", {2.0e200});
    write_solution(dir / "large-valid", {0.0});
    std::cout << "tool ownership and binary I/O checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
