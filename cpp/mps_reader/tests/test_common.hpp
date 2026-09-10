#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace mps_reader_tests {

using TestFn = void (*)();

void register_test(const char* name, TestFn fn);
int run_all_tests();

class TestFailure : public std::runtime_error {
public:
  explicit TestFailure(const std::string& message) : std::runtime_error(message) {}
};

inline void require(bool condition, const char* expression, const char* file, int line) {
  if (!condition) {
    throw TestFailure(std::string(file) + ":" + std::to_string(line) +
                      ": requirement failed: " + expression);
  }
}

inline void require_near(double actual,
                         double expected,
                         double tolerance,
                         const char* expression,
                         const char* file,
                         int line) {
  if (std::fabs(actual - expected) > tolerance) {
    throw TestFailure(std::string(file) + ":" + std::to_string(line) +
                      ": near requirement failed: " + expression +
                      ", actual=" + std::to_string(actual) +
                      ", expected=" + std::to_string(expected));
  }
}

}  // namespace mps_reader_tests

#define MPS_READER_REQUIRE(expr) \
  ::mps_reader_tests::require((expr), #expr, __FILE__, __LINE__)

#define MPS_READER_REQUIRE_NEAR(actual, expected, tolerance) \
  ::mps_reader_tests::require_near((actual), (expected), (tolerance), #actual, __FILE__, __LINE__)

#define MPS_READER_TEST(name)                                                \
  void name();                                                                 \
  namespace {                                                                  \
  struct name##_registrar {                                                    \
    name##_registrar() { ::mps_reader_tests::register_test(#name, &name); }  \
  };                                                                           \
  static name##_registrar name##_registrar_instance;                           \
  }                                                                            \
  void name()
