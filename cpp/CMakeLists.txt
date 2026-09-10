cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to build")
project(GPUPresolverCpp LANGUAGES CXX CUDA)

foreach(gpu_presolver_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(gpu_presolver_arch MATCHES "^([0-9]+)(-real|-virtual)?$")
    if(CMAKE_MATCH_1 LESS 60)
      message(FATAL_ERROR
        "GPU-Presolver requires CUDA architecture 60 or newer. Set "
        "-DCMAKE_CUDA_ARCHITECTURES for your GPU in the host project (e.g. 80).")
    endif()
  endif()
endforeach()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_library(gpu_presolver_core
  src/presolve/fixed_col_tape.cu
  src/presolve/gpu_postsolve.cu
  src/presolve/gpu_presolve.cu
  src/presolve/gpu_presolve_kernels.cu
  src/presolve/presolve_structs.cu
  src/presolve/rules/rule_antipodal_components.cu
  src/presolve/rules/rule_bounded_two_row_projection.cu
  src/presolve/rules/rule_column_singletons.cu
  src/presolve/rules/rule_covering_cost_dominance.cu
  src/presolve/rules/rule_doubleton_equations.cu
  src/presolve/rules/rule_dual_fix.cu
  src/presolve/rules/rule_duplicate_columns.cu
  src/presolve/rules/rule_duplicate_rows.cu
  src/presolve/rules/rule_empty_cols.cu
  src/presolve/rules/rule_empty_rows.cu
  src/presolve/rules/rule_implied_variable_bounds.cu
  src/presolve/rules/rule_infeasible_fixed_variables.cu
  src/presolve/rules/rule_infeasible_redundant_rows.cu
  src/presolve/rules/rule_linf_components.cu
  src/presolve/rules/rule_orphan_mccormick_projection.cu
  src/presolve/rules/rule_redundant_bounds.cu
  src/presolve/rules/rule_singleton_rows.cu
  src/presolve/rules/rule_structural_l1_substitution.cu
)
target_include_directories(gpu_presolver_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpu_presolver_core PUBLIC CUDA::cusparse Threads::Threads)
set_target_properties(gpu_presolver_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_library(gpu_presolver_mps_reader STATIC
  mps_reader/src/core/lp_model.cpp
  mps_reader/src/core/sparse_matrix.cpp
  mps_reader/src/io/mpsreader.cpp
)
target_include_directories(gpu_presolver_mps_reader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/mps_reader/include
)

foreach(tool IN ITEMS
    gpu_presolver_mps
    gpu_presolver_export_reduced
    gpu_presolver_check_postsolve_solution)
  add_executable(${tool} tools/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE gpu_presolver_core gpu_presolver_mps_reader)
endforeach()

include(CTest)
if(BUILD_TESTING)
  foreach(test IN ITEMS
      test_gpu_direct_port
      test_gpu_postsolve
      test_gpu_presolve_kernels
      test_rule_antipodal_components
      test_rule_column_singletons
      test_rule_covering_cost_dominance
      test_rule_doubleton_equations
      test_rule_dual_fix
      test_rule_duplicate_columns
      test_rule_duplicate_rows
      test_rule_empty_cols
      test_rule_empty_rows
      test_rule_implied_variable_bounds
      test_rule_infeasible_fixed_variables
      test_rule_infeasible_redundant_rows
      test_rule_linf_components
      test_rule_projected_auxiliaries
      test_rule_redundant_bounds
      test_rule_singleton_rows
      test_rule_structural_l1_substitution
  )
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} PRIVATE gpu_presolver_core)
    # Keep the existing assert-based checks active in optimized builds too.
    if(MSVC)
      target_compile_options(${test} PRIVATE /UNDEBUG)
    else()
      target_compile_options(${test} PRIVATE -UNDEBUG)
    endif()
    add_test(NAME ${test} COMMAND ${test})
  endforeach()

  add_executable(test_mpsreader
    mps_reader/tests/test_main.cpp
    mps_reader/tests/test_mpsreader.cpp
    mps_reader/tests/test_lp_model.cpp
    mps_reader/tests/test_sparse_matrix.cpp
  )
  target_link_libraries(test_mpsreader PRIVATE gpu_presolver_mps_reader)
  add_test(NAME test_mpsreader COMMAND test_mpsreader)

  add_executable(test_tool_common tests/test_tool_common.cpp)
  target_include_directories(test_tool_common PRIVATE include tools)
  target_link_libraries(test_tool_common PRIVATE gpu_presolver_mps_reader CUDA::cudart)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Exercise upload_lp's ownership transfer without optional NRVO.
    target_compile_options(test_tool_common PRIVATE -fno-elide-constructors)
  endif()
  add_test(NAME test_tools COMMAND ${CMAKE_COMMAND}
    "-DHELPER=$<TARGET_FILE:test_tool_common>"
    "-DCHECKER=$<TARGET_FILE:gpu_presolver_check_postsolve_solution>"
    "-DEXPORTER=$<TARGET_FILE:gpu_presolver_export_reduced>"
    "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/tool-regression"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tools.cmake"
  )
endif()
