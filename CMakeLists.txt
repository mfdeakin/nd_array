
cmake_minimum_required (VERSION 3.9)
project(ndarray)

set(google_benchmark_path /usr/local CACHE PATH "Path to where Google Benchmark is installed")

enable_testing()

#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

add_executable(unit_tests tests/tests.cpp tests/zip_tests.cpp)
set_target_properties(unit_tests PROPERTIES COMPILE_FLAGS "-g -std=c++17 -Wall")
target_include_directories(unit_tests PUBLIC "${PROJECT_SOURCE_DIR}/include")
add_test(all unit_tests)

set(TEST_PERFORMANCE TRUE CACHE BOOL "Whether to build the performance testing executable")

if(${TEST_PERFORMANCE})
  link_directories("${google_benchmark_path}/lib")

  add_executable(performance tests/performance.cpp)
  set(CMAKE_CXX_FLAGS_RELEASE "-march=native -fstrict-aliasing -ffast-math -O3")
  # CMake still doesn't support C++17 properly...
  target_compile_options(performance PUBLIC -std=c++17)
  target_include_directories(performance PUBLIC "${google_benchmark_path}/include" "${PROJECT_SOURCE_DIR}/include")
  target_link_libraries(performance benchmark pthread)

  find_package(xtl CONFIG)
  find_package(xtensor CONFIG)
  if(NOT ${xtensor_INCLUDE_DIRS})
    target_include_directories(performance PUBLIC "${xtensor_INCLUDE_DIRS}" "${xtl_INCLUDE_DIRS}")
    set_target_properties(performance PROPERTIES COMPILE_FLAGS "-DCOMPARE_XTENSOR=1")
  endif()
endif()
