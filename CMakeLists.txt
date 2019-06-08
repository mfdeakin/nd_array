
cmake_minimum_required (VERSION 3.9)
project(ndarray)

set(google_benchmark_path /usr/local CACHE PATH "Path to where Google Benchmark is installed")

enable_testing()

#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

add_executable(tests tests.cpp zip_tests.cpp)
set_target_properties(tests PROPERTIES COMPILE_FLAGS "-g -std=c++17 -Wall")
add_test(all tests)

set(TEST_PERFORMANCE TRUE CACHE BOOL "Whether to build the performance testing executable")

if(${TEST_PERFORMANCE})
  link_directories("${google_benchmark_path}/lib")

  add_executable(performance performance.cpp timer/timer.cpp)
  set_target_properties(performance PROPERTIES COMPILE_FLAGS "-g -std=c++14 -march=native")
  target_include_directories(performance PUBLIC "${google_benchmark_path}/include")
  target_link_libraries(performance benchmark pthread)
endif()
