
cmake_minimum_required (VERSION 3.3)
project(ndarray)

add_executable(tests tests.cpp)
set_target_properties(tests PROPERTIES COMPILE_FLAGS "-g -std=c++14")

add_executable(performance performance.cpp timer/timer.cpp)
set_target_properties(performance PROPERTIES COMPILE_FLAGS "-g -std=c++14 -march=native")
