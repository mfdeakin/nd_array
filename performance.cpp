
#include <memory>
#include <typeinfo>

#include <benchmark/benchmark.h>

#ifdef COMPARE_KOKKOS
#include <Kokkos_Core.hpp>
#else
namespace Kokkos {
void initialize() {}
}  // namespace Kokkos
#endif

#include "nd_array.hpp"

static void BM_Null(benchmark::State &state) {}

static void BM_ND_Array_Create(benchmark::State &state) {
  ND_Array<double, 5, 7, 11, 13, 17> array;
  while(state.KeepRunning()) {
    benchmark::DoNotOptimize(array);
  }
}

static void BM_ND_Array_Iterate(benchmark::State &state) {
  ND_Array<double, 5, 7, 11, 13, 17> array;
  benchmark::DoNotOptimize(array);
  while(state.KeepRunning()) {
    for(int i1 = 0; i1 < array.extent(0); i1++) {
      for(int i2 = 0; i2 < array.extent(1); i2++) {
        for(int i3 = 0; i3 < array.extent(2); i3++) {
          for(int i4 = 0; i4 < array.extent(3); i4++) {
            for(int i5 = 0; i5 < array.extent(4); i5++) {
              benchmark::DoNotOptimize(
                  array(i1, i2, i3, i4, i5));
            }
          }
        }
      }
    }
  }
}

static void BM_C_Array_Initialize(benchmark::State &state) {
  // ND_Array<double, 5, 7, 11, 13, 17> array;
  constexpr int e1 = 5;
  constexpr int e2 = 7;
  constexpr int e3 = 11;
  constexpr int e4 = 13;
  constexpr int e5 = 17;
  double array[e1][e2][e3][e4][e5];
	double counter = 1.0;
  while(state.KeepRunning()) {
    for(int i1 = 0; i1 < e1; i1++) {
      for(int i2 = 0; i2 < e2; i2++) {
        for(int i3 = 0; i3 < e3; i3++) {
          for(int i4 = 0; i4 < e4; i4++) {
            for(int i5 = 0; i5 < e5; i5++) {
							benchmark::DoNotOptimize(array[i1][i2][i3][i4][i5] = counter);
							counter += 1.0;
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Initialize(
    benchmark::State &state) {
  ND_Array<double, 5, 7, 11, 13, 17> array;
	double counter = 1.0;
  while(state.KeepRunning()) {
    for(int i1 = 0; i1 < array.extent(0); i1++) {
      for(int i2 = 0; i2 < array.extent(1); i2++) {
        for(int i3 = 0; i3 < array.extent(2); i3++) {
          for(int i4 = 0; i4 < array.extent(3); i4++) {
            for(int i5 = 0; i5 < array.extent(4); i5++) {
							benchmark::DoNotOptimize(array(i1, i2, i3, i4, i5) = counter);
							counter += 1.0;
            }
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize();

  benchmark::RegisterBenchmark("BM_Null", BM_Null);
  benchmark::RegisterBenchmark("BM_ND_Array_Create",
                               BM_ND_Array_Create);
  benchmark::RegisterBenchmark("BM_ND_Array_Iterate",
                               BM_ND_Array_Iterate);
  benchmark::RegisterBenchmark("BM_C_Array_Initialize",
                               BM_C_Array_Initialize);
  benchmark::RegisterBenchmark("BM_ND_Array_Initialize",
                               BM_ND_Array_Initialize);
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
