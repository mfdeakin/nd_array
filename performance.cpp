
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

#include "zip.hpp"

static void BM_Null(benchmark::State &state) {
  while(state.KeepRunning()) {
  }
}

double f(int i) { return i + 1; }

static void BM_ND_Array_Create(benchmark::State &state) {
  while(state.KeepRunning()) {
    ND_Array<double, 5, 7, 11, 13, 17> array;
    // double array[5][7][11][13][17];
    benchmark::DoNotOptimize(array);
  }
}

static void BM_C_Array_Iterate(benchmark::State &state) {
  constexpr int e1 = 5;
  constexpr int e2 = 7;
  constexpr int e3 = 11;
  constexpr int e4 = 13;
  constexpr int e5 = 17;
  double array[e1][e2][e3][e4][e5];
  benchmark::DoNotOptimize(array);
  while(state.KeepRunning()) {
    for(int i1 = 0; i1 < e1; i1++) {
      for(int i2 = 0; i2 < e2; i2++) {
        for(int i3 = 0; i3 < e3; i3++) {
          for(int i4 = 0; i4 < e4; i4++) {
            for(int i5 = 0; i5 < e5; i5++) {
              benchmark::DoNotOptimize(
                  array[i1][i2][i3][i4][i5]);
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Iterate_Index(
    benchmark::State &state) {
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

static void BM_ND_Array_Iterate_Pointer(
    benchmark::State &state) {
  ND_Array<double, 5, 7, 11, 13, 17> array;
  benchmark::DoNotOptimize(array);
  while(state.KeepRunning()) {
    double *iter = &array(0, 0, 0, 0, 0);
    for(int i1 = 0; i1 < array.extent(0); i1++) {
      for(int i2 = 0; i2 < array.extent(1); i2++) {
        for(int i3 = 0; i3 < array.extent(2); i3++) {
          for(int i4 = 0; i4 < array.extent(3); i4++) {
            for(int i5 = 0; i5 < array.extent(4); i5++) {
              benchmark::DoNotOptimize(*iter);
              iter++;
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Iterate_Iterator(
    benchmark::State &state) {
  ND_Array<double, 5, 7, 11, 13, 17> array;
  benchmark::DoNotOptimize(array);
  while(state.KeepRunning()) {
    for(auto i = array.begin(); i != array.end(); i++) {
      benchmark::DoNotOptimize(*i);
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
              benchmark::DoNotOptimize(
                  array[i1][i2][i3][i4][i5] = counter);
              counter += 1.0;
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Initialize_Index(
    benchmark::State &state) {
  ND_Array<double, 5, 7, 11, 13, 17> array;
  double counter = 1.0;
  while(state.KeepRunning()) {
    for(int i1 = 0; i1 < array.extent(0); i1++) {
      for(int i2 = 0; i2 < array.extent(1); i2++) {
        for(int i3 = 0; i3 < array.extent(2); i3++) {
          for(int i4 = 0; i4 < array.extent(3); i4++) {
            for(int i5 = 0; i5 < array.extent(4); i5++) {
              benchmark::DoNotOptimize(
                  array(i1, i2, i3, i4, i5) = counter);
              counter += 1.0;
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Initialize_Pointer(
    benchmark::State &state) {
  ND_Array<double, 5, 7, 11, 13, 17> array;
  benchmark::DoNotOptimize(array);
  double counter = 1.0;
  while(state.KeepRunning()) {
    double *iter = &array(0, 0, 0, 0, 0);
    for(int i1 = 0; i1 < array.extent(0); i1++) {
      for(int i2 = 0; i2 < array.extent(1); i2++) {
        for(int i3 = 0; i3 < array.extent(2); i3++) {
          for(int i4 = 0; i4 < array.extent(3); i4++) {
            for(int i5 = 0; i5 < array.extent(4); i5++) {
              benchmark::DoNotOptimize(*iter = counter);
              iter++;
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Initialize_Iterator(
    benchmark::State &state) {
  ND_Array<double, 5, 7, 11, 13, 17> array;
  benchmark::DoNotOptimize(array);
  double counter = 1.0;
  while(state.KeepRunning()) {
    for(auto i = array.begin(); i != array.end(); i++) {
      benchmark::DoNotOptimize(*i = counter);
    }
  }
}

static void BM_ND_Array_Iterate_2_Index(
    benchmark::State &state) {
  using array_t = ND_Array<double, 5, 7, 11, 13, 17>;
  using size_type = array_t::size_type;
  array_t a1, a2;
  while(state.KeepRunning()) {
    for(size_type i1 = 0; i1 < a1.extent(0); i1++) {
      for(size_type i2 = 0; i2 < a1.extent(0); i2++) {
        for(size_type i3 = 0; i3 < a1.extent(0); i3++) {
          for(size_type i4 = 0; i4 < a1.extent(0); i4++) {
            for(size_type i5 = 0; i5 < a1.extent(0); i5++) {
              benchmark::DoNotOptimize(
                  a1(i1, i2, i3, i4, i5));
              benchmark::DoNotOptimize(
                  a2(i1, i2, i3, i4, i5));
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Iterate_2_Pointer(
    benchmark::State &state) {
  using array_t = ND_Array<double, 5, 7, 11, 13, 17>;
  using size_type = array_t::size_type;
  array_t a1, a2;
  while(state.KeepRunning()) {
    double *v1 = &a1(0, 0, 0, 0, 0);
    double *v2 = &a2(0, 0, 0, 0, 0);
    for(size_type i1 = 0; i1 < a1.extent(0); i1++) {
      for(size_type i2 = 0; i2 < a1.extent(0); i2++) {
        for(size_type i3 = 0; i3 < a1.extent(0); i3++) {
          for(size_type i4 = 0; i4 < a1.extent(0); i4++) {
            for(size_type i5 = 0; i5 < a1.extent(0); i5++) {
              benchmark::DoNotOptimize(*v1);
              v1++;
              benchmark::DoNotOptimize(*v2);
              v2++;
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Iterate_2_Iterator(
    benchmark::State &state) {
  using array_t = ND_Array<double, 5, 7, 11, 13, 17>;
  array_t a1, a2;
  while(state.KeepRunning()) {
    for(auto i1 = a1.begin(), i2 = a2.begin();
        i1 != a1.end(); i1++, i2++) {
      benchmark::DoNotOptimize(*i1);
      benchmark::DoNotOptimize(*i2);
    }
  }
}

static void BM_ND_Array_Iterate_2_Zip(
    benchmark::State &state) {
  using array_t = ND_Array<double, 5, 7, 11, 13, 17>;
  array_t a1, a2;
  using zip_t = Zip<array_t, array_t>;
  while(state.KeepRunning()) {
    for(auto [v1, v2] : zip_t(a1, a2)) {
      benchmark::DoNotOptimize(v1);
      benchmark::DoNotOptimize(v2);
    }
  }
}

static void BM_ND_Array_Initialize_2_Index(
    benchmark::State &state) {
  using array_t = ND_Array<double, 5, 7, 11, 13, 17>;
  using size_type = array_t::size_type;
  array_t a1, a2;
  double counter = 1.0;
  while(state.KeepRunning()) {
    for(size_type i1 = 0; i1 < a1.extent(0); i1++) {
      for(size_type i2 = 0; i2 < a1.extent(0); i2++) {
        for(size_type i3 = 0; i3 < a1.extent(0); i3++) {
          for(size_type i4 = 0; i4 < a1.extent(0); i4++) {
            for(size_type i5 = 0; i5 < a1.extent(0); i5++) {
              benchmark::DoNotOptimize(
                  a1(i1, i2, i3, i4, i5) = counter);
              counter += 1.0;
              benchmark::DoNotOptimize(
                  a2(i1, i2, i3, i4, i5) = counter);
              counter += 1.0;
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Initialize_2_Pointer(
    benchmark::State &state) {
  using array_t = ND_Array<double, 5, 7, 11, 13, 17>;
  using size_type = array_t::size_type;
  array_t a1, a2;
  double counter = 1.0;
  while(state.KeepRunning()) {
    double *v1 = &a1(0, 0, 0, 0, 0);
    double *v2 = &a2(0, 0, 0, 0, 0);
    for(size_type i1 = 0; i1 < a1.extent(0); i1++) {
      for(size_type i2 = 0; i2 < a1.extent(0); i2++) {
        for(size_type i3 = 0; i3 < a1.extent(0); i3++) {
          for(size_type i4 = 0; i4 < a1.extent(0); i4++) {
            for(size_type i5 = 0; i5 < a1.extent(0); i5++) {
              benchmark::DoNotOptimize(*v1 = counter);
              v1++;
              counter += 1.0;
              benchmark::DoNotOptimize(*v2 = counter);
              v2++;
              counter += 1.0;
            }
          }
        }
      }
    }
  }
}

static void BM_ND_Array_Initialize_2_Iterator(
    benchmark::State &state) {
  using array_t = ND_Array<double, 5, 7, 11, 13, 17>;
  array_t a1, a2;
  double counter = 1.0;
  while(state.KeepRunning()) {
    for(auto i1 = a1.begin(), i2 = a2.begin();
        i1 != a1.end(); i1++, i2++) {
      benchmark::DoNotOptimize(*i1 = counter);
      counter += 1.0;
      benchmark::DoNotOptimize(*i2 = counter);
      counter += 1.0;
    }
  }
}

static void BM_ND_Array_Initialize_2_Zip(
    benchmark::State &state) {
  using array_t = ND_Array<double, 5, 7, 11, 13, 17>;
  array_t a1, a2;
  double counter = 1.0;
  using zip_t = Zip<array_t, array_t>;
  while(state.KeepRunning()) {
    for(auto [v1, v2] : zip_t(a1, a2)) {
      benchmark::DoNotOptimize(v1 = counter);
      counter += 1.0;
      benchmark::DoNotOptimize(v2 = counter);
      counter += 1.0;
    }
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize();

  benchmark::RegisterBenchmark("BM_Null", BM_Null);
  benchmark::RegisterBenchmark("BM_ND_Array_Create",
                               BM_ND_Array_Create);

  benchmark::RegisterBenchmark("BM_C_Array_Iterate",
                               BM_C_Array_Iterate);
  benchmark::RegisterBenchmark("BM_ND_Array_Iterate_Index",
                               BM_ND_Array_Iterate_Index);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Iterate_Pointer",
      BM_ND_Array_Iterate_Pointer);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Iterate_Iterator",
      BM_ND_Array_Iterate_Iterator);

  benchmark::RegisterBenchmark("BM_C_Array_Initialize",
                               BM_C_Array_Initialize);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Initialize_Index",
      BM_ND_Array_Initialize_Index);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Initialize_Pointer",
      BM_ND_Array_Initialize_Pointer);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Initialize_Iterator",
      BM_ND_Array_Initialize_Iterator);

  benchmark::RegisterBenchmark(
      "BM_ND_Array_Iterate_2_Index",
      BM_ND_Array_Iterate_2_Index);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Iterate_2_Pointer",
      BM_ND_Array_Iterate_2_Pointer);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Iterate_2_Iterator",
      BM_ND_Array_Iterate_2_Iterator);
  benchmark::RegisterBenchmark("BM_ND_Array_Iterate_2_Zip",
                               BM_ND_Array_Iterate_2_Zip);

  benchmark::RegisterBenchmark(
      "BM_ND_Array_Initialize_2_Index",
      BM_ND_Array_Initialize_2_Index);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Initialize_2_Pointer",
      BM_ND_Array_Initialize_2_Pointer);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Initialize_2_Iterator",
      BM_ND_Array_Initialize_2_Iterator);
  benchmark::RegisterBenchmark(
      "BM_ND_Array_Initialize_2_Zip",
      BM_ND_Array_Initialize_2_Zip);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
