
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

#include "nd_array/nd_array.hpp"

#include "nd_array/zip.hpp"

#ifdef COMPARE_XTENSOR
#include "xtensor/xtensor.hpp"
#endif  // COMPARE_XTENSOR

static void BM_Null(benchmark::State &state) {
  while(state.KeepRunning()) {
  }
}

double f(int i) { return i + 1; }

static void BM_ND_Array_Create(benchmark::State &state) {
  while(state.KeepRunning()) {
    ND_Array<double, 5, 7, 11, 13, 17> array;
    array.fill(1.0);
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
              const double *val =
                  &array[i1][i2][i3][i4][i5];
              benchmark::DoNotOptimize(*val);
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
    for(int i1 = 0; i1 < 5; i1++) {
      for(int i2 = 0; i2 < 7; i2++) {
        for(int i3 = 0; i3 < 11; i3++) {
          for(int i4 = 0; i4 < 13; i4++) {
            for(int i5 = 0; i5 < 17; i5++) {
              const double *val =
                  &array(i1, i2, i3, i4, i5);
              benchmark::DoNotOptimize(*val);
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
  array.fill(1.0);
  benchmark::DoNotOptimize(array);
  while(state.KeepRunning()) {
    const double *end =
        &array(array.extent(0) - 1, array.extent(1) - 1,
               array.extent(2) - 1, array.extent(3) - 1,
               array.extent(4) - 1) +
        1;
    for(double *iter = &array(0, 0, 0, 0, 0); iter != end;
        iter++) {
      benchmark::DoNotOptimize(*iter);
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
    const double *end =
        &array(array.extent(0) - 1, array.extent(1) - 1,
               array.extent(2) - 1, array.extent(3) - 1,
               array.extent(4) - 1) +
        1;
    for(double *iter = &array(0, 0, 0, 0, 0); iter != end;
        iter++) {
      benchmark::DoNotOptimize(*iter = counter);
      counter += 1.0;
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
      counter += 1.0;
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
      for(size_type i2 = 0; i2 < a1.extent(1); i2++) {
        for(size_type i3 = 0; i3 < a1.extent(2); i3++) {
          for(size_type i4 = 0; i4 < a1.extent(3); i4++) {
            for(size_type i5 = 0; i5 < a1.extent(4); i5++) {
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
      for(size_type i2 = 0; i2 < a1.extent(1); i2++) {
        for(size_type i3 = 0; i3 < a1.extent(2); i3++) {
          for(size_type i4 = 0; i4 < a1.extent(3); i4++) {
            for(size_type i5 = 0; i5 < a1.extent(4); i5++) {
              benchmark::DoNotOptimize(*v1);
              benchmark::DoNotOptimize(*v2);
              v1++;
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
  while(state.KeepRunning()) {
    for(auto [v1, v2] : zip::make_zip(a1, a2)) {
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
      for(size_type i2 = 0; i2 < a1.extent(1); i2++) {
        for(size_type i3 = 0; i3 < a1.extent(2); i3++) {
          for(size_type i4 = 0; i4 < a1.extent(3); i4++) {
            for(size_type i5 = 0; i5 < a1.extent(4); i5++) {
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
      for(size_type i2 = 0; i2 < a1.extent(1); i2++) {
        for(size_type i3 = 0; i3 < a1.extent(2); i3++) {
          for(size_type i4 = 0; i4 < a1.extent(3); i4++) {
            for(size_type i5 = 0; i5 < a1.extent(4); i5++) {
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
  while(state.KeepRunning()) {
    for(auto [v1, v2] : zip::make_zip(a1, a2)) {
      benchmark::DoNotOptimize(v1 = counter);
      counter += 1.0;
      benchmark::DoNotOptimize(v2 = counter);
      counter += 1.0;
    }
  }
}

#ifdef COMPARE_XTENSOR

static xt::xtensor<double, 2> &mmul_xtensor(
    const xt::xtensor<double, 2> &lhs,
    const xt::xtensor<double, 2> &rhs,
    xt::xtensor<double, 2> &result) {
  assert(lhs.shape()[0] == result.shape()[0]);
  assert(lhs.shape()[1] == rhs.shape()[0]);
  assert(rhs.shape()[1] == result.shape()[1]);
  for(int i = 0; i < lhs.shape()[0]; ++i) {
    for(int j = 0; j < rhs.shape()[1]; ++j) {
      result(i, j) = 0.0;
      for(int k = 0; k < lhs.shape()[1]; ++k) {
        result(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  return result;
}

static void BM_XTensor_MMul(benchmark::State &state) {
  xt::xtensor<double, 2> a1({80, 100});
  xt::xtensor<double, 2> a2({100, 120});
  xt::xtensor<double, 2> a3({80, 120});
  double counter = 1.0;
  for(double &v : a1) {
    v = counter;
    counter += 1.0;
  }
  for(double &v : a2) {
    v = counter;
    counter += 1.0;
  }
  for(double &v : a3) {
    v = std::numeric_limits<double>::quiet_NaN();
  }

  while(state.KeepRunning()) {
    benchmark::DoNotOptimize(mmul_xtensor(a1, a2, a3));
  }
}

#endif  // COMPARE_XTENSOR

template <typename M1, typename M2, typename M3>
static M3 &mmul_nd_array(const M1 &lhs, const M2 &rhs,
                         M3 &result) {
  static_assert(M1::extent(0) == M3::extent(0),
                "Shapes don't match");
  static_assert(M1::extent(1) == M2::extent(0),
                "Shapes don't match");
  static_assert(M2::extent(1) == M3::extent(1),
                "Shapes don't match");
  for(int i = 0; i < lhs.extent(0); ++i) {
    for(int j = 0; j < rhs.extent(1); ++j) {
      result(i, j) = 0.0;
      for(int k = 0; k < lhs.extent(1); ++k) {
        result(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  return result;
}

static void BM_ND_Array_Deref_MMul(
    benchmark::State &state) {
  auto a1 = std::make_unique<ND_Array<double, 80, 100>>();
  auto a2 = std::make_unique<ND_Array<double, 100, 120>>();
  auto a3 = std::make_unique<ND_Array<double, 80, 120>>();
  double counter = 1.0;
  for(double &v : *a1) {
    v = counter;
    counter += 1.0;
  }
  for(double &v : *a2) {
    v = counter;
    counter += 1.0;
  }
  for(double &v : *a3) {
    v = std::numeric_limits<double>::quiet_NaN();
  }

  while(state.KeepRunning()) {
    benchmark::DoNotOptimize(mmul_nd_array(*a1, *a2, *a3));
  }
}

static void BM_ND_Array_MMul(benchmark::State &state) {
  ND_Array<double, 80, 100> a1;
  ND_Array<double, 100, 120> a2;
  ND_Array<double, 80, 120> a3;
  double counter = 1.0;
  for(double &v : a1) {
    v = counter;
    counter += 1.0;
  }
  for(double &v : a2) {
    v = counter;
    counter += 1.0;
  }
  for(double &v : a3) {
    v = std::numeric_limits<double>::quiet_NaN();
  }

  while(state.KeepRunning()) {
    benchmark::DoNotOptimize(mmul_nd_array(a1, a2, a3));
  }
}

template <size_t d>
using array_2d = double (*)[d];

template <size_t D1, size_t D2, size_t D3>
static array_2d<D3> mmul_c_array(const double lhs[D1][D2],
                                 const double rhs[D2][D3],
                                 double result[D1][D3]) {
  for(int i = 0; i < D1; ++i) {
    for(int j = 0; j < D3; ++j) {
      result[i][j] = 0.0;
      for(int k = 0; k < D2; ++k) {
        result[i][j] += lhs[i][k] * rhs[k][j];
      }
    }
  }
  return result;
}

static void BM_C_Array_MMul(benchmark::State &state) {
  constexpr size_t D1 = 80;
  constexpr size_t D2 = 100;
  constexpr size_t D3 = 120;
  double a1[D1][D2];
  double a2[D2][D3];
  double a3[D1][D3];
  double counter = 1.0;
  for(int i = 0; i < D1; i++) {
    for(int j = 0; j < D2; j++) {
      a1[i][j] = counter;
      counter += 1.0;
    }
  }
  for(int i = 0; i < D2; i++) {
    for(int j = 0; j < D3; j++) {
      a2[i][j] = counter;
      counter += 1.0;
    }
  }
  for(int i = 0; i < D1; i++) {
    for(int j = 0; j < D3; j++) {
      a3[i][j] = std::numeric_limits<double>::quiet_NaN();
    }
  }

  while(state.KeepRunning()) {
    benchmark::DoNotOptimize(
        mmul_c_array<D1, D2, D3>(a1, a2, a3));
  }
}

static double *mmul_ptr_array(const double *lhs,
                              const double *rhs,
                              double *result, const int d1,
                              const int d2, const int d3) {
  // lhs[d1][d2], rhs[d2][d3], result[d1][d3]
  for(int i = 0; i < d1; ++i) {
    for(int j = 0; j < d3; ++j) {
      result[i * d3 + j] = 0.0;
      for(int k = 0; k < d2; ++k) {
        result[i * d3 + j] +=
            lhs[i * d2 + k] * rhs[k * d3 + j];
      }
    }
  }
  return result;
}

static void BM_C_Ptr_MMul(benchmark::State &state) {
  constexpr size_t D1 = 80;
  constexpr size_t D2 = 100;
  constexpr size_t D3 = 120;
  double a1[D1][D2];
  double a2[D2][D3];
  double a3[D1][D3];
  double counter = 1.0;
  for(int i = 0; i < D1; i++) {
    for(int j = 0; j < D2; j++) {
      a1[i][j] = counter;
      counter += 1.0;
    }
  }
  for(int i = 0; i < D2; i++) {
    for(int j = 0; j < D3; j++) {
      a2[i][j] = counter;
      counter += 1.0;
    }
  }
  for(int i = 0; i < D1; i++) {
    for(int j = 0; j < D3; j++) {
      a3[i][j] = std::numeric_limits<double>::quiet_NaN();
    }
  }

  while(state.KeepRunning()) {
    benchmark::DoNotOptimize(mmul_ptr_array(
        &a1[0][0], &a2[0][0], &a3[0][0], D1, D2, D3));
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize();

  benchmark::RegisterBenchmark("BM_Null", BM_Null);

#ifdef COMPARE_XTENSOR
  benchmark::RegisterBenchmark("BM_XTensor_MMul",
                               BM_XTensor_MMul);
#endif  // COMPARE_XTENSOR
  benchmark::RegisterBenchmark("BM_ND_Array_Deref_MMul",
                               BM_ND_Array_Deref_MMul);
  benchmark::RegisterBenchmark("BM_ND_Array_MMul",
                               BM_ND_Array_MMul);
  benchmark::RegisterBenchmark("BM_C_Array_MMul",
                               BM_C_Array_MMul);
  benchmark::RegisterBenchmark("BM_C_Ptr_MMul",
                               BM_C_Ptr_MMul);
#ifdef COMPARE_XTENSOR
  benchmark::RegisterBenchmark("BM_XTensor_MMul_2",
                               BM_XTensor_MMul);
#endif  // COMPARE_XTENSOR

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
