
#include <limits>
#include <random>
#include <vector>

#include <iostream>

#ifdef COMPARE_KOKKOS
#include <Kokkos_Core.hpp>
#else
namespace Kokkos {
void initialize() {}
}
#endif

#include "nd_array.hpp"
#include "timer/timer.hpp"

#define ALIGN(vardec) __declspec(align) vardec
#define ALIGNTO(vardec, boundary) __declspec(align(boundary)) vardec

class PFunctor {
public:
  PFunctor() : _time(){};
  virtual ~PFunctor() {}

  virtual const char *name() const = 0;

  void time_functor(int num_runs = 1000) {
    _run_functor();
    _time.startTimer();
    for (int i = 0; i < num_runs; i++) {
      _run_functor();
    }
    _time.stopTimer();
  }

  friend std::ostream &operator<<(std::ostream &os, const PFunctor &p) {
    os << p.name() << std::endl << p._time << std::endl;
    return os;
  }

private:
  virtual void _run_functor() = 0;

  Timer::Timer _time;
};

template <int _scale_dim, int _matrix_dim> class NDFunctor : public PFunctor {
public:
  template <typename rng_alg>
  NDFunctor(rng_alg &engine) : PFunctor(), a1(), a2(), results() {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int h = 0; h < scale_dim; h++) {
      for (int i = 0; i < matrix_dim; i++) {
        for (int j = 0; j < matrix_dim; j++) {
          a1(h, i, j) = dist(engine);
          results(h, i, j) = std::numeric_limits<double>::quiet_NaN();
        }
      }
      a2(h) = dist(engine);
    }
  }

  virtual const char *name() const { return "ND_Array Performance"; }

private:
  static constexpr const int scale_dim = _scale_dim;
  static constexpr const int matrix_dim = _matrix_dim;

  virtual void _run_functor() {
#pragma ivdep
    for (int h = 0; h < scale_dim; h++) {
#pragma ivdep
      for (int i = 0; i < matrix_dim; i++) {
#pragma ivdep
        for (int j = 0; j < matrix_dim; j++) {
          results(h, i, j) = a1(h, i, j) * a2(h);
        }
      }
    }
  }

  ND_Array<double, scale_dim, matrix_dim, matrix_dim> a1;
  ND_Array<double, scale_dim> a2;
  ND_Array<double, scale_dim, matrix_dim, matrix_dim> results;
};

template <int _scale_dim, int _matrix_dim>
class NDPointerFunctor : public PFunctor {
public:
  template <typename rng_alg>
  NDPointerFunctor(rng_alg &engine)
      : PFunctor(),
        a1(new ND_Array<double, scale_dim, matrix_dim, matrix_dim>()),
        a2(new ND_Array<double, scale_dim>()),
        results(new ND_Array<double, scale_dim, matrix_dim, matrix_dim>()) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int h = 0; h < scale_dim; h++) {
      for (int i = 0; i < matrix_dim; i++) {
        for (int j = 0; j < matrix_dim; j++) {
          (*a1)(h, i, j) = dist(engine);
          (*results)(h, i, j) = std::numeric_limits<double>::quiet_NaN();
        }
      }
      (*a2)(h) = dist(engine);
    }
  }

  virtual ~NDPointerFunctor() {
    delete a1;
    delete a2;
    delete results;
  }

  virtual const char *name() const {
    return "ND Array Raw Pointer Performance";
  }

private:
  static constexpr const int scale_dim = _scale_dim;
  static constexpr const int matrix_dim = _matrix_dim;

  virtual void _run_functor() {
#pragma ivdep
    for (int h = 0; h < scale_dim; h++) {
#pragma ivdep
      for (int i = 0; i < matrix_dim; i++) {
#pragma ivdep
        for (int j = 0; j < matrix_dim; j++) {
          (*results)(h, i, j) = (*a1)(h, i, j) * (*a2)(h);
        }
      }
    }
  }

  ND_Array<double, scale_dim, matrix_dim, matrix_dim> *a1;
  ND_Array<double, scale_dim> *a2;
  ND_Array<double, scale_dim, matrix_dim, matrix_dim> *results;
};

template <int _scale_dim, int _matrix_dim> class NDUPFunctor : public PFunctor {
public:
  template <typename rng_alg>
  NDUPFunctor(rng_alg &engine)
      : PFunctor(),
        a1(new ND_Array<double, scale_dim, matrix_dim, matrix_dim>()),
        a2(new ND_Array<double, scale_dim>()),
        results(new ND_Array<double, scale_dim, matrix_dim, matrix_dim>()) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int h = 0; h < scale_dim; h++) {
      for (int i = 0; i < matrix_dim; i++) {
        for (int j = 0; j < matrix_dim; j++) {
          (*a1)(h, i, j) = dist(engine);
          (*results)(h, i, j) = std::numeric_limits<double>::quiet_NaN();
        }
      }
      (*a2)(h) = dist(engine);
    }
  }

  virtual const char *name() const {
    return "ND Array Unique Pointer Performance";
  }

private:
  static constexpr const int scale_dim = _scale_dim;
  static constexpr const int matrix_dim = _matrix_dim;

  virtual void _run_functor() {
#pragma ivdep
    for (int h = 0; h < scale_dim; h++) {
#pragma ivdep
      for (int i = 0; i < matrix_dim; i++) {
#pragma ivdep
        for (int j = 0; j < matrix_dim; j++) {
          (*results)(h, i, j) = (*a1)(h, i, j) * (*a2)(h);
        }
      }
    }
  }

  std::unique_ptr<ND_Array<double, scale_dim, matrix_dim, matrix_dim> > a1;
  std::unique_ptr<ND_Array<double, scale_dim> > a2;
  std::unique_ptr<ND_Array<double, scale_dim, matrix_dim, matrix_dim> > results;
};

#ifdef COMPARE_KOKKOS
template <int _scale_dim, int _matrix_dim>
class KViewFunctor : public PFunctor {
public:
  template <typename rng_alg>
  KViewFunctor(rng_alg &engine)
      : PFunctor(), a1("a1"), a2("a2"), results("results") {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int h = 0; h < scale_dim; h++) {
      for (int i = 0; i < matrix_dim; i++) {
        for (int j = 0; j < matrix_dim; j++) {
          a1(h, i, j) = dist(engine);
          results(h, i, j) = std::numeric_limits<double>::quiet_NaN();
        }
      }
      a2(h) = dist(engine);
    }
  }

  virtual const char *name() const { return "Kokkos View Performance"; }

private:
  static constexpr const int scale_dim = _scale_dim;
  static constexpr const int matrix_dim = _matrix_dim;

  virtual void _run_functor() {
#pragma ivdep
    for (int h = 0; h < scale_dim; h++) {
#pragma ivdep
      for (int i = 0; i < matrix_dim; i++) {
#pragma ivdep
        for (int j = 0; j < matrix_dim; j++) {
          results(h, i, j) = a1(h, i, j) * a2(h);
        }
      }
    }
  }

  typename Kokkos::View<double[scale_dim][matrix_dim][matrix_dim],
                        Kokkos::MemoryUnmanaged>::HostMirror a1;
  typename Kokkos::View<double[scale_dim], Kokkos::MemoryUnmanaged>::HostMirror
      a2;
  typename Kokkos::View<double[scale_dim][matrix_dim][matrix_dim],
                        Kokkos::MemoryUnmanaged>::HostMirror results;
};
#endif

template <int _scale_dim, int _matrix_dim> class CArrFunctor : public PFunctor {
public:
  template <typename rng_alg>
  CArrFunctor(rng_alg &engine) : PFunctor(), a1(), a2(), results() {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int h = 0; h < scale_dim; h++) {
      for (int i = 0; i < matrix_dim; i++) {
        for (int j = 0; j < matrix_dim; j++) {
          a1[h][i][j] = dist(engine);
          results[h][i][j] = std::numeric_limits<double>::quiet_NaN();
        }
      }
      a2[h] = dist(engine);
    }
  }

  virtual const char *name() const { return "C ND Pointer Performance"; }

private:
  static constexpr const int scale_dim = _scale_dim;
  static constexpr const int matrix_dim = _matrix_dim;

  virtual void _run_functor() {
#pragma ivdep
    for (int h = 0; h < scale_dim; h++) {
#pragma ivdep
      for (int i = 0; i < matrix_dim; i++) {
#pragma ivdep
        for (int j = 0; j < matrix_dim; j++) {
          results[h][i][j] = a1[h][i][j] * a2[h];
        }
      }
    }
  }

  double a1[scale_dim][matrix_dim][matrix_dim];
  double a2[scale_dim];
  double results[scale_dim][matrix_dim][matrix_dim];
};

template <int _scale_dim, int _matrix_dim>
class CArrPointerFunctor : public PFunctor {
public:
  template <typename rng_alg>
  CArrPointerFunctor(rng_alg &engine)
      : PFunctor(), a1(new double[scale_dim][matrix_dim][matrix_dim]),
        a2(new double[scale_dim]),
        results(new double[scale_dim][matrix_dim][matrix_dim]) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int h = 0; h < scale_dim; h++) {
      for (int i = 0; i < matrix_dim; i++) {
        for (int j = 0; j < matrix_dim; j++) {
          a1[h][i][j] = dist(engine);
          results[h][i][j] = std::numeric_limits<double>::quiet_NaN();
        }
      }
      a2[h] = dist(engine);
    }
  }

  virtual ~CArrPointerFunctor() {
    delete[] a1;
    delete[] a2;
    delete[] results;
  }

  virtual const char *name() const {
    return "C ND Array Raw Pointer Performance";
  }

private:
  static constexpr const int scale_dim = _scale_dim;
  static constexpr const int matrix_dim = _matrix_dim;

  virtual void _run_functor() {
#pragma ivdep
    for (int h = 0; h < scale_dim; h++) {
#pragma ivdep
      for (int i = 0; i < matrix_dim; i++) {
#pragma ivdep
        for (int j = 0; j < matrix_dim; j++) {
          results[h][i][j] = a1[h][i][j] * a2[h];
        }
      }
    }
  }

  double (*a1)[matrix_dim][matrix_dim];
  double(*a2);
  double (*results)[matrix_dim][matrix_dim];
};

template <int _scale_dim, int _matrix_dim>
class CArrUPFunctor : public PFunctor {
public:
  template <typename rng_alg>
  CArrUPFunctor(rng_alg &engine)
      : PFunctor(), a1(new double[scale_dim][matrix_dim][matrix_dim]),
        a2(new double[scale_dim]),
        results(new double[scale_dim][matrix_dim][matrix_dim]) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int h = 0; h < scale_dim; h++) {
      for (int i = 0; i < matrix_dim; i++) {
        for (int j = 0; j < matrix_dim; j++) {
          a1[h][i][j] = dist(engine);
          results[h][i][j] = std::numeric_limits<double>::quiet_NaN();
        }
      }
      a2[h] = dist(engine);
    }
  }

  virtual const char *name() const {
    return "C ND Array Unique Pointer Performance";
  }

private:
  static constexpr const int scale_dim = _scale_dim;
  static constexpr const int matrix_dim = _matrix_dim;

  virtual void _run_functor() {
#pragma ivdep
    for (int h = 0; h < scale_dim; h++) {
#pragma ivdep
      for (int i = 0; i < matrix_dim; i++) {
#pragma ivdep
        for (int j = 0; j < matrix_dim; j++) {
          results[h][i][j] = a1[h][i][j] * a2[h];
        }
      }
    }
  }

  std::unique_ptr<double[scale_dim][matrix_dim][matrix_dim]> a1;
  std::unique_ptr<double[scale_dim]> a2;
  std::unique_ptr<double[scale_dim][matrix_dim][matrix_dim]> results;
};

int main(int argc, char **argv) {
  Kokkos::initialize();
  std::random_device rd;
  std::mt19937_64 engine(rd());
  std::vector<std::unique_ptr<PFunctor> > p_compare;

  static constexpr int scale_dim = 100;
  static constexpr int matrix_dim = 100;

  p_compare.push_back(std::unique_ptr<PFunctor>(
      new CArrFunctor<scale_dim, matrix_dim>(engine)));
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new CArrPointerFunctor<scale_dim, matrix_dim>(engine)));
  p_compare.push_back(
      std::unique_ptr<PFunctor>(new NDFunctor<scale_dim, matrix_dim>(engine)));
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new NDPointerFunctor<scale_dim, matrix_dim>(engine)));
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new NDUPFunctor<scale_dim, matrix_dim>(engine)));
#ifdef COMPARE_KOKKOS
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new KViewFunctor<scale_dim, matrix_dim>(engine)));
#endif

  p_compare.push_back(std::unique_ptr<PFunctor>(
      new CArrFunctor<scale_dim, matrix_dim>(engine)));
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new CArrPointerFunctor<scale_dim, matrix_dim>(engine)));
  p_compare.push_back(
      std::unique_ptr<PFunctor>(new NDFunctor<scale_dim, matrix_dim>(engine)));
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new NDPointerFunctor<scale_dim, matrix_dim>(engine)));
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new NDUPFunctor<scale_dim, matrix_dim>(engine)));
#ifdef COMPARE_KOKKOS
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new KViewFunctor<scale_dim, matrix_dim>(engine)));
#endif

  for (std::unique_ptr<PFunctor> &functor : p_compare) {
    functor->time_functor();
  }

  for (std::unique_ptr<PFunctor> &functor : p_compare) {
    std::cout << *functor << std::endl;
  }
  return 0;
}
