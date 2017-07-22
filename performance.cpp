
#include <limits>
#include <random>
#include <vector>

#include <iostream>

#include "nd_array.hpp"
#include "timer/timer.hpp"

class PFunctor {
 public:
  PFunctor() : _time(){};

  virtual const char *name() const = 0;

  void time_functor(int num_runs = 1000) {
    _run_functor();
    _time.startTimer();
    for(int i = 0; i < num_runs; i++) {
      _run_functor();
    }
    _time.stopTimer();
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const PFunctor &p) {
    os << p.name() << std::endl << p._time << std::endl;
    return os;
  }

 private:
  virtual void _run_functor() = 0;

  Timer::Timer _time;
};

template <int _scale_dim, int _matrix_dim>
class NDFunctor : public PFunctor {
 public:
  template <typename rng_alg>
  NDFunctor(rng_alg &engine)
      : PFunctor(), a1(), a2(), results() {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(int h = 0; h < scale_dim; h++) {
      for(int i = 0; i < matrix_dim; i++) {
        for(int j = 0; j < matrix_dim; j++) {
          a1(h, i, j) = dist(engine);
          results(h, i, j) =
              std::numeric_limits<double>::quiet_NaN();
        }
      }
      a2(h) = dist(engine);
    }
  }

  virtual const char *name() const {
    return "ND_Array Performance";
  }

 private:
  static constexpr const int scale_dim = _scale_dim;
  static constexpr const int matrix_dim = _matrix_dim;

  virtual void _run_functor() {
    for(int h = 0; h < scale_dim; h++) {
      for(int i = 0; i < matrix_dim; i++) {
        for(int j = 0; j < matrix_dim; j++) {
          results(h, i, j) = a1(h, i, j) * a2(h);
        }
      }
    }
  }

  ND_Array<double, scale_dim, matrix_dim, matrix_dim> a1;
  ND_Array<double, scale_dim> a2;
  ND_Array<double, scale_dim, matrix_dim, matrix_dim>
      results;
};

template <int _scale_dim, int _matrix_dim>
class CArrFunctor : public PFunctor {
 public:
  template <typename rng_alg>
  CArrFunctor(rng_alg &engine)
      : PFunctor(), a1(), a2(), results() {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(int h = 0; h < scale_dim; h++) {
      for(int i = 0; i < matrix_dim; i++) {
        for(int j = 0; j < matrix_dim; j++) {
          a1[h][i][j] = dist(engine);
          results[h][i][j] =
              std::numeric_limits<double>::quiet_NaN();
        }
      }
      a2[h] = dist(engine);
    }
  }

  virtual const char *name() const {
    return "C ND Array Performance";
  }

 private:
  static constexpr const int scale_dim = _scale_dim;
  static constexpr const int matrix_dim = _matrix_dim;

  virtual void _run_functor() {
    for(int h = 0; h < scale_dim; h++) {
      for(int i = 0; i < matrix_dim; i++) {
        for(int j = 0; j < matrix_dim; j++) {
          results[h][i][j] = a1[h][i][j] * a2[h];
        }
      }
    }
  }

  double a1[scale_dim][matrix_dim][matrix_dim];
  double a2[scale_dim];
  double results[scale_dim][matrix_dim][matrix_dim];
};

int main(int argc, char **argv) {
  std::random_device rd;
  std::mt19937_64 engine(rd());
  std::vector<std::unique_ptr<PFunctor> > p_compare;
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new CArrFunctor<10, 100>(engine)));
  p_compare.push_back(std::unique_ptr<PFunctor>(
      new NDFunctor<10, 100>(engine)));

  for(std::unique_ptr<PFunctor> &functor : p_compare) {
    functor->time_functor();
  }

  for(std::unique_ptr<PFunctor> &functor : p_compare) {
    std::cout << *functor << std::endl;
  }
  return 0;
}
