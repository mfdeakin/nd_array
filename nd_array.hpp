
#ifndef _NDARRAY_HPP_
#define _NDARRAY_HPP_

#include <type_traits>

namespace {
template <typename FieldT, FieldT leading, FieldT... others>
struct CT_Array {
  constexpr static const FieldT current = leading;

  static constexpr int len() {
    return sizeof...(others) + 1;
  }

  static constexpr FieldT value(int idx) {
    return (idx == 0 ? current : Next::value(idx - 1));
  }

  static constexpr FieldT sum() {
    return leading + Next::sum();
  }

  static constexpr FieldT product() {
    return leading * Next::product();
  }

  template <typename... indices>
  static constexpr int slice_idx(int idx, indices... tail) {
    return idx * Next::product() + Next::slice_idx(tail...);
  }

  static constexpr int slice_idx(int idx) {
    return idx * Next::product();
  }

  template <typename Idx_Array,
            typename std::enable_if<Idx_Array::len() != 1,
                                    int>::type = 0>
  static constexpr FieldT slice_idx() {
    static_assert(Idx_Array::current < leading,
                  "Index array's indices are too large");
    static_assert(Idx_Array::len() <= Self::len(),
                  "Too many indices");
    return Idx_Array::current * Next::product() +
           Next::template slice_idx<
               typename Idx_Array::Next>();
  }

  template <typename Idx_Array,
            typename std::enable_if<Idx_Array::len() == 1,
                                    int>::type = 0>
  static constexpr FieldT slice_idx() {
    static_assert(Idx_Array::current < leading,
                  "Index array's indices are too large");
    static_assert(Idx_Array::len() <= Self::len(),
                  "Too many indices");
    return Idx_Array::current * Next::product();
  }

  using Self = CT_Array<FieldT, leading, others...>;
  using Next = CT_Array<FieldT, others...>;
};

template <typename FieldT, FieldT val>
struct CT_Array<FieldT, val> {
  constexpr static const FieldT current = val;

  static constexpr int len() { return 1; }
  static constexpr int value(int idx) { return current; }
  static constexpr FieldT sum() { return val; }
  static constexpr FieldT product() { return val; }

  static constexpr int slice_idx(int idx) { return idx; }

  template <typename Idx_Array>
  static constexpr FieldT slice_idx() {
    static_assert(Idx_Array::len() == 1,
                  "Index array not of length 1");
    static_assert(Idx_Array::current < val,
                  "Index array value overflow");
    return Idx_Array::current;
  }
};
}

template <typename A_Type, int... dims>
class ND_Array {
 public:
  constexpr ND_Array() noexcept {}

  template <typename... int_t>
  A_Type &operator()(int_t... indices) noexcept {
    static_assert(sizeof...(int_t) == DIMS::len(),
                  "Number of indices passed is incorrect");
    return vals[DIMS::slice_idx(indices...)];
  }

  template <typename... int_t>
  constexpr A_Type operator()(int_t... indices) const
      noexcept {
    static_assert(sizeof...(int_t) == DIMS::len(),
                  "Number of indices passed is incorrect");
    return vals[DIMS::slice_idx(indices...)];
  }

 private:
  using DIMS = CT_Array<int, dims...>;
  A_Type vals[DIMS::product()];
};

namespace {

/* Compile Time List Tests */

using Arr0 = CT_Array<int, 4>;

static_assert(Arr0::len() == 1,
              "Incorrect CT Array Length");
static_assert(Arr0::sum() == 4, "Incorrect CT Array Sum");
static_assert(Arr0::product() == 4,
              "Incorrect CT Array Product");

using Arr1 = CT_Array<int, 1, 2, 3, 4>;
static_assert(Arr1::len() == 4,
              "Incorrect CT Array Length");
static_assert(Arr1::sum() == 10, "Incorrect CT Array Sum");
static_assert(Arr1::product() == 24,
              "Incorrect CT Array Product");

using Arr2 = CT_Array<int, 0, 1>;
static_assert(Arr1::slice_idx<Arr2>() == 12,
              "Incorrect Template Slice Index");

using Arr3 = CT_Array<int, 0, 0, 1>;
static_assert(Arr1::slice_idx<Arr3>() == 4,
              "Incorrect Template Slice Index");

using Arr4 = CT_Array<int, 0, 0, 0, 1>;
static_assert(Arr1::slice_idx<Arr4>() == 1,
              "Incorrect Template Slice Index");

static_assert(Arr1::slice_idx(0, 0, 0, 0) == 0,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 0, 1) == 1,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 0, 2) == 2,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 0, 3) == 3,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 1, 0) == 4,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 1, 1) == 5,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 1, 2) == 6,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 1, 3) == 7,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 2, 0) == 8,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 2, 1) == 9,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 2, 2) == 10,
              "Incorrect Runtime Slice Index");
static_assert(Arr1::slice_idx(0, 0, 2, 3) == 11,
              "Incorrect Runtime Slice Index");
}

#endif
