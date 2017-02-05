
#ifndef _NDARRAY_HPP_
#define _NDARRAY_HPP_

#include <type_traits>

#include "ct_array.hpp"

template <typename A_Type, typename Dims_CT_Array>
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

  template <typename... int_t>
  constexpr ND_Array<
      A_Type, typename forward_truncate_array<
                  sizeof...(int_t), Dims_CT_Array>::type>
      &outer_slice(int_t... indices) noexcept {
    using truncated_dims =
        typename forward_truncate_array<sizeof...(int_t),
                                        DIMS>::type;
    using ret_type = ND_Array<A_Type, truncated_dims>;
    return *(reinterpret_cast<ret_type *>(
        &vals[0] +
        DIMS::slice_idx(indices...)));
  }

 private:
  using DIMS = Dims_CT_Array;
  A_Type vals[DIMS::product()];
};

#endif
