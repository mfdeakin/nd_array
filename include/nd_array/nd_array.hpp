
#ifndef _NDARRAY_HPP_
#define _NDARRAY_HPP_

#include <type_traits>

#include "ct_array.hpp"

namespace ND_Array_internals_ {

// A better implementation might have all nd_arrays inherit
// from a 1D array (like std::array, but preferably without
// exceptions as they don't work on all platforms; eg GPUs
// https://reviews.llvm.org/D25036)
template <typename value_type_, typename Dims_CT_Array>
class [[nodiscard]] nd_array_ {
 public:
  using DIMS = Dims_CT_Array;

  using value_type = value_type_;
  using reference = value_type &;
  using const_reference = const value_type &;
  using pointer = value_type *;
  using const_pointer = const value_type *;

  using size_type = typename Dims_CT_Array::FieldT;
  using difference_type = std::ptrdiff_t;

  using nd_array_type =
      nd_array_<value_type, Dims_CT_Array>;

  constexpr nd_array_() noexcept {}

  template <
      typename Other_Dims,
      typename std::enable_if<Other_Dims::product() ==
                                  Dims_CT_Array::product(),
                              int>::type = 0>
  explicit constexpr nd_array_(
      const nd_array_<value_type, Other_Dims>
          &src) noexcept {
    for(size_type i = 0; i < size(); i++) {
      vals[i] = src.vals[i];
    }
  }

  template <typename... int_t>
  [[nodiscard]] constexpr const_reference at(
      int_t... indices) const noexcept {
    static_assert(sizeof...(int_t) == DIMS::len(),
                  "Number of indices passed is incorrect");
    return vals[DIMS::slice_idx(indices...)];
  }

  template <typename... int_t>
  [[nodiscard]] constexpr reference at(
      int_t... indices) noexcept {
    static_assert(sizeof...(int_t) == DIMS::len(),
                  "Number of indices passed is incorrect");
    return vals[DIMS::slice_idx(indices...)];
  }

  template <typename... int_t>
  [[nodiscard]] constexpr reference operator()(
      int_t... indices) noexcept {
    return at(indices...);
  }

  template <typename... int_t>
  [[nodiscard]] constexpr const_reference operator()(
      int_t... indices) const noexcept {
    return at(indices...);
  }

  [[nodiscard]] constexpr reference front() noexcept {
    return vals[0];
  }

  [[nodiscard]] constexpr const_reference front()
      const noexcept {
    return vals[0];
  }

  [[nodiscard]] constexpr reference back() noexcept {
    return vals[size() - 1];
  }

  [[nodiscard]] constexpr const_reference back()
      const noexcept {
    return vals[size() - 1];
  }

  template <typename... int_t>
  [[nodiscard]] constexpr const nd_array_<
      value_type,
      typename forward_truncate_array<sizeof...(int_t),
                                      Dims_CT_Array>::type>
      &outer_slice(int_t... indices) const noexcept {
    using truncated_dims =
        typename forward_truncate_array<sizeof...(int_t),
                                        DIMS>::type;
    using ret_type = nd_array_<value_type, truncated_dims>;
    return *(reinterpret_cast<ret_type *const>(
        &vals[0] + DIMS::slice_idx(indices...)));
  }

  template <typename... int_t>
  [[nodiscard]] nd_array_<
      value_type,
      typename forward_truncate_array<sizeof...(int_t),
                                      Dims_CT_Array>::type>
      &outer_slice(int_t... indices) noexcept {
    using truncated_dims =
        typename forward_truncate_array<sizeof...(int_t),
                                        DIMS>::type;
    using ret_type = nd_array_<value_type, truncated_dims>;
    return *(reinterpret_cast<ret_type *>(
        &vals[0] + DIMS::slice_idx(indices...)));
  }

  template <typename Reshaped_Array>
  [[nodiscard]] Reshaped_Array &reshape() noexcept {
    static_assert(Reshaped_Array::size() == size(),
                  "Reshaped array is not the same size");
    return reinterpret_cast<Reshaped_Array &>(*this);
  }

  [[nodiscard]] static constexpr bool empty() noexcept {
    return size() == 0;
  }

  [[nodiscard]] static constexpr int extent(int dim) {
    return DIMS::value(dim);
  }

  [[nodiscard]] static constexpr size_type size() noexcept {
    return DIMS::product();
  }

  [[nodiscard]] static constexpr size_type
  max_size() noexcept {
    return size();
  }

  [[nodiscard]] static constexpr int dimension() {
    return DIMS::len();
  }

  [[nodiscard]] constexpr pointer data() const noexcept {
    return vals;
  }

  constexpr void fill(const_reference value) noexcept {
    for(reference elem : (*this)) {
      elem = value;
    }
  }

  constexpr void swap(nd_array_<value_type, Dims_CT_Array> &
                      rhs) noexcept {
    iterator iter_l = begin();
    iterator iter_r = rhs.begin();
    while(iter_l != end()) {
      std::swap(*iter_l, *iter_r);
      iter_l++;
      iter_r++;
    }
  }

  using iterator = value_type *;

  [[nodiscard]] constexpr iterator begin() noexcept {
    return &vals[0];
  }

  [[nodiscard]] constexpr iterator end() noexcept {
    return &vals[size()];
  }

  using const_iterator = const value_type *;

  [[nodiscard]] constexpr const_iterator cbegin()
      const noexcept {
    return &vals[0];
  }

  [[nodiscard]] constexpr const_iterator cend()
      const noexcept {
    return &vals[size()];
  }

  [[nodiscard]] constexpr size_type index(
      const iterator &itr, const typename DIMS::FieldT dim)
      const noexcept {
    assert(dim < DIMS::len());
    const difference_type idx = itr - cbegin();
    if(dim + 1 == DIMS::len()) {
      return idx % DIMS::value(dim);
    } else {
      return (idx / DIMS::trailing_product(dim + 1)) %
             DIMS::value(dim);
    }
  }

  // WARNING: This function can return an invalid iterator,
  // comparisons of the returned iterator against begin()
  // and end() should be inequalities rather than equalities
  template <typename... int_t>
  [[nodiscard]] static constexpr iterator offset(
      const iterator &origin, const int_t &... offset) {
    iterator itr_offset =
        origin + static_cast<difference_type>(
                     DIMS::slice_idx(offset...));
    return itr_offset;
  }

  template <typename _value_type, typename _Dims_CT_Array>
  friend class nd_array_;

 private:
  value_type vals[size()];
};

}  // namespace ND_Array_internals_

template <typename value_type, int... Dims>
using ND_Array = ND_Array_internals_::nd_array_<
    value_type,
    ND_Array_internals_::CT_Array<size_t, Dims...>>;

#endif
