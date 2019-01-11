
#ifndef _NDARRAY_HPP_
#define _NDARRAY_HPP_

#include <type_traits>

namespace ND_Array_internals {

#include "ct_array.hpp"

template <typename value_type_, typename Dims_CT_Array>
class [[nodiscard]] nd_array_ {
 public:
  using value_type = value_type_;
  using reference = value_type &;
  using const_reference = const value_type &;
  using pointer = value_type *;
  using const_pointer = const value_type *;

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using nd_array_type =
      nd_array_<value_type, Dims_CT_Array>;

  constexpr nd_array_() noexcept {}

  constexpr nd_array_(
      const nd_array_<value_type, Dims_CT_Array>
          &src) noexcept {
    for(int i = 0; i < size(); i++) {
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

  // Provides fast (memory-access wise) iterators
  class const_iterator {
   public:
    using value_type = value_type_;
    using difference_type = int;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;

   protected:
    constexpr const_iterator(pointer pos, int idx_)
        : val(pos), idx(idx_) {}

   public:
    constexpr const_iterator(const const_iterator &src)
        : val(src.val), idx(src.idx) {}

    constexpr const_iterator &operator=(
        const const_iterator &src) {
      val = src.val;
      idx = src.idx;
      return *this;
    }

    ~const_iterator() = default;

    [[nodiscard]] constexpr bool operator==(
        const const_iterator &cmp) const {
      return (val == cmp.val);
    }

    [[nodiscard]] constexpr bool operator!=(
        const const_iterator &cmp) const {
      return (val != cmp.val);
    }

    [[nodiscard]] constexpr bool operator<(
        const const_iterator &cmp) const {
      return idx < cmp.idx;
    }

    [[nodiscard]] constexpr bool operator<=(
        const const_iterator &cmp) const {
      return idx <= cmp.idx;
    }

    [[nodiscard]] constexpr bool operator>(
        const const_iterator &cmp) const {
      return idx > cmp.idx;
    }

    [[nodiscard]] constexpr bool operator>=(
        const const_iterator &cmp) const {
      return idx >= cmp.idx;
    }

    constexpr const_iterator &operator++() {
      val++;
      idx++;
      return *this;
    }

    constexpr const_iterator &operator--() {
      val--;
      idx--;
      assert(idx >= 0);
      return *this;
    }

    // Postfix implementations
    constexpr const_iterator &operator++(int) {
      val++;
      idx++;
      return *this;
    }

    constexpr const_iterator &operator--(int) {
      val--;
      idx--;
      assert(idx >= 0);
      return *this;
    }

    [[nodiscard]] constexpr int index() const noexcept {
      return idx;
    }

    [[nodiscard]] constexpr int index(const int dim) const
        noexcept {
      assert(dim < DIMS::len());
      if(dim + 1 == DIMS::len()) {
        return idx % DIMS::value(dim);
      } else {
        return (idx / DIMS::trailing_product(dim + 1)) %
               DIMS::value(dim);
      }
    }

    [[nodiscard]] constexpr const_reference operator*()
        const {
      return *val;
    }

    constexpr void swap(const_iterator &other) {
      std::swap(val, other.val);
      std::swap(idx, other.idx);
    }

    friend nd_array_type;

   protected:
    pointer val;
    int idx;
  };

  class iterator : public const_iterator {
   protected:
    constexpr iterator(pointer pos, int idx)
        : const_iterator(pos, idx) {}

   public:
    constexpr iterator(const iterator &src)
        : const_iterator(src) {}

    constexpr iterator &operator=(const iterator &src) {
      this->val = src.val;
      this->idx = src.idx;
      return *this;
    }

    ~iterator() = default;

    [[nodiscard]] constexpr reference operator*() const {
      return *this->val;
    }

    friend nd_array_type;
  };

  [[nodiscard]] constexpr iterator begin() noexcept {
    return iterator(&vals[0], 0);
  }

  [[nodiscard]] constexpr const_iterator cbegin()
      const noexcept {
    return iterator(&vals[0], 0);
  }

  [[nodiscard]] constexpr iterator end() noexcept {
    return iterator(&vals[size()], size());
  }

  [[nodiscard]] constexpr const_iterator cend()
      const noexcept {
    return iterator(&vals[size()], size());
  }

  template <typename _value_type, typename _Dims_CT_Array>
  friend class nd_array_;

 private:
  using DIMS = Dims_CT_Array;
  value_type vals[size()];
};

template <typename value_type, typename dims>
constexpr typename nd_array_<
    value_type, dims>::const_iterator::difference_type
operator-(
    const typename nd_array_<value_type,
                             dims>::const_iterator &lhs,
    const typename nd_array_<value_type,
                             dims>::const_iterator &rhs) {
  return lhs.index() - rhs.index();
}

}  // namespace ND_Array_internals

template <typename value_type, int... Dims>
using ND_Array = ND_Array_internals::nd_array_<
    value_type,
    ND_Array_internals::CT_Array<int, Dims...> >;

#endif
