
#ifndef _ZIP_HPP_
#define _ZIP_HPP_

#include "zip_internal.hpp"

// The actual Zip iterator
// WARNING: The lifetime of the Zip object is dependent on
// the lifetime of the containers its constructed with
//
// An example of the intended usage is just
// for(auto [&t1, &t2] : Zip<std::vector<T1>,
//                           std::vector<T2> >(vec_1,
//                                             vec_2))
// {...}
//
// or better
// for(auto [&t1, &t2] : make_zip(vec_1, vec_2))
// {...}
//
// An interesting and probably difficult to implement future
// improvement would enable mixing the types of the
// iterators used, so some could be marked as const
template <typename... containers_>
class Zip {
 public:
  // Container types
  using value_type = typename zip_internal_::tuple_builder<
      zip_internal_::value_converter,
      zip_internal_::tt_list<>, containers_...>::tuple_type;
  using reference = typename zip_internal_::tuple_builder<
      zip_internal_::reference_converter,
      zip_internal_::tt_list<>, containers_...>::tuple_type;
  using const_reference =
      typename zip_internal_::tuple_builder<
          zip_internal_::const_reference_converter,
          zip_internal_::tt_list<>,
          containers_...>::tuple_type;
  using pointer = typename zip_internal_::tuple_builder<
      zip_internal_::pointer_converter,
      zip_internal_::tt_list<>, containers_...>::tuple_type;
  using const_pointer =
      typename zip_internal_::tuple_builder<
          zip_internal_::const_pointer_converter,
          zip_internal_::tt_list<>,
          containers_...>::tuple_type;

  using size_type = typename std::tuple_element<
      0, std::tuple<containers_...> >::type::size_type;
  using difference_type = typename std::tuple_element<
      0,
      std::tuple<containers_...> >::type::difference_type;

  // Iterator
  template <typename iterator_converter,
            typename reference_t>
  class iterator_t {
   public:
    using iterator_tuple =
        typename zip_internal_::tuple_builder<
            iterator_converter, zip_internal_::tt_list<>,
            containers_...>::tuple_type;

    explicit constexpr iterator_t(
        const iterator_tuple &iters) noexcept
        : iters_(iters) {}

    constexpr const reference_t operator*() const noexcept {
      return zip_internal_::ref_tuple_map(
          iters_, zip_internal_::const_iterator_deref());
    }

    constexpr reference_t operator*() noexcept {
      return zip_internal_::ref_tuple_map(
          iters_, zip_internal_::const_iterator_deref());
    }

    constexpr iterator_t &operator=(
        const iterator_t &src) noexcept {
      iters_ = src.iters_;
      return *this;
    }

    constexpr difference_type operator-(
        const iterator_t &rhs) const noexcept {
      return std::get<0>(iters_) - std::get<0>(rhs.iters_);
    }

    // Pre-increment operators
    constexpr iterator_t &operator++() noexcept {
      zip_internal_::ref_tuple_map(
          iters_, zip_internal_::iterator_incr());
      return *this;
    }

    constexpr iterator_t &operator--() noexcept {
      zip_internal_::ref_tuple_map(
          iters_, zip_internal_::iterator_decr());
      return *this;
    }

    // Post-increment operators
    constexpr iterator_t operator++(int) noexcept {
      const auto copy = *this;
      ++(*this);
      return copy;
    }

    constexpr iterator_t operator--(int) noexcept {
      const auto copy = *this;
      --(*this);
      return copy;
    }

    // Very much looking forward to the spaceship operator
    constexpr bool operator==(const iterator_t &cmp) const
        noexcept {
      return ((*this) - cmp) == 0;
    }

    constexpr bool operator!=(const iterator_t &cmp) const
        noexcept {
      return !(*this == cmp);
    }

    constexpr bool operator<(const iterator_t &cmp) const
        noexcept {
      return (*this - cmp) < 0;
    }

    constexpr bool operator<=(const iterator_t &cmp) const
        noexcept {
      return (*this - cmp) <= 0;
    }

    constexpr bool operator>(const iterator_t &cmp) const
        noexcept {
      return !((*this - cmp) <= 0);
    }

    constexpr bool operator>=(const iterator_t &cmp) const
        noexcept {
      return !((*this - cmp) < 0);
    }

   protected:
    iterator_tuple iters_;
  };

  using const_iterator =
      iterator_t<zip_internal_::const_iterator_converter,
                 const_reference>;
  using iterator =
      iterator_t<zip_internal_::iterator_converter,
                 reference>;

  // Constructor - due to the lifetime constraints, rvalues
  // are not permitted as inputs, only lvalue references
  constexpr Zip() = delete;
  constexpr Zip(containers_ &&...) = delete;

  constexpr Zip(containers_ &... contents) noexcept
      : contents_(contents...) {}

  constexpr Zip &operator=(const Zip &src) noexcept {
    contents_ = src.contents_;
    return *this;
  }

  // Methods
  constexpr const_iterator cbegin() const noexcept {
    return const_iterator(tuple_map(
        contents_,
        zip_internal_::const_begin_iterator_converter()));
  }
  constexpr const_iterator cend() const noexcept {
    return const_iterator(tuple_map(
        contents_,
        zip_internal_::const_end_iterator_converter()));
  }

  constexpr iterator begin() const noexcept {
    return iterator(tuple_map(
        contents_,
        zip_internal_::begin_iterator_converter()));
  }
  constexpr iterator end() const noexcept {
    return iterator(
        tuple_map(contents_,
                  zip_internal_::end_iterator_converter()));
  }

  std::tuple<containers_ &...> contents_;
};

#endif  // _ZIP_HPP_
