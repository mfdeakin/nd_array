
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "nd_array.hpp"

TEST_CASE("get, set, slice, reshape", "[ND_Array]") {
  ND_Array<int, 2, 3, 5> arr;
  int count = 1;
  for(decltype(arr)::size_type i = 0; i < arr.extent(0);
      ++i) {
    for(decltype(arr)::size_type j = 0; j < arr.extent(1);
        ++j) {
      for(decltype(arr)::size_type k = 0; k < arr.extent(2);
          ++k) {
        arr(i, j, k) = count;
        count++;
      }
    }
  }
  count = 1;
  for(decltype(arr)::size_type i = 0; i < 2; ++i) {
    for(decltype(arr)::size_type j = 0; j < 3; ++j) {
      for(decltype(arr)::size_type k = 0; k < 5; ++k) {
        REQUIRE(arr(i, j, k) == count);
        count++;
      }
    }
  }
  ND_Array<int, 3, 5> &slice1 = arr.outer_slice(0);
  using Slice1T =
      std::remove_reference<decltype(slice1)>::type;
  count = 1;
  for(Slice1T::size_type j = 0; j < 3; ++j) {
    for(Slice1T::size_type k = 0; k < 5; ++k) {
      REQUIRE(slice1(j, k) == count);
      count++;
    }
  }
  ND_Array<int, 3, 5> &slice2 = arr.outer_slice(1);
  using Slice2T =
      std::remove_reference<decltype(slice2)>::type;
  for(Slice2T::size_type j = 0; j < 3; ++j) {
    for(Slice2T::size_type k = 0; k < 5; ++k) {
      REQUIRE(slice2(j, k) == count);
      count++;
    }
  }
  ND_Array<int, 5, 3> &reshape =
      slice2.template reshape<ND_Array<int, 5, 3>>();
  using ReshapeT =
      std::remove_reference<decltype(reshape)>::type;
  for(ReshapeT::size_type i = 0; i < 3; i++) {
    REQUIRE(&reshape(0, i) == &slice2(0, i));
  }
}

TEST_CASE("iterate 1D", "[ND_Array]") {
  ND_Array<int, 11> arr;
  int count = 0;
  for(decltype(arr)::size_type i = 0; i < arr.extent(0);
      i++) {
    arr(i) = count;
    count++;
  }
  auto itr = arr.begin();
  count = 0;
  for(int i = 0; i < arr.extent(0); i++) {
    REQUIRE(*itr == count);
    REQUIRE(arr.index(itr, 0) == i);
    count++;
    ++itr;
  }
}

TEST_CASE("iterate 3D", "[ND_Array]") {
  ND_Array<int, 5, 7, 11> arr;
  int count = 0;
  for(int i = 0; i < arr.extent(0); i++) {
    for(int j = 0; j < arr.extent(1); j++) {
      for(int k = 0; k < arr.extent(2); k++) {
        arr(i, j, k) = count;
        count++;
      }
    }
  }
  auto itr = arr.begin();
  count = 0;
  for(int i = 0; i < arr.extent(0); i++) {
    for(int j = 0; j < arr.extent(1); j++) {
      for(int k = 0; k < arr.extent(2); k++) {
        REQUIRE(*itr == count);
        REQUIRE(arr.index(itr, 0) == i);
        REQUIRE(arr.index(itr, 1) == j);
        REQUIRE(arr.index(itr, 2) == k);
        count++;
        ++itr;
      }
    }
  }
}

TEST_CASE("fill", "[ND_Array]") {
  ND_Array<int, 7, 2, 3> arr;
  const int fill_val = 756;
  arr.fill(fill_val);
  for(int v : arr) {
    REQUIRE(v == fill_val);
  }
}

TEST_CASE("swap", "[ND_Array]") {
  ND_Array<int, 7, 2, 3> arr1, arr2;

  int i = 23;
  for(int &v : arr1) {
    v = i;
    i++;
  }
  for(int &v : arr2) {
    v = i;
    i++;
  }
  arr1.swap(arr2);

  i = 23;
  for(int v : arr2) {
    REQUIRE(v == i);
    i++;
  }
  for(int v : arr1) {
    REQUIRE(v == i);
    i++;
  }
}

TEST_CASE("castable", "[ND_Array]") {
  ND_Array<int, 6, 3> a1;
  ND_Array<int, 3, 6> a2 =
      static_cast<ND_Array<int, 3, 6>>(a1);
  ND_Array<int, 3, 6> &a3 = a2;
  ND_Array<int, 3, 6> &a4 =
      a1.reshape<ND_Array<int, 3, 6>>();
}

/* Compile Time List Tests */

using ND_Array_0 =
    ND_Array<int, 2, 3, 5, 7, 11, 13, 17, 19>;

static_assert(ND_Array_0::dimension() == 8,
              "Incorrect dimension");
static_assert(ND_Array_0::extent(0) == 2,
              "Incorrect extent");
static_assert(ND_Array_0::extent(1) == 3,
              "Incorrect extent");
static_assert(ND_Array_0::extent(2) == 5,
              "Incorrect extent");
static_assert(ND_Array_0::extent(3) == 7,
              "Incorrect extent");
static_assert(ND_Array_0::extent(4) == 11,
              "Incorrect extent");
static_assert(ND_Array_0::extent(5) == 13,
              "Incorrect extent");
static_assert(ND_Array_0::extent(6) == 17,
              "Incorrect extent");
static_assert(ND_Array_0::extent(7) == 19,
              "Incorrect extent");
static_assert(ND_Array_0::size() == 9699690,
              "Incorrect size");
static_assert(ND_Array_0::max_size() == 9699690,
              "Incorrect max size");
static_assert(
    ND_Array_0::empty() == false,
    "Incorrect empty - ND_Array should never be empty");

using Arr0 = ND_Array_internals::CT_Array<int, 4>;

static_assert(Arr0::len() == 1,
              "Incorrect CT Array Length");
static_assert(Arr0::sum() == 4, "Incorrect CT Array Sum");
static_assert(Arr0::product() == 4,
              "Incorrect CT Array Product");
static_assert(Arr0::slice_idx(0) == 0,
              "Incorrect 1D slice_idx");
static_assert(Arr0::slice_idx(1) == 1,
              "Incorrect 1D slice_idx");
static_assert(Arr0::slice_idx(2) == 2,
              "Incorrect 1D slice_idx");
static_assert(Arr0::slice_idx(3) == 3,
              "Incorrect 1D slice_idx");

using Arr1 = ND_Array_internals::CT_Array<int, 1, 2, 3, 4>;
static_assert(Arr1::len() == 4,
              "Incorrect CT Array Length");
static_assert(Arr1::sum() == 10, "Incorrect CT Array Sum");
static_assert(Arr1::product() == 24,
              "Incorrect CT Array Product");

using Arr2 = ND_Array_internals::CT_Array<int, 0, 1>;
static_assert(Arr1::slice_idx<Arr2>() == 12,
              "Incorrect Template Slice Index");

using Arr3 = ND_Array_internals::CT_Array<int, 0, 0, 1>;
static_assert(Arr1::slice_idx<Arr3>() == 4,
              "Incorrect Template Slice Index");

using Arr4 = ND_Array_internals::CT_Array<int, 0, 0, 0, 1>;
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

static_assert(ND_Array_internals::forward_truncate_array<
                  0, Arr1>::type::len() == 4,
              "forward_truncate_array failed");
static_assert(ND_Array_internals::forward_truncate_array<
                  0, Arr1>::type::value(0) == 1,
              "forward_truncate_array failed");
static_assert(ND_Array_internals::forward_truncate_array<
                  1, Arr1>::type::len() == 3,
              "forward_truncate_array failed");
static_assert(ND_Array_internals::forward_truncate_array<
                  1, Arr1>::type::value(0) == 2,
              "forward_truncate_array failed");
static_assert(ND_Array_internals::forward_truncate_array<
                  2, Arr1>::type::len() == 2,
              "forward_truncate_array failed");
static_assert(ND_Array_internals::forward_truncate_array<
                  2, Arr1>::type::value(0) == 3,
              "forward_truncate_array failed");
static_assert(ND_Array_internals::forward_truncate_array<
                  3, Arr1>::type::len() == 1,
              "forward_truncate_array failed");
static_assert(ND_Array_internals::forward_truncate_array<
                  3, Arr1>::type::value(0) == 4,
              "forward_truncate_array failed");
