
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "nd_array.hpp"

TEST_CASE("get and set", "[ND_Array]") {
  ND_Array<int, 2, 3, 5> arr;
  int count = 1;
  for(int i = 0; i < 2; ++i) {
    for(int j = 0; j < 3; ++j) {
      for(int k = 0; k < 5; ++k) {
        arr(i, j, k) = count;
        count++;
      }
    }
  }
  count = 1;
  for(int i = 0; i < 2; ++i) {
    for(int j = 0; j < 3; ++j) {
      for(int k = 0; k < 5; ++k) {
        REQUIRE(arr(i, j, k) == count);
        count++;
      }
    }
  }
}
