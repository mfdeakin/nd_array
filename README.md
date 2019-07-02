# ND Array
A generic ND array with compile time dimensions
This requires C++14

## Features:

* Type safety and sanity unlike C arrays - preserve array dimensions across function calls
* Fast accessing and slicing - as performant as C arrays + type safety
* Stack allocatable - no new or malloc calls, and also doesn't introduce an extra layer of dereferencing in an array of structures of arrays
* Safe reshaping - ensures shapes are compatible at compile time
* Assert based runtime checks - safe to use on GPUs, and doesn't violate the "only pay for what you use" concept (though maybe exceptions will be fine with C++29+ with respect to this... Until then)
* Header only - not certain this is a feature (ie, compile times); but at least it's easy to snapshot in :)

## Usage:

```c++
#include "nd_array/nd_array.hpp"

ND_Array<Object_type, dim_0, dim_1, ..., dim_k> a;
ND_Array<Object_type, dim_2, dim_3, ..., dim_k> & b = a.outer_slice(idx_dim0, idx_dim1);
ND_Array<Object_type, newdim_1, newdim_2, ..., newdim_m> & c = b.template reshape<ND_Array<Object_type, newdim1, ..., newdim_m> >();

a(idx_1, idx_2, ..., idx_k) = Object_type();

auto itr = a.begin();

// Returns the iterator offset by the specified amounts in each dimension
// This may result in an invalid iterator, comparisons with begin() and end() should be made after
auto itr_offset = a.offset(itr, 5, -2, 4, 6, 7);
```

# Zip Iterator
Also included: a Zip iterator which enables iterating over multiple iterable containers of the same size.
Performance of the iterator was a major concern; tests indicate it's as good as manually iterating over all of the containers simultaneously.
Note that the zip iterator defaults to assuming all of the iterable containers are of the Random Access type, this can be changed by specifying the type as the first template parameter.
This requires C++17

Basic usage:

```c++
#include "nd_array/zip.hpp"

ND_Array<double, 2, 10, 10> vel;
for(auto [vel_x, vel_y] : zip::make_zip(vel.outer_slice(0), vel.outer_slice(1))) {
  // ...
}
// With the iterator tag specified
for(auto [vel_x, vel_y] : zip::make_zip<std::random_access_iterator_tag>(vel.outer_slice(0), vel.outer_slice(1))) {
  // ...
}
```

# Performance results

The performance comparison executable is built by default; it assumes google benchmark is installed in `/usr/local`.
To change this, specify `google_benchmark_path` when configuring with CMake.

The following results are the averrage and standard deviation (in that order) of the CPU time collected by running the performance tests 5 times:

## Gcc 9, `-O3 -fmarch=native -fstrict-aliasing`, `Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz`
```
BM_Null                                  1       0.08
BM_ND_Array_Create                   69758    4400

BM_C_Array_Iterate                   45240     380
BM_ND_Array_Iterate_Index            47020     450
BM_ND_Array_Iterate_Pointer          41116     700
BM_ND_Array_Iterate_Iterator         40658     130

BM_C_Array_Initialize               129758     690
BM_ND_Array_Initialize_Index        130136     710
BM_ND_Array_Initialize_Pointer      129483    1400
BM_ND_Array_Initialize_Iterator     129983    1300

BM_ND_Array_Iterate_2_Index         115402     810
BM_ND_Array_Iterate_2_Pointer       115964    2300
BM_ND_Array_Iterate_2_Iterator      115351    1000
BM_ND_Array_Iterate_2_Zip           116129     290

BM_ND_Array_Initialize_2_Index      259394     920
BM_ND_Array_Initialize_2_Pointer    259251     720
BM_ND_Array_Initialize_2_Iterator   260720    3400
BM_ND_Array_Initialize_2_Zip        259068    1300

BM_ND_Array_Deref_MMul             1206767   98000
BM_ND_Array_MMul                   1259991    3100
BM_C_Array_MMul                     200179   16000
BM_C_Ptr_MMul                       277302    3000
```

The GCC results indicate that performance is mostly as expected; ie equivalent to using a C array.
The major outlier to this is the performance of the matrix multiplication, GCC is able to optimize the C array but not the ND Array; some study is needed to understand why.
Future tests should be done with a less naive implementation of the matrix multiply algorithm to make certain these results hold.

## Clang 8, `-O3 -fmarch=native -fstrict-aliasing`, `Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz`
```
BM_Null                                  0      0.040
BM_ND_Array_Create                   23935   2400

BM_C_Array_Iterate                   41199    530
BM_ND_Array_Iterate_Index            41333    150
BM_ND_Array_Iterate_Pointer          32654    100
BM_ND_Array_Iterate_Iterator         32784    150

BM_C_Array_Initialize               118901  11000
BM_ND_Array_Initialize_Index        115346  16000
BM_ND_Array_Initialize_Pointer      115272  16000
BM_ND_Array_Initialize_Iterator     115270  16000

BM_ND_Array_Iterate_2_Index          23247    610
BM_ND_Array_Iterate_2_Pointer         2808     14
BM_ND_Array_Iterate_2_Iterator       32683    430
BM_ND_Array_Iterate_2_Zip            32751    450

BM_ND_Array_Initialize_2_Index      254585    530
BM_ND_Array_Initialize_2_Pointer    254403    290
BM_ND_Array_Initialize_2_Iterator   234290  27000
BM_ND_Array_Initialize_2_Zip        231021  32000

BM_ND_Array_Deref_MMul              421887  44000
BM_ND_Array_MMul                    439503   2700
BM_C_Array_MMul                     447811   6700
BM_C_Ptr_MMul                       440590   1567
```

The Clang results are somewhat odd, particularly the pointer iteration over 2 ND arrays is clearly wrong, suggesting some issue with Clang optimizing things it shouldn't.
The matrix multiplication performance is essentially identical regardless of the datastructure used. Thus, with Clang, the performance difference between a C array and an ND Array is likely negligible; though perhaps slower than a C array with GCC when performing mathematical operations.
