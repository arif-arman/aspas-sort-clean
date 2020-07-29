// pch.h: This is a precompiled header file.
// Files listed below are compiled only once, improving build performance for future builds.
// This also affects IntelliSense performance, including code completion and many code browsing features.
// However, files listed here are ALL re-compiled if any one of them is updated between builds.
// Do not add files here that you will be updating frequently as this negates the performance advantage.

#ifndef PCH_H
#define PCH_H

// add headers that you want to pre-compile here
#include <Windows.h>
#include <cstdint>
#include <chrono>
using namespace std::chrono;
using hrc = high_resolution_clock;


typedef uint32_t ui;
typedef uint64_t ui64;

#define VALLOC(sz)				(Key*)VirtualAlloc(NULL, (sz), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
#define VFREE(ptr)				(VirtualFree((ptr), 0, MEM_RELEASE))
#define FOR(i,n,k)				for (ui64 (i) = 0; (i) < (n); (i)+=(k))
#define FOR_INIT(i, init, n, k)	for (ui64 (i) = (init); (i) < (n); (i) += (k))
#define PRINT_ARR(arr, n)		{ FOR((i), (n), 1) printf("%10ld ", (arr)[(i)]); printf("\n"); }
#define PRINT_DASH(n)			{ FOR(i, (n), 1) printf("-"); printf("\n"); }
#define ELAPSED_MS(st, en)		( duration_cast<duration<double, std::milli>>(en - st).count() )

 /**
     * Represents the number of elements of various types one vector can hold
     * on different platforms.
     */
enum class simd_width : std::int8_t
{
    /// SIMD width of integer types in AVX ISA (CPU)
    AVX_INT = 8,
    /// SIMD width of float types in AVX ISA (CPU)
    AVX_FLOAT = 8,
    /// SIMD width of double types in AVX ISA (CPU)
    AVX_DOUBLE = 4,
    /// SIMD width of integer types in AVX512 ISA (MIC)
    AVX512_INT = 16,
    /// SIMD width of float types in AVX512 ISA (MIC)
    AVX512_FLOAT = 16,
    /// SIMD width of double types in AVX512 ISA (MIC)
    AVX512_DOUBLE = 8
};


#endif //PCH_H
