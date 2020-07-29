/*
* (c) 2015 Virginia Polytechnic Institute & State University (Virginia Tech)
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, version 2.1
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License, version 2.1, for more details.
*
*   You should have received a copy of the GNU General Public License
*
*/

/**
 * @file aspas_merge_avx.cpp
 *
 * Definiation of the merge function in AVX instruction sets.
 *
 */

#include <immintrin.h>
#include <type_traits>
#include <cstdint>

#include "aspas.h" 
#include "extintrin.h"

namespace aspas
{

    /**
     * Integer vector (__m256i) version:
     * This method performs the in-register merge of two sorted vectors.
     *
     * @param v0 v1 sorted vector registers
     * @return sorted data stored horizontally in the two registers
     *
     */
    /*template <typename T>
    typename std::enable_if<std::is_same<T, __m256i>::value>::type*/
    void    in_register_merge(__m256i& v0, __m256i& v1)
    {
        __m256i l1, h1, l1p, h1p, l2, h2, l2p, h2p, l3, h3, l3p, h3p, l4, h4;
        __m256i ext, ext1, ext2;
        __m128i sl1, sh1, sl2, sh2;
        __m128i max1, min1, max2, min2;

        // reverse register v1 
        ext = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(v1), _mm256_castsi256_ps(v1), _MM_SHUFFLE(0, 1, 2, 3)));
        v1 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(ext), _mm256_castsi256_ps(ext), 0x03));

        // level 1 comparison
        l1 = util::_my_mm256_min_epi32(v0, v1);
        h1 = util::_my_mm256_max_epi32(v0, v1);

        // level 2 comparison
        l1p = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(l1), _mm256_castsi256_ps(h1), 0x30));
        h1p = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(l1), _mm256_castsi256_ps(h1), 0x21));
        l2 = util::_my_mm256_min_epi32(l1p, h1p);
        h2 = util::_my_mm256_max_epi32(l1p, h1p);

        // level 3 comparison
        l2p = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(l2), _mm256_castsi256_ps(h2), _MM_SHUFFLE(3, 2, 1, 0)));
        h2p = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(l2), _mm256_castsi256_ps(h2), _MM_SHUFFLE(1, 0, 3, 2)));
        l3 = util::_my_mm256_min_epi32(l2p, h2p);
        h3 = util::_my_mm256_max_epi32(l2p, h2p);

        // level 4 comparison
        l3p = _mm256_castps_si256(_mm256_blend_ps(_mm256_castsi256_ps(l3), _mm256_castsi256_ps(h3), 0xAA));
        ext = _mm256_castps_si256(_mm256_blend_ps(_mm256_castsi256_ps(l3), _mm256_castsi256_ps(h3), 0x55));
        h3p = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(ext), _mm256_castsi256_ps(ext), _MM_SHUFFLE(2, 3, 0, 1)));
        l4 = util::_my_mm256_min_epi32(l3p, h3p);
        h4 = util::_my_mm256_max_epi32(l3p, h3p);

        // final permute/shuffle
        ext1 = _mm256_castps_si256(_mm256_unpacklo_ps(_mm256_castsi256_ps(l4), _mm256_castsi256_ps(h4)));
        ext2 = _mm256_castps_si256(_mm256_unpackhi_ps(_mm256_castsi256_ps(l4), _mm256_castsi256_ps(h4)));
        v0 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(ext1), _mm256_castsi256_ps(ext2), 0x20));
        v1 = _mm256_castps_si256(_mm256_permute2f128_ps(_mm256_castsi256_ps(ext1), _mm256_castsi256_ps(ext2), 0x31));
    }

   
    /**
     * Float vector _mm256_castsi256_pd( version:
     * This method performs the in-register merge of two sorted vectors.
     *
     * @param v0 v1 sorted vector registers
     * @return sorted data stored horizontally in the two registers
     *
     */
    /*template <typename T>
    typename std::enable_if<std::is_same<T, __m256>::value>::type*/
        void in_register_merge(__m256& v0, __m256& v1)
    {
        __m256 l1, h1, l1p, h1p, l2, h2, l2p, h2p, l3, h3, l3p, h3p, l4, h4;
        __m256 ext, ext1, ext2;
        __m128 sl1, sh1, sl2, sh2;
        __m128 max1, min1, max2, min2;

        // reverse register v1 
        ext = _mm256_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 1, 2, 3));
        v1 = _mm256_permute2f128_ps(ext, ext, 0x03);

        // level 1 comparison
        l1 = _mm256_min_ps(v0, v1);
        h1 = _mm256_max_ps(v0, v1);

        // level 2 comparison
        l1p = _mm256_permute2f128_ps(l1, h1, 0x30);
        h1p = _mm256_permute2f128_ps(l1, h1, 0x21);
        l2 = _mm256_min_ps(l1p, h1p);
        h2 = _mm256_max_ps(l1p, h1p);

        // level 3 comparison
        l2p = _mm256_shuffle_ps(l2, h2, _MM_SHUFFLE(3, 2, 1, 0));
        h2p = _mm256_shuffle_ps(l2, h2, _MM_SHUFFLE(1, 0, 3, 2));
        l3 = _mm256_min_ps(l2p, h2p);
        h3 = _mm256_max_ps(l2p, h2p);

        // level 4 comparison
        l3p = _mm256_blend_ps(l3, h3, 0xAA);
        ext = _mm256_blend_ps(l3, h3, 0x55);
        h3p = _mm256_shuffle_ps(ext, ext, _MM_SHUFFLE(2, 3, 0, 1));
        l4 = _mm256_min_ps(l3p, h3p);
        h4 = _mm256_max_ps(l3p, h3p);

        // final permute/shuffle
        ext1 = _mm256_unpacklo_ps(l4, h4);
        ext2 = _mm256_unpackhi_ps(l4, h4);
        v0 = _mm256_permute2f128_ps(ext1, ext2, 0x20);
        v1 = _mm256_permute2f128_ps(ext1, ext2, 0x31);
    }

    
    /**
     * Double vector (__m256d) version:
     * This method performs the in-register merge of two sorted vectors.
     *
     * @param v0 v1 sorted vector registers
     * @return sorted data stored horizontally in the two registers
     *
     */
   /* template <typename T>
    typename std::enable_if<std::is_same<T, __m256d>::value>::type*/
      void  in_register_merge(__m256d& v0, __m256d& v1)
    {
        __m256d ext, l1, h1, l1p, h1p, l2, h2, l2p, h2p, l3, h3;
        __m256d ext1, ext2;
        // reverse register v1 
        ext = _mm256_shuffle_pd(v1, v1, 0x5);
        v1 = _mm256_permute2f128_pd(ext, ext, 0x03);
        // level 1 comparison
        l1 = _mm256_min_pd(v0, v1);
        h1 = _mm256_max_pd(v0, v1);
        // level 2 comparison
        l1p = _mm256_permute2f128_pd(l1, h1, 0x30);
        h1p = _mm256_permute2f128_pd(l1, h1, 0x21);
        l2 = _mm256_min_pd(l1p, h1p);
        h2 = _mm256_max_pd(l1p, h1p);
        // level 3 comparison
        l2p = _mm256_shuffle_pd(l2, h2, 0x0);
        h2p = _mm256_shuffle_pd(l2, h2, 0xf);
        l3 = _mm256_min_pd(l2p, h2p);
        h3 = _mm256_max_pd(l2p, h2p);
        // final permute/shuffle
        ext1 = _mm256_unpacklo_pd(l3, h3);
        ext2 = _mm256_unpackhi_pd(l3, h3);
        v0 = _mm256_permute2f128_pd(ext1, ext2, 0x20);
        v1 = _mm256_permute2f128_pd(ext1, ext2, 0x31);
    }

   

    /*template <typename T>
    typename std::enable_if<std::is_same<T, int>::value>::type*/
    void    merge(int* inputA, uint32_t sizeA, int* inputB, uint32_t sizeB, int* output)
    {
        __m256i vec0;
        __m256i vec1;

        const uint8_t stride = (uint8_t)simd_width::AVX_INT;
        uint32_t i0 = 0;
        uint32_t i1 = 0;
        uint32_t iout = 0;
        int buffer[stride];
        uint32_t i3 = 0;

        if (sizeA >= stride && sizeB >= stride)
        {
            vec0 = _mm256_loadu_si256((__m256i*)inputA);
            vec1 = _mm256_loadu_si256((__m256i*)inputB);

            in_register_merge(vec0, vec1);

            _mm256_storeu_si256((__m256i*)(output + iout), vec0);
            i0 += stride;
            i1 += stride;
            iout += stride;

            while (i0 + stride <= sizeA && i1 + stride <= sizeB)
            {
                if (inputA[i0] <= inputB[i1])
                {
                    vec0 = _mm256_loadu_si256((__m256i*)(inputA + i0));
                    i0 += stride;
                }
                else
                {
                    vec0 = _mm256_loadu_si256((__m256i*)(inputB + i1));
                    i1 += stride;
                }
                in_register_merge(vec0, vec1);
                _mm256_storeu_si256((__m256i*)(output + iout), vec0);
                iout += stride;
            }
            while (i0 + stride <= sizeA)
            {
                if (i1 < sizeB && inputA[i0] <= inputB[i1] || i1 == sizeB)
                {
                    vec0 = _mm256_loadu_si256((__m256i*)(inputA + i0));
                    i0 += stride;
                    in_register_merge(vec0, vec1);
                    _mm256_storeu_si256((__m256i*)(output + iout), vec0);
                    iout += stride;
                }
                else
                    break;
            }
            while (i1 + stride <= sizeB)
            {
                if (i0 < sizeA && inputB[i1] <= inputA[i0] || i0 == sizeA)
                {
                    vec0 = _mm256_loadu_si256((__m256i*)(inputB + i1));
                    i1 += stride;
                    in_register_merge(vec0, vec1);
                    _mm256_storeu_si256((__m256i*)(output + iout), vec0);
                    iout += stride;
                }
                else
                    break;
            }
            _mm256_storeu_si256((__m256i*)buffer, vec1);

            while (i0 < sizeA && i1 < sizeB && i3 < stride)
            {
                if (inputA[i0] <= inputB[i1] && inputA[i0] <= buffer[i3])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else if (inputB[i1] <= inputA[i0] && inputB[i1] <= buffer[i3])
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
                else if (buffer[i3] <= inputA[i0] && buffer[i3] <= inputB[i1])
                {
                    output[iout] = buffer[i3];
                    i3++;
                    iout++;
                }
            }
            while (i0 < sizeA && i1 < sizeB)
            {
                if (inputA[i0] <= inputB[i1])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
            }
            while (i1 < sizeB && i3 < stride)
            {
                if (inputB[i1] <= buffer[i3])
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
                else
                {
                    output[iout] = buffer[i3];
                    i3++;
                    iout++;
                }

            }
            while (i0 < sizeA && i3 < stride)
            {
                if (inputA[i0] <= buffer[i3])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else
                {
                    output[iout] = buffer[i3];
                    i3++;
                    iout++;
                }
            }
            while (i0 < sizeA)
            {
                output[iout] = inputA[i0];
                i0++;
                iout++;
            }
            while (i1 < sizeB)
            {
                output[iout] = inputB[i1];
                i1++;
                iout++;
            }
            while (i3 < stride)
            {
                output[iout] = buffer[i3];
                i3++;
                iout++;
            }

        }
        else
        {
            while (i0 < sizeA && i1 < sizeB)
            {
                if (inputA[i0] <= inputB[i1])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
            }
            while (i0 < sizeA)
            {
                output[iout] = inputA[i0];
                i0++;
                iout++;
            }
            while (i1 < sizeB)
            {
                output[iout] = inputB[i1];
                i1++;
                iout++;
            }
        }
    }

    

    /*template <typename T>
    typename std::enable_if<std::is_same<T, float>::value>::type*/
    void    merge(float* inputA, uint32_t sizeA, float* inputB, uint32_t sizeB, float* output)
    {
        __m256 vec0;
        __m256 vec1;

        const uint8_t stride = (uint8_t)simd_width::AVX_FLOAT;
        uint32_t i0 = 0;
        uint32_t i1 = 0;
        uint32_t iout = 0;
        float buffer[stride];
        uint32_t i3 = 0;

        if (sizeA >= stride && sizeB >= stride)
        {
            vec0 = _mm256_loadu_ps(inputA);
            vec1 = _mm256_loadu_ps(inputB);

            in_register_merge(vec0, vec1);

            _mm256_storeu_ps((output + iout), vec0);
            i0 += stride;
            i1 += stride;
            iout += stride;

            while (i0 + stride <= sizeA && i1 + stride <= sizeB)
            {
                if (inputA[i0] <= inputB[i1])
                {
                    vec0 = _mm256_loadu_ps((inputA + i0));
                    i0 += stride;
                }
                else
                {
                    vec0 = _mm256_loadu_ps((inputB + i1));
                    i1 += stride;
                }
                in_register_merge(vec0, vec1);
                _mm256_storeu_ps((output + iout), vec0);
                iout += stride;
            }
            while (i0 + stride <= sizeA)
            {
                if (i1 < sizeB && inputA[i0] <= inputB[i1] || i1 == sizeB)
                {
                    vec0 = _mm256_loadu_ps((inputA + i0));
                    i0 += stride;
                    in_register_merge(vec0, vec1);
                    _mm256_storeu_ps((output + iout), vec0);
                    iout += stride;
                }
                else
                    break;
            }
            while (i1 + stride <= sizeB)
            {
                if (i0 < sizeA && inputB[i1] <= inputA[i0] || i0 == sizeA)
                {
                    vec0 = _mm256_loadu_ps((inputB + i1));
                    i1 += stride;
                    in_register_merge(vec0, vec1);
                    _mm256_storeu_ps((output + iout), vec0);
                    iout += stride;
                }
                else
                    break;
            }
            _mm256_storeu_ps(buffer, vec1);

            while (i0 < sizeA && i1 < sizeB && i3 < stride)
            {
                if (inputA[i0] <= inputB[i1] && inputA[i0] <= buffer[i3])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else if (inputB[i1] <= inputA[i0] && inputB[i1] <= buffer[i3])
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
                else if (buffer[i3] <= inputA[i0] && buffer[i3] <= inputB[i1])
                {
                    output[iout] = buffer[i3];
                    i3++;
                    iout++;
                }
            }
            while (i0 < sizeA && i1 < sizeB)
            {
                if (inputA[i0] <= inputB[i1])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
            }
            while (i1 < sizeB && i3 < stride)
            {
                if (inputB[i1] <= buffer[i3])
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
                else
                {
                    output[iout] = buffer[i3];
                    i3++;
                    iout++;
                }

            }
            while (i0 < sizeA && i3 < stride)
            {
                if (inputA[i0] <= buffer[i3])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else
                {
                    output[iout] = buffer[i3];
                    i3++;
                    iout++;
                }
            }
            while (i0 < sizeA)
            {
                output[iout] = inputA[i0];
                i0++;
                iout++;
            }
            while (i1 < sizeB)
            {
                output[iout] = inputB[i1];
                i1++;
                iout++;
            }
            while (i3 < stride)
            {
                output[iout] = buffer[i3];
                i3++;
                iout++;
            }

        }
        else
        {
            while (i0 < sizeA && i1 < sizeB)
            {
                if (inputA[i0] <= inputB[i1])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
            }
            while (i0 < sizeA)
            {
                output[iout] = inputA[i0];
                i0++;
                iout++;
            }
            while (i1 < sizeB)
            {
                output[iout] = inputB[i1];
                i1++;
                iout++;
            }
        }
    }

    

    /*template <typename T>
    typename std::enable_if<std::is_same<T, double>::value>::type*/
    void    merge(double* inputA, uint32_t sizeA, double* inputB, uint32_t sizeB, double* output)
    {
        __m256d vec0;
        __m256d vec1;

        const uint8_t stride = (uint8_t)simd_width::AVX_DOUBLE;
        uint32_t i0 = 0;
        uint32_t i1 = 0;
        uint32_t iout = 0;
        double buffer[stride];
        uint32_t i3 = 0;

        if (sizeA >= stride && sizeB >= stride)
        {
            vec0 = _mm256_loadu_pd(inputA);
            vec1 = _mm256_loadu_pd(inputB);

            in_register_merge(vec0, vec1);

            _mm256_storeu_pd((output + iout), vec0);
            i0 += stride;
            i1 += stride;
            iout += stride;

            while (i0 + stride <= sizeA && i1 + stride <= sizeB)
            {
                if (inputA[i0] <= inputB[i1])
                {
                    vec0 = _mm256_loadu_pd((inputA + i0));
                    i0 += stride;
                }
                else
                {
                    vec0 = _mm256_loadu_pd((inputB + i1));
                    i1 += stride;
                }
                in_register_merge(vec0, vec1);
                _mm256_storeu_pd((output + iout), vec0);
                iout += stride;
            }
            while (i0 + stride <= sizeA)
            {
                if (i1 < sizeB && inputA[i0] <= inputB[i1] || i1 == sizeB)
                {
                    vec0 = _mm256_loadu_pd((inputA + i0));
                    i0 += stride;
                    in_register_merge(vec0, vec1);
                    _mm256_storeu_pd((output + iout), vec0);
                    iout += stride;
                }
                else
                    break;
            }
            while (i1 + stride <= sizeB)
            {
                if (i0 < sizeA && inputB[i1] <= inputA[i0] || i0 == sizeA)
                {
                    vec0 = _mm256_loadu_pd((inputB + i1));
                    i1 += stride;
                    in_register_merge(vec0, vec1);
                    _mm256_storeu_pd((output + iout), vec0);
                    iout += stride;
                }
                else
                    break;
            }
            _mm256_storeu_pd(buffer, vec1);

            while (i0 < sizeA && i1 < sizeB && i3 < stride)
            {
                if (inputA[i0] <= inputB[i1] && inputA[i0] <= buffer[i3])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else if (inputB[i1] <= inputA[i0] && inputB[i1] <= buffer[i3])
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
                else if (buffer[i3] <= inputA[i0] && buffer[i3] <= inputB[i1])
                {
                    output[iout] = buffer[i3];
                    i3++;
                    iout++;
                }
            }
            while (i0 < sizeA && i1 < sizeB)
            {
                if (inputA[i0] <= inputB[i1])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
            }
            while (i1 < sizeB && i3 < stride)
            {
                if (inputB[i1] <= buffer[i3])
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
                else
                {
                    output[iout] = buffer[i3];
                    i3++;
                    iout++;
                }

            }
            while (i0 < sizeA && i3 < stride)
            {
                if (inputA[i0] <= buffer[i3])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else
                {
                    output[iout] = buffer[i3];
                    i3++;
                    iout++;
                }
            }
            while (i0 < sizeA)
            {
                output[iout] = inputA[i0];
                i0++;
                iout++;
            }
            while (i1 < sizeB)
            {
                output[iout] = inputB[i1];
                i1++;
                iout++;
            }
            while (i3 < stride)
            {
                output[iout] = buffer[i3];
                i3++;
                iout++;
            }

        }
        else
        {
            while (i0 < sizeA && i1 < sizeB)
            {
                if (inputA[i0] <= inputB[i1])
                {
                    output[iout] = inputA[i0];
                    i0++;
                    iout++;
                }
                else
                {
                    output[iout] = inputB[i1];
                    i1++;
                    iout++;
                }
            }
            while (i0 < sizeA)
            {
                output[iout] = inputA[i0];
                i0++;
                iout++;
            }
            while (i1 < sizeB)
            {
                output[iout] = inputB[i1];
                i1++;
                iout++;
            }
        }
    }

    
} // end namespace aspas
