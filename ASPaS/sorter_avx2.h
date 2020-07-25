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

#include "pch.h"
#include <immintrin.h> 
#include <type_traits> 
#include <cstdint>

#include "extintrin.h"

namespace aspas
{

    namespace internal
    {

        /**
         * This method compares two values from index i and j.
         * If value at i is larger than j, do the swap.
         *
         * @param a data array
         * @param i first index
         * @param j second index
         * @return the values in i and j are sorted
         *
         */
        template <typename T>
        void swap(T* a, uint32_t i, uint32_t j)
        {
            if (a[i] > a[j])
            {
                T tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }

       
        /**
         * Float vector version (__m256):
         * This method performs the in-register sort. The sorted elements
         * are stored vertically accross the registers.
         *
         * @param v0-v7 vector data registers
         * @return sorted data stored vertically among the registers
         *
         */
       /* template <typename T>
        typename std::enable_if<std::is_same<T, __m256>::value>::type*/
        void    in_register_sort(__m256& v0, __m256& v1, __m256& v2, __m256& v3,
                __m256& v4, __m256& v5, __m256& v6, __m256& v7)
        {
            __m256 l, h;

            /** odd-even sorting network */
            /** step 1 */
            l = _mm256_min_ps(v0, v1);
            h = _mm256_max_ps(v0, v1); v0 = l; v1 = h;
            l = _mm256_min_ps(v2, v3);
            h = _mm256_max_ps(v2, v3); v2 = l; v3 = h;
            l = _mm256_min_ps(v4, v5);
            h = _mm256_max_ps(v4, v5); v4 = l; v5 = h;
            l = _mm256_min_ps(v6, v7);
            h = _mm256_max_ps(v6, v7); v6 = l; v7 = h;
            /** step 2 */
            l = _mm256_min_ps(v0, v2);
            h = _mm256_max_ps(v0, v2); v0 = l; v2 = h;
            l = _mm256_min_ps(v1, v3);
            h = _mm256_max_ps(v1, v3); v1 = l; v3 = h;
            l = _mm256_min_ps(v4, v6);
            h = _mm256_max_ps(v4, v6); v4 = l; v6 = h;
            l = _mm256_min_ps(v5, v7);
            h = _mm256_max_ps(v5, v7); v5 = l; v7 = h;
            /** step 3 */
            l = _mm256_min_ps(v1, v2);
            h = _mm256_max_ps(v1, v2); v1 = l; v2 = h;
            l = _mm256_min_ps(v5, v6);
            h = _mm256_max_ps(v5, v6); v5 = l; v6 = h;
            /** step 4 */
            l = _mm256_min_ps(v0, v4);
            h = _mm256_max_ps(v0, v4); v0 = l; v4 = h;
            l = _mm256_min_ps(v1, v5);
            h = _mm256_max_ps(v1, v5); v1 = l; v5 = h;
            l = _mm256_min_ps(v2, v6);
            h = _mm256_max_ps(v2, v6); v2 = l; v6 = h;
            l = _mm256_min_ps(v3, v7);
            h = _mm256_max_ps(v3, v7); v3 = l; v7 = h;
            /** step 5 */
            l = _mm256_min_ps(v2, v4);
            h = _mm256_max_ps(v2, v4); v2 = l; v4 = h;
            l = _mm256_min_ps(v3, v5);
            h = _mm256_max_ps(v3, v5); v3 = l; v5 = h;
            /** step 6 */
            l = _mm256_min_ps(v1, v2);
            h = _mm256_max_ps(v1, v2); v1 = l; v2 = h;
            l = _mm256_min_ps(v3, v4);
            h = _mm256_max_ps(v3, v4); v3 = l; v4 = h;
            l = _mm256_min_ps(v5, v6);
            h = _mm256_max_ps(v5, v6); v5 = l; v6 = h;
        }

        /**
         * Integer vector version (__m256i):
         * This method performs the in-register sort. The sorted elements
         * are stored vertically accross the registers.
         *
         * @param v0-v7 vector data registers
         * @return sorted data stored vertically among the registers
         *
         */
        /*template <typename T>
        typename std::enable_if<std::is_same<T, __m256i>::value>::type*/
        void    in_register_sort(__m256i& v0, __m256i& v1, __m256i& v2, __m256i& v3,
                __m256i& v4, __m256i& v5, __m256i& v6, __m256i& v7)
        {
            __m256i l, h;

            /** odd-even sorting network */
            /** step 1 */
            l = _mm256_min_epi32(v0, v1);
            h = _mm256_max_epi32(v0, v1); v0 = l; v1 = h;
            l = _mm256_min_epi32(v2, v3);
            h = _mm256_max_epi32(v2, v3); v2 = l; v3 = h;
            l = _mm256_min_epi32(v4, v5);
            h = _mm256_max_epi32(v4, v5); v4 = l; v5 = h;
            l = _mm256_min_epi32(v6, v7);
            h = _mm256_max_epi32(v6, v7); v6 = l; v7 = h;
            /** step 2 */
            l = _mm256_min_epi32(v0, v2);
            h = _mm256_max_epi32(v0, v2); v0 = l; v2 = h;
            l = _mm256_min_epi32(v1, v3);
            h = _mm256_max_epi32(v1, v3); v1 = l; v3 = h;
            l = _mm256_min_epi32(v4, v6);
            h = _mm256_max_epi32(v4, v6); v4 = l; v6 = h;
            l = _mm256_min_epi32(v5, v7);
            h = _mm256_max_epi32(v5, v7); v5 = l; v7 = h;
            /** step 3 */
            l = _mm256_min_epi32(v1, v2);
            h = _mm256_max_epi32(v1, v2); v1 = l; v2 = h;
            l = _mm256_min_epi32(v5, v6);
            h = _mm256_max_epi32(v5, v6); v5 = l; v6 = h;
            /** step 4 */
            l = _mm256_min_epi32(v0, v4);
            h = _mm256_max_epi32(v0, v4); v0 = l; v4 = h;
            l = _mm256_min_epi32(v1, v5);
            h = _mm256_max_epi32(v1, v5); v1 = l; v5 = h;
            l = _mm256_min_epi32(v2, v6);
            h = _mm256_max_epi32(v2, v6); v2 = l; v6 = h;
            l = _mm256_min_epi32(v3, v7);
            h = _mm256_max_epi32(v3, v7); v3 = l; v7 = h;
            /** step 5 */
            l = _mm256_min_epi32(v2, v4);
            h = _mm256_max_epi32(v2, v4); v2 = l; v4 = h;
            l = _mm256_min_epi32(v3, v5);
            h = _mm256_max_epi32(v3, v5); v3 = l; v5 = h;
            /** step 6 */
            l = _mm256_min_epi32(v1, v2);
            h = _mm256_max_epi32(v1, v2); v1 = l; v2 = h;
            l = _mm256_min_epi32(v3, v4);
            h = _mm256_max_epi32(v3, v4); v3 = l; v4 = h;
            l = _mm256_min_epi32(v5, v6);
            h = _mm256_max_epi32(v5, v6); v5 = l; v6 = h;
        }

       
        /**
         * Double vector version (__m256d):
         * This method performs the in-register sort. The sorted elements
         * are stored vertically accross the registers.
         *
         * @param v0-v7 vector data registers
         * @return sorted data stored vertically among the registers
         *
         */
        /*template <typename T>
        typename std::enable_if<std::is_same<T, __m256d>::value>::type*/
        void    in_register_sort(__m256d& v0, __m256d& v1, __m256d& v2, __m256d& v3)
        {
            __m256d l, h;
            /** odd-even sorting network */
            /** step 1 */
            l = _mm256_min_pd(v0, v1);
            h = _mm256_max_pd(v0, v1); v0 = l; v1 = h;
            l = _mm256_min_pd(v2, v3);
            h = _mm256_max_pd(v2, v3); v2 = l; v3 = h;
            /** step 2 */
            l = _mm256_min_pd(v0, v2);
            h = _mm256_max_pd(v0, v2); v0 = l; v2 = h;
            l = _mm256_min_pd(v1, v3);
            h = _mm256_max_pd(v1, v3); v1 = l; v3 = h;
            /** step 3 */
            l = _mm256_min_pd(v1, v2);
            h = _mm256_max_pd(v1, v2); v1 = l; v2 = h;
        }

       
        /**
         * Float vector version (__m256):
         * This method performs the in-register transpose. The sorted elements
         * are stored horizontally within the registers.
         *
         * @param v0-v7 vector data registers
         * @return sorted data stored horizontally in the registers
         *
         */
       /* template <typename T>
        typename std::enable_if<std::is_same<T, __m256>::value>::type*/
        void    in_register_transpose(__m256& v0, __m256& v1, __m256& v2, __m256& v3,
                __m256& v4, __m256& v5, __m256& v6, __m256& v7)
        {
            __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
            __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
            __t0 = _mm256_unpacklo_ps(v0, v1);
            __t1 = _mm256_unpackhi_ps(v0, v1);
            __t2 = _mm256_unpacklo_ps(v2, v3);
            __t3 = _mm256_unpackhi_ps(v2, v3);
            __t4 = _mm256_unpacklo_ps(v4, v5);
            __t5 = _mm256_unpackhi_ps(v4, v5);
            __t6 = _mm256_unpacklo_ps(v6, v7);
            __t7 = _mm256_unpackhi_ps(v6, v7);
            __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
            __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
            __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
            __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
            __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
            __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
            __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
            __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
            v0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
            v1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
            v2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
            v3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
            v4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
            v5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
            v6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
            v7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
        }

       

        __m256d _my_unpacklo_pd(__m256i vpl0, __m256i vpl1)
        {
            __m256d tpl0 = _mm256_permute2f128_pd(_mm256_castsi256_pd(vpl0), _mm256_castsi256_pd(vpl1), 0x20);
            __m256d trl0 = _mm256_permute_pd(_mm256_permute2f128_pd(tpl0, tpl0, 0x21), 0x5);
            tpl0 = _mm256_blend_pd(tpl0, trl0, 0x6);
            return tpl0;
        }

        __m256d _my_unpackhi_pd(__m256i vpl0, __m256i vpl1)
        {
            __m256d tpl1 = _mm256_permute2f128_pd(_mm256_castsi256_pd(vpl0), _mm256_castsi256_pd(vpl1), 0x31);
            __m256d trl1 = _mm256_permute_pd(_mm256_permute2f128_pd(tpl1, tpl1, 0x21), 0x5);
            tpl1 = _mm256_blend_pd(tpl1, trl1, 0x6);
            return tpl1;
        }

        
        /**
         * Integer vector version (__m256i):
         * This method performs the in-register transpose. The sorted elements
         * are stored horizontally within the registers.
         *
         * @param v0-v7 vector data registers
         * @return sorted data stored horizontally in the registers
         *
         */
        /*template <typename T>
        typename std::enable_if<std::is_same<T, __m256i>::value>::type*/
        void    in_register_transpose(__m256i& v0, __m256i& v1, __m256i& v2, __m256i& v3,
                __m256i& v4, __m256i& v5, __m256i& v6, __m256i& v7)
        {
            __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
            __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
            __t0 = _mm256_unpacklo_ps(_mm256_castsi256_ps(v0), _mm256_castsi256_ps(v1));
            __t1 = _mm256_unpackhi_ps(_mm256_castsi256_ps(v0), _mm256_castsi256_ps(v1));
            __t2 = _mm256_unpacklo_ps(_mm256_castsi256_ps(v2), _mm256_castsi256_ps(v3));
            __t3 = _mm256_unpackhi_ps(_mm256_castsi256_ps(v2), _mm256_castsi256_ps(v3));
            __t4 = _mm256_unpacklo_ps(_mm256_castsi256_ps(v4), _mm256_castsi256_ps(v5));
            __t5 = _mm256_unpackhi_ps(_mm256_castsi256_ps(v4), _mm256_castsi256_ps(v5));
            __t6 = _mm256_unpacklo_ps(_mm256_castsi256_ps(v6), _mm256_castsi256_ps(v7));
            __t7 = _mm256_unpackhi_ps(_mm256_castsi256_ps(v6), _mm256_castsi256_ps(v7));
            __tt0 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(1, 0, 1, 0));
            __tt1 = _mm256_shuffle_ps(__t0, __t2, _MM_SHUFFLE(3, 2, 3, 2));
            __tt2 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(1, 0, 1, 0));
            __tt3 = _mm256_shuffle_ps(__t1, __t3, _MM_SHUFFLE(3, 2, 3, 2));
            __tt4 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(1, 0, 1, 0));
            __tt5 = _mm256_shuffle_ps(__t4, __t6, _MM_SHUFFLE(3, 2, 3, 2));
            __tt6 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(1, 0, 1, 0));
            __tt7 = _mm256_shuffle_ps(__t5, __t7, _MM_SHUFFLE(3, 2, 3, 2));
            v0 = _mm256_castps_si256(_mm256_permute2f128_ps(__tt0, __tt4, 0x20));
            v1 = _mm256_castps_si256(_mm256_permute2f128_ps(__tt1, __tt5, 0x20));
            v2 = _mm256_castps_si256(_mm256_permute2f128_ps(__tt2, __tt6, 0x20));
            v3 = _mm256_castps_si256(_mm256_permute2f128_ps(__tt3, __tt7, 0x20));
            v4 = _mm256_castps_si256(_mm256_permute2f128_ps(__tt0, __tt4, 0x31));
            v5 = _mm256_castps_si256(_mm256_permute2f128_ps(__tt1, __tt5, 0x31));
            v6 = _mm256_castps_si256(_mm256_permute2f128_ps(__tt2, __tt6, 0x31));
            v7 = _mm256_castps_si256(_mm256_permute2f128_ps(__tt3, __tt7, 0x31));
        }

       
        /**
         * Double vector version (__m256d):
         * This method performs the in-register transpose. The sorted elements
         * are stored horizontally within the registers.
         *
         * @param v0-v7 vector data registers
         * @return sorted data stored horizontally in the registers
         *
         */
       /* template <typename T>
        typename std::enable_if<std::is_same<T, __m256d>::value>::type*/
        void    in_register_transpose(__m256d& v0, __m256d& v1, __m256d& v2, __m256d& v3)
        {
            __m256d __t0, __t1, __t2, __t3;
            __t0 = _mm256_unpacklo_pd(v0, v1);
            __t1 = _mm256_unpackhi_pd(v0, v1);
            __t2 = _mm256_unpacklo_pd(v2, v3);
            __t3 = _mm256_unpackhi_pd(v2, v3);
            v0 = _mm256_permute2f128_pd(__t0, __t2, 0x20);
            v1 = _mm256_permute2f128_pd(__t1, __t3, 0x20);
            v2 = _mm256_permute2f128_pd(__t0, __t2, 0x31);
            v3 = _mm256_permute2f128_pd(__t1, __t3, 0x31);
        }

       

        /*template <typename T>
        typename std::enable_if<std::is_same<T, int>::value>::type*/
        void    sorter(int*& orig, uint32_t size)
        {
            uint32_t i, j;
            __m256i vec0;
            __m256i vec1;
            __m256i vec2;
            __m256i vec3;
            __m256i vec4;
            __m256i vec5;
            __m256i vec6;
            __m256i vec7;
            uint8_t stride = (uint8_t)simd_width::AVX_INT;
            for (i = 0; i + stride * stride - 1 < size; i += stride * stride) {
                vec0 = _mm256_loadu_si256((__m256i*)(orig + i + 0 * stride));
                vec1 = _mm256_loadu_si256((__m256i*)(orig + i + 1 * stride));
                vec2 = _mm256_loadu_si256((__m256i*)(orig + i + 2 * stride));
                vec3 = _mm256_loadu_si256((__m256i*)(orig + i + 3 * stride));
                vec4 = _mm256_loadu_si256((__m256i*)(orig + i + 4 * stride));
                vec5 = _mm256_loadu_si256((__m256i*)(orig + i + 5 * stride));
                vec6 = _mm256_loadu_si256((__m256i*)(orig + i + 6 * stride));
                vec7 = _mm256_loadu_si256((__m256i*)(orig + i + 7 * stride));

                in_register_sort(vec0, vec1, vec2, vec3,
                    vec4, vec5, vec6, vec7);

                in_register_transpose(vec0, vec1, vec2, vec3,
                    vec4, vec5, vec6, vec7);

                _mm256_storeu_si256((__m256i*)(orig + i + 0 * stride), vec0);
                _mm256_storeu_si256((__m256i*)(orig + i + 1 * stride), vec1);
                _mm256_storeu_si256((__m256i*)(orig + i + 2 * stride), vec2);
                _mm256_storeu_si256((__m256i*)(orig + i + 3 * stride), vec3);
                _mm256_storeu_si256((__m256i*)(orig + i + 4 * stride), vec4);
                _mm256_storeu_si256((__m256i*)(orig + i + 5 * stride), vec5);
                _mm256_storeu_si256((__m256i*)(orig + i + 6 * stride), vec6);
                _mm256_storeu_si256((__m256i*)(orig + i + 7 * stride), vec7);
            }

            // Batcher odd-even mergesort
            for (/*cont'd*/; i + stride - 1 < size; i += stride)
            {
                swap(orig, i, i + 1);
                swap(orig, i + 2, i + 3);
                swap(orig, i + 4, i + 5);
                swap(orig, i + 6, i + 7);
                swap(orig, i, i + 2);
                swap(orig, i + 1, i + 3);
                swap(orig, i + 4, i + 6);
                swap(orig, i + 5, i + 7);
                swap(orig, i + 1, i + 2);
                swap(orig, i + 5, i + 6);
                swap(orig, i, i + 4);
                swap(orig, i + 1, i + 5);
                swap(orig, i + 2, i + 6);
                swap(orig, i + 3, i + 7);
                swap(orig, i + 2, i + 4);
                swap(orig, i + 3, i + 5);
                swap(orig, i + 1, i + 2);
                swap(orig, i + 3, i + 4);
                swap(orig, i + 5, i + 6);
            }

            // bubble sort 
            for (/*cont'd*/; i < size; i++)
            {
                for (j = i + 1; j < size; j++)
                {
                    swap(orig, i, j);
                }
            }
        }

        

        /*template <typename T>
        typename std::enable_if<std::is_same<T, float>::value>::type*/
        void    sorter(float*& orig, uint32_t size)
        {
            uint32_t i, j;
            __m256 vec0;
            __m256 vec1;
            __m256 vec2;
            __m256 vec3;
            __m256 vec4;
            __m256 vec5;
            __m256 vec6;
            __m256 vec7;
            uint8_t stride = (uint8_t)simd_width::AVX_FLOAT;
            for (i = 0; i + stride * stride - 1 < size; i += stride * stride) {
                vec0 = _mm256_loadu_ps((orig + i + 0 * stride));
                vec1 = _mm256_loadu_ps((orig + i + 1 * stride));
                vec2 = _mm256_loadu_ps((orig + i + 2 * stride));
                vec3 = _mm256_loadu_ps((orig + i + 3 * stride));
                vec4 = _mm256_loadu_ps((orig + i + 4 * stride));
                vec5 = _mm256_loadu_ps((orig + i + 5 * stride));
                vec6 = _mm256_loadu_ps((orig + i + 6 * stride));
                vec7 = _mm256_loadu_ps((orig + i + 7 * stride));

                in_register_sort(vec0, vec1, vec2, vec3,
                    vec4, vec5, vec6, vec7);

                in_register_transpose(vec0, vec1, vec2, vec3,
                    vec4, vec5, vec6, vec7);

                _mm256_storeu_ps((orig + i + 0 * stride), vec0);
                _mm256_storeu_ps((orig + i + 1 * stride), vec1);
                _mm256_storeu_ps((orig + i + 2 * stride), vec2);
                _mm256_storeu_ps((orig + i + 3 * stride), vec3);
                _mm256_storeu_ps((orig + i + 4 * stride), vec4);
                _mm256_storeu_ps((orig + i + 5 * stride), vec5);
                _mm256_storeu_ps((orig + i + 6 * stride), vec6);
                _mm256_storeu_ps((orig + i + 7 * stride), vec7);
            }

            // Batcher odd-even mergesort
            for (/*cont'd*/; i + stride - 1 < size; i += stride)
            {
                swap(orig, i, i + 1);
                swap(orig, i + 2, i + 3);
                swap(orig, i + 4, i + 5);
                swap(orig, i + 6, i + 7);
                swap(orig, i, i + 2);
                swap(orig, i + 1, i + 3);
                swap(orig, i + 4, i + 6);
                swap(orig, i + 5, i + 7);
                swap(orig, i + 1, i + 2);
                swap(orig, i + 5, i + 6);
                swap(orig, i, i + 4);
                swap(orig, i + 1, i + 5);
                swap(orig, i + 2, i + 6);
                swap(orig, i + 3, i + 7);
                swap(orig, i + 2, i + 4);
                swap(orig, i + 3, i + 5);
                swap(orig, i + 1, i + 2);
                swap(orig, i + 3, i + 4);
                swap(orig, i + 5, i + 6);
            }

            // bubble sort 
            for (/*cont'd*/; i < size; i++)
            {
                for (j = i + 1; j < size; j++)
                {
                    swap(orig, i, j);
                }
            }
        }

       
        /*template <typename T>
        typename std::enable_if<std::is_same<T, double>::value>::type*/
        void    sorter(double*& orig, uint32_t size)
        {
            uint32_t i, j;
            __m256d vec0;
            __m256d vec1;
            __m256d vec2;
            __m256d vec3;
            uint8_t stride = (uint8_t)simd_width::AVX_DOUBLE;
            for (i = 0; i + stride * stride - 1 < size; i += stride * stride) {
                vec0 = _mm256_loadu_pd((orig + i + 0 * stride));
                vec1 = _mm256_loadu_pd((orig + i + 1 * stride));
                vec2 = _mm256_loadu_pd((orig + i + 2 * stride));
                vec3 = _mm256_loadu_pd((orig + i + 3 * stride));

                in_register_sort(vec0, vec1, vec2, vec3);

                in_register_transpose(vec0, vec1, vec2, vec3);

                _mm256_storeu_pd((orig + i + 0 * stride), vec0);
                _mm256_storeu_pd((orig + i + 1 * stride), vec1);
                _mm256_storeu_pd((orig + i + 2 * stride), vec2);
                _mm256_storeu_pd((orig + i + 3 * stride), vec3);
            }

            // Batcher odd-even mergesort
            for (/*cont'd*/; i + stride - 1 < size; i += stride)
            {
                swap(orig, i, i + 1);
                swap(orig, i + 2, i + 3);
                swap(orig, i, i + 2);
                swap(orig, i + 1, i + 3);
                swap(orig, i + 1, i + 2);
            }

            // bubble sort 
            for (/*cont'd*/; i < size; i++)
            {
                for (j = i + 1; j < size; j++)
                {
                    swap(orig, i, j);
                }
            }
        }

       

    } // end namespace internal

} // end namespace aspas
