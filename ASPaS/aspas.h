#pragma once
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

#include <cstdint>
#include <thread>
//#include <unistd.h>
#define __AVX2__

#include "sorter.h"
#include "merger.h"


/**
 * This namespace contains parallel sorting functions that operate on various
 * primitive data types (i.e. int, float, double) on both of AVX and AVX512 ISA.
 */
 //! Auto-generated SIMD parallel sorting tools interfaces and implementations. 
namespace aspas
{

    /// thread_num is set to be the value of the logical cores of the current platform.
    uint32_t thread_num = 16;       //sysconf(_SC_NPROCESSORS_ONLN);

   


    /**
     * This method sorts the given input array. Currently the input array can be of the type
     * of int, float, and double.
     *
     * @param array the pointer to the first element of the input array
     * @param size the size of the input array
     * @return the sorted elements are stored in the pointer of array
     *
     */
     //! This method sorts the given input array.
    template <class T>
    __forceinline void sort(T* array, uint32_t size)
    {
        internal::sorter(array, size);
        internal::merger(array, size);
    }

  
    /**
     * Integer version <br>
     * This method merges two sorted input arrays pointed by inputA and inputB respectively.
     *
     * @param inputA the first sorted array
     * @param sizeA the size of the first array
     * @param inputB the second sorted array
     * @param sizeB the size of the second array
     * @param output the saving target of the merged array
     * @return
     *
     */
     //! This method merges two sorted input arrays into one.
    /*template <typename T>
    typename std::enable_if<std::is_same<T, int>::value>::type*/
        void merge(int* inputA, uint32_t sizeA, int* inputB, uint32_t sizeB, int* output);

    /**
     * Float version <br>
     * This method merges two sorted input arrays pointed by inputA and inputB respectively.
     *
     * @param inputA the first sorted array
     * @param sizeA the size of the first array
     * @param inputB the second sorted array
     * @param sizeB the size of the second array
     * @param output the saving target of the merged array
     * @return
     *
     */
     //! This method merges two sorted input arrays into one.
    /*template <typename T>
    typename std::enable_if<std::is_same<T, float>::value>::type*/
        void merge(float* inputA, uint32_t sizeA, float* inputB, uint32_t sizeB, float* output);

    /**
     * Double version <br>
     * This method merges two sorted input arrays pointed by inputA and inputB respectively.
     *
     * @param inputA the first sorted array
     * @param sizeA the size of the first array
     * @param inputB the second sorted array
     * @param sizeB the size of the second array
     * @param output the saving target of the merged array
     * @return
     *
     */
     //! This method merges two sorted input arrays into one.
    /*template <typename T>
    typename std::enable_if<std::is_same<T, double>::value>::type*/
        void merge(double* inputA, uint32_t sizeA, double* inputB, uint32_t sizeB, double* output);



    //////////////// parallel sort stuff
        template<class T>
        struct args_t
        {
            uint32_t start;
            uint32_t end;
            uint32_t tid;
            T* input;
        };

        template<class T>
        struct args_split
        {
            uint32_t start;
            uint32_t mid;
            uint32_t end;
            uint32_t tid;
            uint32_t factor;
            T* input;
            T* inputa;
            int* startA;
            int* startB;
        };

        template<class T>
        void thread_sort_kernel(void* arguments)
        {
            args_t<T>* args = (args_t<T>*)arguments;
            uint32_t start = args->start;
            uint32_t end = args->end;

            //SetThreadAffinityMask(GetCurrentThread(), 1LU << (args->tid << 1));

            sort<T>((args->input) + start, end - start);
            /*
            // check for sort correctness
            for (uint32_t i = start; i < end - 1; ++i) {
                if (args->input[i + 1] < args->input[i]) {
                    printf("Error @ %lu\n", i);
                    break;
                }
            }*/
        }

        template<class T>
        void thread_merge_kernel(void* arguments)
        {
            args_split<T>* args = (args_split<T>*)arguments;

            //SetThreadAffinityMask(GetCurrentThread(), 1LU << (args->tid << 1));

            uint32_t i, j, k;
            i = args->start;
            j = args->mid;
            k = args->end;
            int SIZEA = j - i;
            int SIZEB = k - j;
            int SIZEC = k - i;
            int tid = args->tid;
            int factor = args->factor;
            int* startA = args->startA;
            int* startB = args->startB;

            long target = (long)SIZEC * tid / factor;
            int offset = (int)target;
            // std::merge(args->input+startA[tid], args->input+startA[tid+1], args->input+SIZEA+startB[tid], args->input+SIZEA+startB[tid+1], args->inputa+offset);
            merge(args->input + startA[tid], startA[tid + 1] - startA[tid], args->input + SIZEA + startB[tid], startB[tid + 1] - startB[tid], args->inputa + offset/*SIZEC/factor*tid*/);
        }


        template<typename iT>
        void find_kth2(iT* key_left_start,
            iT* key_left_end,
            iT* key_right_start,
            iT* key_right_end,
            int Kth,
            int& key_left_idx_i,
            int& key_right_idx_i)
        {
            int k = Kth;
            int key_right_top = k < (key_right_end - key_right_start) ? k : (key_right_end - key_right_start);
            int key_right_bottom = (key_left_end - key_left_start) < k ? (k - (key_left_end - key_left_start)) : 0;
            int key_left_top = k > (key_right_end - key_right_start) ? (k - (key_right_end - key_right_start)) : 0;
            int antidiag_middle;

            if (k == (key_right_end - key_right_start + key_left_end - key_left_start))
            {
                key_right_idx_i = key_right_end - key_right_start - 1;
                key_left_idx_i = key_left_end - key_left_start - 1;
                return;
            }
            //if(key_right_top == key_right_bottom) std::cout<<"error0:"<<k<<", top = "<<key_right_top<<", bottom = "<<key_right_bottom<<", left_length = "<<key_left_end - key_left_start<<", right_length = "<<key_right_end - key_right_start<<std::endl;

            if ((key_left_end - key_left_start) == 0 || (key_right_end - key_right_start) == 0)
            {
                if ((key_left_end - key_left_start) == 0)
                {
                    key_left_idx_i = -1;
                    key_right_idx_i = k - 1;
                    return;
                }
                else if ((key_right_end - key_right_start) == 0)
                {
                    key_right_idx_i = -1;
                    key_left_idx_i = k - 1;
                    return;
                }
                else
                {
                    std::cerr << "ERROR!: look for " << k << "th element, total input length is 0!" << std::endl;
                }
            }
            while (true)
            {
                // handle different corner cases
                if (key_right_top == key_right_bottom + 1)
                {
                    if (*(key_right_start + key_right_bottom) > *(key_left_start + key_left_top))
                    {
                        key_right_idx_i = key_right_bottom;
                        key_left_idx_i = key_left_top + 1;
                        break;
                    }
                    else
                    {
                        key_right_idx_i = key_right_top;
                        key_left_idx_i = key_left_top;
                        break;
                    }
                }

                antidiag_middle = (int)ceil((double)(key_right_top - key_right_bottom) / 2.0);
                //if(antidiag_middle == 0) std::cout<<"error1: top = "<<key_right_top<<", bottom = "<<key_right_bottom<<std::endl;
                //if(key_right_bottom == (key_right_top - antidiag_middle)) std::cout<<"error2, top = "<<key_right_top<<", bottom = "<<key_right_bottom<<std::endl;
                if (*(key_right_start + key_right_top - antidiag_middle) > *(key_left_start + key_left_top + antidiag_middle - 1))
                {
                    // check the lower p
                    if (*(key_right_start + key_right_top - antidiag_middle - 1) <= *(key_left_start + key_left_top + antidiag_middle))
                    {
                        // find intersection point
                        key_left_idx_i = key_left_top + antidiag_middle;
                        key_right_idx_i = key_right_top - antidiag_middle;
                        break;
                    }
                    else
                    {
                        // move top point
                        key_left_top = key_left_top + antidiag_middle;
                        key_right_top = key_right_top - antidiag_middle;
                    }
                }
                else
                {
                    key_right_bottom = key_right_top - antidiag_middle;
                }
            }
            key_left_idx_i--;
            key_right_idx_i--;
        }

        template<class T>
        void thread_findkth_kernel(void* arguments)
        {
            args_split<T>* args = (args_split<T>*)arguments;

            //SetThreadAffinityMask(GetCurrentThread(), 1LU << (args->tid << 1));

            uint32_t i, j, k;
            i = args->start;
            j = args->mid;
            k = args->end;
            int SIZEA = j - i;
            int SIZEB = k - j;
            int SIZEC = k - i;
            int tid = args->tid;
            int factor = args->factor;
            int* startA = args->startA;
            int* startB = args->startB;

            int indA, indB;

            long target = (long)SIZEC * (tid + 1) / factor; // avoid int overflow
            int kth = (int)target;

            // find_kth(args->input, args->input, SIZEA, args->input+SIZEA, args->input+SIZEA, SIZEB, kth/*SIZEC/factor*(tid+1)*/, indA, indB);
            find_kth2(args->input, args->input + SIZEA, args->input + SIZEA, args->input + SIZEA + SIZEB, kth/*SIZEC/factor*(tid+1)*/, indA, indB);
            args->startA[tid + 1] = indA + 1;
            args->startB[tid + 1] = indB + 1;
        }
   
       template <class T>
        void parallel_sort(T*& array, uint32_t size)
        {
            std::thread** threads = new std::thread*[thread_num];
            args_t<T>* thread_args = new args_t<T>[thread_num];

            uint32_t b = size / thread_num;
            uint32_t m = size % thread_num;
            for (uint32_t i = 0; i < thread_num; i++)
            {
                if (i < m)
                {
                    thread_args[i].start = i * (b + 1);
                    thread_args[i].end = thread_args[i].start + b + 1;
                }
                else
                {
                    thread_args[i].start = i * b + m;
                    thread_args[i].end = thread_args[i].start + b;
                }
                thread_args[i].input = array;
                thread_args[i].tid = i;
                threads[i] = new std::thread(thread_sort_kernel<T>, &thread_args[i]);
            }

            for (uint32_t i = 0; i < thread_num; i++)
                threads[i]->join();

            // delete threads
            for (uint32_t i = 0; i < thread_num; i++)
                delete threads[i];


            // merge sort buffer 
            T* inputa = new T[size];
            // works on inputa if true 
            bool flaga = false;
            uint32_t segments = thread_num;
            uint32_t tnum;
            args_split<T>* as_old;
            args_split<T>* as_new;

            as_old = new args_split<T>[thread_num];
            for (uint32_t i = 0; i < thread_num; i++)
            {
                as_old[i].start = thread_args[i].start;
                as_old[i].end = thread_args[i].end;
            }

            as_new = new args_split<T>[thread_num];
            int factor = 2;
            int v = thread_num; // v is the closest power of 2
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v++;

            int* startA = (int*)malloc(sizeof(int) * (((thread_num + (factor - 1)) / factor) * (factor + 1)));
            int* startB = (int*)malloc(sizeof(int) * (((thread_num + (factor - 1)) / factor) * (factor + 1)));
            while (factor <= v)
            {

                for (uint32_t i = 0; i < thread_num; i++)
                {
                    as_new[i].start = as_old[(i / factor) * factor].start;

                    if ((i / factor) * factor + factor / 2 < thread_num)
                        as_new[i].mid = as_old[(i / factor) * factor + factor / 2].start;
                    else
                        as_new[i].mid = as_old[thread_num - 1].end;

                    if ((i / factor) * factor + factor < thread_num)
                        as_new[i].end = as_old[(i / factor) * factor + factor].start;
                    else
                        as_new[i].end = as_old[thread_num - 1].end;

                    int tfactor = min(thread_num, (i / factor + 1) * factor) - (i / factor) * factor;
                    as_new[i].tid = i % tfactor;
                    as_new[i].factor = tfactor;
                    as_new[i].startA = startA + (i / factor) * (factor + 1);
                    as_new[i].startB = startB + (i / factor) * (factor + 1);
                    if (as_new[i].tid == 0)
                    {
                        as_new[i].startA[0] = 0;
                        as_new[i].startB[0] = 0;
                    }
                    if (flaga)
                    {
                        as_new[i].input = inputa + as_new[i].start;
                        as_new[i].inputa = array + as_new[i].start;
                    }
                    else
                    {
                        as_new[i].input = array + as_new[i].start;
                        as_new[i].inputa = inputa + as_new[i].start;
                    }

                    threads[i] = new std::thread(thread_findkth_kernel<T>, &as_new[i]);
                }

                for (uint32_t i = 0; i < thread_num; ++i)
                    threads[i]->join();
                for (uint32_t i = 0; i < thread_num; ++i)
                    delete threads[i];

                for (uint32_t i = 0; i < thread_num; ++i)
                    threads[i] = new std::thread(thread_merge_kernel<T>, &as_new[i]);
                
                for (uint32_t i = 0; i < thread_num; ++i)
                    threads[i]->join();
                for (uint32_t i = 0; i < thread_num; ++i)
                    delete threads[i];

                if (flaga)
                {
                    flaga = false;
                }
                else
                {
                    flaga = true;
                }
                factor *= 2;
            }
            free(startA);
            free(startB);

            if (flaga)
                util::copy_array(array, size, inputa, size);

            delete[] inputa;
            delete[] threads;
            return;
        }
}

//#include "aspas.hpp"






