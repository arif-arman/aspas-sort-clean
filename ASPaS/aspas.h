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
//#include <unistd.h>
#define __AVX__

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
    uint32_t thread_num = 12;       //sysconf(_SC_NPROCESSORS_ONLN);

   


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
    void sort(T* array, uint32_t size)
    {
        internal::sorter(array, size);
        internal::merger(array, size);
        return;
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

}

//#include "aspas.hpp"






