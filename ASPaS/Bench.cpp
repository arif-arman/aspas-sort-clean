// ASPaS.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "aspas.h"
#include <iostream>
#include <random>

typedef int Key;
#define Keysize sizeof(Key)

void Sort() {
    uint64_t sz = 1LLU << 30;
    uint64_t n = sz / Keysize;

    Key* A = (Key*)VirtualAlloc(NULL, sz, MEM_COMMIT, PAGE_READWRITE);
    
    std::mt19937 g;
    std::uniform_int_distribution<int> d;
    FOR(i, n, 1) A[i] = d(g);

    printf("Running aspas::sort ... ");
    hrc::time_point st = hrc::now();
    aspas::sort(A, n);
    hrc::time_point en = hrc::now();
    double el = ELAPSED_MS(st, en);
    printf("done\n");
    printf("Elapsed: %.2f ms, Speed: %.2f M/s\n", el, n / el / 1e3);

    if (A[132] & 0x123 == 0) printf("Dummy\n");

    VirtualFree(A, 0, MEM_RELEASE);
}

int main()
{
    
    Sort();
    return 0;
}
