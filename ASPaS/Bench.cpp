// ASPaS.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "aspas.h"
#include <iostream>
#include <random>

typedef float Key;
#define Keysize sizeof(Key)

void Sort(uint64_t n) {
    uint64_t sz = 1LLU << 30;
    uint64_t tot_n = sz / Keysize;

    Key* A = (Key*)VirtualAlloc(NULL, sz, MEM_COMMIT, PAGE_READWRITE);
    Key* A_copy = (Key*)VirtualAlloc(NULL, sz, MEM_COMMIT, PAGE_READWRITE);
    Key* S = (Key*)VirtualAlloc(NULL, sz, MEM_COMMIT, PAGE_READWRITE);
    
    std::mt19937 g;
    //std::uniform_int_distribution<Key> d;
    std::uniform_real_distribution<Key> d;
    FOR(i, n, 1) A[i] = d(g);
    memcpy(A_copy, A, sz);
    memcpy(S, A, sz);
    std::sort(S,  S + n);

    const int repeat = 10;

    hrc::time_point st, en; double el = 0;
    printf("Running aspas::sort on N: %llu ...\n", n);
    FOR(i, repeat, 1) {
        printf("Iter: %3lu ... ", i);
        memcpy(A, A_copy, sz);
        Key* p = A, * endp = A + tot_n;
        st = hrc::now();
        while (p < endp) {
            aspas::sort(p, n); 
            p += n;
        }
        en = hrc::now();
        el += ELAPSED_MS(st, en);
        printf("\r");
    }
    
    printf("\r                                 \r");
    printf("Done\n");
    printf("Elapsed: %.2f ms/iter, Speed: %.2f M/s\n", el / repeat, tot_n * repeat / el / 1e3);

    if (A[132] == 123) printf("\n");

    printf("Checking correctness ... ");
    FOR(i, n, 1) {
        if (A[i] != S[i]) {
            printf("Incorrect @ %llu\n", i);
            break;
        }
    }
    printf("done\n");

    VirtualFree(A, 0, MEM_RELEASE);
    VirtualFree(A_copy, 0, MEM_RELEASE);
    VirtualFree(S, 0, MEM_RELEASE);
}

int main()
{
    
    FOR_INIT(i, 4, 29, 1)
        Sort(1LLU << i);
    return 0;
}
