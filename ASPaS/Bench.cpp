// ASPaS.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "aspas.h"
#include <iostream>
#include <random>
#include <omp.h>

typedef double Key;
#define Keysize sizeof(Key)
#define N_LOGICAL   12

void Sort(uint64_t n) {
    printf("---------------------------------\n");
    uint64_t sz = 1LLU << 30;
    uint64_t tot_n = sz / Keysize;

    Key* A =        (Key*)VirtualAlloc(NULL, sz, MEM_COMMIT, PAGE_READWRITE);
    Key* A_copy =   (Key*)VirtualAlloc(NULL, sz, MEM_COMMIT, PAGE_READWRITE);
    Key* S =        (Key*)VirtualAlloc(NULL, sz, MEM_COMMIT, PAGE_READWRITE);
    
    /*std::mt19937 g;
    std::uniform_int_distribution<Key> d;*/

    std::default_random_engine g;
    std::uniform_real_distribution<Key> d; 

    FOR(i, n, 1) A[i] = d(g);
    memcpy(A_copy, A, sz);
    memcpy(S, A, sz);
    omp_set_num_threads(N_LOGICAL);
    printf("Running std::sort for correctness ... ");
#pragma omp parallel for
    for(int i = 0; i < (tot_n / n); ++i)
        std::sort(S + i * n, S + (i + 1) * n);
    printf("done\n");

    const int repeat = 3;

    hrc::time_point st, en; double el = 0;
    printf("Running aspas::sort on N: %llu, Keysize: %lu bytes ...\n", n, Keysize);
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
    printf("Elapsed: %.2f ms/iter, Speed: %.2f M/s\n", el / repeat, tot_n * repeat / el / 1e3);

    if (A[132] == 123) printf("\n");

    printf("Checking correctness ... ");
#pragma omp parallel for
    for (int i = 0; i < (tot_n / n); ++i) {
        Key* a = A + i * n, * s = S + i * n;
        FOR(j, n, 1) {
            if (a[j] != s[j]) {
                printf("Incorrect @ idx %llu @ segment %llu\n", j, i);
                break;
            }
        }
    }
    printf("done\n");

    VirtualFree(A, 0, MEM_RELEASE);
    VirtualFree(A_copy, 0, MEM_RELEASE);
    VirtualFree(S, 0, MEM_RELEASE);
}

int main()
{
    SetThreadAffinityMask(GetCurrentThread(), 1 << 4);
    
    FOR_INIT(i, 4, 29, 1)
        Sort(1LLU << i);
    return 0;
}
