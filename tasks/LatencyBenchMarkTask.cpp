//
// Created by Jonas on 7/01/2021.
//

#include "LatencyBenchMarkTask.h"
#include <cuda_runtime.h>

// Larger data transfers, 128 kBytes
//#define DSIZE 32 * 1024
//#define LOOPS 100000

// Small data transfers, 128 bytes
#define DSIZE 32
#define LOOPS 100000

void LatencyBenchMarkTask::run() {
    {
        int *d_data;

        int *h_data;
        cudaMallocHost(&h_data, DSIZE*sizeof(int));
//        int h_data[DSIZE];

        cudaFree(nullptr);
        cudaMalloc(&d_data, DSIZE*sizeof(int));

        auto startmicros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        for (int i=0; i < LOOPS; i++)
            cudaMemcpy(h_data, d_data, DSIZE*sizeof(int), cudaMemcpyDeviceToHost);
        auto stopmicros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        auto totalmicros = stopmicros - startmicros;
        auto averagemicros = totalmicros/LOOPS;
        printf("Total microseconds: %lld\n", totalmicros);
        printf("Average microseconds per loop:  %lld\n", averagemicros);
    }
}
