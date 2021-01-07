//
// Created by Jonas on 7/01/2021.
//

#include "LatencyBenchMarkTask.h"
#include <cuda_runtime.h>

#define DSIZE 32
#define LOOPS 100000

void LatencyBenchMarkTask::run() {
    {
        auto startmicros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        int *d_data, *h_data;

        cudaMallocHost(&h_data, DSIZE*sizeof(int));
        cudaMalloc(&d_data, DSIZE*sizeof(int));

        for (int i=0; i < LOOPS; i++)
            cudaMemcpy(h_data, d_data, DSIZE*sizeof(int), cudaMemcpyDeviceToHost);
        auto stopmicros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        auto totalmicros = stopmicros - startmicros;
        auto averagemicros = totalmicros/LOOPS;
        printf("Total microseconds: %lld\n", totalmicros);
        printf("Average microseconds per loop:  %lld\n", averagemicros);
    }
}
