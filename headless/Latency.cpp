//
// Created by jonas on 11.01.21.
//

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

// Larger data transfers, 128 kBytes
#define DSIZE 32 * 1024
#define LOOPS 100000

// Small data transfers, 128 bytes
//#define DSIZE 32
//#define LOOPS 100000

int main(int argc, char *argv[]){
    {
        int *d_data;

//        int *h_data;
//        cudaMallocHost(&h_data, DSIZE*sizeof(int));
        int h_data[DSIZE];

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
