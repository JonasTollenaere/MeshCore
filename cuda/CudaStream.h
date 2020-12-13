//
// Created by Jonas on 13/12/2020.
//

#ifndef MESHCORE_CUDASTREAM_H
#define MESHCORE_CUDASTREAM_H

#include <driver_types.h>

class CudaStream {
private:
    cudaStream_t stream;
public:
    CudaStream();
    CudaStream(const CudaStream& stream) = delete;
    ~CudaStream();

    const cudaStream_t &getStream() const;

};


#endif //MESHCORE_CUDASTREAM_H
