//
// Created by tolle on 12/11/2020.
//

#ifndef MESHCORE_INDEXBUFFER_H
#define MESHCORE_INDEXBUFFER_H

// TODO later we can define what index type to use; (probably only interesting to save on VRAM)
//#define INDEXTYPE = GL_UNSIGNED_INT;
//typedef unsigned int indexType;

//unsigned long long int;   8 byte  0 to 18,446,744,073,709,551,615
//unsigned long int;        8 bytes 0 to 4,294,967,295
//unsigned int;             4 bytes 0 to 4,294,967,295 -- seems a good choice
//unsigned short int;       2 bytes 0 to 65,535
//unsigned char;            0 to 255     only very small files can be rendered

#include "../core/Triangle.h"
#include <vector>

class IndexBuffer {
private:
    unsigned int m_RendererId;
    unsigned int m_Count;
public:
    IndexBuffer();
    IndexBuffer(const IndexBuffer& other);
    IndexBuffer(IndexBuffer&& other) noexcept ;
    IndexBuffer& operator=(IndexBuffer&& other);
    ~IndexBuffer();

    IndexBuffer(const unsigned int* indices, unsigned int count);
    IndexBuffer(const std::vector<Triangle>& triangles);

    void bind() const;

    inline unsigned int getCount() const { return m_Count; }
};


#endif //MESHCORE_INDEXBUFFER_H
