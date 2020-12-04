//
// Created by tolle on 12/11/2020.
//

#ifndef MESHCORE_INDEXBUFFER_H
#define MESHCORE_INDEXBUFFER_H

#include "../core/Triangle.h"
#include <vector>

class IndexBuffer {
private:
    unsigned int m_RendererId;
    unsigned int m_Count;
public:
    IndexBuffer();
    IndexBuffer(const IndexBuffer& other);
    IndexBuffer(IndexBuffer&& other) noexcept;
    IndexBuffer& operator=(IndexBuffer&& other) noexcept;
    ~IndexBuffer();

    IndexBuffer(const unsigned int* indices, unsigned int count);
    IndexBuffer(const std::vector<Triangle>& triangles);

    void bind() const;

    inline unsigned int getCount() const { return m_Count; }
};


#endif //MESHCORE_INDEXBUFFER_H
