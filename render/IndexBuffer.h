//
// Created by tolle on 12/11/2020.
//

#ifndef MESHCORE_INDEXBUFFER_H
#define MESHCORE_INDEXBUFFER_H

// TODO later we can define what index type to use;
//#define INDEXTYPE = GL_UNSIGNED_INT;
//typedef unsigned int indexType;

class IndexBuffer {
private:
    unsigned int m_RendererId;
    unsigned int m_Count;
public:
    IndexBuffer(const unsigned int* data, unsigned int count);
    ~IndexBuffer();

    void bind() const;
    void unbind() const;

    inline unsigned int getCount() const { return m_Count; }
};


#endif //MESHCORE_INDEXBUFFER_H
