//
// Created by Jonas on 12/11/2020.
//

#ifndef MESHCORE2_VERTEXBUFFERLAYOUT_H
#define MESHCORE2_VERTEXBUFFERLAYOUT_H

#include <vector>
#include <GL/glew.h>
#include "Renderer.h"

struct VertexBufferElement {

    unsigned int type;
    unsigned int count;
    unsigned char normalized;

    static unsigned int getSizeOfType(unsigned int type){
        switch(type){
            case GL_FLOAT:          return sizeof(GLfloat);
            case GL_UNSIGNED_INT:   return sizeof(GLuint);
            case GL_UNSIGNED_BYTE:  return sizeof(GLbyte);
            default: ASSERT(false);
        }
        ASSERT(false);
        return 0;
    }
};

class VertexBufferLayout {
private:
    std::vector<VertexBufferElement> m_Elements;
    unsigned int m_Stride;
public:
    VertexBufferLayout() : m_Stride(0) {
    }

    template<typename T>
    void push(unsigned int count){
        static_assert(false);
    }

    template<>
    void push<float>(unsigned int count){
        m_Elements.push_back({GL_FLOAT, count, GL_FALSE});
        m_Stride += count * VertexBufferElement::getSizeOfType(GL_FLOAT);
    }

    template<>
    void push<unsigned int>(unsigned int count){
        m_Elements.push_back({GL_UNSIGNED_INT, count, GL_FALSE});
        m_Stride += count * VertexBufferElement::getSizeOfType(GL_UNSIGNED_INT);
    }

    template<>
    void push<unsigned char>(unsigned int count){
        m_Elements.push_back({GL_UNSIGNED_BYTE, count, GL_TRUE});
        m_Stride += count * VertexBufferElement::getSizeOfType(GL_UNSIGNED_BYTE);
    }

    inline const std::vector<VertexBufferElement> getElements() const& { return m_Elements;}
    inline unsigned int getStride() const { return m_Stride;}
};


#endif //MESHCORE2_VERTEXBUFFERLAYOUT_H
