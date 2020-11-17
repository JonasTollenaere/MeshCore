//
// Created by Jonas on 12/11/2020.
//

#ifndef MESHCORE_RENDERER_H
#define MESHCORE_RENDERER_H

#include <GL/glew.h>
#include "VertexArray.h"
#include "Shader.h"
#include "IndexBuffer.h"

#define ASSERT(x) if (!(x)) __debugbreak();  //Compiler specific
#define GLCall(x) GLClearError();\
    x;\
    ASSERT(GLLogCall(#x, __FILE__, __LINE__))

void GLClearError();
bool GLLogCall(const char* function, const char* file, int line);

class Renderer {
public:
    Renderer();
    void clear();
    void drawTriangles(const VertexArray& vertexArray, const IndexBuffer& indexBuffer, const Shader& shader) const;
    void drawLines(const VertexArray &vertexArray, const IndexBuffer &indexBuffer, const Shader &shader) const;
};


#endif //MESHCORE_RENDERER_H
