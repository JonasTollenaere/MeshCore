//
// Created by Jonas on 12/11/2020.
//

#include "Renderer.h"

#include <iostream>

void GLClearError(){
    while(glGetError() != GL_NO_ERROR){}
}

bool GLLogCall(const char* function, const char* file, int line){
    while(GLenum error = glGetError()){
        std::cout<< "[OpenGL Error] (" << error << "): " << function << " " << file << ":" << line << std::endl;
        return false;
    }
    return true;
}

void Renderer::draw(const VertexArray &vertexArray, const IndexBuffer &indexBuffer, const Shader &shader) const {

    shader.bind();
    indexBuffer.bind();
    vertexArray.bind();
    GLCall(glDrawElements(GL_TRIANGLES, indexBuffer.getCount(), GL_UNSIGNED_INT, nullptr));

}

void Renderer::clear() {
    GLCall(glClear(GL_COLOR_BUFFER_BIT));
}

Renderer::Renderer() {
    GLCall(glEnable(GL_BLEND))
    GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
}
