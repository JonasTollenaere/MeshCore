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

void Renderer::drawTriangles(const VertexArray &vertexArray, const IndexBuffer &indexBuffer, const Shader &shader) const {

    ASSERT(sizeof(GLuint) == sizeof(unsigned int));

    shader.bind();
    vertexArray.bind();
    indexBuffer.bind();
    GLCall(glDrawElements(GL_TRIANGLES, indexBuffer.getCount(), GL_UNSIGNED_INT, nullptr));

}

void Renderer::drawLines(const VertexArray &vertexArray, const IndexBuffer &indexBuffer, const Shader &shader) const {

    shader.bind();
    indexBuffer.bind();
    vertexArray.bind();
    GLCall(glDrawElements(GL_LINES, indexBuffer.getCount(), GL_UNSIGNED_INT, nullptr));

}

void Renderer::clear() {
    GLCall(glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT));
}

Renderer::Renderer() {
    GLCall(glEnable(GL_DEPTH_TEST));
    GLCall(glDepthFunc(GL_LESS));
    GLCall(glEnable(GL_BLEND))
    GLCall(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    GLCall(glEnable(GL_MULTISAMPLE));

    // Options (should be available in GUI, per model)
//    GLCall(glClearColor(100.0f/255.0f, 149.0f/255.0f, 237.0f/255.0f, 1.0f));
    GLCall(glEnable(GL_CULL_FACE));
//    GLCall(glDisable(GL_CULL_FACE));
    GLCall(glCullFace(GL_BACK));
//    GLCall(glPolygonMode(GL_FRONT_AND_BACK,GL_LINE));

//    GLCall(glDisable(GL_POLYGON_SMOOTH));
}