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

Renderer::Renderer(): shader("../../glfw/res/shaders/Intermediate.shader"){

    GLCall(std::cout << glGetString(GL_VERSION) << std::endl)

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

}

void Renderer::doRenderIteration() {
    this->clear();
    this->window.processInput();

    glm::mat4 projectionMatrix = window.getProjectionMatrix();
    glm::mat4 viewMatrix = window.getViewMatrix();
    glm::mat4 projectionViewMatrix = projectionMatrix * viewMatrix;
    glm::vec3 viewSpaceLightDirection = glm::vec4(glm::vec3(0, 0, 1), 1.0f) * viewMatrix;

    for(RenderModel const& renderModel: renderModels){
        renderModel.draw(shader, projectionViewMatrix, viewSpaceLightDirection);
    }

    this->window.update();
}

void Renderer::addWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh){
    this->renderModels.emplace_back(RenderModel(worldSpaceMesh));
}

void Renderer::addWorldSpaceMesh(const WorldSpaceMesh& worldSpaceMesh, const Color& color){
    RenderModel renderModel = RenderModel(worldSpaceMesh);
    renderModel.setColor(color);
    this->renderModels.emplace_back(std::move(renderModel));
}

void Renderer::start() {
    std::cout << "Start glfw thread" << std::endl;
    this->renderThread = std::thread([this]{this->run();});
    std::cout << "Continue main thread" << std::endl;
}

void Renderer::stop(){
    this->renderThread.join();
}

void Renderer::run() {
    int counter = 0;
    std::cout << "Run called" << std::endl;
    while(!window.shouldClose()){
        doRenderIteration();
        if(counter++%60 == 0) std::cout << "Render thread going" << std::endl;
    }
    std::cout << "end of run()" << std::endl;
}
