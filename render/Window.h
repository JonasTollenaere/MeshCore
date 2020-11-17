//
// Created by Jonas on 15/11/2020.
//

#ifndef MESHCORE2_WINDOW_H
#define MESHCORE2_WINDOW_H

#include <GL/glew.h>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

class Window {
private:
    GLFWwindow *window;
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatrix;
public:
    Window();
    bool shouldClose();
    void processInput();
    void update();
    glm::mat4 getProjectionViewMatrix();

    void mouseInput(GLFWwindow *window2, double xPosition, double yPosition);
};


#endif //MESHCORE2_WINDOW_H
