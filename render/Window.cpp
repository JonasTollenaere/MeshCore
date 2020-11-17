//
// Created by Jonas on 15/11/2020.
//

#include <iostream>
#include <algorithm>
#include "Window.h"

float fov = 75.0f;
float distance = 200.0f;
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    fov -= (float)yoffset;
    if (fov < 0.1f)
        fov = 0.1f;
    if (fov > 90.0f)
        fov = 90.0f;
}

double currentX;
double currentY;
static void cursorPositionCallback(GLFWwindow* window, double newX, double newY)
{
    currentX = newX;
    currentY = newY;
}

bool leftMouseButtonPressed = false;
double oldX;
double oldY;
static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if(GLFW_PRESS == action){
            glfwGetCursorPos(window, &oldX, &oldY);
            leftMouseButtonPressed = true;
        }
        else if(GLFW_RELEASE == action)
            leftMouseButtonPressed = false;
    }
}

Window::Window():
    viewMatrix(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f,0.0f, - distance))),
    projectionMatrix(glm::perspective(glm::radians(75.0f), 1280.0f / 720.0f, 0.1f, 10000.0f))
{

    /* Initialize the library */
    if (!glfwInit())
        std::cout << "Error while initialising the GLFW Library" << std::endl;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(1280, 720, "MeshCore", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        std::cout << "Error while creating the window" << std::endl;
    }

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    if (glewInit() != GLEW_OK) {
        std::cout << "Error whilst initialising GLEW!" << std::endl;
    }
}

bool Window::shouldClose() {
    return !glfwWindowShouldClose(window);
}

void Window::processInput() {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    const float translateSpeed = 5.0f;
    const float rotationSpeed = 0.01f;
    const float fovSpeed = 2.0f;

    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
    glm::vec3 cameraLeft    = glm::vec3(- 1.0f, 0.0f,  0.0f);

    bool translateMode = false;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){
        // std::cout << "Z" << std::endl;
        if(translateMode) viewMatrix = glm::translate(viewMatrix, translateSpeed * cameraUp);
        else viewMatrix = glm::rotate(viewMatrix, rotationSpeed, glm::vec3(glm::inverse(viewMatrix) * glm::vec4(cameraLeft, 0.0f)));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
        // std::cout << "S" << std::endl;
        if(translateMode) viewMatrix = glm::translate(viewMatrix, - translateSpeed * cameraUp);
        else viewMatrix = glm::rotate(viewMatrix, - rotationSpeed, glm::vec3(glm::inverse(viewMatrix) * glm::vec4(cameraLeft,0.0f)));
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS|| glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
        // std::cout << "Q" << std::endl;
        if(translateMode) viewMatrix = glm::translate(viewMatrix, translateSpeed * cameraLeft);
        else viewMatrix = glm::rotate(viewMatrix, - rotationSpeed, glm::vec3(glm::inverse(viewMatrix) * glm::vec4(cameraUp,0.0f)));
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS){
        // std::cout << "D" << std::endl;
        if(translateMode) viewMatrix = glm::translate(viewMatrix, - translateSpeed * cameraLeft);
        else viewMatrix = glm::rotate(viewMatrix, rotationSpeed, glm::vec3(glm::inverse(viewMatrix) * glm::vec4(cameraUp,0.0f)));
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        // std::cout << "A" << std::endl;
        viewMatrix = glm::rotate(viewMatrix, - rotationSpeed, glm::vec3(glm::inverse(viewMatrix) * glm::vec4(cameraFront,0.0f)));
    }

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        // std::cout << "E" << std::endl;
        viewMatrix = glm::rotate(viewMatrix, rotationSpeed, glm::vec3(glm::inverse(viewMatrix) * glm::vec4(cameraFront,0.0f)));
    }

    if (glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS) {
        // std::cout << "+" << std::endl;
        fov = std::clamp(fov -= fovSpeed, 1.0f, 90.0f);
    }

    if (glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS) {
        // std::cout << "-" << std::endl;
        fov = std::clamp(fov += fovSpeed, 1.0f, 90.0f);
    }

    if(leftMouseButtonPressed){
        auto offsetX = float(currentX - oldX) ;
        auto offsetY = float(currentY - oldY);
        oldX = currentX;
        oldY = currentY;
        viewMatrix = glm::rotate(viewMatrix, rotationSpeed * offsetX, glm::vec3(glm::inverse(viewMatrix) * glm::vec4(cameraUp,0.0f)));
        viewMatrix = glm::rotate(viewMatrix, - rotationSpeed * offsetY, glm::vec3(glm::inverse(viewMatrix) * glm::vec4(cameraLeft,0.0f)));
    }
}

glm::mat4 Window::getProjectionViewMatrix() {

    projectionMatrix = glm::perspective(glm::radians(fov), 1280.0f / 720.0f, 0.1f, 10000.0f);
    return projectionMatrix * viewMatrix;
}

void Window::update() {
    /* Swap front and back buffers */
    glfwSwapBuffers(window);

    /* Poll for and process events */
    glfwPollEvents();
}