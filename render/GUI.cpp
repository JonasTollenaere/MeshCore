//
// Created by Jonas on 10/11/2020.
//

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "Renderer.h"
#include "VertexBufferLayout.h"

//#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

int main() {

    GLFWwindow *window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(1280, 720, "MeshCore", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK) {
        std::cout << "Error!" << std::endl;
    }

    {
        float positions[]{
                -0.1f, -0.1f,
                +0.1f, -0.1f,
                +0.1f, +0.1f,
                -0.1f, +0.1f,
                +0.2f, +0.1f,
        };

        unsigned int indices[] = {
                0, 1, 2,
                2, 3, 0,
                1, 2, 4,
        };


        // Vertex Buffer and Vertex Array
        VertexArray vertexArray;
        VertexBuffer vertexBuffer(positions, 5 * 2 * sizeof(float));
        VertexBufferLayout layout;
        layout.push<float>(2);
        vertexArray.addBuffer(vertexBuffer, layout);

        // Index Buffer
        IndexBuffer indexBuffer(indices, 9);

//        // Index Buffer
        unsigned int indices2[] = {
                0, 1, 2,
        };

        IndexBuffer indexBuffer2(indices2, 3);

        glm::mat4 projectionMatrix = glm::ortho(-1.0f, 1.0f, -9.0f/16.0f, 9.0f/16.0f, -1.0f, 1.0f);

        // Shader
        Shader shader("../../render/res/shaders/Basic.shader");
        shader.bind();
        shader.setUniform4f("u_Color", 1.0f, 0.0f, 1.0, 1.0);
        shader.setUniformMat4f("u_ModelViewProjectionMatrix", projectionMatrix);
        shader.unbind();

        Shader shader2(shader);
        shader2.bind();
        shader2.setUniform4f("u_Color", 1.0f, 1.0f, 1.0f, 0.5f);
        shader2.setUniformMat4f("u_ModelViewProjectionMatrix", projectionMatrix);
        shader2.unbind();

        vertexArray.unbind();
        indexBuffer.unbind();
        vertexBuffer.unbind();

        Renderer renderer;

        float r = 0.0f;
        float increment = 0.05f;

        /* Loop until the user closes the window */
        while (!glfwWindowShouldClose(window)) {
            /* Render here */

            shader.bind();
            shader.setUniform4f("u_Color", r, 0.0f, 1.0f, 1.0f);

            renderer.draw(vertexArray, indexBuffer, shader);

            renderer.draw(vertexArray, indexBuffer2, shader2);


            r += increment;
            if (r > 1.0f || r < 0.0f) {
                increment = -increment;
            }


            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
        }
    }

    glfwTerminate();
    return 0;
}
