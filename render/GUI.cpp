//
// Created by Jonas on 10/11/2020.
//

#include <iostream>

#include "Renderer.h"
#include "VertexBufferLayout.h"
#include "Window.h"
#include "../core/FileParser.h"


int main() {

    std::string path = "../../data/models/DIAMCADbr1.obj";
    ModelSpaceMesh brilliantMesh = FileParser::parseFile(path);

    std::string path2 = "../../data/models/dragon.obj";
    ModelSpaceMesh roughMesh = FileParser::parseFile(path2);

    std::string path3 = "../../data/models/DIAMCADbr2.obj";
    ModelSpaceMesh brilliantMesh2 = FileParser::parseFile(path3);

    Window window;
    {
        float positions[]{
                -50.0f, -50.0f, 50.0f, 0.6627f, 0.6627f, 0.6627f, 0.5f,
                +50.0f, -50.0f, 50.0f, 0.6627f, 0.6627f, 0.6627f, 0.5f,
                +50.0f, +50.0f, 50.0f, 0.6627f, 0.6627f, 0.6627f, 0.5f,
                -50.0f, +50.1f, 50.0f, 0.6627f, 0.6627f, 0.6627f, 0.5f,
                +150.0f, +50.0f,50.0f, 0.6627f, 0.6627f, 0.6627f, 0.5f,
                0.0f, 0.0f, 100.0f, 1.0f, 0.0f, 0.0f, 0.8f,
                0.0f, 100.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.8f,
                100.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.8f,
                000.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.8f,
        };

        unsigned int indices[] = {
                0, 1, 2,
                2, 3, 0,
                1, 2, 4,
                5, 6, 7,
        };
        unsigned int indices2[] = {
                0, 1, 2,
        };

        /// Axis
        float axisPositions[]{
                0.0f, 0.0f, 0.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 200.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 200.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                200.0f, 0.0f, 0.0f,  1.0f, 1.0f, 1.0f, 1.0f,
        };

//        unsigned int axisIndices[]{
//                5,8,
//                6,8,
//                8,7,
//        };
        unsigned int axisIndices[]{
                0,1,
                0,2,
                0,3,
        };

        // Vertex Buffer and Vertex Array
        VertexArray triangleVertexArray;
        VertexBuffer triangleVertexBuffer(positions, sizeof(positions));
        VertexBufferLayout triangleLayout;
        triangleLayout.push<float>(3);
        triangleLayout.push<float>(4);
        triangleVertexArray.addBuffer(triangleVertexBuffer, triangleLayout);

        // Index Buffer
        IndexBuffer triangleIndexBuffer(indices, std::size(indices));
        IndexBuffer triangleIndexBufffer2(indices2, std::size(indices2));

        VertexArray axisVertexArray;
        VertexBuffer axisVertexBuffer(axisPositions, sizeof(axisPositions));
        VertexBufferLayout axisLayout;
        axisLayout.push<float>(3);
        axisLayout.push<float>(4);
        axisVertexArray.addBuffer(axisVertexBuffer, axisLayout);
        IndexBuffer axisIndexBuffer(axisIndices, std::size(axisIndices));

        // Vertex Buffer and Vertex Array
        VertexArray roughVertexArray;
        VertexBuffer roughVertexBuffer(roughMesh, glm::vec4(0.6627f, 0.6627f, 0.6627f, 0.5f));
//        VertexBuffer roughVertexBuffer(brMesh.vertices, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
        VertexBufferLayout roughLayout;
        roughLayout.push<float>(3);
        roughLayout.push<float>(3);
        roughLayout.push<float>(4);
        roughVertexArray.addBuffer(roughVertexBuffer, roughLayout);
        IndexBuffer roughIndexBuffer(roughMesh.triangles);

        VertexArray brilliantVertexArray;
        VertexBuffer brilliantVertexBuffer(brilliantMesh, glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
        VertexBufferLayout brilliantLayout;
        brilliantLayout.push<float>(3);
        brilliantLayout.push<float>(3);
        brilliantLayout.push<float>(4);
        brilliantVertexArray.addBuffer(brilliantVertexBuffer, brilliantLayout);
        IndexBuffer brilliantIndexBuffer(brilliantMesh.triangles);

        VertexArray brilliantVertexArray2;
        VertexBuffer brilliantVertexBuffer2(brilliantMesh2, glm::vec4(0.0f, 0.5f, 0.0f, 1.0f));
        VertexBufferLayout brilliantLayout2;
        brilliantLayout2.push<float>(3);
        brilliantLayout2.push<float>(3);
        brilliantLayout2.push<float>(4);
        brilliantVertexArray2.addBuffer(brilliantVertexBuffer2, brilliantLayout2);
        IndexBuffer brilliantIndexBuffer2(brilliantMesh2.triangles);

        // Shader
        Shader shader("../../render/res/shaders/Basic.shader");
        Shader shader2("../../render/res/shaders/Intermediate.shader");

        Renderer renderer;

        /* Loop until the user closes the window */
        while (window.shouldClose()) {

            window.processInput();
            glm::mat4 projectionViewMatrix = window.getProjectionViewMatrix();
            renderer.clear();

            // TODO: 4 draw calls from some rendering class

            // static and dynamic are a seperate vertex buffer
            // transparent and dynamic should be seperate indexbuffer (because solids should be drawn first, then transparent in order)

            // 1. DRAW STATIC SOLID (includes axis?)
            // 2. DYNAMIC SOLID
            // 3. STATIC TRANSPARENT -- after sorting by depth?
            // 4. DYNAMIC TRANSPARENT -- after sorting by depth?


            shader.bind();
            shader.setUniformMat4f("u_ViewProjectionMatrix", projectionViewMatrix);

            shader2.setUniformMat4f("u_ViewProjectionMatrix", projectionViewMatrix);
            glm::vec3 lightSource = glm::vec4(glm::vec3(0,-0,-200), 1.0f) *  window.getViewMatrix();
            shader2.setUniform3fv("u_LightSource", lightSource);
            shader2.setUniform1f("u_Ambient", 0.15);

            renderer.drawLines(axisVertexArray, axisIndexBuffer, shader);

//            renderer.drawTriangles(triangleVertexArray, triangleIndexBuffer, shader);
//
//            renderer.drawTriangles(triangleVertexArray, triangleIndexBufffer2, shader);

//            renderer.drawTriangles(brilliantVertexArray, brilliantIndexBuffer, shader2);
//
//            renderer.drawTriangles(brilliantVertexArray2, brilliantIndexBuffer2, shader2);

            renderer.drawTriangles(roughVertexArray, roughIndexBuffer, shader2);

            window.update();

        }
    }

    glfwTerminate();
    return 0;
}



