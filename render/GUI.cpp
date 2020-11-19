//
// Created by Jonas on 10/11/2020.
//

#include <iostream>

#include "Renderer.h"
#include "VertexBufferLayout.h"
#include "Window.h"
#include "../core/FileParser.h"
#include "RenderModel.h"


int main() {

    std::string path = "../../data/models/DIAMCADbr1.obj";
    ModelSpaceMesh brilliantMesh = FileParser::parseFile(path);

    std::string path2 = "../../data/models/DIAMCADrough.obj";
    const ModelSpaceMesh roughMesh = FileParser::parseFile(path2);

    std::string path3 = "../../data/models/DIAMCADbr2.obj";
    ModelSpaceMesh brilliantMesh2 = FileParser::parseFile(path3);

    Window window;
    {
        /// Axis
        float axisPositions[]{
                0.0f, 0.0f, 0.0f,  1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 0.0f, 200.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                0.0f, 200.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                200.0f, 0.0f, 0.0f,  1.0f, 1.0f, 1.0f, 1.0f,
        };
        unsigned int axisIndices[]{
                0,1,
                0,2,
                0,3,
        };

        VertexArray axisVertexArray;
        VertexBuffer axisVertexBuffer(axisPositions, sizeof(axisPositions));
        VertexBufferLayout axisLayout;
        axisLayout.push<float>(3);
        axisLayout.push<float>(4);
        axisVertexArray.addBuffer(axisVertexBuffer, axisLayout);
        IndexBuffer axisIndexBuffer(axisIndices, (unsigned int) std::size(axisIndices));

        const WorldSpaceMesh roughWorldSpaceMesh(roughMesh);
//        RenderModel roughRenderModel(roughWorldSpaceMesh);
//        roughRenderModel.setColor(glm::vec4(0.6627f, 0.6627f, 0.6627f, 0.3f));


        // Vertex Buffer and Vertex Array
        VertexArray roughVertexArray;
        VertexBuffer roughVertexBuffer(roughWorldSpaceMesh);
        VertexBufferLayout roughLayout;
        roughLayout.push<float>(3);
        roughLayout.push<float>(3);
        roughVertexArray.addBuffer(roughVertexBuffer, roughLayout);
        IndexBuffer roughIndexBuffer(roughWorldSpaceMesh.modelSpaceMesh.triangles);

        VertexArray brilliantVertexArray;
        VertexBuffer brilliantVertexBuffer((WorldSpaceMesh(brilliantMesh)));
        VertexBufferLayout brilliantLayout;
        brilliantLayout.push<float>(3);
        brilliantLayout.push<float>(3);
        brilliantVertexArray.addBuffer(brilliantVertexBuffer, brilliantLayout);
        IndexBuffer brilliantIndexBuffer(brilliantMesh.triangles);

        VertexArray brilliantVertexArray2;
//        VertexBuffer brilliantVertexBuffer2((WorldSpaceMesh(brilliantMesh2)));
        VertexBuffer brilliantVertexBuffer2(brilliantVertexBuffer);
        IndexBuffer brilliantIndexBuffer2(brilliantIndexBuffer);
        VertexBufferLayout brilliantLayout2;
        brilliantLayout2.push<float>(3);
        brilliantLayout2.push<float>(3);
        brilliantVertexArray2.addBuffer(brilliantVertexBuffer2, brilliantLayout2);
//        IndexBuffer brilliantIndexBuffer2(brilliantMesh2.triangles);

        // Shader
        Shader shader("../../render/res/shaders/Basic.shader");
        Shader shader2("../../render/res/shaders/Intermediate.shader");

        Renderer renderer;

        /* Loop until the user closes the window */
        while (window.shouldClose()) {

            window.processInput();
            glm::mat4 modelTransformationMatrix = glm::mat4(1.0f);
            modelTransformationMatrix = glm::scale(modelTransformationMatrix, glm::vec3(2,2,2));
            modelTransformationMatrix = glm::translate(modelTransformationMatrix, glm::vec3(10,10,10));
            modelTransformationMatrix = glm::rotate(modelTransformationMatrix, 50.0f, glm::vec3(0,1,0));

            glm::mat4 projectionViewMatrix = window.getProjectionViewMatrix();
            glm::mat4 modelViewProjectionMatrix = projectionViewMatrix * modelTransformationMatrix;
            renderer.clear();

            // 1. DRAW STATIC SOLID (includes axis?)
            // 2. DRAW STATIC TRANSPARENT -- after sorting by depth?
            // No dynamic buffers needed

            shader.setUniform4fv("u_Color", glm::vec4(1, 1, 1, 1));
            shader.setUniformMat4f("u_ModelViewProjectionMatrix", projectionViewMatrix);

            // Bring the light source in model space
            glm::vec3 lightSource = glm::vec4(glm::vec3(0,-0,-200), 1.0f) *  window.getViewMatrix() * modelTransformationMatrix;
            shader2.setUniform3fv("u_LightSource", lightSource);
            shader2.setUniform1f("u_Ambient", 0.15f);
            shader2.setUniformMat4f("u_ModelViewProjectionMatrix", modelViewProjectionMatrix);

            renderer.drawLines(axisVertexArray, axisIndexBuffer, shader);

            shader2.setUniform4fv("u_Color", glm::vec4(1, 0, 0, 1));
            renderer.drawTriangles(brilliantVertexArray, brilliantIndexBuffer, shader2);

            shader2.setUniform4fv("u_Color", glm::vec4(0, 0.5, 0, 1));
            shader2.setUniformMat4f("u_ModelViewProjectionMatrix", glm::translate(modelViewProjectionMatrix, glm::vec3(0,15,0)));
            renderer.drawTriangles(brilliantVertexArray2, brilliantIndexBuffer2, shader2);
//
            shader2.setUniform4fv("u_Color", glm::vec4(0.6627f, 0.6627f, 0.6627f, 0.4f));
            shader2.setUniformMat4f("u_ModelViewProjectionMatrix", modelViewProjectionMatrix);
            renderer.drawTriangles(roughVertexArray, roughIndexBuffer, shader2);

//            roughRenderModel.setColor(glm::vec4(0.6627f, 0.6627f, 0.6627f, 1.0f));
//            roughRenderModel.draw(shader2);

            window.update();

        }
    }

    glfwTerminate();
    return 0;
}



