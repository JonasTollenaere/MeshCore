//
// Created by Jonas on 10/11/2020.
//

#include "Renderer.h"
#include "VertexBufferLayout.h"
#include "Window.h"
#include "../core/FileParser.h"
#include "RenderModel.h"


int main() {

    const std::string path = "../../data/models/DIAMCADbr1.obj";
    const ModelSpaceMesh brilliantMesh = FileParser::parseFile(path);

    const std::string path2 = "../../data/models/DIAMCADrough.obj";
    const ModelSpaceMesh roughMesh = FileParser::parseFile(path2);

    const std::string path3 = "../../data/models/DIAMCADbr2.obj";
    const ModelSpaceMesh brilliantMesh2 = FileParser::parseFile(path3);

    const std::string path4 = "../../data/models/dragon.obj";
    const ModelSpaceMesh dragonMesh = FileParser::parseFile(path4);
    const WorldSpaceMesh dragonWorldSpaceMesh(dragonMesh);

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

        WorldSpaceMesh roughWorldSpaceMesh(roughMesh);
        RenderModel roughRenderModel(roughWorldSpaceMesh);
        roughRenderModel.setColor(glm::vec4(0.6627f, 0.6627f, 0.6627f, 0.4f));

        WorldSpaceMesh brilliantWorldSpaceMesh(brilliantMesh);
        RenderModel brilliantModel = RenderModel(brilliantWorldSpaceMesh);
        brilliantModel.setColor(Color(1,0,0,1));

        WorldSpaceMesh brilliantWorldSpaceMesh2(brilliantMesh2);
        RenderModel brilliant2Model = RenderModel(brilliantWorldSpaceMesh2);
        brilliant2Model.setColor(Color(0,0.5,0,1));

        RenderModel dragonRenderModel = RenderModel(dragonWorldSpaceMesh);
        // Shader
        Shader shader("../../render/res/shaders/Basic.shader");
        Shader shader2("../../render/res/shaders/Intermediate.shader");

        Renderer renderer;

        /* Loop until the user closes the window */
        while (window.shouldClose()) {

            window.processInput();
//            glm::mat4 modelTransformationMatrix = glm::mat4(1.0f);
//            modelTransformationMatrix = glm::scale(modelTransformationMatrix, glm::vec3(2,2,2));
//            modelTransformationMatrix = glm::translate(modelTransformationMatrix, glm::vec3(10,10,10));
//            modelTransformationMatrix = glm::rotate(modelTransformationMatrix, 50.0f, glm::vec3(0,1,0));

            glm::mat4 projectionViewMatrix = window.getProjectionViewMatrix();renderer.clear();

            shader.setUniform4fv("u_Color", glm::vec4(1, 1, 1, 1));
            shader.setUniformMat4f("u_ModelViewProjectionMatrix", projectionViewMatrix);
            renderer.drawLines(axisVertexArray, axisIndexBuffer, shader);

            glm::vec3 viewSpaceLightDirection = glm::vec4(glm::vec3(0, 0, 1), 1.0f) * window.getViewMatrix();

//            brilliantWorldSpaceMesh.transform(glm::rotate(glm::mat4(1.0f), 0.005f, glm::vec3(1.0f,0,0)));

            dragonRenderModel.draw(shader2, projectionViewMatrix, viewSpaceLightDirection);

            brilliantModel.draw(shader2, projectionViewMatrix, viewSpaceLightDirection);
            brilliant2Model.draw(shader2, projectionViewMatrix, viewSpaceLightDirection);
            roughRenderModel.draw(shader2, projectionViewMatrix, viewSpaceLightDirection); // Drawing transparent objects last stays inmportant

            window.update();

        }
    }

    glfwTerminate();
    return 0;
}



