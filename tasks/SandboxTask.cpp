//
// Created by Jonas on 4/12/2020.
//

#include <QtWidgets>
#include <glm/ext/matrix_transform.hpp>
#include "SandboxTask.h"
#include "../core/FileParser.h"
#include "../core/TriangleTriangleIntersectModified.h"
#include "../core/TriangleTriangleIntersect.h"
#include <chrono>
#include <glm/gtc/type_ptr.hpp>

void SandboxTask::run(){

//    Vertex v0(-0.843433,-0.804675,-0.826513);
//    Vertex v1(-0.930694,0.0654775,-0.0837993);
//    Vertex v2(-0.151629,0.0382991,0.174226);
//    Vertex u0(-0.751919,0.660254,-0.302284);
//    Vertex u1(0.340251,-0.525589,0.642898);
//    Vertex u2(0.372815,0.0277787,-0.177663);
//
//    bool test = MIntersection::ExperimentalTriangleTriangleIntersection(v0, v1, v2, u0, u1, u2);
//    std::cout << "Experimental: " << test << std::endl;
//
//
//    bool test0 = Intersection::NoDivTriTriIsect(glm::value_ptr(v0), glm::value_ptr(v1), glm::value_ptr(v2),
//                                                glm::value_ptr(u0), glm::value_ptr(u1), glm::value_ptr(u2));
//    std::cout << "Default: " << test0 << std::endl;
//
//
//    std::mt19937 randomEngine(0); // TODO set back to 0
//    auto nextFloat = std::uniform_real_distribution<float>(-1, 1);
//    auto iterations = (1<<24);
//    for(int i=0; i<iterations; i++) {
//
//        Vertex v0(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex v1(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex v2(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//
//        Vertex u0(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex u1(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex u2(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//
//
//        bool test1 = MIntersection::ExperimentalTriangleTriangleIntersection(v0, v1, v2, u0, u1, u2);
//        bool test2 = Intersection::NoDivTriTriIsect(glm::value_ptr(v0), glm::value_ptr(v1), glm::value_ptr(v2),
//                                                    glm::value_ptr(u0), glm::value_ptr(u1), glm::value_ptr(u2));
//
//        if(test1!=test2){
//            std::cout << "Iteration " << i << " - Results not equal!" << std::endl;
//            std::cout << "Triangle 1:" << v0 << v1 << v2 << std::endl;
//            std::cout << "Triangle 2:" << u0 << u1 << u2 << std::endl;
//
//            std::cout << "Default: " << test2 << std::endl;
//            std::cout << "Experimental: " << test1 << std::endl;
//
//            test1 = MIntersection::ExperimentalTriangleTriangleIntersection(v0, v1, v2, u0, u1, u2);
//            test2 = Intersection::NoDivTriTriIsect(glm::value_ptr(v0), glm::value_ptr(v1), glm::value_ptr(v2),
//                                                        glm::value_ptr(u0), glm::value_ptr(u1), glm::value_ptr(u2));
//
////            experimental is about 20% slower in random [-1;1] tests
//
//            break;
//        }
//        else if(i%100000==0){
//            std::cout << "Iteration " << i << std::endl;
//        }
//    }
//
//    auto startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//    randomEngine = std::mt19937(0);
//    for(int i=0; i<iterations; i++) {
//
//        Vertex v0(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex v1(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex v2(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//
//        Vertex u0(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex u1(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex u2(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//
//
//        bool test1 = MIntersection::ExperimentalTriangleTriangleIntersection(v0, v1, v2, u0, u1, u2);
//    }
//
//    auto stopms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//    auto totalms = stopms - startms;
//    std::cout << totalms << std::endl;
//    std::cout << "Experimental: " << iterations << " calculations in " << (float)totalms/1000.0f << " s." << std::endl;
//
//    startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//    randomEngine = std::mt19937(0);
//    for(int i=0; i<iterations; i++) {
//
//        Vertex v0(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex v1(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex v2(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//
//        Vertex u0(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex u1(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex u2(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//
//
//        bool test2 = Intersection::NoDivTriTriIsect(glm::value_ptr(v0), glm::value_ptr(v1), glm::value_ptr(v2),
//                                                    glm::value_ptr(u0), glm::value_ptr(u1), glm::value_ptr(u2));
//    }
//
//    stopms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//    totalms = stopms - startms;
//    std::cout << totalms << std::endl;
//    std::cout << "Default: " << iterations << " calculations in " << (float)totalms/1000.0f << " s." << std::endl;
//
//    startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//    randomEngine = std::mt19937(0);
//    for(int i=0; i<iterations; i++) {
//
//        Vertex v0(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex v1(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex v2(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//
//        Vertex u0(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex u1(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//        Vertex u2(nextFloat(randomEngine),nextFloat(randomEngine),nextFloat(randomEngine));
//
//
//        bool test2 = MIntersection::NoDivTriTriIsect(v0, v1, v2, u0, u1, u2);
//    }
//
//    stopms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//    totalms = stopms - startms;
//    std::cout << totalms << std::endl;
//    std::cout << "GLM: " << iterations << " calculations in " << (float)totalms/1000.0f << " s." << std::endl;

    const ModelSpaceMesh innerModelMesh = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
//    const ModelSpaceMesh innerMesh = FileParser::parseFile("../../data/models/DIAMCADbr1.obj");
    WorldSpaceMesh innerMesh = WorldSpaceMesh(innerModelMesh, glm::scale(Transformation(1.0f), glm::vec3(0.2f)));
    this->renderMesh(innerMesh, glm::vec4(1, 0, 0, 1));
    const ModelSpaceMesh modelSpaceMesh5 = FileParser::parseFile("../../data/models/DIAMCADrough.obj");
    WorldSpaceMesh roughMesh = WorldSpaceMesh(modelSpaceMesh5/*, glm::translate(Transformation(1.0f), glm::vec3(0, -1, 0))*/);
//    roughMesh.transform(glm::scale(Transformation(1.0f), glm::vec3(1.2f)));
    this->renderMesh(roughMesh, glm::vec4(1, 1, 1, 0.4));

    std::cout << "Starting Sandbox Test" << std::endl;

    int value = roughMesh.calculateNumberOfIntersections(Ray(Vertex(0,0,0), glm::vec3(1,0,0)));

    std::cout << "Intersections: " << value << std::endl;

    std::cout << "Starting Random Walk CPU" << std::endl;

//    std::mt19937 randomEngine(0); // TODO set back to 0
//    auto nextFloat = std::uniform_real_distribution<float>(0, 1);
//
//    Transformation currentTransformation = innerMesh.getModelTransformation();
//    std::cout << std::boolalpha;
//    int moves = 1000;
//
//    auto startms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//    for(int i=0; i<moves; i++){
//
//        std::cout << i << std::endl;
//
//        Transformation newTransformation = glm::scale(currentTransformation, glm::vec3(0.85 + 0.4 * nextFloat(randomEngine)));
//        newTransformation = glm::rotate(newTransformation, nextFloat(randomEngine), glm::vec3(nextFloat(randomEngine) - 0.5, nextFloat(randomEngine)-0.5, nextFloat(randomEngine)-0.5));
//        newTransformation = glm::translate(newTransformation, glm::vec3(nextFloat(randomEngine) - 0.5f, nextFloat(randomEngine) - 0.5f,nextFloat(randomEngine)  - 0.5f));
//        innerMesh.setModelTransformation(newTransformation);
//
//        // Ray triangle test
//        bool feasible;
//        if(!innerMesh.rayTriangleInside(roughMesh)){
//            feasible = false;
//        }
//        else{
//            feasible = !innerMesh.triangleTriangleIntersects(roughMesh);
//        }
//
//        if(feasible){
//            currentTransformation = newTransformation;
//            this->updateRenderMesh(innerMesh);
//        }
//        else{
////            innerMesh.setModelTransformation(currentTransformation);
//        }
//    }
//
//    auto stopms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//    auto totalms = stopms - startms;
//
//    std::cout << totalms << std::endl;
//
//    std::cout << currentTransformation << std::endl;
//
//    std::cout << "MPS: " << float(moves)/float(totalms)*1000.0f << std::endl;
}