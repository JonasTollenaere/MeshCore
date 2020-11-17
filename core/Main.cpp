//
// Created by Jonas on 9/11/2020.
//
#include <iostream>
#include "Vertex.h"
#include "MeshTriangle.h"
#include "Ray.h"
#include <iomanip>
#include "Vertex.h"
#include "Transformation.h"
#include "FileParser.h"
#include <glm/gtx/normal.hpp>


#include <fstream>
#include <string>

#define print(x) std::cout << x << std::endl;

int mainz(){
//    std::cout << "Hello World!" << std::endl;
//
//    Vector vertex1 = Vector(1.23456789, 1.555555555, 1.0);
//    std::cout << vertex1 << std::endl;
//
//    Vector vertex2 = Vector(1.23456789, 1.555555555, 2.0);
//    Vector vertex3 = Vector(1.23456789, 1.555555555, 3.0);

//    MeshTriangle triangle = MeshTriangle(vertex1, vertex2, vertex3);
////    std::cout << triangle << std::endl;
//
//    Vector point0 = Vector(0,6,3);
//    Vector point1 = Vector(1,7,4);
//    Vector point2 = Vector(2,8,5);
//    Vector point3 = Vector(3,9,6);
//    Vector point4 = Vector(4,1,7);
//    Vector point5 = Vector(5,2,8);
//
//
//    std::cout << std::setprecision(4);
//
//    Ray ray1 = Ray(glm::dvec3(5.123456789,5,5.123456789),glm::dvec3(5.0,5,5));
//
//    std::cout << ray1 << std::endl;

//
//
//    Vertex vertex0 = Vertex(1, 0, 0);
//    Vertex vertex1 = Vertex(0, 1, 0);
//    Vertex vertex2 = Vertex(0, 0, 1);
//    MeshTriangle triangle = MeshTriangle(vertex0, vertex1, vertex2);
//
//    Vertex origin = Vertex(0, 0, 0);
//    Vertex direction = Vertex(1, 1, 1);
//    Ray ray = Ray(origin, direction);
//
//    std::cout << triangle << std::endl;
//    std::cout << ray << std::endl;
//
//    std::cout << "Distance: " << ray.intersects(triangle) << std::endl;
//
//    Transformation transformation = Transformation(5);
//    std::cout << transformation << std::endl;

    Vertex vertex0 = Vertex(8.513557e-01,  2.310529e-01,  4.467032e-02);
    Vertex vertex1 = Vertex(1.464905e+00,  1.392931e-02,  3.821188e-02);
    Vertex vertex2 = Vertex(1.119580e+00, -2.779771e-01,  1.327147e+00);
    Vertex normal = glm::triangleNormal(vertex0, vertex1, vertex2);
    std::cout << normal << std::endl;

    vertex0 = Vertex(8.513557e-01,  2.310529e-01,  4.467032e-02);
    vertex1 = Vertex(1.119580e+00, -2.779771e-01,  1.327147e+00);
    vertex2 = Vertex(3.202746e-01, -1.042207e-01,  1.323172e+00);
    normal = glm::triangleNormal(vertex0, vertex1, vertex2);
    std::cout << normal << std::endl;

    vertex0 = Vertex(3.202746e-01, -1.042207e-01,  1.323172e+00);
    vertex1 = Vertex(1.005635e-01,  3.097333e-01,  7.271256e-02);
    vertex2 = Vertex(8.513557e-01,  2.310529e-01,  4.467032e-02);
    normal = glm::triangleNormal(vertex0, vertex1, vertex2);
    std::cout << normal << std::endl;


    std::string path = "../../data/models/pig.obj";

    FileParser::parseFile(path);

//    FileParser::parseFile(stl_path);

    return 0;
}