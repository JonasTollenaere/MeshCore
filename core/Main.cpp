//
// Created by Jonas on 9/11/2020.
//
#include <iostream>
#include "Vector.h"
#include "Triangle.h"
#include "Ray.h"
#include <iomanip>
#include "Vector.h"
#include "Transformation.h"

int main(){
//    std::cout << "Hello World!" << std::endl;
//
//    Vector vertex1 = Vector(1.23456789, 1.555555555, 1.0);
//    std::cout << vertex1 << std::endl;
//
//    Vector vertex2 = Vector(1.23456789, 1.555555555, 2.0);
//    Vector vertex3 = Vector(1.23456789, 1.555555555, 3.0);

//    Triangle triangle = Triangle(vertex1, vertex2, vertex3);
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



    Vector vertex0 = Vector(1,0,0);
    Vector vertex1 = Vector(0,1,0);
    Vector vertex2 = Vector(0,0,1);
    Triangle triangle = Triangle(vertex0, vertex1,vertex2);

    Vector origin = Vector(0,0,0);
    Vector direction = Vector(1,1,1);
    Ray ray = Ray(origin, direction);

    std::cout << triangle << std::endl;
    std::cout << ray << std::endl;

    std::cout << "Distance: " << ray.intersects(triangle) << std::endl;

//    Transformation transformation = Transformation(5);
//    std::cout << transformation << std::endl;

}