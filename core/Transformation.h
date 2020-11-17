//
// Created by Jonas on 9/11/2020.
//

#ifndef MESHCORE_TRANSFORMATION_H
#define MESHCORE_TRANSFORMATION_H
#include <glm/glm.hpp>
#include <iostream>
#include "Vertex.h"

typedef glm::dmat4 TMatrix;

class Transformation {
private:

    const TMatrix matrix;
    const TMatrix inverseMatrix{}; // TODO advantage of this way: not recalculating inverse matrix when not necessary

public:
    explicit Transformation();
    explicit Transformation(TMatrix matrix);
    explicit Transformation(TMatrix matrix, TMatrix inverseMatrix);

    // These can all be done trough glm transformations => only interesting to cache inverse?
    static Transformation fromScaleFactor(double scaleFactor);
    static Transformation fromEulerAngles(double yaw, double pitch, double roll);
    static Transformation fromTranslationVector(Vertex shift);
    static Transformation fromTranslation(double dx, double dy, double dz);
    Transformation operator*(const double & multiplier) const;
    Transformation operator*(const Transformation & multiplier) const;
    friend std::ostream & operator<<(std::ostream & os, const Transformation& transformation);

};

#endif //MESHCORE_TRANSFORMATION_H
