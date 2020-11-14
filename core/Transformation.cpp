//
// Created by Jonas on 9/11/2020.
//

#include "Transformation.h"

/// Constructs the identity transformation containing identity matrices for the transformation and its inverse
Transformation::Transformation(): matrix(TMatrix(1)), inverseMatrix(TMatrix(1)) {}

/// Constructs a transformation from a given transformation matrix
Transformation::Transformation(TMatrix matrix): matrix(matrix), inverseMatrix(glm::inverse(matrix)) {}

/// Constructs a transformation from a given transformation and inverse matrix
Transformation::Transformation(TMatrix matrix, TMatrix inverseMatrix): matrix(matrix), inverseMatrix(inverseMatrix) {}

/// Constructs a transformation from a given scaling factor
Transformation Transformation::fromScaleFactor(double scaleFactor) {
    return  Transformation(TMatrix(scaleFactor));
}

/// Constructs a transformation from given Euler angles
Transformation Transformation::fromEulerAngles(double yaw, double pitch, double roll) {

    // According to the matrix and quaternion FAQ -- http://web.archive.org/web/20060914224155/http://web.archive.org/web/20041029003853/http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q32
    double A = glm::cos(pitch);
    double B = glm::sin(pitch);
    double C = glm::cos(yaw);
    double D = glm::sin(yaw);
    double E = glm::cos(roll);
    double F = glm::sin(roll);

    double AD = A * D;
    double BD = B * D;

    TMatrix matrix = TMatrix(C*E, -C * F, D, 0,
                             BD * E + A * F, -BD * F + A * E, - B * C, 0,
                             - AD * E + B * F, AD * F + B * E, A * C, 0,
                             0, 0,0, 1);

    return Transformation(matrix);
}

/// Constructs a transformation from a given translation vector
Transformation Transformation::fromTranslationVector(Vector shift){
    TMatrix matrix = TMatrix (1,0,0,shift.x,
                              0,1,0,shift.y,
                              0,0,1,shift.z,
                              0,0,0,1);

    return Transformation(matrix);
}

/// Constructs a transformation from given translation values along each axis
Transformation Transformation::fromTranslation(double dx, double dy, double dz) {

    TMatrix matrix = TMatrix (1,0,0,dx,
                              0,1,0,dy,
                              0,0,1,dz,
                              0,0,0,1);

    return Transformation(matrix);
}

std::ostream &operator<<(std::ostream &os, const Transformation &transformation) {
    TMatrix m = transformation.matrix;
    return os
    << "[" << m[0][0] << "," << m[0][1]  << "," << m[0][2]  << "," <<  m[0][3] << std::endl
    << " " << m[1][0] << "," << m[1][1]  << "," << m[1][2]  << "," <<  m[1][3] << std::endl
    << " " << m[2][0] << "," << m[2][1]  << "," << m[2][2]  << "," <<  m[2][3] << std::endl
    << " " << m[3][0] << "," << m[3][1]  << "," << m[3][2]  << "," <<  m[3][3] << "]" << std::endl;
}

Transformation Transformation::operator*(const double & multiplier) const{
    return Transformation(this->matrix*multiplier);
}

Transformation Transformation::operator*(const Transformation & multiplier) const{
    return Transformation(this->matrix*multiplier.matrix);
}