//
// Created by Jonas on 17/11/2020.
//

#ifndef MESHCORE2_TRIANGLE_H
#define MESHCORE2_TRIANGLE_H

struct IndexTriangle {
    size_t vertexIndex0{};
    size_t vertexIndex1{};
    size_t vertexIndex2{};

    IndexTriangle(const size_t vertexIndex0, const size_t vertexIndex1, const size_t vertexIndex2): vertexIndex0(vertexIndex0), vertexIndex1(vertexIndex1), vertexIndex2(vertexIndex2) {}
    IndexTriangle(const IndexTriangle& other): vertexIndex0(other.vertexIndex0), vertexIndex1(other.vertexIndex1), vertexIndex2(other.vertexIndex2) {}
};

#endif //MESHCORE2_TRIANGLE_H
