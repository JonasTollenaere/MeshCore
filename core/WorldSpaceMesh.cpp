//
// Created by Jonas on 9/11/2020.
//

#include "WorldSpaceMesh.h"

WorldSpaceMesh::WorldSpaceMesh(ModelSpaceMesh& modelSpaceMesh): modelSpaceMesh(modelSpaceMesh), transformation(Transformation()) {
}
//
//void WorldSpaceMesh::transform(const Transformation &t) {
////    this->transformation = this->transformation * t;
//}
//
//void WorldSpaceMesh::setTransformation(const Transformation &t) {
//    // TODO
//}
//
//Transformation WorldSpaceMesh::getTransformation() {
//    // TODO
//}
