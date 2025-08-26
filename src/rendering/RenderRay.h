//
// Created by Jonas on 13/06/2024.
//

#ifndef EXTENDEDMESHCORE_RENDERRAY_H
#define EXTENDEDMESHCORE_RENDERRAY_H

#include "AbstractRenderModel.h"
#include "meshcore/core/Ray.h"

class RenderRay: public AbstractRenderModel {
public:
    void setTransformation(const Transformation &transformation) override;

    explicit RenderRay(const Ray& ray, const Transformation& transformation=Transformation{}, float widthLengthRatio=0.10);

    void setMaterial(const PhongMaterial &material) override;

    void draw(const OpenGLWidget *openGLWidget, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix, bool lightMode) override;
};

#endif //EXTENDEDMESHCORE_RENDERRAY_H
