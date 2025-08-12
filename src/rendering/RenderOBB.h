//
// Created by Jonas on 26/08/2022.
//

#ifndef MESHCORE_RENDEROBB_H
#define MESHCORE_RENDEROBB_H

#include <QOpenGLShaderProgram>
#include "AbstractRenderModel.h"
#include "meshcore/core/OBB.h"

class RenderOBB: public AbstractRenderModel {
private:
    Quaternion obbRotation;
    float unscaledSurfaceArea;
    float unscaledVolume;
public:

    RenderOBB(const OBB &obb, const Transformation& transformation);

    RenderModelDetailDialog *createRenderModelDetailDialog(QWidget* parent) override;

    void draw(const OpenGLWidget* openGLWidget, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix, bool lightMode) override;
};

#endif //MESHCORE_RENDEROBB_H
