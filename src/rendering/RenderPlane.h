//
// Created by Jonas on 15/05/2024.
//

#ifndef EXTENDEDMESHCORE_RENDERPLANE_H
#define EXTENDEDMESHCORE_RENDERPLANE_H

#include <QOpenGLShaderProgram>
#include "AbstractRenderModel.h"
#include "meshcore/core/Plane.h"
#include "RenderRay.h"

class RenderPlane: public AbstractRenderModel {

    bool cullingEnabled = true;
    bool normalEnabled = false;

    RenderRay renderNormal;

public:
    explicit RenderPlane(const Plane& plane, const Transformation& transformation=Transformation{});
    void draw(const OpenGLWidget *openGLWidget, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix, bool lightMode) override;

    QMenu* getContextMenu() override;

    bool isCullingEnabled() const;
    void setCullingEnabled(bool cullingEnabled);

    void setTransformation(const Transformation &transformation) override;
};

#endif //EXTENDEDMESHCORE_RENDERPLANE_H
