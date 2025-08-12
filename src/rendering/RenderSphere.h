//
// Created by Jonas on 18/07/2022.
//

#ifndef MESHCORE_RENDERSPHERE_H
#define MESHCORE_RENDERSPHERE_H

#include <QOpenGLShaderProgram>
#include "AbstractRenderModel.h"
#include "meshcore/core/Sphere.h"

class RenderSphere: public AbstractRenderModel { // Due to this AbstractRenderModel inheritance,
private:

    class SingletonUnitySphere: QOpenGLFunctions{
    private:
        SingletonUnitySphere();
    public:
        static std::shared_ptr<SingletonUnitySphere>& getInstance();
        QOpenGLBuffer* indexBuffer;
        QOpenGLBuffer* vertexBuffer;
        QOpenGLVertexArrayObject* vertexArray;
    };

    const Sphere sphere;

    bool cullingEnabled = true;
    bool wireframeEnabled = false;

public:

    RenderSphere(const Sphere &sphere, const Transformation& transformation);

    void draw(const OpenGLWidget* openGLWidget, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix, bool lightMode) override;

    RenderModelDetailDialog *createRenderModelDetailDialog(QWidget *parent) override;

    QMenu *getContextMenu() override;

    bool isCullingEnabled() const;

    void setCullingEnabled(bool newCullingEnabled);

    bool isWireframeEnabled() const;

    void setWireframeEnabled(bool newWireframeEnabled);

};


#endif //MESHCORE_RENDERSPHERE_H
