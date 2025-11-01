//
// Created by Jonas on 18/05/2022.
//

#ifndef MESHCORE_RENDERMESH_H
#define MESHCORE_RENDERMESH_H


#include <QOpenGLShaderProgram>
#include "AbstractRenderModel.h"
#include "RenderAABB.h"
#include "RenderLine.h"
#include "meshcore/rendering/PhongMaterial.h"
#include "RenderRay.h"

class RenderMesh: public AbstractRenderModel {
private:

    enum MeshTexture {
        HIDDEN,
        UNIFORM,
        TRIANGLES,
        FACES
    };

    enum WireframeMode {
        DISABLED,
        TRIANGLE_EDGES,
        FACE_EDGES
    };

    MeshTexture renderedTexture = MeshTexture::UNIFORM;
    WireframeMode wireframeMode = WireframeMode::DISABLED;

    QOpenGLBuffer* faceIndexBuffer;
    QOpenGLBuffer* faceVertexBuffer;
    QOpenGLVertexArrayObject* faceVertexArray;

    QOpenGLBuffer* faceEdgeIndexBuffer;
    QOpenGLBuffer* faceEdgeVertexBuffer;
    QOpenGLVertexArrayObject* faceEdgeVertexArray;

    bool cullingEnabled = true;
    bool boundingBoxEnabled = false;
    bool axisEnabled = false;
    bool normalsEnabled = false;

    RenderAABB boundingBox;
    std::vector<std::shared_ptr<AbstractRenderModel>> axisRenderModels;
    std::vector<std::shared_ptr<AbstractRenderModel>> normalRenderModels;

    unsigned int numberOfVertices;
    unsigned int numberOfFaces;
    unsigned int numberOfTriangles;
    float unscaledVolume;
    float unscaledSurfaceArea;

public:
//    RenderModel();
//    RenderModel(const RenderModel& other);
//    RenderMesh(RenderMesh&& other) noexcept;
    RenderMesh& operator=(RenderMesh&& other) noexcept;
    ~RenderMesh() override = default;

    explicit RenderMesh(const WorldSpaceMesh &worldSpaceMesh);
    void draw(const OpenGLWidget* openGLWidget, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix, bool lightMode) override;

    [[nodiscard]] bool isCullingEnabled() const;
    [[nodiscard]] bool isBoundingBoxEnabled() const;
    [[nodiscard]] bool isAxisEnabled() const;
    [[nodiscard]] bool isNormalsEnabled() const;
    [[nodiscard]] MeshTexture getRenderedTexture() const;
    [[nodiscard]] WireframeMode getWireframeMode() const;

    void setCullingEnabled(bool newCullingEnabled);
    void setBoundingBoxEnabled(bool newBoundingBoxEnabled);
    void setAxisEnabled(bool axisEnabled);
    void setNormalsEnabled(bool newNormalsEnabled);
    void setRenderedTexture(MeshTexture newRenderedTexture);
    void setWireframeMode(WireframeMode newWireframeMode);

    QMenu* getContextMenu() override;

    RenderModelDetailDialog* createRenderModelDetailDialog(QWidget* parent) override;

    void setMaterial(const PhongMaterial& newMaterial) override;
    void setTransformation(const Transformation &newTransformation) override;
};


#endif //MESHCORE_RENDERMESH_H
