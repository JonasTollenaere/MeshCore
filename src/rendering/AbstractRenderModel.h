//
// Created by Jonas on 8/11/2021.
//

#ifndef MESHCORE_ABSTRACTRENDERMODEL_H
#define MESHCORE_ABSTRACTRENDERMODEL_H

#include <glm/glm.hpp>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <Qmenu.h>
#include "RenderModelDetailDialog.h"
#include "meshcore/rendering/PhongMaterial.h"
#include "RenderListeners.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

class OpenGLWidget;

class AbstractRenderModel: protected QOpenGLFunctions {

    std::string name;
    RenderModelDetailDialog* detailDialog = nullptr;
    bool visible = true;
    PhongMaterial material = PhongMaterial(Color(1.0f));

protected:
    QOpenGLBuffer* indexBuffer;
    QOpenGLBuffer* vertexBuffer;
    QOpenGLVertexArrayObject* vertexArray;
    Transformation transformation;
    std::vector<std::shared_ptr<AbstractRenderModelListener>> listeners;

public:
//    AbstractRenderModel() = delete;
    explicit AbstractRenderModel(const Transformation& transformation, const std::string& name);
    AbstractRenderModel(AbstractRenderModel&& other) noexcept;
    AbstractRenderModel& operator=(AbstractRenderModel&& other) noexcept;
    virtual ~AbstractRenderModel();

    virtual void draw(const OpenGLWidget* openGLWidget, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix, bool lightMode) = 0;

public:
    [[nodiscard]] const std::string &getName() const;
    void setName(const std::string &newName);
    [[nodiscard]] bool isVisible() const;
    void setVisible(bool newVisible);
    [[nodiscard]] const PhongMaterial &getMaterial() const;
    virtual void setMaterial(const PhongMaterial &material);
    [[nodiscard]] glm::mat4 getTransformationMatrix() const;
    virtual void setTransformation(const Transformation &transformation);
    const Transformation& getTransformation() const;
    Transformation& getTransformation();

    RenderModelDetailDialog* getDetailsDialog(QWidget* parent);
    virtual RenderModelDetailDialog* createRenderModelDetailDialog(QWidget* parent);
    virtual QMenu* getContextMenu();

    void addListener(const std::shared_ptr<AbstractRenderModelListener> &listener) {
        this->listeners.push_back(listener);
    }

    void removeListener(const std::shared_ptr<AbstractRenderModelListener> &listener) {
        this->listeners.erase(std::remove(this->listeners.begin(), this->listeners.end(), listener), this->listeners.end());
    }
};

#endif //MESHCORE_ABSTRACTRENDERMODEL_H
