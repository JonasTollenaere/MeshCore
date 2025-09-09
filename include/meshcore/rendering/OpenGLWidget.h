//
// Created by Jonas on 18/05/2022.
//

#ifndef MESHCORE_OPENGLWIDGET_H
#define MESHCORE_OPENGLWIDGET_H

#include "meshcore/core/Plane.h"
#include "meshcore/core/Ray.h"
#include "meshcore/core/Sphere.h"
#include "meshcore/core/VertexTriangle.h"
#include "meshcore/core/WorldSpaceMesh.h"
#include "meshcore/rendering/KeyFrame.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <glm/glm.hpp>
#include <unordered_map>

#define INITIAL_VIEW_DISTANCE 50.0f
#define INITIAL_FOV 86.0f

class AbstractRenderModel;
class AbstractRenderGroupListener;
class RenderWidget;
class GifWrapper;

class OpenGLWidget: public QOpenGLWidget, protected QOpenGLFunctions {
Q_OBJECT

private:

    bool lightMode = false;
    bool usePerspective = false;
    bool axisEnabled = false;

    int width{};
    int height{};
    QPoint lastMousePosition;

    Transformation viewTransformation;
    glm::mat4 projectionMatrix{};

    // Render models
    mutable std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<AbstractRenderModel>>> groupedRenderModelsMap;
    std::unordered_map<std::string, std::vector<std::shared_ptr<AbstractRenderGroupListener>>> groupListeners;
    std::vector<std::shared_ptr<AbstractRenderModel>> sortedRenderModels;
    std::vector<std::shared_ptr<AbstractRenderModel>> axisRenderModels;

    // Animation
    std::shared_ptr<GifWrapper> gifWrapper;

private:
    std::shared_ptr<QOpenGLShaderProgram> ambientShader;
    std::shared_ptr<QOpenGLShaderProgram> diffuseShader;
    std::shared_ptr<QOpenGLShaderProgram> phongShader;
    std::shared_ptr<QOpenGLShaderProgram> polyChromeShader;
public:
    const std::shared_ptr<QOpenGLShaderProgram> &getAmbientShader() const;
    const std::shared_ptr<QOpenGLShaderProgram> &getDiffuseShader() const;
    const std::shared_ptr<QOpenGLShaderProgram> &getPhongShader() const;
    const std::shared_ptr<QOpenGLShaderProgram> &getPolyChromeShader() const;

    const Transformation &getViewTransformation() const;

public:
    [[maybe_unused]] explicit OpenGLWidget(QWidget *parent = nullptr);
    ~OpenGLWidget() override;

public:

    [[nodiscard]] bool isUsePerspective() const;
    [[nodiscard]] bool isLightMode() const;
    [[nodiscard]] bool isAxisEnabled() const;
    void setUsePerspective(bool newUsePerspective);
    void setLightMode(bool newLightMode);
    void setAxisEnabled(bool enabled);
    void addGroupListener(const std::string& group, const std::shared_ptr<AbstractRenderGroupListener>& listener);
    void removeGroupListener(const std::string& group, const std::shared_ptr<AbstractRenderGroupListener>& listener);

    int getWidth() const;
    int getHeight() const;

    std::vector<uint8_t> capturePixelBufferSlot();

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void calculateProjectionMatrix();

    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void resetView();
    void setView(size_t viewIndex);
    void clear();
    void clearGroup(const std::string &group);
    void setViewTransformation(const Transformation &newViewTransformation);
    void setGroupVisible(const std::string &group, bool visible);
    void renderWorldSpaceMeshSlot(const std::string &group, const std::shared_ptr<WorldSpaceMesh> &worldSpaceMesh, const PhongMaterial &material, RenderWidget* renderWidget);
    void renderPlaneSlot(const std::string &group, const std::string &name, const Plane &plane, const PhongMaterial& material, RenderWidget* renderWidget);
    void renderBoxSlot(const std::string &group, const std::string &name, const AABB &aabb, const Transformation& transformation, const PhongMaterial& material, RenderWidget *renderWidget);
    void renderRaySlot(const std::string &group, const std::string &name, const Ray &ray, const PhongMaterial& material, float widthLengthRatio, RenderWidget* renderWidget);
    void renderSphereSlot(const std::string &group, const std::string &name, const Sphere &sphere, const PhongMaterial& material, RenderWidget* renderWidget);
    void renderTriangleSlot(const std::string &group, const std::string &name, const VertexTriangle &triangle, const PhongMaterial& material, RenderWidget* renderWidget);
    void renderLineSlot(const std::string &group, const std::string &name, const glm::vec3 &start, const glm::vec3 &end, const PhongMaterial& material, RenderWidget* renderWidget);

    void addOrUpdateRenderModelSlot(const std::string& group, const std::string& id, std::shared_ptr<AbstractRenderModel> sharedPtr, RenderWidget* renderWidget);
    void captureSceneSlot();
    void captureAnimationSlot();
    void captureSceneToFileSlot(const QString& fileName);

    void captureLinearAnimationSlot(const Transformation& initialViewTransformation, const Transformation& finalViewTransformation,
                                    const KeyFrame& initialKeyFrame, const KeyFrame& finalKeyFrame,
                                    const QString& fileName, int steps, int delay, RenderWidget* renderWidget);

    void openAnimationSlot(const std::string& path);
    void appendAnimationSlot();
    void closeAnimationSlot();

private:
    std::unordered_map<std::string, std::shared_ptr<AbstractRenderModel>>& getOrInsertRenderModelsMap(const std::string &group) const;
    void updateSortedRenderModels();
    void addRenderModelListeners(const std::string& group, const std::shared_ptr<AbstractRenderModel>& renderModel);
};


#endif //MESHCORE_OPENGLWIDGET_H
