//
// Created by Jonas on 29/03/2022.
//

#ifndef MESHCORE_RENDERWIDGET_H
#define MESHCORE_RENDERWIDGET_H

#include <QApplication>
#include <unordered_map>
#include <iostream>
#include <QTreeWidgetItem>

#include "meshcore/acceleration/AbstractBoundsTree.h"
#include "meshcore/rendering/OpenGLWidget.h"
#include "meshcore/rendering/KeyFrame.h"
#include "meshcore/core/WorldSpaceMesh.h"
#include "meshcore/tasks/AbstractTaskObserver.h"
#include "meshcore/tasks/AbstractTask.h"
#include "meshcore/core/Plane.h"

QT_BEGIN_NAMESPACE
namespace Ui { class RenderWidget; }
QT_END_NAMESPACE

class RenderWidget: public QWidget, public AbstractTaskObserver {
Q_OBJECT

public:
    explicit RenderWidget(QWidget *parent = nullptr);
    ~RenderWidget() override;
    [[nodiscard]] OpenGLWidget* getOpenGLWidget() const;

private:
    Ui::RenderWidget *ui;
    std::unordered_map<std::string, QTreeWidgetItem *> groupTreeWidgetItems;
    QTreeWidgetItem * getOrAddGroupWidget(const std::string &group);

public:

    void captureScene() const;
    void captureSceneToFile(const std::string& fileName) const;
    void captureAnimation() const;

    void captureLinearAnimation(const Transformation &initialViewTransformation, const Transformation &finalViewTransformation,
                                const KeyFrame &initialKeyFrame, const KeyFrame &finalKeyFrame,
                                const std::string& fileName, int steps, int delay);

    void captureLinearAnimation(const Transformation &initialViewTransformation, const Transformation &finalViewTransformation, const std::string& fileName, int steps, int delay);


    void clear();
    void clearGroup(const std::string &group);

    void setViewTransformation(const Transformation& transformation) const;
    void resetViewTransformation() const;
    void setView(size_t i) const;

private:
    // Render generic objects that extend AbstractRenderModel
    // This would imply that programmers make their own render models, which can't happen outside the main thread, this is not a good practice
    void addOrUpdateRenderModel(const std::string& group, const std::string& id, const std::shared_ptr<AbstractRenderModel> &renderModel);

public:
    // Render objects with ids like meshes
    void renderWorldSpaceMesh(const std::string &group, const std::shared_ptr<WorldSpaceMesh> &worldSpaceMesh, const Color& color = Color::White());
    void renderWorldSpaceMesh(const std::string &group, const std::shared_ptr<WorldSpaceMesh> &worldSpaceMesh,  const PhongMaterial& material);

    void renderBox(const std::string &group, const std::string& name, const AABB &aabb, const Transformation& transformation=Transformation(), const Color& = Color::White());
    void renderPlane(const std::string &group, const std::string &name, const Plane &plane, const Color &color);
    void renderRay(const std::string &group, const std::string& name, const Ray &ray, const Color &color = Color::White(), float widthLengthRatio=0.1f);
    void renderSphere(const std::string &group, const std::string& name, const Sphere &sphere, const Color &color = Color::White());
    void renderSphere(const std::string &group, const std::string& name, const Sphere &sphere, const PhongMaterial& material = PhongMaterial(Color::White()));
    void renderTriangle(const std::string &group, const std::string& name, const VertexTriangle &triangle, const Color &color = Color::White());
    void renderLine(const std::string &group,  const std::string& name, const Vertex &vertexA, const Vertex &vertexB, const Color &color = Color::White());

    // For the moment, the classes extending AbstractBoundsTree have to be added explicitly to be able to render them
    void renderBoundsTree(const std::string &group, const std::string& name, const std::shared_ptr<AbstractBoundsTree<AABB, 8, false>>& aabbTree, const Transformation& transformation, const Color &color=Color::White());
    void renderBoundsTree(const std::string &group, const std::string& name, const std::shared_ptr<AbstractBoundsTree<AABB, 2, true>>& aabbTree, const Transformation& transformation, const Color &color=Color::White());
    void renderBoundsTree(const std::string &group, const std::string& name, const std::shared_ptr<AbstractBoundsTree<OBB, 8, true>>& aabbTree, const Transformation& transformation, const Color &color=Color::White());
    void renderBoundsTree(const std::string &group, const std::string& name, const std::shared_ptr<AbstractBoundsTree<OBB, 2, true>>& aabbTree, const Transformation& transformation, const Color &color=Color::White());
    void renderBoundsTree(const std::string &group, const std::string& name, const std::shared_ptr<AbstractBoundsTree<Sphere, 8, true>>& aabbTree, const Transformation& transformation, const Color &color=Color::White());
    void renderBoundsTree(const std::string &group, const std::string& name, const std::shared_ptr<AbstractBoundsTree<Sphere, 2, true>>& aabbTree, const Transformation& transformation, const Color &color=Color::White());

    void addControlWidget(const std::string &group, const std::shared_ptr<AbstractRenderModel> &renderModel);

    void observeTask(AbstractTask* task);
    void observeTask(AbstractTask* task, const std::function<void(RenderWidget* renderWidget, std::shared_ptr<const AbstractSolution> solution)>& solutionRenderCallback);

    void setSolutionRenderCallback(const std::function<void(RenderWidget *renderWidget, const std::shared_ptr<const AbstractSolution> &solution)> &solutionRenderCallback);
    void setDefaultSolutionRenderCallback();

    void startCurrentTask() const;
    void stopCurrentTask() const;

    void notifyStarted() override;
    void notifyFinished() override;
    void notifyProgress(float progress) override;
    void notifyStatus(const std::string &status) override;
    void notifySolution(const std::shared_ptr<const AbstractSolution>& solution) override;

private:
    AbstractTask* currentTask = nullptr;
    std::function<void(RenderWidget* renderWidget, const std::shared_ptr<const AbstractSolution>& solution)> solutionRenderCallback  = {};
    std::thread timerThread{};
    std::atomic<bool> taskRunning = false;
};

#endif //MESHCORE_RENDERWIDGET_H
