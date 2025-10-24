//
// Created by Jonas on 29/03/2022.
//

#include <iomanip>
#include <sstream>
#include <qthread.h>

#include "meshcore/rendering/RenderWidget.h"


#include "RenderBoundsTree.h"
#include "RenderModelControlWidget.h"
#include "RenderGroupControlWidget.h"
#include "meshcore/core/Plane.h"

#include "forms/ui_renderwidget.h"
#include "meshcore/optimization/SingleVolumeMaximisationSolution.h"
#include "meshcore/optimization/StripPackingSolution.h"

RenderWidget::RenderWidget(QWidget *parent): QWidget(parent), ui(new Ui::RenderWidget) {

    ui->setupUi(this);
    ui->progressBar->setMinimumWidth(350);

    this->ui->taskSection->setVisible(false);

    // Connect the start and stop buttons
    connect(this->ui->startButton, &QPushButton::clicked, this, &RenderWidget::startCurrentTask);
    connect(this->ui->stopButton, &QPushButton::clicked, this, &RenderWidget::stopCurrentTask);

    // Hide the header of the tree widget
    auto& tree = this->ui->treeWidget;
    tree->headerItem()->setHidden(true);
    tree->setColumnCount(1);

    // Set the default SolutionRenderCallback
    setDefaultSolutionRenderCallback();
}

RenderWidget::~RenderWidget() {
    delete ui;
    if(this->currentTask != nullptr){
        currentTask->stop();
        currentTask->join();
    }
    if(timerThread.joinable()){
        this->timerThread.join();
    }
}

OpenGLWidget *RenderWidget::getOpenGLWidget() const {
    return ui->openGLWidget;
}

void RenderWidget::renderWorldSpaceMesh(const std::string &group, const std::shared_ptr<WorldSpaceMesh> &worldSpaceMesh, const Color& color) {
    renderWorldSpaceMesh(group, *worldSpaceMesh, PhongMaterial(color));
}

void RenderWidget::renderWorldSpaceMesh(const std::string &group, const std::shared_ptr<WorldSpaceMesh> &worldSpaceMesh,  const PhongMaterial& material) {
    renderWorldSpaceMesh(group, *worldSpaceMesh, material);
}

void RenderWidget::renderWorldSpaceMesh(const std::string &group, const WorldSpaceMesh &worldSpaceMesh,  const Color& color) {
    renderWorldSpaceMesh(group, worldSpaceMesh, PhongMaterial(color));
}

void RenderWidget::renderWorldSpaceMesh(const std::string &group, const WorldSpaceMesh &worldSpaceMesh,  const PhongMaterial& material) {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "renderWorldSpaceMeshSlot",
                              Qt::AutoConnection,
                              Q_ARG(std::string, group),
                              Q_ARG(WorldSpaceMesh, WorldSpaceMesh(worldSpaceMesh)), // We should copy the actual worldSpaceMesh object here, otherwise the transformation could change before the render thread reads it
                              Q_ARG(PhongMaterial, material),
                              Q_ARG(RenderWidget*, this));
}

void RenderWidget::captureLinearAnimation(const Transformation& initialViewTransformation, const Transformation& finalViewTransformation,
                                          const KeyFrame& initialKeyFrame, const KeyFrame& finalKeyFrame,
                                          const std::string& fileName, int steps, int delay){
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "captureLinearAnimationSlot",
                              Qt::BlockingQueuedConnection,
                              Q_ARG(Transformation , initialViewTransformation),
                              Q_ARG(Transformation , finalViewTransformation),
                              Q_ARG(KeyFrame , initialKeyFrame),
                              Q_ARG(KeyFrame , finalKeyFrame),
                              Q_ARG(QString, QString::fromStdString(fileName)),
                              Q_ARG(int, steps),
                              Q_ARG(int, delay),
                              Q_ARG(RenderWidget*, this));
}

void RenderWidget::captureLinearAnimation(const Transformation& initialViewTransformation, const Transformation& finalViewTransformation,
                                          const std::string& fileName, int steps, int delay){
    this->captureLinearAnimation(initialViewTransformation, finalViewTransformation, {}, {}, fileName, steps, delay);
}

void RenderWidget::addControlWidget(const std::string &group, const std::shared_ptr<AbstractRenderModel> &renderModel) {
    auto groupTreeWidgetItem = this->getOrAddGroupWidget(group);
    auto childTreeWidgetItem = new QTreeWidgetItem();
    groupTreeWidgetItem->addChild(childTreeWidgetItem);
    this->ui->treeWidget->setItemWidget(childTreeWidgetItem, 0, new RenderModelControlWidget(renderModel, this));

}

void RenderWidget::clear() {

//    // clear all layouts, on the UI thread
    QMetaObject::invokeMethod(this, [&]{
        ui->treeWidget->clear();
        groupTreeWidgetItems.clear();
        QMetaObject::invokeMethod(this->getOpenGLWidget(), "clear", Qt::AutoConnection);
    });
}

void RenderWidget::clearGroup(const std::string &group) {
    QMetaObject::invokeMethod(this, [&, group] {

        auto& tree = this->ui->treeWidget;

        if(groupTreeWidgetItems.find(group) != groupTreeWidgetItems.end()){
            auto groupTreeWidgetItem = groupTreeWidgetItems.at(group);
            for (const auto &item: groupTreeWidgetItem->takeChildren()){
                delete item;
            }

            groupTreeWidgetItems.erase(group);
            for (int i = 0; i < tree->topLevelItemCount(); ++i){
                auto item = tree->topLevelItem(i);
                if (item == groupTreeWidgetItem){
                    delete tree->takeTopLevelItem(i);
                    break;
                }
            }
        }
        QMetaObject::invokeMethod(this->getOpenGLWidget(), "clearGroup", Qt::AutoConnection, Q_ARG(std::string, group));
    });
}

void RenderWidget::setViewTransformation(const Transformation &transformation) const {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "setViewTransformation", Qt::AutoConnection, Q_ARG(Transformation, transformation));
}

void RenderWidget::resetViewTransformation() const {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "resetView", Qt::AutoConnection);
}

void RenderWidget::setView(size_t i) const {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "setView", Qt::AutoConnection, Q_ARG(size_t, i));
}

QTreeWidgetItem *RenderWidget::getOrAddGroupWidget(const std::string &group) {
    // Find the group
    auto iterator = groupTreeWidgetItems.find(group);

    // Add new group if not found
    if(iterator == groupTreeWidgetItems.end()){
        auto groupTreeWidgetItem = new QTreeWidgetItem();
//        groupTreeWidgetItem->setCheckState(0, Qt::CheckState::Checked);

//        groupTreeWidgetItem->setText(0, QString::fromStdString(group));

        const auto groupHeader = new RenderGroupControlWidget(group, this, this->getOpenGLWidget());


        this->ui->treeWidget->addTopLevelItem(groupTreeWidgetItem);
        this->ui->treeWidget->setItemWidget(groupTreeWidgetItem, 0, groupHeader);
        groupTreeWidgetItem->setExpanded(true);

        iterator = groupTreeWidgetItems.insert({group, groupTreeWidgetItem}).first;
    }
    return iterator->second;
}

void RenderWidget::renderBox(const std::string &group, const std::string& name,  const AABB &aabb, const Transformation& transformation, const Color& color) {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "renderBoxSlot", Qt::AutoConnection,
                              Q_ARG(std::string, group),
                              Q_ARG(std::string, name),
                              Q_ARG(AABB, aabb),
                              Q_ARG(Transformation, transformation),
                              Q_ARG(PhongMaterial, PhongMaterial(color)),
                              Q_ARG(RenderWidget*, this));
}

void RenderWidget::renderPlane(const std::string &group, const std::string& name, const Plane &plane, const Color& color) {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "renderPlaneSlot", Qt::AutoConnection,
                              Q_ARG(std::string, group),
                              Q_ARG(std::string, name),
                              Q_ARG(Plane, plane),
                              Q_ARG(PhongMaterial, PhongMaterial(color)),
                              Q_ARG(RenderWidget*, this));
}

void RenderWidget::renderSphere(const std::string &group, const std::string& name, const Sphere &sphere, const Color& color) {
    this->renderSphere(group, name, sphere, PhongMaterial(color));
}

void RenderWidget::renderSphere(const std::string &group, const std::string& name, const Sphere &sphere, const PhongMaterial& material) {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "renderSphereSlot", Qt::AutoConnection,
                              Q_ARG(std::string, group),
                              Q_ARG(std::string, name),
                              Q_ARG(Sphere, sphere),
                              Q_ARG(PhongMaterial, material),
                              Q_ARG(RenderWidget*, this));
}

void RenderWidget::renderTriangle(const std::string& group, const std::string& name, const VertexTriangle& vertexTriangle, const Color& color) {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "renderTriangleSlot", Qt::AutoConnection,
                              Q_ARG(std::string, group),
                              Q_ARG(std::string, name),
                              Q_ARG(VertexTriangle, vertexTriangle),
                              Q_ARG(PhongMaterial, PhongMaterial(color)),
                              Q_ARG(RenderWidget*, this));
}

void RenderWidget::renderLine(const std::string &group, const std::string& name, const Vertex &vertexA, const Vertex &vertexB, const Color& color) {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "renderLineSlot", Qt::AutoConnection,
                                Q_ARG(std::string, group),
                                Q_ARG(std::string, name),
                                Q_ARG(glm::vec3, vertexA),
                                Q_ARG(glm::vec3, vertexB),
                                Q_ARG(PhongMaterial, PhongMaterial(color)),
                                Q_ARG(RenderWidget*, this));
}

void RenderWidget::notifySolution(const std::shared_ptr<const AbstractSolution>& solution) {
    if(this->solutionRenderCallback) solutionRenderCallback(this, solution);
}

void RenderWidget::notifyProgress(float progress) {
    QMetaObject::invokeMethod(this, [this,progress] {
        const int value = static_cast<int>(100 * progress);
        this->ui->progressBar->setValue(value);
    });
}

void RenderWidget::notifyStarted(const std::string& taskName) {

    QMetaObject::invokeMethod(this, [this, taskName] {
        this->ui->statusLabel->setText(QString::fromStdString("-"));
        this->ui->taskNameLabel->setText(QString::fromStdString(taskName));
        this->ui->progressBar->setValue(0);
        this->ui->startButton->setEnabled(false);
        this->ui->stopButton->setEnabled(true);
    });

    this->taskRunning = true;

    // Start timer thread
    timerThread = std::thread([&]{
        auto startMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        while(taskRunning){
            auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            auto elapsedMs = nowMs - startMs;
            //Format as HH:MM:SS
            auto hours = std::chrono::duration_cast<std::chrono::hours>(elapsedMs);
            elapsedMs -= std::chrono::milliseconds(std::chrono::hours(hours));
            auto minutes = std::chrono::duration_cast<std::chrono::minutes>(elapsedMs);
            elapsedMs -= std::chrono::milliseconds(std::chrono::minutes(minutes));
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsedMs);
//            elapsedMs -= std::chrono::milliseconds(std::chrono::seconds(seconds));
//            auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedMs);
            std::stringstream ss;
            ss << std::setfill('0') << std::setw(2) << hours.count()
               << ":" << std::setfill('0') << std::setw(2) << minutes.count()
               << ":" << std::setfill('0') << std::setw(2) << seconds.count();
//               << "." << std::setfill('0') << std::setw(3) << milliseconds.count();

            QMetaObject::invokeMethod(this, [this, &ss] {
                if (this->taskRunning) this->ui->timeLabel->setText(QString::fromStdString(ss.str()));
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });
    this->clear();
}

void RenderWidget::notifyFinished() {
    this->taskRunning = false;
    if(timerThread.joinable()) timerThread.join();
    QMetaObject::invokeMethod(this, [this] {
        // Update the start and stop buttons
        this->ui->startButton->setEnabled(true);
        this->ui->stopButton->setEnabled(false);
    });
}

void RenderWidget::notifyStatus(const std::string &status) {
    QMetaObject::invokeMethod(this, [this, status] {
        QString qStatus = QString::fromUtf8(status.data(), status.size());
        this->ui->statusLabel->setText(qStatus);
    });
}

void RenderWidget::observeTask(AbstractTask *task) {

    // Check if we're already running on the GUI thread
    auto connectionType = QThread::currentThread() == this->thread()? Qt::AutoConnection : Qt::BlockingQueuedConnection;

    // Task observation should happen on the GUI thread
    QMetaObject::invokeMethod(this, [this,task]{

        // Clear currently observed task if needed
        if(this->currentTask!=nullptr){
            currentTask->unregisterObserver(this);
            this->clear();
            this->ui->taskSection->setVisible(false);
        }

        // Set and observe new task
        this->currentTask = task;
        if(task!=nullptr){
            currentTask->registerObserver(this);

            this->ui->taskSection->setVisible(true); // This is not threadsafe
        }
    }, connectionType);
}

void RenderWidget::observeTask(AbstractTask *task, const std::function<void(RenderWidget* renderWidget, const std::shared_ptr<const AbstractSolution> solution)>& newSolutionRenderCallback) {

    auto connectionType = QThread::currentThread() == this->thread()? Qt::AutoConnection : Qt::BlockingQueuedConnection;

    QMetaObject::invokeMethod(this, [this,&task,&newSolutionRenderCallback]{

        // Clear currently observed task if needed
        if(this->currentTask!=nullptr){
            currentTask->unregisterObserver(this);
            this->clear();
            this->ui->taskSection->setVisible(false);
            this->solutionRenderCallback = {};
        }

        // Set and observe new task
        this->currentTask = task;
        this->solutionRenderCallback = newSolutionRenderCallback;
        if(task!=nullptr){
            currentTask->registerObserver(this);
            this->ui->taskSection->setVisible(true);
        }
    }, connectionType);
}

void RenderWidget::setSolutionRenderCallback(const std::function<void(RenderWidget *renderWidget, const std::shared_ptr<const AbstractSolution> &solution)> &
    newSolutionRenderCallback) {
    this->solutionRenderCallback = newSolutionRenderCallback;
}

void RenderWidget::setDefaultSolutionRenderCallback() {
    solutionRenderCallback = {
        [](RenderWidget* renderWidget, const std::shared_ptr<const AbstractSolution>& solution) {
            if (!solution) return;

            if (const auto& sol = std::dynamic_pointer_cast<const StripPackingSolution>(solution)) {
                renderWidget->clearGroup("MinimalContainer");
                float maximumHeight = 0.0f;
                for (size_t itemIndex = 0; itemIndex < sol->getItems().size(); ++itemIndex) {
                    const auto& item = sol->getItem(itemIndex);
                    const auto& itemName = sol->getItemName(itemIndex);

                    // Update the maximum height
                    const auto& itemAABB = sol->getItemAABB(itemIndex);
                    maximumHeight = std::max(maximumHeight,itemAABB.getMaximum().z);

                    renderWidget->renderWorldSpaceMesh("Items", item, StripPackingProblem::getItemColor(itemName));
                }
                auto min = sol->getProblem()->getContainer().getMinimum();
                auto max = sol->getProblem()->getContainer().getMaximum();

                renderWidget->renderBox("MinimalContainer", "AABB", {min, {max.x, max.y, maximumHeight}});

            }
            else if (const auto& svms = std::dynamic_pointer_cast<const SingleVolumeMaximisationSolution>(solution)) {
                renderWidget->renderWorldSpaceMesh("Items", svms->getItemWorldSpaceMesh(), Color::Red());
                renderWidget->renderWorldSpaceMesh("Container", svms->getContainerWorldSpaceMesh(), Color(1, 1, 1, 0.4));
            }
            else {
                std::cout << "The default solutionRenderCallback could not resolve the solution type. Please override the callback using RenderWidget::setSolutionRenderCallback()" << std::endl;
            }
        }
    };
}

void RenderWidget::startCurrentTask() const {
    if(this->currentTask!=nullptr) this->currentTask->start();
}

void RenderWidget::stopCurrentTask() const {
    if(this->currentTask!=nullptr) this->currentTask->stop();
}

void RenderWidget::addOrUpdateRenderModel(const std::string& group, const std::string& id, const std::shared_ptr<AbstractRenderModel> &renderModel) {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "addOrUpdateRenderModelSlot",
                              Qt::AutoConnection,
                              Q_ARG(std::string, group),
                              Q_ARG(std::string, id),
                              Q_ARG(std::shared_ptr<AbstractRenderModel>, renderModel), // We should copy the actual worldSpaceMesh object here, otherwise the transformation could change before the render thread reads it
                              Q_ARG(RenderWidget*, this));
}

void RenderWidget::captureScene() const {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "captureSceneSlot",
                              Qt::AutoConnection);

}


void RenderWidget::captureSceneToFile(const std::string &fileName) const {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "captureSceneToFileSlot",
                              Qt::AutoConnection,
                              Q_ARG(QString, QString::fromStdString(fileName)));
}



void RenderWidget::captureAnimation() const {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "captureAnimationSlot",
                              Qt::AutoConnection);
}

void RenderWidget::renderRay(const std::string &group, const std::string &name, const Ray &ray, const Color &color, float widthLengthRatio) {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "renderRaySlot", Qt::AutoConnection,
                              Q_ARG(std::string, group),
                              Q_ARG(std::string, name),
                              Q_ARG(Ray, ray),
                              Q_ARG(PhongMaterial, PhongMaterial(color)),
                              Q_ARG(float, widthLengthRatio),
                              Q_ARG(RenderWidget*, this));
}

void RenderWidget::renderBoundsTree(const std::string &group, const std::string &name, const std::shared_ptr<AbstractBoundsTree<AABB, 8, false>>& aabbTree, const Transformation &transformation, const Color &color) {
    QMetaObject::invokeMethod(qApp, [group, color, name, transformation, aabbTree, this] {
        auto renderModel = std::make_shared<RenderBoundsTree>(*aabbTree, transformation);
        renderModel->setName(name);
        renderModel->setMaterial(PhongMaterial(color));
        this->addOrUpdateRenderModel(group, name, renderModel);
    });
}

void RenderWidget::renderBoundsTree(const std::string &group, const std::string &name, const std::shared_ptr<AbstractBoundsTree<AABB, 2, true>>& aabbTree, const Transformation &transformation, const Color &color) {
    QMetaObject::invokeMethod(qApp, [group, color, name, transformation, aabbTree, this] {
        auto renderModel = std::make_shared<RenderBoundsTree>(*aabbTree, transformation);
        renderModel->setName(name);
        renderModel->setMaterial(PhongMaterial(color));
        this->addOrUpdateRenderModel(group, name, renderModel);
    });
}

void RenderWidget::renderBoundsTree(const std::string &group, const std::string &name, const std::shared_ptr<AbstractBoundsTree<OBB, 8, true>>& aabbTree, const Transformation &transformation, const Color &color) {
    QMetaObject::invokeMethod(qApp, [group, color, name, transformation, aabbTree, this] {
        auto renderModel = std::make_shared<RenderBoundsTree>(*aabbTree, transformation);
        renderModel->setName(name);
        renderModel->setMaterial(PhongMaterial(color));
        this->addOrUpdateRenderModel(group, name, renderModel);
    });
}

void RenderWidget::renderBoundsTree(const std::string &group, const std::string &name, const std::shared_ptr<AbstractBoundsTree<OBB, 2, true>>& aabbTree, const Transformation &transformation, const Color &color) {
    QMetaObject::invokeMethod(qApp, [group, color, name, transformation, aabbTree, this] {
        auto renderModel = std::make_shared<RenderBoundsTree>(*aabbTree, transformation);
        renderModel->setName(name);
        renderModel->setMaterial(PhongMaterial(color));
        this->addOrUpdateRenderModel(group, name, renderModel);
    });
}

void RenderWidget::renderBoundsTree(const std::string &group, const std::string &name, const std::shared_ptr<AbstractBoundsTree<Sphere, 8, true>>& aabbTree, const Transformation &transformation, const Color &color) {
    QMetaObject::invokeMethod(qApp, [group, color, name, transformation, aabbTree, this] {
        auto renderModel = std::make_shared<RenderBoundsTree>(*aabbTree, transformation);
        renderModel->setName(name);
        renderModel->setMaterial(PhongMaterial(color));
        this->addOrUpdateRenderModel(group, name, renderModel);
    });
}

void RenderWidget::renderBoundsTree(const std::string &group, const std::string &name, const std::shared_ptr<AbstractBoundsTree<Sphere, 2, true>>& aabbTree, const Transformation &transformation, const Color &color) {
    QMetaObject::invokeMethod(qApp, [group, color, name, transformation, aabbTree, this] {
        auto renderModel = std::make_shared<RenderBoundsTree>(*aabbTree, transformation);
        renderModel->setName(name);
        renderModel->setMaterial(PhongMaterial(color));
        this->addOrUpdateRenderModel(group, name, renderModel);
    });
}