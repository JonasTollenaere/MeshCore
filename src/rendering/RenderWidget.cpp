//
// Created by Jonas on 29/03/2022.
//

#include <QtWidgets>
#include <iomanip>
#include <sstream>

#include "RenderWidget.h"
#include "RenderModelControlWidget.h"
#include "RenderGroupControlWidget.h"
#include "meshcore/core/Plane.h"

#include "forms/ui_renderwidget.h"

RenderWidget::RenderWidget(QWidget *parent):
    QWidget(parent), ui(new Ui::RenderWidget)
    {
        ui->setupUi(this);
        ui->progressBar->setMinimumWidth(350);

        this->ui->taskSection->setVisible(false);

        // Connect the start and stop buttons
        connect(this->ui->startButton, &QPushButton::clicked, this, &RenderWidget::startCurrentTask);
        connect(this->ui->stopButton, &QPushButton::clicked, this, &RenderWidget::stopCurrentTask);

        // Hide the header of the tree widget
        auto& tree = this->ui->treeWidget;
        tree->headerItem()->setHidden(true);
//        tree->setIndentation(0);
        tree->setColumnCount(1);
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

void RenderWidget::renderWorldSpaceMesh(const std::string &group, const std::shared_ptr<WorldSpaceMesh> &worldSpaceMesh,  const Color& color) {
    renderWorldSpaceMesh(group, worldSpaceMesh, PhongMaterial(color));
}

void RenderWidget::renderWorldSpaceMesh(const std::string &group, const std::shared_ptr<WorldSpaceMesh> &worldSpaceMesh,  const PhongMaterial& material) {
    QMetaObject::invokeMethod(this->getOpenGLWidget(), "renderWorldSpaceMeshSlot",
                              Qt::AutoConnection,
                              Q_ARG(std::string, group),
                              Q_ARG(std::shared_ptr<WorldSpaceMesh>, std::make_shared<WorldSpaceMesh>(*worldSpaceMesh)), // We should copy the actual worldSpaceMesh object here, otherwise the transformation could change before the render thread reads it
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
    if(this->onSolutionNotified) onSolutionNotified(this, solution);
}

void RenderWidget::notifyProgress(float progress) {
    int value = int(100*progress);
    QMetaObject::invokeMethod(this, "updateProgressBarSlot", Qt::AutoConnection, Q_ARG(int, value));
}

[[maybe_unused]] void RenderWidget::updateProgressBarSlot(int progress) {
    this->ui->progressBar->setValue(progress);
}

void RenderWidget::notifyStarted() {
    QMetaObject::invokeMethod(this, "setStatusLabelSlot", Qt::AutoConnection, Q_ARG(QString, QString::fromStdString("-")));
    QMetaObject::invokeMethod(this, "updateProgressBarSlot", Qt::AutoConnection, Q_ARG(int, 0));
    QMetaObject::invokeMethod(this, "setStopButtonEnabledSlot", Qt::AutoConnection, Q_ARG(bool, true));
    QMetaObject::invokeMethod(this, "setStartButtonEnabledSlot", Qt::AutoConnection, Q_ARG(bool, false));

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

            QMetaObject::invokeMethod(this, "setTimeLabelSlot", Qt::AutoConnection, Q_ARG(QString, QString::fromStdString(ss.str())));

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });
    this->clear();
}

void RenderWidget::notifyFinished() {
    this->taskRunning = false;
    if(timerThread.joinable()) timerThread.join();
    QMetaObject::invokeMethod(this, "setStopButtonEnabledSlot", Qt::AutoConnection, Q_ARG(bool, false));
    QMetaObject::invokeMethod(this, "setStartButtonEnabledSlot", Qt::AutoConnection, Q_ARG(bool, true));
}

void RenderWidget::notifyStatus(const std::string &status) {
    QString qStatus = QString::fromUtf8(status.data(), status.size());
    QMetaObject::invokeMethod(this, "setStatusLabelSlot", Qt::AutoConnection, Q_ARG(QString, qStatus));
}

[[maybe_unused]] void RenderWidget::setStartButtonEnabledSlot(bool enabled) {
    this->ui->startButton->setEnabled(enabled);
}

[[maybe_unused]] void RenderWidget::setStopButtonEnabledSlot(bool enabled) {
    this->ui->stopButton->setEnabled(enabled);
}

[[maybe_unused]] void RenderWidget::setStatusLabelSlot(const QString& status) {
    this->ui->statusLabel->setText(status);
}

void RenderWidget::observeTask(AbstractTask *task, const std::function<void(RenderWidget* renderWidget, const std::shared_ptr<const AbstractSolution> solution)>& onTaskSolutionNotified) {

    // Clear currently observed task if needed
    if(this->currentTask!=nullptr){
        currentTask->unregisterObserver(this);
        this->clear();
        this->ui->taskSection->setVisible(false);
        this->onSolutionNotified = {};
    }

    // Set and observe new task
    this->currentTask = task;
    this->onSolutionNotified = onTaskSolutionNotified;
    if(task!=nullptr){
        currentTask->registerObserver(this);
        this->ui->taskSection->setVisible(true);
    }
}

void RenderWidget::startCurrentTask() {
    if(this->currentTask!=nullptr) this->currentTask->start();
}

void RenderWidget::stopCurrentTask() {
    if(this->currentTask!=nullptr) this->currentTask->stop();
}

void RenderWidget::addOrUpdateRenderModel(const std::string& group, const std::string& id, std::shared_ptr<AbstractRenderModel> renderModel) {
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

[[maybe_unused]] void RenderWidget::setTimeLabelSlot(const QString &time) {
    this->ui->timeLabel->setText(time);
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
