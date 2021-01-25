#include "TaskWidget.h"
#include "./res/forms/ui_taskrenderwidget.h"

TaskWidget::TaskWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::TaskWidget)
{
    ui->setupUi(this);
}

TaskWidget::~TaskWidget()
{
    delete ui;
}

/*** This is threadsafe, can be called from anywhere ***/
void TaskWidget::addWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh) {
    this->addWorldSpaceMesh(worldSpaceMesh, Color(1, 1, 1, 1));
}

/*** This is threadsafe, can be called from anywhere ***/
Q_DECLARE_METATYPE(WorldSpaceMesh)
Q_DECLARE_METATYPE(Color)
void TaskWidget::addWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh, const Color& color) {

    // This way the actions are executed on the main thread
    qRegisterMetaType<const WorldSpaceMesh&>();
    qRegisterMetaType<const Color&>();
    QMetaObject::invokeMethod(this->ui->openGLWidget, "addWorldSpaceMeshSlot", Qt::AutoConnection, Q_ARG(WorldSpaceMesh, worldSpaceMesh), Q_ARG(const Color&, color));
}

void TaskWidget::updateWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh) {
    this->ui->openGLWidget->updateWorldSpaceMesh(worldSpaceMesh);
}

void TaskWidget::notify(){
    std::cout << "This notify is overridden" << std::endl;
}

void TaskWidget::notifyProgress(float progress) {
    int value = int(100*progress);
    QMetaObject::invokeMethod(this, "updateProgressBar", Qt::AutoConnection, Q_ARG(int, value));
}

[[maybe_unused]] void TaskWidget::updateProgressBar(int progress) {
    this->ui->progressBar->setValue(progress);
}

void TaskWidget::notifyStarted() {
    QMetaObject::invokeMethod(this, "updateProgressBar", Qt::AutoConnection, Q_ARG(int, 0));
    QMetaObject::invokeMethod(this, "setStopButtonEnabled", Qt::AutoConnection, Q_ARG(bool, true));
    QMetaObject::invokeMethod(this, "setStartButtonEnabled", Qt::AutoConnection, Q_ARG(bool, false));
}

void TaskWidget::notifyFinished() {
    QMetaObject::invokeMethod(this, "setStopButtonEnabled", Qt::AutoConnection, Q_ARG(bool, false));
    QMetaObject::invokeMethod(this, "setStartButtonEnabled", Qt::AutoConnection, Q_ARG(bool, true));
}

void TaskWidget::notifyStatus(const std::string &status) {
    QString qStatus = QString::fromUtf8(status.data(), status.size());
    QMetaObject::invokeMethod(this, "setStatusLabel", Qt::AutoConnection, Q_ARG(const QString&, qStatus));
}

[[maybe_unused]] void TaskWidget::setStartButtonEnabled(bool enabled) {
    this->ui->startButton->setEnabled(enabled);
}

[[maybe_unused]] void TaskWidget::setStopButtonEnabled(bool enabled) {
    this->ui->stopButton->setEnabled(enabled);
}

[[maybe_unused]] void TaskWidget::setStatusLabel(const QString& status) {
    this->ui->statusLabel->setText(status);
}
