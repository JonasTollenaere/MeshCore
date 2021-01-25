#include "TaskWidget.h"
#include "./res/forms/ui_taskrenderwidget.h"
#include "../tasks/SandboxTask.h"

TaskWidget::TaskWidget(QWidget *parent, SandboxTask* sandboxTask)
    : QWidget(parent)
    , ui(new Ui::TaskWidget)
{
    ui->setupUi(this);
    this->task = sandboxTask;
//    this->ui->openGLWidget->addWorldSpaceMesh(this->task->getOuterWorldSpaceMesh(), Color(1,1,1,0.4));
//    this->ui->openGLWidget->addWorldSpaceMesh(this->task->getInnerWorldSpaceMesh(), Color(1,0,0,1));
}

TaskWidget::~TaskWidget()
{
    delete ui;
}

void TaskWidget::notify(){
    if(this->task!=nullptr){
        this->ui->openGLWidget->updateWorldSpaceMesh(this->task->getOuterWorldSpaceMesh());
        this->ui->openGLWidget->updateWorldSpaceMesh(this->task->getInnerWorldSpaceMesh());
    }
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
