#ifndef TASKRENDERWIDGET_H
#define TASKRENDERWIDGET_H

#include <QWidget>
#include "../core/WorldSpaceMesh.h"
#include "RenderModel.h"
#include "../tasks/AbstractTaskObserver.h"
#include "../solutions/AbstractSolution.h"
#include "../tasks/AbstractTask.h"
#include "../tasks/SandboxTask.h"

QT_BEGIN_NAMESPACE
namespace Ui { class TaskWidget; }
QT_END_NAMESPACE

class TaskWidget : public QWidget, public AbstractTaskObserver
{
    Q_OBJECT

public:
    explicit TaskWidget(QWidget *parent = nullptr, SandboxTask* sandboxTask = nullptr);
    ~TaskWidget() override;
    void notify() override;
    void notifyStarted() override;
    void notifyFinished() override;
    void notifyProgress(float progress) override;
    void notifyStatus(const std::string &status) override;
//    void notifySolution(const AbstractSolution& solution) override;

private:
    SandboxTask* task = nullptr;
    Ui::TaskWidget *ui;

public slots:
    Q_INVOKABLE [[maybe_unused]] void updateProgressBar(int progress);
    Q_INVOKABLE [[maybe_unused]] void setStartButtonEnabled(bool enabled);
    Q_INVOKABLE [[maybe_unused]] void setStopButtonEnabled(bool enabled);
    Q_INVOKABLE [[maybe_unused]] void setStatusLabel(const QString& status);

};
#endif // TASKRENDERWIDGET_H
