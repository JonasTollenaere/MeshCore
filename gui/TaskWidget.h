#ifndef TASKRENDERWIDGET_H
#define TASKRENDERWIDGET_H

#include <QWidget>
#include "../core/WorldSpaceMesh.h"
#include "RenderModel.h"
#include "../tasks/AbstractTaskObserver.h"
#include "../solutions/AbstractSolution.h"

QT_BEGIN_NAMESPACE
namespace Ui { class TaskWidget; }
QT_END_NAMESPACE

class TaskWidget : public QWidget, public AbstractTaskObserver
{
    Q_OBJECT

public:
    explicit TaskWidget(QWidget *parent = nullptr);
    ~TaskWidget() override;
    void notify() override;
    void notifyStarted() override;
    void notifyFinished() override;
    void notifyProgress(float progress) override;
    void notifyStatus(const std::string &status) override;

    AbstractSolution* currentSolution;

    // TODO Make all these thread safe, should address slots
    void updateWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh);
    void addWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh, const Color& color);
    void addWorldSpaceMesh(const WorldSpaceMesh &worldSpaceMesh);

private:
    Ui::TaskWidget *ui;

public slots:
    Q_INVOKABLE [[maybe_unused]] void updateProgressBar(int progress);
    Q_INVOKABLE [[maybe_unused]] void setStartButtonEnabled(bool enabled);
    Q_INVOKABLE [[maybe_unused]] void setStopButtonEnabled(bool enabled);
    Q_INVOKABLE [[maybe_unused]] void setStatusLabel(const QString& status);

};
#endif // TASKRENDERWIDGET_H
