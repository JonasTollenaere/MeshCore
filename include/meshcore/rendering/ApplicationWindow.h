//
// Created by Jonas on 30/11/2020.
//

#ifndef MESHCORE_NEWAPPLICATIONWINDOW_H
#define MESHCORE_NEWAPPLICATIONWINDOW_H

#include <QMainWindow>
#include <QDialog>

#include "RenderWidget.h"

QT_BEGIN_NAMESPACE
class QAction;
class QMenu;
class QMenuBar;
QT_END_NAMESPACE

class ApplicationWindow: public QMainWindow {
private:
    QMenuBar *menuBar;
    RenderWidget* renderWidget;
public:
    ApplicationWindow();
    [[nodiscard]] RenderWidget* getRenderWidget() const;

protected:
    void loadMesh();
    void displayApplicationInfo();
};


#endif //MESHCORE_NEWAPPLICATIONWINDOW_H
