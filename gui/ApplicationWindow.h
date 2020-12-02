//
// Created by Jonas on 30/11/2020.
//

#ifndef MESHCORE_APPLICATIONWINDOW_H
#define MESHCORE_APPLICATIONWINDOW_H

#include <QMainWindow>
#include <QDialog>
#include "OpenGLWidget.h"

QT_BEGIN_NAMESPACE
class QAction;
class QDialogButtonBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QMenu;
class QMenuBar;
class QPushButton;
class QTextEdit;
QT_END_NAMESPACE

class ApplicationWindow: public QMainWindow {
private:
    QMenuBar *menuBar;
    OpenGLWidget openGlWidget;
public:
    ApplicationWindow();
protected:
    void keyPressEvent(QKeyEvent *event) override;
};


#endif //MESHCORE_APPLICATIONWINDOW_H
