//
// Created by Jonas on 30/11/2020.
//

#include <QtWidgets>
#include "ApplicationWindow.h"
#include "../tasks/SandboxTask.h"
#include "../tasks/CudaTask.h"


Q_DECLARE_METATYPE(WorldSpaceMesh)

int main(int argc, char *argv[]){
    QApplication app(argc, argv);


    ApplicationWindow window;
    window.show();


    CudaTask task;
    task.setRenderWidget(window.getRenderWidget());
    task.run();

//    SandboxTask task2;
//    task2.setRenderWidget(window.getRenderWidget());
//    task2.run();

    return QApplication::exec();

}