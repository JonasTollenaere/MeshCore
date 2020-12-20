//
// Created by Jonas on 30/11/2020.
//

#include <QtWidgets>
#include "ApplicationWindow.h"
#include "../tasks/SandboxTask.h"
#include "../tasks/CudaTask.h"
#include "../tasks/OptixTask.h"


Q_DECLARE_METATYPE(WorldSpaceMesh)

int main(int argc, char *argv[]){

//    QApplication app(argc, argv);
//    ApplicationWindow window;
//    window.show();
//
//
//    OptixTask optixTask(window.getRenderWidget());
//    optixTask.start();
//
//
//    int returnCode = QApplication::exec();
//
//    return returnCode;

    OptixTask optixTask(nullptr);
    optixTask.run();
    return 0;
}