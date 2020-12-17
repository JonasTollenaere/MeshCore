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

    QApplication app(argc, argv);
    ApplicationWindow window;
    window.show();

//    SandboxTask task(window.getRenderWidget());
//    task.start();

    OptixTask optixTask(window.getRenderWidget());
    optixTask.start();

//    CudaTask task1(window.getRenderWidget());
//    task1.start();
//
////    for(int i=0; i<2; i++){
////        CudaTask task2;
////        task2.start();
////    }

    int returnCode = QApplication::exec();

    return returnCode;
}