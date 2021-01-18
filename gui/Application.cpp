//
// Created by Jonas on 30/11/2020.
//

#include <QtWidgets>
#include "ApplicationWindow.h"
#include "../tasks/SandboxTask.h"
#include "../tasks/CudaTask.h"
#include "../tasks/OptixTask.h"
#include "../tasks/LatencyBenchMarkTask.h"

Q_DECLARE_METATYPE(WorldSpaceMesh)

int main(int argc, char *argv[]){
//
    QApplication app(argc, argv);
    ApplicationWindow window;
    window.show();

//    OptixTask optixTask(window.getRenderWidget());
//    optixTask.start();

    SandboxTask sandboxTask(window.getRenderWidget());
    sandboxTask.start();

    int returnCode = QApplication::exec();

    return returnCode;
//
//    OptixTask task1;
//    task1.run();

//    LatencyBenchMarkTask latencyBenchMarkTask;
//    latencyBenchMarkTask.run();

    return 0;

}