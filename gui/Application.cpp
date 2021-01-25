//
// Created by Jonas on 30/11/2020.
//

#include <QtWidgets>
#include "ApplicationWindow.h"
#include "../tasks/SandboxTask.h"
#include "../tasks/CudaTask.h"
#include "../tasks/OptixTask.h"
#include "../tasks/LatencyBenchMarkTask.h"
#include "TaskWidget.h"

Q_DECLARE_METATYPE(WorldSpaceMesh)

int main(int argc, char *argv[]){

    QApplication app(argc, argv);

    ApplicationWindow window;
    window.show();

//    OptixTask optixTask(window.getRenderWidget());
//    optixTask.start();

    SandboxTask sandboxTask(window.getTaskRenderWidget());
    sandboxTask.registerObserver(window.getTaskRenderWidget());
    sandboxTask.start();

    int returnCode = QApplication::exec();
    sandboxTask.unregisterObserver(window.getTaskRenderWidget());
    return returnCode;

//    OptixTask task1;
//    task1.run();
//
//    LatencyBenchMarkTask latencyBenchMarkTask;
//    latencyBenchMarkTask.run();

//    SandboxTask sandboxTask;
//    sandboxTask.run();
//
//    return 0;

}