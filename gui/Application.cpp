//
// Created by Jonas on 30/11/2020.
//

#include <QtWidgets>
#include "ApplicationWindow.h"
#include "../tasks/SingleModelOptimization.h"

int main(int argc, char *argv[]){
    QApplication app(argc, argv);


    ApplicationWindow window;
    window.show();


    SingleModelOptimization task;
    task.setRenderWidget(window.getRenderWidget());
    task.run();


    return QApplication::exec();

}