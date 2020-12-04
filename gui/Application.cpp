//
// Created by Jonas on 30/11/2020.
//

#include <QtWidgets>
#include "ApplicationWindow.h"

int main(int argc, char *argv[]){
    QApplication app(argc, argv);



    ApplicationWindow window;
#ifdef Q_OS_ANDROID
    dialog.showMaximized();
#else
    window.show();
#endif
    return QApplication::exec();

}