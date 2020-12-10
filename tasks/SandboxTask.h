//
// Created by Jonas on 4/12/2020.
//

#ifndef MESHCORE_SANDBOXTASK_H
#define MESHCORE_SANDBOXTASK_H


#include <RenderWidget.h>

class SandboxTask {
private:
    RenderWidget* renderWidget;
public:
    void run();

    void setRenderWidget(RenderWidget *renderWidget);
};


#endif //MESHCORE_SANDBOXTASK_H
