//
// Created by Jonas on 12/12/2020.
//

#ifndef MESHCORE_CUDATASK_H
#define MESHCORE_CUDATASK_H


#include <RenderWidget.h>

class CudaTask {
private:
    RenderWidget* renderWidget;
public:
    void run();

    void setRenderWidget(RenderWidget *renderWidget);
};


#endif //MESHCORE_CUDATASK_H
