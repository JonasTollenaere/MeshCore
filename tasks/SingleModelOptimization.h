//
// Created by Jonas on 4/12/2020.
//

#ifndef MESHCORE_SINGLEMODELOPTIMIZATION_H
#define MESHCORE_SINGLEMODELOPTIMIZATION_H


#include <RenderWidget.h>

class SingleModelOptimization {
private:
    RenderWidget* renderWidget;
public:
    void run();

    void setRenderWidget(RenderWidget *renderWidget);
};


#endif //MESHCORE_SINGLEMODELOPTIMIZATION_H
