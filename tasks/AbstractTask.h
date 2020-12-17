//
// Created by Jonas on 14/12/2020.
//

#ifndef MESHCORE_ABSTRACTTASK_H
#define MESHCORE_ABSTRACTTASK_H


#include <RenderWidget.h>
#include <thread>
#include <random>

class AbstractTask {
private:
    std::thread* thread;
    RenderWidget* renderWidget;
    std::mt19937 randomEngine;

public:
    AbstractTask();
    explicit AbstractTask(RenderWidget *renderWidget);

    virtual void run() = 0;
    void start();
    void finalize();


    void renderMesh(const WorldSpaceMesh& worldSpaceMesh, const Color& color) const;
    void updateRenderMesh(const WorldSpaceMesh& worldSpaceMesh) const;

    void setSeed(unsigned int seed);
    float getRandomFloat(float maxValue);
};

#endif //MESHCORE_ABSTRACTTASK_H
