//
// Created by Jonas on 14/12/2020.
//

#ifndef MESHCORE_ABSTRACTTASK_H
#define MESHCORE_ABSTRACTTASK_H


#include <RenderWidget.h>
#include <thread>

class AbstractTask {
private:
    std::thread* thread;
    RenderWidget* renderWidget;

public:
    AbstractTask();
    explicit AbstractTask(RenderWidget *renderWidget);

    virtual void run() = 0;
    void start();
    void finalize();

    void renderMesh(const WorldSpaceMesh& worldSpaceMesh, const Color& color) const;
    void updateRenderMesh(const WorldSpaceMesh& worldSpaceMesh) const;
};

#endif //MESHCORE_ABSTRACTTASK_H
