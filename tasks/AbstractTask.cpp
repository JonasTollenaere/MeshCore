//
// Created by Jonas on 14/12/2020.
//

#include "AbstractTask.h"

static void run_static(AbstractTask* task){
    task->run();
}

void AbstractTask::start() {
    if(thread == nullptr){
        thread = new std::thread(run_static, this);
    }
    else std::cout << "Task already started" << std::endl;
}

void AbstractTask::finalize() {
    thread->join();
    delete thread;
}

AbstractTask::AbstractTask(): renderWidget(nullptr), thread(nullptr) {}

AbstractTask::AbstractTask(RenderWidget* renderWidget): renderWidget(renderWidget), thread(nullptr) {}

void AbstractTask::renderMesh(const WorldSpaceMesh &worldSpaceMesh, const Color &color) const {
    if(this->renderWidget!= nullptr) this->renderWidget->addWorldSpaceMesh(worldSpaceMesh, color);
}

void AbstractTask::updateRenderMesh(const WorldSpaceMesh &worldSpaceMesh) const {
    if(this->renderWidget!= nullptr) this->renderWidget->updateWorldSpaceMesh(worldSpaceMesh);
}