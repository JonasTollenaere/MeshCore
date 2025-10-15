//
// Created by Jonas on 14/12/2020.
//

#include "meshcore/tasks/AbstractTask.h"

void AbstractTask::run_static(AbstractTask* task){
    task->notifyObserversStarted();
    task->run();
    if(task->stopCalled){
        task->notifyObserversStatus("Stopped");
    }
    else{
        task->notifyObserversStatus("Finished");
    }
    task->notifyObserversFinished();
}

void AbstractTask::start() {
    this->join();
    this->stopCalled = false;
    thread = new std::thread(run_static, this);
}

void AbstractTask::join() {
    if(thread != nullptr){
        thread->join();
        delete thread;
        thread = nullptr;
    }
}

AbstractTask::AbstractTask():
    stopCalled(false),
    thread(nullptr)
{}

void AbstractTask::notifyObserversUpdate() const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notify();
    }
}

void AbstractTask::unregisterObserver(AbstractTaskObserver* observer) {
    auto iterator = taskObservers.begin();
    while(iterator!=taskObservers.end()){
        if(*iterator==observer){
            taskObservers.erase(iterator);
            std::cout << "Unregistered observer" << std::endl;
            return;
        }
        iterator++;
    }
}

void AbstractTask::registerObserver(AbstractTaskObserver* observer) {
    taskObservers.emplace_back(observer);
}

void AbstractTask::notifyObserversSolution(const std::shared_ptr<const AbstractSolution>& solution) const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notifySolution(solution);
    }
}

void AbstractTask::notifyObserversProgress(float progress) const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notifyProgress(progress);
    }
}

void AbstractTask::notifyObserversFinished() const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notifyFinished();
    }
}

void AbstractTask::notifyObserversStarted() const {
    auto taskName = this->getTaskName();
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notifyStarted(taskName);
    }
}

std::string AbstractTask::getTaskName() const {
    return "-";
}

void AbstractTask::notifyObserversStatus(const std::string& status) const {
    for(AbstractTaskObserver* observer: taskObservers){
        observer->notifyStatus(status);
    }
}

void AbstractTask::stop() {
    this->stopCalled = true;
}

void AbstractTask::run() {
    this->stopCalled = false;
    this->notifyObserversStarted();
}