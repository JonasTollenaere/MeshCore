//
// Created by Jonas on 25/01/2021.
//

#ifndef MESHCORE_CONSOLETASKOBSERVER_H
#define MESHCORE_CONSOLETASKOBSERVER_H
class ConsoleTaskObserver: public AbstractTaskObserver {
public:
    virtual void notify() {};

    virtual void notifyProgress(float progress) {
        std::cout << "Progress: ";
        printf("%.2f\n", progress);
    }

    virtual void notifyFinished() {
        std::cout << "Task finished!" << std::endl;
    }

    virtual void notifyStarted() {
        std::cout << "Task started" << std::endl;
    }

    virtual void notifyStatus(const std::string& status) {
        std::cout << "Task status: " << status << std::endl;
    }
};
#endif //MESHCORE_CONSOLETASKOBSERVER_H
