//
// Created by jonas on 01.03.21.
//

#include "meshcore/tasks/ConsoleTaskObserver.h"

void ConsoleTaskObserver::notifyStatus(const std::string &newStatus) {
    this->status = newStatus;
    this->update();
}

void ConsoleTaskObserver::notifySolution(const std::shared_ptr<const AbstractSolution>& solution) {
    std::cout << "notifySolution not implemented yet" << std::endl;
}

void ConsoleTaskObserver::notifyProgress(float newProgress) {
    this->progress = newProgress;
    this->update();
}

void ConsoleTaskObserver::notifyStarted(const std::string& taskName) {
    std::cout << "Starting task: " << taskName << std::endl;
    this->progress = 0.0f;
    this->update();
}

void ConsoleTaskObserver::update() {
    std::cout << status;
    std::cout << " [";
    const int barWidth = 50;
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

void ConsoleTaskObserver::notifyFinished() {
    this->progress = 1;
    std::cout << status;
    std::cout << " [";
    const int barWidth = 50;
    for (int i = 0; i < barWidth; ++i) {
        std::cout << "=";
    }
    std::cout << "] " << int(progress * 100.0) << " %\n";
    std::cout.flush();
}
