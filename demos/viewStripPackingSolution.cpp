//
// Created by Jonas on 26/03/2023.
//

#include "meshcore/rendering/ApplicationWindow.h"
#include "meshcore/optimization/StripPackingSolution.h"

class LoadSolutionTask: public AbstractTask {
public:
    void run() override {

        // Create and notify the solution
        notifyObserversStatus("Loading solution");
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        std::string path = std::string(MESHCORE_DATA_DIR) + "Stoyan et al. 2005/Solution1.json";

        auto solution = StripPackingSolution::fromJson(path);
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout << "Creating solution took " << (end-start) << "ms" << std::endl;

        notifyObserversSolution(solution);

        auto result = solution->toJson();

        std::cout << "Feasible: " << solution->isFeasible() << std::endl; // The mathematical solution is not feasible due to constraint tolerances
    }
};

int main(int argc, char *argv[]){

    LoadSolutionTask task;

    QApplication app(argc, argv);
    ApplicationWindow window;
    window.show();
    RenderWidget* renderWidget = window.getRenderWidget();
    renderWidget->observeTask(&task); // The default solution render callback will be used
    task.start();
    int returnCode = QApplication::exec();
    task.stop();
    task.join();
    return returnCode;
}