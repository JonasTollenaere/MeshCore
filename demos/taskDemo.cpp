//
// Created by Jonas on 27/09/2022.
//

#include <thread>

#include "meshcore/rendering/ApplicationWindow.h"
#include "meshcore/utility/FileParser.h"
#include "meshcore/utility/random.h"

class DummySolution: public AbstractSolution{

    std::shared_ptr<WorldSpaceMesh> item;
    std::shared_ptr<WorldSpaceMesh> container;

public:
    DummySolution(const std::shared_ptr<WorldSpaceMesh> &item, const std::shared_ptr<WorldSpaceMesh> &container) : item(item), container(container) {}

    [[nodiscard]] std::shared_ptr<AbstractSolution> clone() const override {
        return std::make_shared<DummySolution>(std::make_shared<WorldSpaceMesh>(*item), std::make_shared<WorldSpaceMesh>(*container));
    }

    [[nodiscard]] const std::shared_ptr<WorldSpaceMesh> &getItem() const {
        return item;
    }

    [[nodiscard]] const std::shared_ptr<WorldSpaceMesh> &getContainer() const {
        return container;
    }

    [[nodiscard]] bool isFeasible() const override {
        // Check if the item doesn't intersect the container
        return !Intersection::intersect(*item, *container);
        // This also doesn't check if the item is actually inside the container, but that's not a problem for this demo
    }
};

class DummyTask: public AbstractTask {
public:
    void run() override {

        // Load files and create a solution
        notifyObserversStatus("Loading files");
        auto item = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/apple.stl")));
        auto container = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/stone_1.obj")));

        container->getModelTransformation().factorScale(2.0f);
        container->getModelTransformation().deltaPosition(-2.0f*container->getModelSpaceMesh()->getBounds().getCenter()); // Put the container in the center of the world
        item->getModelTransformation().factorScale(0.1f);
        item->getModelTransformation().deltaPosition(-0.1f * item->getModelSpaceMesh()->getBounds().getCenter()); // Put the item in the center of the world

        auto solution = std::make_shared<DummySolution>(item, container);

        // Notify the solution so the RenderWidget will observe this and render it
        this->notifyObserversSolution(solution);

        // Loop over a bunch of random transformations
        auto iterations = 2e5;
        Random random;
        notifyObserversStatus("Executing random walk");
        for (int i = 0; i < iterations; ++i){

            if(this->stopCalled){
                return;
            }

            // 0. Sample a random change to position and orientation
            const Transformation originalTransformation = item->getModelTransformation();
            auto translation = container->getModelSpaceMesh()->getBounds().getHalf() * glm::vec3(random.nextFloat(-0.1f, 0.1f), random.nextFloat(-0.1f, 0.1f), random.nextFloat(-0.1f, 0.1f));
            item->getModelTransformation().deltaPosition(translation);
            Quaternion randomRotation(random.nextFloat(-0.1f, 0.1f), random.nextFloat(-0.1f, 0.1f), random.nextFloat(-0.1f, 0.1f));
            item->getModelTransformation().factorRotation(randomRotation);

            // 1. Check if the new transformation is feasible
            if (solution->isFeasible()){
                // 2. If it is, notify the solution so the RenderWidget will observe this and render it
                this->notifyObserversSolution(solution);
            } else {
                // 3. If it isn't, revert the transformation
                item->getModelTransformation() = originalTransformation;
            }

            if(i%10==0) notifyObserversProgress(i/iterations);

            // Sleep 5 ms to avoid overloading render thread
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        notifyObserversProgress(1.0f);
    }
};

int main(int argc, char *argv[]){

    QApplication app(argc, argv);
    ApplicationWindow window;
    window.show();

    DummyTask dummyTask;

    // Pass a function that is called when a solution has to be rendered
    std::function<void(RenderWidget* renderWidget, std::shared_ptr<const AbstractSolution> solution)> onSolutionNotified = [](RenderWidget* renderWidget, const std::shared_ptr<const AbstractSolution>& solution){
        auto dummySolution = std::dynamic_pointer_cast<const DummySolution>(solution);
        renderWidget->renderWorldSpaceMesh("Container", dummySolution->getContainer(), Color(1.0,1.0,1.0,0.4));
        renderWidget->renderWorldSpaceMesh("Items", dummySolution->getItem(), Color::Red());
    };

    window.getRenderWidget()->observeTask(&dummyTask, onSolutionNotified);

    dummyTask.start();
    int returnCode = QApplication::exec();
    dummyTask.stop();
    dummyTask.join();
    return returnCode;
}