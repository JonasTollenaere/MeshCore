//
// Created by Jonas on 13/02/2023.
//

#include <memory>

#include "meshcore/core/ModelSpaceMesh.h"
#include "meshcore/utility/FileParser.h"
#include "meshcore/acceleration/BoundingVolumeHierarchy.h"
#include "meshcore/acceleration/CachingBoundsTreeFactory.h"
#include "meshcore/rendering/ApplicationWindow.h"

int main(int argc, char *argv[]){

    QApplication app(argc, argv);
    ApplicationWindow window;
    window.show();
    RenderWidget* renderWidget = window.getRenderWidget();

    std::thread t([&]() {
        auto meshA = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/pig.obj")));
        auto meshB = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("E. F. Silva et al. 2021/dragon.obj")));
        // auto meshA = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("Tollenaere, J. et al/Containers/rock_125k.obj")));
        // auto meshB = std::make_shared<WorldSpaceMesh>(FileParser::loadMeshFile(MESHCORE_DATA_DIR + std::string("Tollenaere, J. et al/Containers/rock_118k.obj")));
        meshB->getModelTransformation().setPositionX(10);

        Quaternion dRotationA(0.08f, 0.03f, 0.02f);
        Quaternion dRotationB(0.01f, 0.02f, 0.03f);

        long long totalDuration = 0;

        while(true){


            meshA->getModelTransformation().factorRotation(dRotationA);
            meshB->getModelTransformation().factorRotation(dRotationB);

            glm::vec3 closestPointA, closestPointB;

            auto start = std::chrono::high_resolution_clock::now();
            auto distance = Distance::distance(*meshA, *meshB, &closestPointA, &closestPointB);
            auto end = std::chrono::high_resolution_clock::now();
            totalDuration += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            std::cout << "Distance: " << distance << " in " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;

            renderWidget->clearGroup("Closest Points");
            renderWidget->renderWorldSpaceMesh("Meshes", meshA, Color(1, 1, 1, 0.6));
            renderWidget->renderWorldSpaceMesh("Meshes", meshB, Color(1, 1, 1, 0.6));

            renderWidget->renderLine("Closest Points", "Line", closestPointA, closestPointB, Color(1, 0, 0, 1));

            if(Intersection::intersect(*meshA, *meshB)) {
                std::cout << "Distance on intersection: " << distance << std::endl;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    QApplication::exec();
}