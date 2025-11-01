//
// Created by Jonas on 26/08/2022.
//

#ifndef MESHCORE_ABSTRACTRENDERBOUNDSTREE_H
#define MESHCORE_ABSTRACTRENDERBOUNDSTREE_H

#include "AbstractRenderModel.h"
#include "meshcore/acceleration/AbstractBoundsTree.h"
#include "RenderAABB.h"
#include "RenderOBB.h"
#include "RenderSphere.h"
#include <QCheckBox>
#include <thread>

class RenderBoundsTree: public AbstractRenderModel{
    unsigned int renderDepth = 0;
    std::atomic<bool> animating = false; // For cycling through the render depth
    std::vector<std::shared_ptr<RenderBoundsTree>> children;
    std::shared_ptr<AbstractRenderModel> renderNodeModel;

public:

    template <unsigned int Degree, bool UniqueAssignment>
    RenderBoundsTree(const AbstractBoundsTree<AABB, Degree, UniqueAssignment>& aabbTree, const Transformation& transformation);

    template <unsigned int Degree, bool UniqueAssignment>
    RenderBoundsTree(const AbstractBoundsTree<OBB, Degree, UniqueAssignment>& obbTree, const Transformation& transformation);

    template <unsigned int Degree, bool UniqueAssignment>
    RenderBoundsTree(const AbstractBoundsTree<Sphere, Degree, UniqueAssignment>& sphereTree, const Transformation& transformation);

    void draw(const OpenGLWidget* openGLWidget, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix, bool lightMode) override;
    void setTransformation(const Transformation &transformation) override;
    void setRenderDepth(unsigned int newRenderDepth);
    unsigned int getMaximumDepth();
    void setMaterial(const PhongMaterial &newMaterial) override;

    RenderModelDetailDialog *createRenderModelDetailDialog(QWidget* parent) override;

    QMenu *getContextMenu() override;

    unsigned int getRenderDepth() const;

private:
    void drawRecursive(const OpenGLWidget* openGLWidget, const glm::mat4 &viewMatrix, const glm::mat4 &projectionMatrix, bool lightMode, unsigned int depth);
};

// Only define templated functions in this header file:

template <unsigned int Degree, bool UniqueAssignment>
RenderBoundsTree::RenderBoundsTree(const AbstractBoundsTree<AABB, Degree, UniqueAssignment> &aabbTree, const Transformation& transformation): AbstractRenderModel(transformation, "aabbTree"), renderNodeModel(std::make_shared<RenderAABB>(aabbTree.getBounds(), transformation)) {
    if(aabbTree.isSplit()){
        for (const auto &child : aabbTree.getChildren()){
            if(child && !child->isEmpty()) this->children.emplace_back(std::make_shared<RenderBoundsTree>(*child, transformation));
        }
    }
}

template <unsigned int Degree, bool UniqueAssignment>
RenderBoundsTree::RenderBoundsTree(const AbstractBoundsTree<OBB, Degree, UniqueAssignment> &obbTree, const Transformation& transformation): AbstractRenderModel(transformation, "obbTree"), renderNodeModel(std::make_shared<RenderOBB>(obbTree.getBounds(), transformation)){
    if(obbTree.isSplit()){
        for (const auto &child : obbTree.getChildren()){
            if(child && !child->isEmpty()) this->children.emplace_back(std::make_shared<RenderBoundsTree>(*child, transformation));
        }
    }
}

template <unsigned int Degree, bool UniqueAssignment>
RenderBoundsTree::RenderBoundsTree(const AbstractBoundsTree<Sphere, Degree, UniqueAssignment> &sphereTree, const Transformation &transformation): AbstractRenderModel(transformation, "sphereTree"), renderNodeModel(std::make_shared<RenderSphere>(sphereTree.getBounds(), transformation)){
    this->setMaterial(PhongMaterial(Color(1.0, 1.0, 1.0, 0.1)));
    if(sphereTree.isSplit()){
        for (const auto &child : sphereTree.getChildren()){
            if(child && !child->isEmpty()) this->children.emplace_back(std::make_shared<RenderBoundsTree>(*child, transformation));
        }
    }
}

#endif //MESHCORE_ABSTRACTRENDERBOUNDSTREE_H