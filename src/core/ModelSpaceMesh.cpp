//
// Created by Jonas on 27/01/2021.
//

#include "meshcore/core/ModelSpaceMesh.h"

#include <algorithm>
#include <utility>
#include <unordered_set>
#include "meshcore/factories/AABBFactory.h"
#include "src/external/quickhull/QuickHull.hpp"
#include "meshcore/core/Plane.h"
#include "meshcore/utility/hash.h"

#define EPSILON 1e-4

ModelSpaceMesh::ModelSpaceMesh(std::vector<Vertex> vertices) {
    this->vertices = std::move(vertices);
}

ModelSpaceMesh::ModelSpaceMesh(std::vector<Vertex> mVertices, std::vector<IndexTriangle> moveableTriangles):
vertices(std::move(mVertices)), triangles(std::move(moveableTriangles)){}

const std::vector<Vertex>& ModelSpaceMesh::getVertices() const {
    return vertices;
}

const std::vector<IndexTriangle>& ModelSpaceMesh::getTriangles() const {
    return triangles;
}

const std::vector<IndexFace>& ModelSpaceMesh::getFaces() const {
    if(!faces.has_value()){
        computeFaces();
        assert(faces.has_value());
    }
    return faces.value();
}

const std::vector<IndexEdge>& ModelSpaceMesh::getEdges() const {
    if(!triangleEdges.has_value()){
        // Set up the hash and equals in a way that the order of vertexIndex0 and vertexIndex1 doesn't matter
        auto hash = [](const IndexEdge& edge) { return std::hash<unsigned int>()(edge.vertexIndex0 + edge.vertexIndex1); }; // Hashes should remain equal if vertices are swapped
        auto equal = [](const IndexEdge& edge1, const IndexEdge& edge2) {
            return (edge1.vertexIndex0 == edge2.vertexIndex0 && edge1.vertexIndex1 == edge2.vertexIndex1) ||
                   (edge1.vertexIndex1 == edge2.vertexIndex0 && edge1.vertexIndex0 == edge2.vertexIndex1); };
        std::unordered_set<IndexEdge, decltype(hash), decltype(equal)> edgeSet(8, hash, equal);
        for(const IndexTriangle& triangle: this->triangles){
            edgeSet.insert(IndexEdge{triangle.vertexIndex0, triangle.vertexIndex1});
            edgeSet.insert(IndexEdge{triangle.vertexIndex1, triangle.vertexIndex2});
            edgeSet.insert(IndexEdge{triangle.vertexIndex2, triangle.vertexIndex0});
        }

        this->triangleEdges = std::vector<IndexEdge>(edgeSet.begin(), edgeSet.end());
    }
    return triangleEdges.value();
}


const std::vector<IndexEdge>& ModelSpaceMesh::getFaceEdges() const {
    if(!faceEdges.has_value()){
        if(!faces.has_value()){
            computeFaces();
        }

        // Determine unique edges
        {
            std::unordered_set<IndexEdge> edgeSet;
            for(const IndexFace& face: this->faces.value()){
                for (int i = 0; i < face.vertexIndices.size(); ++i){
                    edgeSet.insert(IndexEdge{face.vertexIndices[i], face.vertexIndices[(i + 1)%face.vertexIndices.size()]});
                }
            }
            this->faceEdges = std::vector<IndexEdge>(edgeSet.begin(), edgeSet.end());
        }
        assert(faceEdges.has_value());
    }
    return faceEdges.value();
}

std::vector<IndexEdge> ModelSpaceMesh::getSufficientIntersectionEdges() const {

    std::unordered_set<IndexEdge> edgeSet;
    std::unordered_set<IndexEdge> preferNotUsed;
    for(const IndexTriangle& triangle: this->triangles){

        auto edge0 = IndexEdge{triangle.vertexIndex0, triangle.vertexIndex1};
        auto edge1 = IndexEdge{triangle.vertexIndex1, triangle.vertexIndex2};
        auto edge2 = IndexEdge{triangle.vertexIndex2, triangle.vertexIndex0};

        bool present0 = (edgeSet.find(edge0) != edgeSet.end());
        bool present1 = (edgeSet.find(edge1) != edgeSet.end());
        bool present2 = (edgeSet.find(edge2) != edgeSet.end());

        unsigned int nPresent = 0;
        if(present0) nPresent++;
        if(present1) nPresent++;
        if(present2) nPresent++;

        while(nPresent<2){
            if(!present0 && !(preferNotUsed.find(edge0) != preferNotUsed.end())){
                edgeSet.insert(edge0);
                present0=true;
                nPresent++;
            }
            else if(!present1 && !(preferNotUsed.find(edge1) != preferNotUsed.end())){
                edgeSet.insert(edge1);
                present1=true;
                nPresent++;
            }
            else if(!present2 && !(preferNotUsed.find(edge2) != preferNotUsed.end())){
                edgeSet.insert(edge2);
                present2=true;
                nPresent++;
            }
            else if(!present0){
                edgeSet.insert(edge0);
                present0=true;
                nPresent++;
            }
            else if(!present1){
                edgeSet.insert(edge1);
                present1=true;
                nPresent++;
            }
            else if(!present2){
                edgeSet.insert(edge2);
                present2=true;
                nPresent++;
            }
            else assert(false);

        }
        if(!present0) preferNotUsed.insert(edge0);
        if(!present1) preferNotUsed.insert(edge1);
        if(!present2) preferNotUsed.insert(edge2);
    }

    std::vector<IndexEdge> returnVector(edgeSet.begin(), edgeSet.end());

#if !NDEBUG
    for(const IndexTriangle& triangle: this->triangles){
        auto edge0 = IndexEdge{triangle.vertexIndex0, triangle.vertexIndex1};
        auto edge1 = IndexEdge{triangle.vertexIndex1, triangle.vertexIndex2};
        auto edge2 = IndexEdge{triangle.vertexIndex2, triangle.vertexIndex0};

        bool present0 = (edgeSet.find(edge0) != edgeSet.end());
        bool present1 = (edgeSet.find(edge1) != edgeSet.end());
        bool present2 = (edgeSet.find(edge2) != edgeSet.end());

        unsigned int nPresent = 0;
        if(present0) nPresent++;
        if(present1) nPresent++;
        if(present2) nPresent++;

        assert(nPresent>=2);
    }

    std::cout << std::endl;
    std::cout << "Generated sufficient set of edges" << std::endl;
    std::cout << "\tDefault number of edges:\t" << this->getEdges().size() << std::endl;
    std::cout << "\tSufficient number of edges:\t" << returnVector.size() << std::endl;
    std::cout << "\tSufficient on default ratio:\t" << float(returnVector.size())/this->getEdges().size() << std::endl;
    std::cout << "\tPercentage theoretical minimum:\t" << (float(returnVector.size())/this->getEdges().size())/(2.0f/3) << std::endl;
    std::cout << std::endl;

#endif
    return returnVector;
}

const std::shared_ptr<ModelSpaceMesh>& ModelSpaceMesh::getConvexHull() const {

    // Return the value if already calculated before
    if(convexHull != nullptr){
        return convexHull;
    }

    // Otherwise, use the quickhull library to calculate the convex hull
    quickhull::QuickHull<float> qh;
    std::vector<quickhull::Vector3<float>> qhVertices;

    qhVertices.reserve(vertices.size());
    for(Vertex vertex: vertices){
        qhVertices.emplace_back(vertex.x, vertex.y, vertex.z);
    }

    auto qhHull = qh.getConvexHull(qhVertices, false, true);

    // Get the triangles that should be part of the convex hull
    const auto& indexBuffer = qhHull.getIndexBuffer();

    // Helper function to keep track of the vertices that should be present in the final result
    std::vector<Vertex> hullVertices;
    std::vector<IndexTriangle> hullTriangles;
    {
        std::vector<unsigned int> vertexMapping(this->vertices.size(), -1); // Element at position i contains the index of vertex i in the hullVertices vector, -1 if not present in hullVertices
        auto addVertexIfNotPresent = [&](unsigned int index){
            if(vertexMapping.at(index)==-1){
                vertexMapping.at(index) = hullVertices.size();
                hullVertices.emplace_back(this->vertices.at(index));
            }
            return vertexMapping.at(index);
        };

        hullTriangles.reserve(indexBuffer.size() / 3);
        for(int i = 0; i < indexBuffer.size(); i += 3){

            // Put the vertices in the hullVertices vector if not present yet
            unsigned int newIndex0 = addVertexIfNotPresent(indexBuffer[i]);
            unsigned int newIndex1 = addVertexIfNotPresent(indexBuffer[i + 1]);
            unsigned int newIndex2 = addVertexIfNotPresent(indexBuffer[i + 2]);
            hullTriangles.emplace_back(newIndex0, newIndex1, newIndex2);
        }
    }

    // For increased robustness in some problematic cases, we take the convex hull again
    std::vector<quickhull::Vector3<float>> qhVertices2;
    qhVertices2.reserve(hullVertices.size());
    for(Vertex vertex: hullVertices){
        qhVertices2.emplace_back(vertex.x, vertex.y, vertex.z);
    }

    auto qhHull2 = qh.getConvexHull(qhVertices2, false, true);

    // Create the convex hull mesh
    const auto& indexBuffer2 = qhHull2.getIndexBuffer();
    std::vector<Vertex> hullVertices2;
    std::vector<IndexTriangle> hullTriangles2;

    {
        std::vector<unsigned int> vertexMapping(hullVertices.size(), -1); // Element at position i contains the index of vertex i in the hullVertices vector, -1 if not present in hullVertices
        auto addVertexIfNotPresent = [&](unsigned int index){
            if(vertexMapping.at(index)==-1){
                vertexMapping.at(index) = hullVertices2.size();
                hullVertices2.emplace_back(hullVertices.at(index));
            }
            return vertexMapping.at(index);
        };

        hullTriangles2.reserve(indexBuffer2.size() / 3);
        for(int i = 0; i < indexBuffer2.size(); i += 3){

            // Put the vertices in the hullVertices vector if not present yet
            unsigned int newIndex0 = addVertexIfNotPresent(indexBuffer2[i]);
            unsigned int newIndex1 = addVertexIfNotPresent(indexBuffer2[i + 1]);
            unsigned int newIndex2 = addVertexIfNotPresent(indexBuffer2[i + 2]);
            hullTriangles2.emplace_back(newIndex0, newIndex1, newIndex2);
        }
    }

    auto hull = std::make_shared<ModelSpaceMesh>(hullVertices2, hullTriangles2);
    assert(glm::all(glm::epsilonEqual(hull->getBounds().getMinimum(), this->getBounds().getMinimum(), 1e-4f)) && "The convex hull should have the same bounding box as the original mesh");
    assert(glm::all(glm::epsilonEqual(hull->getBounds().getMaximum(), this->getBounds().getMaximum(), 1e-4f)) && "The convex hull should have the same bounding box as the original mesh");
    hull->setName("Convex hull of " + this->getName());
    this->convexHull = std::move(hull);
    return this->convexHull;
}

const std::string &ModelSpaceMesh::getName() const {
    return name;
}

void ModelSpaceMesh::setName(const std::string &newName) {
    ModelSpaceMesh::name = newName;
}

float ModelSpaceMesh::getVolume() const {
    if(!volume.has_value()){
        computeVolumeAndCentroid();
    }
    return volume.value();
}

bool ModelSpaceMesh::isConvex() const{

    // Return the value if already calculated before
    if(!convex.has_value()){
        computeConvexity();
    }

    return convex.value();
}

void ModelSpaceMesh::computeConvexity() const {
    // Calculate the convexity
    convex = true;
    for (const auto &indexTriangle: this->triangles){

        Vertex vertex0 = this->vertices.at(indexTriangle.vertexIndex0);
        Vertex vertex1 = this->vertices.at(indexTriangle.vertexIndex1);
        Vertex vertex2 = this->vertices.at(indexTriangle.vertexIndex2);

        VertexTriangle triangle({vertex0, vertex1, vertex2});

        for (const auto &vertex : this->vertices){

            // The dot product between the triangle's normal and the vector from the triangle to the vertex should be negative
            auto normal = glm::normalize(triangle.normal);
            auto delta = glm::normalize(vertex - vertex0);
            auto dot = glm::dot(normal, delta);

            if(dot > EPSILON){
                convex = false;
                return;
            }
        }
    }
}

void ModelSpaceMesh::computeSurfaceAreaAndCentroid() const {
    auto meshSurfaceArea = 0.0f;
    auto centroid = Vertex(0.0f, 0.0f, 0.0f);
    for (const auto &indexTriangle: this->triangles){

        const auto v0 = this->vertices.at(indexTriangle.vertexIndex0);
        const auto v1 = this->vertices.at(indexTriangle.vertexIndex1);
        const auto v2 = this->vertices.at(indexTriangle.vertexIndex2);

        auto edge0(v1-v0);
        auto edge1(v2-v1);
        auto triangleArea = glm::length(glm::cross(edge0, edge1)); // 2.0f * the area of the triangle
        auto center = (v0 + v1 + v2); // 3.0f * the center of the triangle

        centroid += center * triangleArea;
        meshSurfaceArea += glm::length(glm::cross(edge0, edge1));
    }
    surfaceArea = meshSurfaceArea / 2.0f;
    surfaceCentroid = centroid / (meshSurfaceArea*3.0f);
    assert(getBounds().containsPoint(surfaceCentroid.value()) && "The surface centroid should be inside the bounding box");
}

Vertex ModelSpaceMesh::getVolumeCentroid() const {
    if(!volumeCentroid.has_value()){
        computeVolumeAndCentroid();
        assert(volumeCentroid.has_value());
    }
    return volumeCentroid.value();
}

Vertex ModelSpaceMesh::getSurfaceCentroid() const {
    if(!surfaceCentroid.has_value()){
        computeSurfaceAreaAndCentroid();
        assert(surfaceCentroid.has_value());
    }
    return surfaceCentroid.value();
}

const AABB &ModelSpaceMesh::getBounds() const {
    if(!this->bounds.has_value()){
        this->bounds = AABBFactory::createAABB(this->vertices);
        assert(this->bounds.has_value());
    }
    return this->bounds.value();
}

float ModelSpaceMesh::getSurfaceArea() const {
    if(!surfaceArea.has_value()){
        computeSurfaceAreaAndCentroid();
        assert(surfaceArea.has_value());
    }
    return surfaceArea.value();
}

void ModelSpaceMesh::computeVolumeAndCentroid() const {
    auto meshVolume = 0.0f;
    auto centroid = Vertex(0.0f, 0.0f, 0.0f);

    for (const auto &triangle: triangles){
        auto v0 = vertices.at(triangle.vertexIndex0);
        auto v1 = vertices.at(triangle.vertexIndex1);
        auto v2 = vertices.at(triangle.vertexIndex2);

        auto center = (v0 + v1 + v2);         // 4.0f * the center of tetrahedron resembled by triangle and origin
        auto _volume = dot(v0, cross(v1, v2));  // 6.0f * the signed volume of the

        meshVolume += _volume;
        centroid += center * _volume;
    }
    volume = meshVolume/6.0f;
    assert(volume.value()/this->getBounds().getVolume() <= (1.0 + 1e-6)  && "Volume of mesh should be smaller than volume of its bounding box");
    volumeCentroid = centroid / (meshVolume*4.0f);
    assert(this->volume < 1e-5f || getBounds().containsPoint(volumeCentroid.value()) && "Volume centroid should be inside the bounding box of the mesh"); // The volume centroid is ill-defined for very small volumes
}

const std::vector<std::vector<size_t>>& ModelSpaceMesh::getConnectedVertexIndices() const {
    if(!connectedVertexIndices.has_value()){
        std::vector<std::unordered_set<size_t>> connectedVertexIndicesSet;
        connectedVertexIndicesSet.resize(vertices.size());
        for(size_t i = 0; i < vertices.size(); i++){
            connectedVertexIndicesSet.emplace_back();
        }
        for(const auto& triangle: triangles){
            connectedVertexIndicesSet.at(triangle.vertexIndex0).insert(triangle.vertexIndex1);
            connectedVertexIndicesSet.at(triangle.vertexIndex0).insert(triangle.vertexIndex2);
            connectedVertexIndicesSet.at(triangle.vertexIndex1).insert(triangle.vertexIndex0);
            connectedVertexIndicesSet.at(triangle.vertexIndex1).insert(triangle.vertexIndex2);
            connectedVertexIndicesSet.at(triangle.vertexIndex2).insert(triangle.vertexIndex0);
            connectedVertexIndicesSet.at(triangle.vertexIndex2).insert(triangle.vertexIndex1);
        }
        connectedVertexIndices = std::vector<std::vector<size_t>>();
        connectedVertexIndices->reserve(vertices.size());
        for(size_t i = 0; i < vertices.size(); i++){
            connectedVertexIndices->emplace_back(connectedVertexIndicesSet.at(i).begin(), connectedVertexIndicesSet.at(i).end());
        }
    }
    return connectedVertexIndices.value();
}

glm::vec3 ModelSpaceMesh::computeSupport(const glm::vec3 &direction) const {
    auto bestSupport = -std::numeric_limits<float>::max();
    auto bestVertex = glm::vec3(0.0f);
    for (const auto &vertex: this->vertices){
        auto support = glm::dot(vertex, direction);
        if(support > bestSupport){
            bestSupport = support;
            bestVertex = vertex;
        }
    }
    return bestVertex;
}

glm::vec3 ModelSpaceMesh::getCenter() const {
    return getBounds().getCenter(); // Alternatively, we could use something like the volume centroid here
}

void ModelSpaceMesh::computeFaces() const {

    // 0. Sort triangles by area
    auto sortedTriangles = this->triangles;
    std::sort(sortedTriangles.begin(), sortedTriangles.end(), [&](const IndexTriangle& a, const IndexTriangle& b) {
        auto areaA = glm::length(glm::cross(vertices[a.vertexIndex1] - vertices[a.vertexIndex0], vertices[a.vertexIndex2] - vertices[a.vertexIndex0]));
        auto areaB = glm::length(glm::cross(vertices[b.vertexIndex1] - vertices[b.vertexIndex0], vertices[b.vertexIndex2] - vertices[b.vertexIndex0]));
        return areaA > areaB;
    });

    // 1. Determine faces by merging triangles
    {
        // a. Keep track of which triangles are already included in a face
        std::vector<IndexFace> createdFaces;
        std::vector<bool> triangleIncluded(sortedTriangles.size(), false);

        // b. Merge triangles into createdFaces
        for (size_t t = 0; t < sortedTriangles.size(); ++t){

            // Continue to the next triangle if already marked as included ...
            if(triangleIncluded[t]) continue;

            // ... otherwise mark it as included and create a new face
            triangleIncluded[t] = true;
            std::vector<size_t> currentFaceIndices = {sortedTriangles[t].vertexIndex0, sortedTriangles[t].vertexIndex1, sortedTriangles[t].vertexIndex2};

            // Keep looking for triangles that can be merged with the current face as long as we find them
            bool mergeFound;
            do{
                mergeFound = false;

                // Compute the current plane of the face
                // Compute the normal using Newell's method
                glm::vec3 normal(0.0f);
                glm::vec3 centroid(0.0f);
                for (int i = 0; i < currentFaceIndices.size(); ++i){
                    auto j = (i + 1)%currentFaceIndices.size();
                    normal.x += (vertices[currentFaceIndices[i]].y - vertices[currentFaceIndices[j]].y) * (vertices[currentFaceIndices[i]].z + vertices[currentFaceIndices[j]].z);
                    normal.y += (vertices[currentFaceIndices[i]].z - vertices[currentFaceIndices[j]].z) * (vertices[currentFaceIndices[i]].x + vertices[currentFaceIndices[j]].x);
                    normal.z += (vertices[currentFaceIndices[i]].x - vertices[currentFaceIndices[j]].x) * (vertices[currentFaceIndices[i]].y + vertices[currentFaceIndices[j]].y);
                    centroid += vertices[currentFaceIndices[i]];
                }
                normal = glm::normalize(normal);
                centroid /= currentFaceIndices.size();

                Plane currentPlane(normal, centroid);

                for (size_t t2 = t+1; t2 < sortedTriangles.size(); ++t2){

                    // Check if the triangle is already included in the list of createdFaces
                    if(triangleIncluded[t2]) continue;

                    auto distance0 = currentPlane.distance(vertices[sortedTriangles[t2].vertexIndex0]);
                    auto distance1 = currentPlane.distance(vertices[sortedTriangles[t2].vertexIndex1]);
                    auto distance2 = currentPlane.distance(vertices[sortedTriangles[t2].vertexIndex2]);

                    float eps = 1e-3f;
                    if(distance0 > eps || distance1 > eps || distance2 > eps){
                        continue; // Triangle is not in the same plane
                    }

                    // Look for an edge that match one of the edges of the face
                    auto &triangle = sortedTriangles[t2];
                    for (int fi = 0; fi < currentFaceIndices.size(); ++fi){

                        // fi-th edge of the face
                        size_t vertexIndexA = currentFaceIndices[fi];
                        size_t vertexIndexB = currentFaceIndices[(fi + 1)%currentFaceIndices.size()];

                        // We can assume the winding order is correct, therefore matching indices as follows
                        if(triangle.vertexIndex0 == vertexIndexB && triangle.vertexIndex1 == vertexIndexA){
                            currentFaceIndices.insert(currentFaceIndices.begin() + fi + 1, triangle.vertexIndex2);
                            triangleIncluded[t2] = true;
                            mergeFound = true;
                            break;
                        }
                        else if(triangle.vertexIndex1 == vertexIndexB && triangle.vertexIndex2 == vertexIndexA){
                            currentFaceIndices.insert(currentFaceIndices.begin() + fi + 1, triangle.vertexIndex0);
                            triangleIncluded[t2] = true;
                            mergeFound = true;
                            break;
                        }
                        else if(triangle.vertexIndex2 == vertexIndexB && triangle.vertexIndex0 == vertexIndexA){
                            currentFaceIndices.insert(currentFaceIndices.begin() + fi + 1, triangle.vertexIndex1);
                            triangleIncluded[t2] = true;
                            mergeFound = true;
                            break;
                        }
                    }

                    if(mergeFound) break;
                }

            } while(mergeFound);


            // Postprocess: Remove unnecessary vertices
            bool duplicateRemoved;
            do{
                duplicateRemoved = false;
                for (auto i = 0; i < currentFaceIndices.size(); ++i){
                    auto j = (i + 2)%currentFaceIndices.size();

                    if(currentFaceIndices[i] == currentFaceIndices[j]){
                        // Remove indices in between
                        if(j > i){
                            currentFaceIndices.erase(currentFaceIndices.begin() + i, currentFaceIndices.begin() + j);
                        }
                        else {
                            // Remove ones at the back first
                            currentFaceIndices.erase(currentFaceIndices.begin() + i, currentFaceIndices.begin() + currentFaceIndices.size());
                            // Then at the front
                            currentFaceIndices.erase(currentFaceIndices.begin(), currentFaceIndices.begin() + j);
                        }
                        duplicateRemoved = true;
                        break;
                    }
                }
            } while(duplicateRemoved);

            // Compute the normal using Newell's method again and check if the face is facing the right direction
            glm::vec3 normal(0.0f);
            for (int i = 0; i < currentFaceIndices.size(); ++i){
                auto j = (i + 1)%currentFaceIndices.size();
                normal.x += (vertices[currentFaceIndices[i]].y - vertices[currentFaceIndices[j]].y) * (vertices[currentFaceIndices[i]].z + vertices[currentFaceIndices[j]].z);
                normal.y += (vertices[currentFaceIndices[i]].z - vertices[currentFaceIndices[j]].z) * (vertices[currentFaceIndices[i]].x + vertices[currentFaceIndices[j]].x);
                normal.z += (vertices[currentFaceIndices[i]].x - vertices[currentFaceIndices[j]].x) * (vertices[currentFaceIndices[i]].y + vertices[currentFaceIndices[j]].y);
            }
            auto n = glm::normalize(normal);
            auto d = -glm::dot(normal, vertices[currentFaceIndices[0]]);

            // Add the face to the list of createdFaces
            createdFaces.emplace_back(currentFaceIndices);
        }
        this->faces = std::move(createdFaces);
    }
}
