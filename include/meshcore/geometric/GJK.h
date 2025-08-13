//
// File created by Jonas on 15/06/2022. Core algorithm mostly copied from an implementation by Wim Vancroonenburg
//

#ifndef MESHCORE_GJK_H
#define MESHCORE_GJK_H

#include <optional>
#include <array>
#include "meshcore/core/AABB.h"
#include "meshcore/core/OBB.h"
#include "meshcore/core/VertexTriangle.h"
#include "meshcore/core/Sphere.h"

#define GJK_EPSILON 0.0001f

class GJKConvexShape {
public:
    virtual ~GJKConvexShape() = default;

    /**
     * Computes the support point of the shape in the given direction.
     * @param direction
     * @return
     */
    [[nodiscard]] virtual glm::vec3 computeSupport(const glm::vec3& direction) const = 0;

    /**
     * Returns (an approximation of) the center of the shape. This is used for finding a good initial direction in GJK queries.
     * @return The center of the shape
     */
    [[nodiscard]] virtual glm::vec3 getCenter() const = 0;
};

class GJK {
public:
    static std::optional<float> computeDistanceSqr(const GJKConvexShape &shapeA, const GJKConvexShape &shapeB);
    static bool hasSeparation(const GJKConvexShape &shapeA, const GJKConvexShape &shapeB, float minimumSeparationDistanceSqr=0.0f);
    static std::optional<std::pair<Vertex,Vertex>> computeClosestPoints(const GJKConvexShape &shapeA, const GJKConvexShape &shapeB);

private:

    /**
     * Support point class for use in the GJK algorithm.
     * Not only the support point itself is stored, but also the two points from which the support point was calculated.
     * This facilitates the reconstruction of the closest points on the two shapes.
     */
    class SupportPoint {
    public:
        SupportPoint() = default;
        SupportPoint(glm::vec3 supportPointA, glm::vec3 supportPointB) : supportPointA(supportPointA), supportPointB(supportPointB), point(supportPointA - supportPointB) {}
        glm::vec3 point;
        glm::vec3 supportPointA;
        glm::vec3 supportPointB;

    };

    /**
     * Simplex class for use in the GJK algorithm. The simplex is a set of support points.
     * Originally being a vector to store 0-4 support points,
     * this was abstracted into a class instead of a vector to try and avoid unnecessary heap allocations.
     */
    class Simplex {
        std::array<SupportPoint, 4> points = {};
        unsigned char pointsCount = 0;
    public:
        [[nodiscard]] unsigned char size() const {
            assert(pointsCount <= 4);
            return pointsCount;
        }

        [[nodiscard]] bool empty() const {
            return pointsCount == 0;
        }

        SupportPoint& operator[](unsigned char i) {
            assert(i < pointsCount);
            return points[i];
        }

        const SupportPoint& operator[](unsigned char i) const {
            assert(i < pointsCount);
            return points[i];
        }

        bool addSupportPoint(const SupportPoint& supportPoint) {

            // Check if the support point is already in the simplex
            for (unsigned char i = 0; i < pointsCount; i++) {
                if (glm::all(glm::epsilonEqual(points[i].point, supportPoint.point, 1e-4f))) {
                    return false;
                }
            }

            // Add the support point to the simplex if not already present
            assert(pointsCount < 4);
            points[pointsCount] = supportPoint;
            pointsCount++;
            return true;
        }

        void removeSupportPoint(unsigned char i) {
            assert(i < pointsCount);

            // Swap the last element with the element to remove
            points[i] = points[pointsCount - 1];

            // "Remove" the last element by decrementing the count
            pointsCount--;
        }
    };

    // Core GJK functionality: computing support points and updating the simplex
    static SupportPoint support(const GJKConvexShape& convex_shape_A, const GJKConvexShape& convex_shape_B, const glm::vec3& D);
    static std::optional<glm::vec3> doSimplex(Simplex& simplex);
    static glm::vec3 doSimplex1(Simplex& simplex);
    static glm::vec3 doSimplex2(Simplex& simplex);
    static glm::vec3 doSimplex3(Simplex& simplex);
    static glm::vec3 doSimplex4(Simplex& simplex);

    // Supporting functions for the GJK algorithm
    static std::pair<glm::vec3, glm::vec3> computeClosestPoints(const Simplex& simplex);
    static glm::vec3 estimateSeparatingDirection(const GJKConvexShape& shapeA, const GJKConvexShape& shapeB);
};

class GJKVertex: public GJKConvexShape {
    Vertex vertex;
public:
    explicit GJKVertex(const Vertex& vertex);
    [[nodiscard]] glm::vec3 computeSupport(const glm::vec3 &direction) const override;
    [[nodiscard]] glm::vec3 getCenter() const override;
};

class GJKAABB: public GJKConvexShape{
    Vertex center;
    glm::vec3 half;
public:
    explicit GJKAABB(const AABB& aabb);
    [[nodiscard]] glm::vec3 computeSupport(const glm::vec3 &direction) const override;
    [[nodiscard]] glm::vec3 getCenter() const override;
};

class GJKOBB: public GJKConvexShape {
    Quaternion rotation;
    GJKAABB gjkaabb;

public:
    explicit GJKOBB(const OBB& obb);
    [[nodiscard]] glm::vec3 computeSupport(const glm::vec3 &direction) const override;
    [[nodiscard]] glm::vec3 getCenter() const override;
};

class GJKSphere: public GJKConvexShape {
    Vertex center;
    float radius;
public:
    explicit GJKSphere(const Sphere& sphere);
    [[nodiscard]] glm::vec3 computeSupport(const glm::vec3 &direction) const override;
    [[nodiscard]] glm::vec3 getCenter() const override;
};

#endif //MESHCORE_GJK_H