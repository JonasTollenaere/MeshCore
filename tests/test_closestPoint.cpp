//
// Created by Jonas on 6/10/2025.
//

#include <gtest/gtest.h>

#include "meshcore/core/Sphere.h"


TEST(ClosestPoint, Sphere) {

    auto center = Vertex(1.23, 4.56, 7.89);
    auto radius = 2.5f;

    Sphere sphere(center, radius);

    auto closestPoint = sphere.getClosestPoint(Vertex(10.0, -10.0, 10.0));

    // Assert this point lies on the sphere
    EXPECT_FLOAT_EQ(glm::distance(closestPoint, center), radius);
}