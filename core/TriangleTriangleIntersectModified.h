#pragma once
namespace MIntersection
{
	bool NoDivTriTriIsect(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                          const glm::vec3& u0, const glm::vec3& u1, const glm::vec3& u2);

	int coplanar_tri_tri(const glm::vec3& n,
                         const glm::vec3& V0, const glm::vec3& V1, const glm::vec3& V2,
                         const glm::vec3& U0, const glm::vec3& U1, const glm::vec3& U2);

	bool ExperimentalTriangleTriangleIntersection(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                                                  const glm::vec3& u0, const glm::vec3& u1, const glm::vec3& u2);
}