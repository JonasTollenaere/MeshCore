//
// Created by Jonas on 23/12/2020.
//

#ifndef MESHCORE_INDEXEDGE_H
#define MESHCORE_INDEXEDGE_H

#include <algorithm>

/**
 * @brief Represents an (undirected) edge as a pair of vertex indices.
 *
 * IndexEdge stores the two vertex indices in canonical (non-decreasing)
 * order: vertexIndex0 <= vertexIndex1. The constructor normalizes the
 * provided indices so that the same undirected edge (a,b) and (b,a)
 * produce identical IndexEdge objects.
 *
 * Invariants:
 * - vertexIndex0 and vertexIndex1 are immutable (const) after construction.
 * - vertexIndex0 <= vertexIndex1 (canonical ordering).
 *
 * Rationale / usage notes:
 * - Because indices are canonicalized on construction, equality is a
 *   simple member-wise comparison (see operator==).
 * - To use IndexEdge as a key in unordered containers you may either
 *   provide a `std::hash<IndexEdge>` specialization or pass a custom
 *   hasher. Since the indices are canonical, a straightforward hash is to
 *   pack the two indices (e.g. ((uint64_t)vertexIndex0 << 32) | vertexIndex1)
 *   and hash that value.
 *
 * Example:
 *   IndexEdge e(5, 2); // stored as (2,5)
 */
struct IndexEdge{
    const size_t vertexIndex0;
    const size_t vertexIndex1;

    IndexEdge(const size_t vertexIndex0, const size_t vertexIndex1):
    vertexIndex0(std::min(vertexIndex0, vertexIndex1)),
    vertexIndex1(std::max(vertexIndex0, vertexIndex1))
    {}
};

// Inline equality and inequality operators. IndexEdge stores indices in a canonical
// (min, max) order, so equality is a simple member-wise comparison.
constexpr bool operator==(const IndexEdge &a, const IndexEdge &b) noexcept {
    return a.vertexIndex0 == b.vertexIndex0 && a.vertexIndex1 == b.vertexIndex1;
}

constexpr bool operator!=(const IndexEdge &a, const IndexEdge &b) noexcept {
    return !(a == b);
}

#endif //MESHCORE_INDEXEDGE_H
