/*** Triangle/triangle intersection test routine,
 * by Tomas Moller, 1997.
 * See article "A Fast Triangle-Triangle Intersection Test",
 * Journal of Graphics Tools, 2(2), 1997
 *
 * Updated June 1999: removed the divisions -- a little faster now!
 * Updated October 1999: added {} to CROSS and SUB macros
 *
 * int NoDivTriTriIsect(float V0[3],float V1[3],float V2[3],
 *                      float U0[3],float U1[3],float U2[3])
 *
 * parameters: modelSpaceVertices of triangle 1: V0,V1,V2
 *             modelSpaceVertices of triangle 2: U0,U1,U2
 * result    : returns 1 if the triangles intersect, otherwise 0
 *
 */

#include <glm/glm.hpp>

#include "meshcore/geometric/Intersection.h"

#define FABS(x) (abs(x))        /* implement as is fastest on your machine */

/* if USE_EPSILON_TEST is true then we do a check:
         if |dv|<EPSILON then dv=0.0;
   else no check is done (which is less robust)
*/
#define USE_EPSILON_TEST TRUE
#define EPSILON 0.000001f

/* sort so that a<=b */
#define SORT(a,b)       \
             if(a>b)    \
             {          \
               float temporary; \
               temporary=a;     \
               a=b;     \
               b=temporary;     \
             }

/* this edge to edge test is based on Franlin Antonio's gem:
   "Faster Line Segment Intersection", in Graphics Gems III,
   pp. 199-202 */
#define EDGE_EDGE_TEST(V0,U0,U1)                      \
  Bx=U0[i0]-U1[i0];                                   \
  By=U0[i1]-U1[i1];                                   \
  Cx=V0[i0]-U0[i0];                                   \
  Cy=V0[i1]-U0[i1];                                   \
  f=Ay*Bx-Ax*By;                                      \
  d=By*Cx-Bx*Cy;                                      \
  if((f>0 && d>=0 && d<=f) || (f<0 && d<=0 && d>=f))  \
  {                                                   \
    e=Ax*Cy-Ay*Cx;                                    \
    if(f>0)                                           \
    {                                                 \
      if(e>=0 && e<=f) return 1;                      \
    }                                                 \
    else                                              \
    {                                                 \
      if(e<=0 && e>=f) return 1;                      \
    }                                                 \
  }

#define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
{                                              \
  float Ax,Ay,Bx,By,Cx,Cy,e,d,f;               \
  Ax=V1[i0]-V0[i0];                            \
  Ay=V1[i1]-V0[i1];                            \
  /* test edge U0,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U0,U1);                    \
  /* test edge U1,U2 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U1,U2);                    \
  /* test edge U2,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U2,U0);                    \
}

#define POINT_IN_TRI(V0,U0,U1,U2)           \
{                                           \
  float a,b,c,d0,d1,d2;                     \
  /* is T1 completly inside T2? */          \
  /* check if V0 is inside tri(U0,U1,U2) */ \
  a=U1[i1]-U0[i1];                          \
  b=-(U1[i0]-U0[i0]);                       \
  c=-a*U0[i0]-b*U0[i1];                     \
  d0=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U2[i1]-U1[i1];                          \
  b=-(U2[i0]-U1[i0]);                       \
  c=-a*U1[i0]-b*U1[i1];                     \
  d1=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U0[i1]-U2[i1];                          \
  b=-(U0[i0]-U2[i0]);                       \
  c=-a*U2[i0]-b*U2[i1];                     \
  d2=a*V0[i0]+b*V0[i1]+c;                   \
  if(d0*d1>0.0)                             \
  {                                         \
    if(d0*d2>0.0) return 1;                 \
  }                                         \
}

#define NEWCOMPUTE_INTERVALS(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,A,B,C,X0,X1) \
{ \
        if(D0D1>0.0f) \
        { \
                /* here we know that D0D2<=0.0 */ \
            /* that is D0, D1 are on the same side, D2 on the other or on the plane */ \
                A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
        } \
        else if(D0D2>0.0f)\
        { \
                /* here we know that d0d1<=0.0 */ \
            A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
        } \
        else if(D1*D2>0.0f || D0!=0.0f) \
        { \
                /* here we know that d0d1<=0.0 or that D0!=0.0 */ \
                A=VV0; B=(VV1-VV0)*D0; C=(VV2-VV0)*D0; X0=D0-D1; X1=D0-D2; \
        } \
        else if(D1!=0.0f) \
        { \
                A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
        } \
        else if(D2!=0.0f) \
        { \
                A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
        } \
        else \
        { \
                /* triangles are coplanar */ \
                return coplanar_tri_tri(triangleA.normal,triangleA.vertices[0],triangleA.vertices[1],triangleA.vertices[2],triangleB.vertices[0],triangleB.vertices[1],triangleB.vertices[2]); \
        } \
}

int coplanar_tri_tri(const glm::vec3& n,
                     const glm::vec3& V0, const glm::vec3& V1, const glm::vec3& V2,
                     const glm::vec3& U0, const glm::vec3& U1, const glm::vec3& U2)
{
    glm::vec3 A;
    short i0, i1;
    /* first project onto an axis-aligned plane, that maximizes the area */
    /* of the triangles, compute indices: i0,i1. */
    A[0] = FABS(n[0]);
    A[1] = FABS(n[1]);
    A[2] = FABS(n[2]);
    if (A[0] > A[1])
    {
        if (A[0] > A[2])
        {
            i0 = 1;      /* A[0] is greatest */
            i1 = 2;
        }
        else
        {
            i0 = 0;      /* A[2] is greatest */
            i1 = 1;
        }
    }
    else   /* A[0]<=A[1] */
    {
        if (A[2] > A[1])
        {
            i0 = 0;      /* A[2] is greatest */
            i1 = 1;
        }
        else
        {
            i0 = 0;      /* A[1] is greatest */
            i1 = 2;
        }
    }

    /* test all edges of triangle 1 against the edges of triangle 2 */
    EDGE_AGAINST_TRI_EDGES(V0, V1, U0, U1, U2)
    EDGE_AGAINST_TRI_EDGES(V1, V2, U0, U1, U2)
    EDGE_AGAINST_TRI_EDGES(V2, V0, U0, U1, U2)

    /* finally, test if tri1 is totally contained in tri2 or vice versa */
    POINT_IN_TRI(V0, U0, U1, U2)
    POINT_IN_TRI(U0, V0, V1, V2)

    return 0;
}


namespace Intersection
{
    bool intersect(const VertexTriangle& triangleA, const VertexTriangle& triangleB)
    {
        /* compute plane equation of triangle(V0,V1,V2) */

        float d1 = - glm::dot(triangleA.normal, triangleA.vertices[0]);
        /* plane equation 1: N1.X+d1=0 */

        /* put U0,U1,U2 into plane equation 1 to compute signed distances to the plane*/
        float du0 = glm::dot(triangleA.normal, triangleB.vertices[0]) + d1;
        float du1 = glm::dot(triangleA.normal, triangleB.vertices[1]) + d1;
        float du2 = glm::dot(triangleA.normal, triangleB.vertices[2]) + d1;

        /* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
        if (FABS(du0) < EPSILON) du0 = 0.0;
        if (FABS(du1) < EPSILON) du1 = 0.0;
        if (FABS(du2) < EPSILON) du2 = 0.0;
#endif
        float du0du1 = du0 * du1;
        float du0du2 = du0 * du2;

        if (du0du1 > 0.0f && du0du2 > 0.0f) /* same sign on all of them + not equal 0 ? */
            return false; /* no intersection occurs */

        /* compute plane of triangle (U0,U1,U2) */
        float d2 = - glm::dot(triangleB.normal, triangleB.vertices[0]);
        /* plane equation 2: N2.X+d2=0 */

        /* put V0,V1,V2 into plane equation 2 */
        float dv0 = glm::dot(triangleB.normal, triangleA.vertices[0]) + d2;
        float dv1 = glm::dot(triangleB.normal, triangleA.vertices[1]) + d2;
        float dv2 = glm::dot(triangleB.normal, triangleA.vertices[2]) + d2;

#if USE_EPSILON_TEST==TRUE
        if (FABS(dv0) < EPSILON) dv0 = 0.0;
        if (FABS(dv1) < EPSILON) dv1 = 0.0;
        if (FABS(dv2) < EPSILON) dv2 = 0.0;
#endif

        float dv0dv1 = dv0 * dv1;
        float dv0dv2 = dv0 * dv2;

        if (dv0dv1 > 0.0f && dv0dv2 > 0.0f) /* same sign on all of them + not equal 0 ? */
            return false;                    /* no intersection occurs */

        /* compute direction of intersection line */
        glm::vec3 dir = glm::cross(triangleA.normal,triangleB.normal);

        /* compute and index to the largest component of D */
        float max = FABS(dir[0]);
        short index = 0;
        float bb = FABS(dir[1]);
        float cc = FABS(dir[2]);
        if (bb > max) max = bb, index = 1;
        if (cc > max) index = 2;

        /* this is the simplified projection onto L*/
        float vp0 = triangleA.vertices[0][index];
        float vp1 = triangleA.vertices[1][index];
        float vp2 = triangleA.vertices[2][index];

        float up0 = triangleB.vertices[0][index];
        float up1 = triangleB.vertices[1][index];
        float up2 = triangleB.vertices[2][index];

        /* compute interval for triangle 1 */
        float a, b, c, x0, x1;
        NEWCOMPUTE_INTERVALS(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, a, b, c, x0, x1)

        /* compute interval for triangle 2 */
        float d, e, f, y0, y1;
        NEWCOMPUTE_INTERVALS(up0, up1, up2, du0, du1, du2, du0du1, du0du2, d, e, f, y0, y1)

        float xx, yy, xxyy, tmp;
        float isect1[2], isect2[2];
        xx = x0 * x1;
        yy = y0 * y1;
        xxyy = xx * yy;

        tmp = a * xxyy;

        isect1[0] = tmp + b * x1 * yy;
        isect1[1] = tmp + c * x0 * yy;

        tmp = d * xxyy;
        isect2[0] = tmp + e * xx * y1;
        isect2[1] = tmp + f * xx * y0;

        SORT(isect1[0], isect1[1])
        SORT(isect2[0], isect2[1])

        if (isect1[1] < isect2[0] || isect2[1] < isect1[0]) return false;
        return true;
    }

    bool edgeIntersectsTriangle(const glm::vec3& v0, const glm::vec3& v1,
                                const glm::vec3& u0, const glm::vec3& u1, const glm::vec3& u2){
        Ray ray(v0,  v1 - v0);
        float t = Intersection::intersectionDistance(ray, VertexTriangle(u0, u1, u2));
        if(t > 0 && t < 1 + EPSILON) return true;
        return false;

    }

    bool ExperimentalTriangleTriangleIntersection(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                                                  const glm::vec3& u0, const glm::vec3& u1, const glm::vec3& u2){

        if(edgeIntersectsTriangle(v0, v1, u0, u1, u2)) return true;
        if(edgeIntersectsTriangle(u0, u1, v0, v1, v2)) return true;
        if(edgeIntersectsTriangle(v1, v2, u0, u1, u2)) return true;
        if(edgeIntersectsTriangle(u1, u2, v0, v1, v2)) return true;
        if(edgeIntersectsTriangle(v2, v0, u0, u1, u2)) return true;
//        if(edgeIntersectsTriangle(u2, u0, v0, v1, v2)) return true; // Checking all 6 edges is redundant!
//        In case of a triangle-triangle intersection there will always be 2 edges that ray-intersect the other triangle
        return false;
    }
}
