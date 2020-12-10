#pragma once
namespace Intersection
{
	int NoDivTriTriIsect(float V0[3], float V1[3], float V2[3],
		float U0[3], float U1[3], float U2[3]);

	int coplanar_tri_tri(float N[3], float V0[3], float V1[3], float V2[3],
		float U0[3], float U1[3], float U2[3]);
}