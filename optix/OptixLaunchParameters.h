//
// Created by Jonas on 16/12/2020.
//

#ifndef MESHCORE_OPTIXLAUNCHPARAMETERS_H
#define MESHCORE_OPTIXLAUNCHPARAMETERS_H

struct OptixLaunchParameters
{
    OptixTraversableHandle handle;
    bool* d_returnValue;
};

#endif //MESHCORE_OPTIXLAUNCHPARAMETERS_H
