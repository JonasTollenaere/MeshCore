nvcc -ptx -arch sm_61 --use_fast_math OptixPrograms.cu -o OptixPrograms.ptx -I"C:/ProgramData/NVIDIA Corporation/Optix SDK 7.2.0/include"
nvcc -ptx -arch sm_61 --use_fast_math OptixPrograms2.cu -o OptixPrograms2.ptx -I"C:/ProgramData/NVIDIA Corporation/Optix SDK 7.2.0/include"
