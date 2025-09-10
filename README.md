[![Latest Release](https://img.shields.io/github/v/release/JonasTollenaere/MeshCore?label=release)](https://github.com/JonasTollenaere/MeshCore/releases)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jonastollenaere.github.io/MeshCore/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/311904459.svg)](https://doi.org/10.5281/zenodo.16533653)

![](icons/logo4.svg)

#### A C++ library aimed at rapid prototyping of 3D irregular cutting and packing algorithms. Created by Jonas Tollenaere (jonas.tollenaere@kuleuven.be)

## Public optimization projects that use MeshCore
* [strip-milp-3d](https://github.com/JonasTollenaere/strip-milp-3d): Constructing MILP models of 3D irregular strip packing problems with convex items
* [svmp-heuristic](https://github.com/JonasTollenaere/svmp-heuristic): Local search heuristics for single volume maximisation problems
* [svmp-quaternion-qcp](https://github.com/JonasTollenaere/svmp-quaternion-qcp): Quadratically constrained formulations for the single volume maximisation problem
* [sparrow-3d](https://github.com/JonasTollenaere/sparrow-3d): A 3D adaptation of the state-of-the-art [sparrow](https://github.com/JeroenGar/sparrow) nesting algorithm, animated below

![Packing Animation](icons/example.gif)

## Using MeshCore

MeshCore uses CMake as its build system.
We recommend using vcpkg as a package manager to install the dependencies that are defined in the `vcpkg.json` file.
To set up a development environment, we refer to the [vcpkg documentation](https://vcpkg.io/en/getting-started).
Our aim is to keep the library cross-platform, and is currently developed on Windows, Linux, and macOS.

We suggest integrating MeshCore into your project through the CMake `FetchContent` module or as a submodule.
Two sample projects are provided as an example:
* [fetchcontent-sample](https://github.com/JonasTollenaere/sample-meshcore-through-cmake-fetchcontent): Minimal project that uses MeshCore through CMake's `FetchContent` module
* [submodule-sample](https://github.com/JonasTollenaere/sample-meshcore-through-git-submodule): Minimal project that uses MeshCore as a git submodule

At this point in time, the documentation is rather limited.
The `demos` folder contains a number of example applications that demonstrate basic usage of the library. 
For more examples we refer to the repositories listed in the section below.

## Acknowledgements
Development of this project began under [CODeS](https://numa.cs.kuleuven.be/research/combinatorial-optimization), part of NUMA, the Numerical Analysis and Applied Mathematics research unit of KU Leuven.
