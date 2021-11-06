//
// Created by Jonas on 1/12/2020.
//

#ifndef MESHCORE_SHADERPROGRAMSOURCE_H
#define MESHCORE_SHADERPROGRAMSOURCE_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <qstring.h>
#include <unordered_map>

class ShaderProgramSource{
public:
    const QString VertexSource;
    const QString FragmentSource;
    static ShaderProgramSource parseShader(const std::string& filepath);
};

#endif //MESHCORE_SHADERPROGRAMSOURCE_H
