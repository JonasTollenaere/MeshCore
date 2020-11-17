//
// Created by Jonas on 12/11/2020.
//

#ifndef MESHCORE2_SHADER_H
#define MESHCORE2_SHADER_H

#include <iostream>
#include <glm/glm.hpp>
#include <unordered_map>

struct ShaderProgramSource{
    std::string VertexSource;
    std::string FragmentSource;
};

class Shader {
private:
    ShaderProgramSource m_Source;
    std::string m_FilePath;
    unsigned int m_RendererId;
    std::unordered_map<std::string, unsigned int> m_UniformLocationCache;
public:
    explicit Shader(const std::string& filepath);
    Shader(const Shader& shader);
    ~Shader();

    void bind() const;
    void unbind() const;

    // Set uniforms
//    void setUniform4f(const std::string& name, float v0, float v1, float v2, float v3);
    void setUniformMat4f(const std::string& name, const glm::mat4& matrix);
    void setUniform1f(const std::string& name, float value);
    void setUniform3fv(const std::string& name, const glm::vec3& vector);

private:

    unsigned int getUniformLocation(const std::string& name);


    ShaderProgramSource ParseShader(const std::string &filepath);
    static unsigned int CompileShader(unsigned int type, const std::string &source);
    static unsigned int CreateShader(const std::string &vertexShader, const std::string &fragmentShader);

};


#endif //MESHCORE2_SHADER_H
