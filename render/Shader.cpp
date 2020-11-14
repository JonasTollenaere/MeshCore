//
// Created by Jonas on 12/11/2020.
//

#include "Shader.h"
#include "Renderer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

Shader::Shader(const std::string& filepath): m_FilePath(filepath) {
    this->m_Source = ParseShader(filepath);
    this->m_RendererId = CreateShader(this->m_Source.VertexSource, this->m_Source.FragmentSource);
}

Shader::Shader(const Shader& shader): m_FilePath(shader.m_FilePath) {
    this->m_Source = shader.m_Source;
    this->m_RendererId = CreateShader(this->m_Source.VertexSource, this->m_Source.FragmentSource);
    this->m_UniformLocationCache = std::unordered_map(shader.m_UniformLocationCache);
}

Shader::~Shader() {
    GLCall(glDeleteProgram(m_RendererId));
}

ShaderProgramSource Shader::ParseShader(const std::string& filepath){
    std::ifstream stream(filepath);

    enum class ShaderType{
        NONE = -1, VERTEX = 0, FRAGMENT = 1
    };

    std::string line;
    std::stringstream ss[2];
    ShaderType type = ShaderType::NONE;
    while(getline(stream, line)){
        if(line.find("#shader") != std::string::npos){
            if(line.find("vertex") != std::string::npos){
                type = ShaderType::VERTEX;
            }
            else if(line.find("fragment") != std::string::npos){
                type = ShaderType::FRAGMENT;
            }
        }
        else{
            ss[(int) type] << line << "\n";
        }
    }

    return{ ss[0].str(), ss[1].str()};
}

unsigned int Shader::CompileShader(unsigned int type, const std::string& source){
    GLCall(unsigned int id = glCreateShader(type));
    const char* src = source.c_str();
    GLCall(glShaderSource(id, 1, &src, nullptr));
    GLCall(glCompileShader(id));

    int result;
    GLCall(glGetShaderiv(id, GL_COMPILE_STATUS, &result));
    if(result == GL_FALSE){
        int length;
        GLCall(glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length));
        char* message = (char*) alloca(length * sizeof(char));
        GLCall(glGetShaderInfoLog(id, length, &length, message));
        std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
        std::cout << message << std::endl;
        GLCall(glDeleteShader(id));
        return 0;
    }
    return id;
}


unsigned int Shader::CreateShader(const std::string& vertexShader, const std::string& fragmentShader){

    GLCall(unsigned int program =   glCreateProgram());
    GLCall(unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader));
    GLCall(unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader));

    GLCall(glAttachShader(program, vs));
    GLCall(glAttachShader(program, fs));
    GLCall(glLinkProgram(program));
    GLCall(glValidateProgram(program));

    GLCall(glDeleteShader(vs));
    GLCall(glDeleteShader(fs));

    return program;
}


void Shader::bind() const {
    GLCall(glUseProgram(m_RendererId));
}

void Shader::unbind() const {
    GLCall(glUseProgram(0));
}

void Shader::setUniform4f(const std::string &name, float v0, float v1, float v2, float v3) {
    GLCall(glUniform4f(getUniformLocation(name), v0, v1, v2, v3));
}

unsigned int Shader::getUniformLocation(const std::string &name) {

    if(m_UniformLocationCache.find(name) != m_UniformLocationCache.end()){
        return m_UniformLocationCache[name];
    }

    GLCall(int location = glGetUniformLocation(m_RendererId, name.c_str()));
    if (location == -1){
        std::cout << "Warning: uniform " << name << " doesn't exist!" << std::endl;
    }
    m_UniformLocationCache[name] = location;
    return location;
}

void Shader::setUniformMat4f(const std::string& name, const glm::mat4& matrix) {
    GLCall(glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &matrix[0][0]));
}
