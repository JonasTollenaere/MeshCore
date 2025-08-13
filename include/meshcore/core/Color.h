//
// Created by Jonas on 17/09/2022.
//

#ifndef MESHCORE_COLOR_H
#define MESHCORE_COLOR_H

#include <glm/glm.hpp>

class Color: public glm::vec4 {
    using glm::vec4::vec4;
public:
    // Define some colors as static public
    explicit Color(const glm::vec4& color): glm::vec4(color){};
    [[maybe_unused]] static Color Red(float alpha=1.0f){return {1.0f, 0.0f, 0.0f, alpha};}
    [[maybe_unused]] static Color Green(float alpha=1.0f){return {0.0f, 1.0f, 0.0f, alpha};}
    [[maybe_unused]] static Color Blue(float alpha=1.0f){return {0.0f, 0.0f, 1.0f, alpha};}
    [[maybe_unused]] static Color White(float alpha=1.0f){return {1.0f, 1.0f, 1.0f, alpha};}
    [[maybe_unused]] static Color Black(float alpha=1.0f){return {0.0f, 0.0f, 0.0f, alpha};}
    [[maybe_unused]] static Color Yellow(float alpha=1.0f){return {1.0f, 1.0f, 0.0f, alpha};}
    [[maybe_unused]] static Color Cyan(float alpha=1.0f){return {0.0f, 1.0f, 1.0f, alpha};}
    [[maybe_unused]] static Color Magenta(float alpha=1.0f){return {1.0f, 0.0f, 1.0f, alpha};}
    [[maybe_unused]] static Color Orange(float alpha=1.0f){return {1.0f, 0.5f, 0.0f, alpha};}
    [[maybe_unused]] static Color LightBlue(float alpha=1.0f){return {0.333, 1.0f, 1.0f, alpha};}
    [[maybe_unused]] static Color Brown(float alpha=1.0f){return {0.666f, 0.333f, 0.0f, alpha};}
    [[maybe_unused]] static Color Locust(float alpha=1.0f){return {0.666f, 0.666f, 0.5f, alpha};}
};

#endif //MESHCORE_COLOR_H
