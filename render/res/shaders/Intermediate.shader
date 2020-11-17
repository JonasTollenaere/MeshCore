#shader vertex
#version 330 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;
layout(location = 2) in vec4 a_Color;

uniform mat4 u_ViewProjectionMatrix;

out vec3 v_Position;
out vec3 v_Normal;
out vec4 v_Color;

void main(){
    gl_Position = u_ViewProjectionMatrix * vec4(a_Position, 1.0);
    v_Position = a_Position;
    v_Normal = a_Normal;
    v_Color = a_Color;
}

    #shader fragment
    #version 330 core

layout(location = 0) out vec4 o_Color;

in vec3 v_Position;
in vec3 v_Normal;
in vec4 v_Color;

uniform vec3 u_LightSource;
uniform float u_Ambient;
uniform mat4 u_ViewMatrix;

void main(){
    vec3 light = - normalize(u_LightSource);
    vec3 normal = normalize(v_Normal); // View space
    float diffuse = max(dot(normal, light), 0.0); // View space
    o_Color = (u_Ambient + diffuse * (1-u_Ambient)) * v_Color;
    o_Color[3] = v_Color[3];
}