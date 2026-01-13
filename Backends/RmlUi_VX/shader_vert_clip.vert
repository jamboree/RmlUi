#version 460
#extension GL_GOOGLE_include_directive : require

#include "VsInput.glsl"

layout(location = 0) in vec2 inPosition;

void main() {
    gl_Position = matrices[transformIdx] * vec4(inPosition + translate, 0, 1);
}