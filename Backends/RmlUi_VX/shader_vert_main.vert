#version 460
#extension GL_GOOGLE_include_directive : require

#include "VsInput.glsl"

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec4 fragColor;

void main() {
	fragTexCoord = inTexCoord;
	fragColor = inColor;
    gl_Position = matrices[transformIdx] * vec4(inPosition + translate, 0, 1);
}