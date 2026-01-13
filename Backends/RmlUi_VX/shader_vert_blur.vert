#version 460
#extension GL_GOOGLE_include_directive : require

#include "BlurDefines.h"

layout(push_constant) uniform VsInput {
	vec2 texelOffset;
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord[BLUR_SIZE];

void main() {
	for (int i = 0; i < BLUR_SIZE; ++i) {
		fragTexCoord[i] = inTexCoord - float(i - BLUR_NUM_WEIGHTS + 1) * texelOffset;
    }
    gl_Position = vec4(inPosition, 1.0);
}