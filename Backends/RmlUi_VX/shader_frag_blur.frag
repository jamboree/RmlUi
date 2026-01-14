#version 460
#extension GL_GOOGLE_include_directive : require

#include "BlurDefines.h"
#include "BindlessTextures.glsl"

layout(push_constant) uniform FsInput {
    TEX_IDX texIdx;
	vec2 texCoordMin;
	vec2 texCoordMax;
	float weights[BLUR_NUM_WEIGHTS];
};

layout(location = 0) in vec2 fragTexCoord[BLUR_SIZE];
layout(location = 0) out vec4 finalColor;

void main() {
	vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
	for (int i = 0; i < BLUR_SIZE; ++i) {
		vec2 in_region = step(texCoordMin, fragTexCoord[i]) * step(fragTexCoord[i], texCoordMax);
		color += texture(sampler2D(textures[texIdx], mySampler), fragTexCoord[i]) * (in_region.x * in_region.y * weights[abs(i - BLUR_NUM_WEIGHTS + 1)]);
	}
	finalColor = color;
}