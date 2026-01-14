#version 460
#extension GL_GOOGLE_include_directive : require

#include "BindlessTextures.glsl"

layout(push_constant) uniform FsInput {
    TEX_IDX texIdx;
	vec2 texCoordMin;
	vec2 texCoordMax;
	vec4 color;
};

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 finalColor;

void main() {
	vec2 in_region = step(texCoordMin, fragTexCoord) * step(fragTexCoord, texCoordMax);
	finalColor = texture(sampler2D(textures[texIdx], mySampler), fragTexCoord).a * in_region.x * in_region.y * color;
}