#version 460
#extension GL_GOOGLE_include_directive : require

#include "BindlessTextures.glsl"

layout(push_constant) uniform FsInput {
    TEX_IDX texIdx;
};

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 finalColor;

void main() {
	vec4 texColor = texture(sampler2D(textures[texIdx], mySampler), fragTexCoord);
	float maskAlpha = texture(sampler2D(textures[3], mySampler), fragTexCoord).a;
	finalColor = texColor * maskAlpha;
}