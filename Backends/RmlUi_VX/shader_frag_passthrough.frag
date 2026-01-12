#version 460
#extension GL_GOOGLE_include_directive : require

#include "BindlessTextures.h"

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 finalColor;

void main() {
	finalColor = texture(sampler2D(textures[texIdx], mySampler), fragTexCoord);
}