#version 460
#extension GL_GOOGLE_include_directive : require

#include "BindlessTextures.h"

layout(set = 1, binding = 0) uniform FsInput {
    mat4 colorMatrix;
};

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 finalColor;

void main() {
	// The general case uses a 4x5 color matrix for full rgba transformation, plus a constant term with the last column.
	// However, we only consider the case of rgb transformations. Thus, we could in principle use a 3x4 matrix, but we
	// keep the alpha row for simplicity.
	// In the general case we should do the matrix transformation in non-premultiplied space. However, without alpha
	// transformations, we can do it directly in premultiplied space to avoid the extra division and multiplication
	// steps. In this space, the constant term needs to be multiplied by the alpha value, instead of unity.
	vec4 texColor = texture(sampler2D(textures[texIdx], mySampler), fragTexCoord);
	vec3 transformedColor = vec3(colorMatrix * texColor);
	finalColor = vec4(transformedColor, texColor.a);
}