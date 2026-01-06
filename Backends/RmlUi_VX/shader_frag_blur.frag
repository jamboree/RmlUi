#version 460

uniform sampler2D tex;
uniform float weights[BLUR_NUM_WEIGHTS];
uniform vec2 texCoordMin;
uniform vec2 texCoordMax;

layout(location = 0) in vec2 fragTexCoord[BLUR_SIZE];
layout(location = 0) out vec4 finalColor;

void main() {
	vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
	for (int i = 0; i < BLUR_SIZE; ++i) {
		vec2 in_region = step(texCoordMin, fragTexCoord[i]) * step(fragTexCoord[i], texCoordMax);
		color += texture(tex, fragTexCoord[i]) * in_region.x * in_region.y * weights[abs(i - BLUR_NUM_WEIGHTS + 1)];
	}
	finalColor = color;
}