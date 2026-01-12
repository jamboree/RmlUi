#version 460

layout(set = 0, binding = 0) uniform sampler mySampler;
layout(set = 0, binding = 1) uniform texture2D tex;
layout(set = 0, binding = 2) uniform texture2D texMask;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 finalColor;

void main() {
	vec4 texColor = texture(sampler2D(tex, mySampler), fragTexCoord);
	float maskAlpha = texture(sampler2D(texMask, mySampler), fragTexCoord).a;
	finalColor = texColor * maskAlpha;
}