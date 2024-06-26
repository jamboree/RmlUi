#version 460

#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 finalColor;

void main() {
	vec4 texColor = texture(tex, fragTexCoord);
	finalColor = fragColor * texColor;
}