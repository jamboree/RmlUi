#version 460

#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform VsInput {
	mat4 transform;
	vec2 translate;
};

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec4 fragColor;

void main() {
	fragTexCoord = inTexCoord;
	fragColor = inColor;
	vec2 translatedPos = inPosition + translate.xy;
	vec4 outPos = transform * vec4(translatedPos, 0, 1);
    gl_Position = outPos;
}