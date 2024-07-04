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
    gl_Position = transform * vec4(inPosition + translate, 0, 1);
}