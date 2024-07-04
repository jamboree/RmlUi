#version 460

#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform VsInput {
	mat4 transform;
	vec2 translate;
};

layout(location = 0) in vec2 inPosition;

void main() {
    gl_Position = transform * vec4(inPosition + translate, 0, 1);
}