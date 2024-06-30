#version 460

#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform VsInput {
	mat4 transform;
	vec2 pt1;
	vec2 pt2;
};

void main() {
    float x = (gl_VertexIndex & 1) == 0 ? pt1.x : pt2.x;
    float y = (gl_VertexIndex & 2) == 0 ? pt1.y : pt2.y;
    gl_Position = transform * vec4(x, y, 0, 1);
}