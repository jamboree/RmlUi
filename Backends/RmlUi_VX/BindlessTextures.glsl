layout(set = 0, binding = 0) readonly buffer MatrixBuffer {
	mat4 matrices[];
};
layout(set = 0, binding = 1) uniform sampler mySampler;
layout(set = 0, binding = 2) uniform texture2D textures[4];

#define TEX_IDX layout(offset = 12) uint