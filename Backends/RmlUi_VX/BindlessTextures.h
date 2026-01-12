layout(set = 0, binding = 0) uniform sampler mySampler;
layout(set = 0, binding = 1) uniform texture2D textures[4];

layout(push_constant) uniform TexInput {
    layout(offset = 72) uint texIdx;
};
