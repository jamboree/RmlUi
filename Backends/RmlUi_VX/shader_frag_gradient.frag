#version 460

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_scalar_block_layout : require

#define MAX_NUM_STOPS 16
#define LINEAR 0
#define RADIAL 1
#define CONIC 2
#define REPEATING 1
#define PI 3.14159265

layout(std430, binding = 0) uniform GradientUniform {
    // one of the above definitions
    int func;
    // linear: starting point, radial: center, conic: center
    int numStops;
    vec2 pos;
    // linear: vector to ending point, radial: 2d curvature (inverse radius), conic: angled unit vector
    vec2 vec;
    // normalized, 0 -> starting point, 1 -> ending point
    float stopPositions[MAX_NUM_STOPS];
    vec4 stopColors[MAX_NUM_STOPS];
};

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 finalColor;

vec4 mixStopColors(float t) {
    vec4 color = stopColors[0];
    float position = stopPositions[0];

    for (int i = 1; i < numStops; ++i) {
        color = mix(color, stopColors[i], smoothstep(position, stopPositions[i], t));
        position = stopPositions[i];
    }

    return color;
}

void main() {
    float t = 0.0;

    switch (func >> 1) {
    case LINEAR: {
        float dist_square = dot(vec, vec);
        vec2 V = fragTexCoord - pos;
        t = dot(vec, V) / dist_square;
        break;
    }
    case RADIAL: {
        vec2 V = fragTexCoord - pos;
        t = length(vec * V);
        break;
    }
    case CONIC: {
        mat2 R = mat2(vec.x, -vec.y, vec.y, vec.x);
        vec2 V = R * (fragTexCoord - pos);
        t = 0.5 + atan(-V.x, V.y) / (2.0 * PI);
        break;
    }
    }

    if ((func & REPEATING) != 0) {
        float t0 = stopPositions[0];
        float t1 = stopPositions[numStops - 1];
        t = t0 + mod(t - t0, t1 - t0);
    }

    finalColor = fragColor * mixStopColors(t);
}