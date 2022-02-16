#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform Uniforms {
    uint particle_count;
    float width;
    float height;
    float time;
};

// https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float sdEquilateralTriangle(in vec2 p) {
    const float k = sqrt(3.0);
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0 / k;

    if (p.x + k * p.y > 0.0) {
        p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
    }

    p.x -= clamp(p.x, -2.0, 0.0);
    return -length(p) * sign(p.y);
}

vec3 circle(in vec2 st) {
    float d = fract(length(st) * 1.0);
    float w = 0.003;
    float pos = 0.19;
    float l = smoothstep(pos + w, pos, d); // - smoothstep(pos, pos - w, d);
    vec3 color = vec3(l);
    return color;
}

vec3 square(in vec2 st) {
    float w = 0.01;
    float pos = 0.1;
    float scale = 5.0;
    float d = smoothstep(pos + w, pos, fract(abs(st.x) * scale)) 
        * smoothstep(pos + w, pos, fract(abs(st.y) * scale));
    vec3 color = vec3(d);
    return color;
}

vec3 triangle(in vec2 st) {
    float w = 0.001;
    float pos = 0.01;
    float d = sdEquilateralTriangle(st * 5.0);
    float l = smoothstep(pos + w, pos, d);
    vec3 color = vec3(l);
    return color;
}

vec3 vertical_line(in vec2 st) {
    float w = 0.01;
    float f = 0.001;
    float d = smoothstep(w + f, w, abs(st.x)) * smoothstep(0.41, 0.4, abs(st.y));
    vec3 color = vec3(d);
    return color;
}

void main() {
    vec2 st = tex_coords;
    st.y = 1.0 - st.y;
    st -= 0.5;
    st.x *= width / height;

    vec3 color = vec3(0.0);

    // color = circle(st); // + circle(st - vec2(0.2, 0.0)) + circle(st + vec2(0.2, 0.0));
    // color = square(st);
    // color = triangle(st);
    color = vertical_line(st);

    f_color = vec4(color, 1.0);
}
