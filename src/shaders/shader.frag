// NOTE: This shader requires being manually compiled to SPIR-V in order to
// avoid having downstream users require building shaderc and compiling the
// shader themselves. If you update this shader, be sure to also re-compile it
// and update `frag.spv`. You can do so using `glslangValidator` with the
// following command: `glslangValidator -V shader.frag`

#version 450

precision highp int;
precision mediump float;

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) buffer PositionBuffer { vec2[] positions; };
layout(set = 0, binding = 1) uniform texture2D tex;
layout(set = 0, binding = 2) uniform sampler tex_sampler;
layout(set = 0, binding = 3) uniform Uniforms {
    uint particle_count;
    float width;
    float height;
    float time;
};

void main() {
    vec3 color = texture(sampler2D(tex, tex_sampler), tex_coords).rgb;

    color *= 0.9;

    // todo: aggregate the particles

    f_color = vec4(color, 1.0);
}
