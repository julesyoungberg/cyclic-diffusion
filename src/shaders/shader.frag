// NOTE: This shader requires being manually compiled to SPIR-V in order to
// avoid having downstream users require building shaderc and compiling the
// shader themselves. If you update this shader, be sure to also re-compile it
// and update `frag.spv`. You can do so using `glslangValidator` with the
// following command: `glslangValidator -V shader.frag`

#version 450

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
    float threshold;
};

void main() {
    // vec3 color = texture(sampler2D(tex, tex_sampler), tex_coords).rgb;
    // color *= 0.99;
    
    vec2 position = tex_coords;
    position.y = 1.0 - position.y;
    position -= 0.5;
    // position *= vec2(width, height);

    // vec3 sum = vec3(0.0);

    // float hwidth = width * 0.5;
    // float hheight = height * 0.5;

    // // todo: use sorted pixels
    // for (uint i = 0; i < particle_count; i++) {
    //     vec2 particle_position = positions[i];
    //     vec2 diff = abs(position - particle_position);

    //     if (length(diff) < 1.0) {
    //         vec2 pixel_position = particle_position / vec2(hwidth, hheight);
    //         pixel_position += 0.5;
    //         sum += texture(sampler2D(tex, tex_sampler), pixel_position).rgb;
    //         color = vec3(1.0);
    //     }
    // }

    // float avg = (sum.r + sum.g + sum.b) / 3.0;
    // if (avg > threshold) {
    //     color = vec3(1.0);
    // }

    vec2 particle_position = positions[0] / vec2(width * 0.5, height * 0.5);
    // particle_position += 0.5;

    float d = distance(position, particle_position);
    f_color = vec4(d);
}
