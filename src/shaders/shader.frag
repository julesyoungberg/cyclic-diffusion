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
    float limitation_threshold;
    float decay;
    float range;
};

void main() {
    vec3 color = texture(sampler2D(tex, tex_sampler), tex_coords).rgb;
    float current = (color.r + color.g + color.b) / 3.0;
    color *= decay;
    
    // get the corresponding world position
    vec2 position = tex_coords;
    position.y = 1.0 - position.y;
    position -= 0.5;
    position *= vec2(width, height);

    float hwidth = width * 0.5;
    float hheight = height * 0.5;

    // check if a particle is on the pixel 
    bool has_particle = false;
    for (uint i = 0; i < particle_count; i++) {
        vec2 particle_position = positions[i];
        vec2 diff = abs(position - particle_position);

        if (length(diff) < range) {
            has_particle = true;
            break;
        }
    }

    if (has_particle && current <= limitation_threshold) {
        // make position nonnegative
        position += vec2(hwidth, hheight);

        // sum neighboring pixels
        vec3 sum = vec3(0.0);
        for (int x = -1; x < 2; x++) {
            for (int y = -1; y < 2; y++) {
                vec2 coord = position + vec2(x, y);
                coord /= vec2(width, height);
                sum += texture(sampler2D(tex, tex_sampler), coord).rgb;
            }
        }

        // if average is above threshold then add to aggregate
        float avg = (sum.r + sum.g + sum.b) / 3.0;
        if (avg > threshold) {
            color = vec3(1.0);
        }
    } 
    // else {
    //     vec2 particle_position = positions[0] / vec2(width * 0.5, height * 0.5);
    //     particle_position += 0.5;

    //     float d = distance(position, particle_position);
    //     f_color = vec4(d);
    // }

    f_color = vec4(color, 1.0);
}
