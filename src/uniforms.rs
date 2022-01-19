use glsl_layout::float;
use glsl_layout::*;
use nannou::prelude::*;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Uniform)]
pub struct Uniforms {
    pub particle_count: uint,
    pub width: float,
    pub height: float,
    pub time: float,
    pub threshold: float,
    pub limitation_threshold: float,
    pub decay: float,
    pub bin_size: float,
}

impl Uniforms {
    pub fn new(particle_count: uint, width: float, height: float, time: f32) -> Self {
        Uniforms {
            particle_count,
            width,
            height,
            time,
            threshold: 0.8,
            limitation_threshold: 0.001,
            decay: 0.97,
            bin_size: 2.0,
        }
    }
}

pub struct UniformBuffer {
    pub data: Uniforms,
    pub buffer: wgpu::Buffer,
}

impl UniformBuffer {
    pub fn new(
        device: &wgpu::Device,
        particle_count: uint,
        width: float,
        height: float,
        time: f32,
    ) -> Self {
        let data = Uniforms::new(particle_count, width, height, time);

        let std140_uniforms = data.std140();
        let uniforms_bytes = std140_uniforms.as_raw();
        let usage = wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST;
        let buffer = device.create_buffer_with_data(uniforms_bytes, usage);

        Self { data, buffer }
    }

    pub fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, time: f32) {
        self.data.time = time;

        // An update for the uniform buffer with the current time.
        let std140_uniforms = self.data.std140();
        let uniforms_bytes = std140_uniforms.as_raw();
        let uniforms_size = uniforms_bytes.len();
        let usage = wgpu::BufferUsage::COPY_SRC;
        let new_uniform_buffer = device.create_buffer_with_data(uniforms_bytes, usage);

        encoder.copy_buffer_to_buffer(
            &new_uniform_buffer,
            0,
            &self.buffer,
            0,
            uniforms_size as u64,
        );
    }
}
