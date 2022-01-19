use nannou::prelude::*;
use nannou::wgpu::CommandEncoder;
use rand;
use rand::Rng;

use crate::compute::*;
use crate::uniforms::*;
use crate::util::*;

pub struct ParticleSystem {
    pub position_buffer: wgpu::Buffer,
    pub velocity_buffer: wgpu::Buffer,
    pub buffer_size: u64,
    pub initial_positions: Vec<Point2>,
    pub compute: Compute,
    pub particle_count: u32,
}

impl ParticleSystem {
    pub fn new(app: &App, device: &wgpu::Device, uniforms: &UniformBuffer) -> Self {
        let hwidth = uniforms.data.width * 0.5;
        let hheight = uniforms.data.height * 0.5;

        let mut positions = vec![];
        let mut velocities = vec![];

        for _ in 0..uniforms.data.particle_count {
            let position_x = rand::thread_rng().gen_range(-hwidth, hwidth);
            let position_y = rand::thread_rng().gen_range(-hheight, hheight);
            let position = pt2(position_x, position_y);
            positions.push(position);

            let velocity_x = rand::thread_rng().gen_range(-1.0, 1.0);
            let velocity_y = rand::thread_rng().gen_range(-1.0, 1.0);
            let velocity = pt2(velocity_x, velocity_y);
            velocities.push(velocity);
        }

        let position_bytes = vectors_as_byte_vec(&positions);
        let velocity_bytes = vectors_as_byte_vec(&velocities);

        // Create the buffers that will store the result of our compute operation.
        let buffer_size = (uniforms.data.particle_count as usize * std::mem::size_of::<Point2>())
            as wgpu::BufferAddress;

        let position_buffer = device.create_buffer_with_data(
            &position_bytes[..],
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        );

        let velocity_buffer = device.create_buffer_with_data(
            &velocity_bytes[..],
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        );

        // Create the compute shader module.
        let update_cs_mod =
            compile_shader(app, device, "update.comp", shaderc::ShaderKind::Compute);

        let buffers = vec![&position_buffer, &velocity_buffer];
        let buffer_sizes = vec![buffer_size, buffer_size];

        let compute = Compute::new::<Uniforms>(
            device,
            Some(buffers),
            Some(buffer_sizes),
            Some(&uniforms.buffer),
            &update_cs_mod,
        )
        .unwrap();

        Self {
            position_buffer,
            velocity_buffer,
            buffer_size,
            initial_positions: positions,
            compute,
            particle_count: uniforms.data.particle_count,
        }
    }

    pub fn update(&self, encoder: &mut CommandEncoder) {
        self.compute.compute(encoder, self.particle_count);
    }
}

pub fn float_as_bytes(data: &f32) -> &[u8] {
    unsafe { wgpu::bytes::from(data) }
}

pub fn vectors_as_byte_vec(data: &[Point2]) -> Vec<u8> {
    let mut bytes = vec![];
    data.iter().for_each(|v| {
        bytes.extend(float_as_bytes(&v.x));
        bytes.extend(float_as_bytes(&v.y));
    });
    bytes
}
