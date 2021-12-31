use glsl_layout::float;
use glsl_layout::*;
use nannou::prelude::*;
use nannou::wgpu::BufferInitDescriptor;
use rand;
use rand::Rng;
use std::sync::{Arc, Mutex};

struct Model {
    compute: Compute,
    positions: Arc<Mutex<Vec<Vec2>>>,
    threadpool: futures::executor::ThreadPool,
    // The texture that we will draw to.
    texture: wgpu::Texture,
    // Create a `Draw` instance for drawing to our texture.
    draw: nannou::Draw,
    // The type used to render the `Draw` vertices to our texture.
    renderer: nannou::draw::Renderer,
    // The type used to resize our texture to the window texture.
    texture_reshaper: wgpu::TextureReshaper,
}

struct Compute {
    position_buffer: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Uniform)]
pub struct Uniforms {
    particle_count: uint,
    width: float,
    height: float,
    time: float,
}

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const PARTICLE_COUNT: u32 = 5000;

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let window_id = app
        .new_window()
        .size(WIDTH, HEIGHT)
        .view(view)
        .build()
        .unwrap();
    let window = app.window(window_id).unwrap();
    let device = window.swap_chain_device();

    let mut positions = vec![];
    let mut velocities = vec![];

    let hwidth = WIDTH as f32 * 0.5;
    let hheight = HEIGHT as f32 * 0.5;

    for _ in 0..PARTICLE_COUNT {
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
    let buffer_size =
        (PARTICLE_COUNT as usize * std::mem::size_of::<Vec2>()) as wgpu::BufferAddress;

    let position_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
        label: Some("particle-positions"),
        contents: &position_bytes[..],
        usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
    });

    let velocity_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
        label: Some("particle-velocities"),
        contents: &velocity_bytes[..],
        usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
    });

    // Create the buffer that will store the uniforms.
    let uniforms = create_uniforms(app.time);
    println!("uniforms: {:?}", uniforms);
    let std140_uniforms = uniforms.std140();
    let uniforms_bytes = std140_uniforms.as_raw();
    let usage = wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST;
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-buffer"),
        contents: uniforms_bytes,
        usage,
    });

    // Create the compute shader module.
    let cs_mod = wgpu::shader_from_spirv_bytes(device, include_bytes!("shaders/comp.spv"));

    // Create the bind group and pipeline.
    let bind_group_layout = create_bind_group_layout(device);
    let bind_group = create_bind_group(
        device,
        &bind_group_layout,
        &position_buffer,
        &velocity_buffer,
        buffer_size,
        &uniform_buffer,
    );
    let pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
    let pipeline = create_compute_pipeline(device, &pipeline_layout, &cs_mod);

    let compute = Compute {
        position_buffer,
        velocity_buffer,
        buffer_size,
        uniform_buffer,
        bind_group,
        pipeline,
    };

    // Create a thread pool capable of running our GPU buffer read futures.
    let threadpool = futures::executor::ThreadPool::new().unwrap();

    // create our custom texture for rendering
    let sample_count = window.msaa_samples();
    let size = pt2(WIDTH as f32, HEIGHT as f32);
    let texture = create_app_texture(&device, size, sample_count);
    let texture_reshaper = create_texture_reshaper(&device, &texture, sample_count);

    // Create our `Draw` instance and a renderer for it.
    let draw = nannou::Draw::new();
    let descriptor = texture.descriptor();
    let mut renderer =
        nannou::draw::RendererBuilder::new().build_from_texture_descriptor(device, descriptor);

    // draw initial aggregate
    draw.reset();
    draw.background().color(BLACK);
    draw.line()
        .start(pt2(0.0, HEIGHT as f32 * 0.3))
        .end(pt2(0.0, HEIGHT as f32 * -0.3))
        .weight(4.0)
        .color(WHITE);

    // Render our drawing to the texture.
    let ce_desc = wgpu::CommandEncoderDescriptor {
        label: Some("texture renderer"),
    };
    let mut encoder = device.create_command_encoder(&ce_desc);
    renderer.render_to_texture(device, &mut encoder, &draw, &texture);
    window.swap_chain_queue().submit([encoder.finish()]);

    Model {
        compute,
        positions: Arc::new(Mutex::new(positions)),
        threadpool,
        texture,
        texture_reshaper,
        draw,
        renderer,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let window = app.main_window();
    let device = window.swap_chain_device();
    let compute = &mut model.compute;

    // create a buffer for reading the particle positions
    let read_position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-positions"),
        size: compute.buffer_size,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    // An update for the uniform buffer with the current time.
    let uniforms = create_uniforms(app.time);
    let std140_uniforms = uniforms.std140();
    let uniforms_bytes = std140_uniforms.as_raw();
    let uniforms_size = uniforms_bytes.len();
    let usage = wgpu::BufferUsage::COPY_SRC;
    let new_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-data-transfer"),
        contents: uniforms_bytes,
        usage,
    });

    // The encoder we'll use to encode the compute pass.
    let desc = wgpu::CommandEncoderDescriptor {
        label: Some("particle-compute"),
    };
    let mut encoder = device.create_command_encoder(&desc);

    encoder.copy_buffer_to_buffer(
        &new_uniform_buffer,
        0,
        &compute.uniform_buffer,
        0,
        uniforms_size as u64,
    );

    {
        let pass_desc = wgpu::ComputePassDescriptor {
            label: Some("compute-pass"),
        };
        let mut cpass = encoder.begin_compute_pass(&pass_desc);
        cpass.set_pipeline(&compute.pipeline);
        cpass.set_bind_group(0, &compute.bind_group, &[]);
        cpass.dispatch(PARTICLE_COUNT, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &compute.position_buffer,
        0,
        &read_position_buffer,
        0,
        compute.buffer_size,
    );

    // Submit the compute pass to the device's queue.
    window.swap_chain_queue().submit(Some(encoder.finish()));

    // Spawn a future that reads the result of the compute pass.
    let positions = model.positions.clone();
    let read_positions_future = async move {
        let slice = read_position_buffer.slice(..);
        if let Ok(_) = slice.map_async(wgpu::MapMode::Read).await {
            if let Ok(mut positions) = positions.lock() {
                let bytes = &slice.get_mapped_range()[..];
                // "Cast" the slice of bytes to a slice of Vec2 as required.
                let slice = {
                    let len = bytes.len() / std::mem::size_of::<Vec2>();
                    let ptr = bytes.as_ptr() as *const Vec2;
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                };

                positions.copy_from_slice(slice);
            }
        }
    };

    model.threadpool.spawn_ok(read_positions_future);
}

fn view(app: &App, model: &Model, frame: Frame) {
    // frame.clear(BLACK);
    // let draw = app.draw();

    // if let Ok(positions) = model.positions.lock() {
    //     for &p in positions.iter() {
    //         draw.ellipse().radius(2.0).color(WHITE).x_y(p.x, p.y);
    //     }
    // }

    // draw.to_frame(app, &frame).unwrap();

    // Sample the texture and write it to the frame.
    let mut encoder = frame.command_encoder();
    model
        .texture_reshaper
        .encode_render_pass(frame.texture_view(), &mut *encoder);
}

fn create_uniforms(time: f32) -> Uniforms {
    Uniforms {
        particle_count: PARTICLE_COUNT,
        width: WIDTH as f32,
        height: HEIGHT as f32,
        time,
    }
}

pub fn float_as_bytes(data: &f32) -> &[u8] {
    unsafe { wgpu::bytes::from(data) }
}

pub fn vectors_as_byte_vec(data: &[Vec2]) -> Vec<u8> {
    let mut bytes = vec![];
    data.iter().for_each(|v| {
        bytes.extend(float_as_bytes(&v.x));
        bytes.extend(float_as_bytes(&v.y));
    });
    bytes
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage_dynamic = false;
    let storage_readonly = false;
    let uniform_dynamic = false;
    wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(
            wgpu::ShaderStage::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
        .storage_buffer(
            wgpu::ShaderStage::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
        .uniform_buffer(wgpu::ShaderStage::COMPUTE, uniform_dynamic)
        .build(device)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    position_buffer: &wgpu::Buffer,
    velocity_buffer: &wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let buffer_size_bytes = std::num::NonZeroU64::new(buffer_size).unwrap();
    wgpu::BindGroupBuilder::new()
        .buffer_bytes(position_buffer, 0, Some(buffer_size_bytes))
        .buffer_bytes(velocity_buffer, 0, Some(buffer_size_bytes))
        .buffer::<Uniforms>(uniform_buffer, 0..1)
        .build(device, layout)
}

fn create_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    cs_mod: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let desc = wgpu::ComputePipelineDescriptor {
        label: Some("compute-pipeline"),
        layout: Some(layout),
        module: &cs_mod,
        entry_point: "main",
    };
    device.create_compute_pipeline(&desc)
}

fn create_app_texture(device: &wgpu::Device, size: Point2, msaa_samples: u32) -> wgpu::Texture {
    wgpu::TextureBuilder::new()
        .size([size[0] as u32, size[1] as u32])
        .usage(wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::SAMPLED)
        .sample_count(msaa_samples)
        .format(Frame::TEXTURE_FORMAT)
        .build(device)
}

fn create_texture_reshaper(
    device: &wgpu::Device,
    texture: &wgpu::Texture,
    msaa_samples: u32,
) -> wgpu::TextureReshaper {
    let texture_view = texture.view().build();
    let texture_component_type = texture.sample_type();
    let dst_format = Frame::TEXTURE_FORMAT;
    wgpu::TextureReshaper::new(
        device,
        &texture_view,
        msaa_samples,
        texture_component_type,
        msaa_samples,
        dst_format,
    )
}
