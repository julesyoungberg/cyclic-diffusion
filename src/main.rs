use glsl_layout::float;
use glsl_layout::*;
use nannou::prelude::*;
use nannou::wgpu::BufferDescriptor;
use rand;
use rand::Rng;
use std::sync::{Arc, Mutex};

struct Model {
    compute: Compute,
    positions: Arc<Mutex<Vec<Vector2>>>,
    threadpool: futures::executor::ThreadPool,
    app_texture: wgpu::Texture,
    uniform_texture: wgpu::Texture,
    draw: nannou::Draw,
    renderer: nannou::draw::Renderer,
    texture_reshaper: wgpu::TextureReshaper,
    bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
}

struct Compute {
    position_buffer: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

// The vertex type that we will use to represent a point on our triangle.
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
}

// The vertices that make up the rectangle to which the image will be drawn.
const VERTICES: [Vertex; 4] = [
    Vertex {
        position: [-1.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
    Vertex {
        position: [1.0, -1.0],
    },
];

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

    println!("generating particles");

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
        (PARTICLE_COUNT as usize * std::mem::size_of::<Vector2>()) as wgpu::BufferAddress;

    let position_buffer = device.create_buffer_with_data(
        &position_bytes[..],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
    );

    let velocity_buffer = device.create_buffer_with_data(
        &velocity_bytes[..],
        wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
    );

    // Create the buffer that will store the uniforms.
    let uniforms = create_uniforms(app.time);
    println!("uniforms: {:?}", uniforms);
    let std140_uniforms = uniforms.std140();
    let uniforms_bytes = std140_uniforms.as_raw();
    let uniform_buffer = device.create_buffer_with_data(
        uniforms_bytes,
        wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
    );

    // Create the compute shader module.
    println!("loading shaders");
    let cs_mod = wgpu::shader_from_spirv_bytes(device, include_bytes!("shaders/comp.spv"));
    let vs_mod = wgpu::shader_from_spirv_bytes(device, include_bytes!("shaders/vert.spv"));
    let fs_mod = wgpu::shader_from_spirv_bytes(device, include_bytes!("shaders/frag.spv"));

    // create our custom texture for rendering
    println!("creating app texure and reshaper");
    let sample_count = window.msaa_samples();
    let size = pt2(WIDTH as f32, HEIGHT as f32);
    let app_texture = create_app_texture(&device, size, sample_count);
    let texture_reshaper = create_texture_reshaper(&device, &app_texture, sample_count);

    println!("creating uniform texture");
    let uniform_texture = create_uniform_texture(&device, size);
    let uniform_texture_view = uniform_texture.view().build();

    // Create the sampler for sampling from the source texture.
    println!("creating sampler");
    let sampler = wgpu::SamplerBuilder::new().build(device);

    println!("creating bind group layout");
    let bind_group_layout = create_bind_group_layout(device, uniform_texture_view.component_type());
    println!("bind group layout: {:?}", bind_group_layout);
    println!("creating bind group");
    let bind_group = create_bind_group(
        device,
        &bind_group_layout,
        &position_buffer,
        buffer_size,
        &uniform_texture_view,
        &sampler,
        &uniform_buffer,
    );
    println!("creating pipeline layout");
    let pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
    println!("pipeline layout: {:?}", pipeline_layout);
    println!("creating render pipeline");
    let render_pipeline =
        create_render_pipeline(device, &pipeline_layout, &vs_mod, &fs_mod, sample_count);

    println!("creating vertex buffer");
    let vertices_bytes = vertices_as_bytes(&VERTICES[..]);
    let vertex_buffer = device.create_buffer_with_data(vertices_bytes, wgpu::BufferUsage::VERTEX);

    // Create the bind group and pipeline.
    println!("creating compute bind group layout");
    let compute_bind_group_layout = create_compute_bind_group_layout(device);
    println!("creating compute bind group");
    let compute_bind_group = create_compute_bind_group(
        device,
        &compute_bind_group_layout,
        &position_buffer,
        &velocity_buffer,
        buffer_size,
        &uniform_buffer,
    );
    println!("creating compute pipeline layout");
    let compute_pipeline_layout =
        create_compute_pipeline_layout(device, &compute_bind_group_layout);
    println!("create compute pipeline");
    let compute_pipeline = create_compute_pipeline(device, &compute_pipeline_layout, &cs_mod);

    // Create our `Draw` instance and a renderer for it.
    println!("creating renderer");
    let draw = nannou::Draw::new();
    let descriptor = app_texture.descriptor();
    let mut renderer =
        nannou::draw::RendererBuilder::new().build_from_texture_descriptor(device, descriptor);

    // draw initial aggregate
    println!("drawing initial design");
    draw.reset();
    draw.background().color(BLACK);
    draw.line()
        .start(pt2(0.0, HEIGHT as f32 * 0.3))
        .end(pt2(0.0, HEIGHT as f32 * -0.3))
        .weight(4.0)
        .color(WHITE);

    // Render our drawing to the texture.
    println!("rendering");
    let ce_desc = wgpu::CommandEncoderDescriptor {
        label: Some("texture-renderer"),
    };
    let mut encoder = device.create_command_encoder(&ce_desc);
    renderer.render_to_texture(device, &mut encoder, &draw, &app_texture);

    // copy app texture to uniform texture
    println!("copying app texture to buffer");
    copy_texture(&mut encoder, &app_texture, &uniform_texture);

    // submit encoded command buffer
    println!("submitting encoded command buffer");
    window.swap_chain_queue().submit(&[encoder.finish()]);

    let compute = Compute {
        position_buffer,
        velocity_buffer,
        buffer_size,
        uniform_buffer,
        bind_group: compute_bind_group,
        pipeline: compute_pipeline,
    };

    // Create a thread pool capable of running our GPU buffer read futures.
    let threadpool = futures::executor::ThreadPool::new().unwrap();

    Model {
        compute,
        positions: Arc::new(Mutex::new(positions)),
        threadpool,
        app_texture,
        uniform_texture,
        texture_reshaper,
        draw,
        renderer,
        bind_group,
        render_pipeline,
        vertex_buffer,
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
    });

    // An update for the uniform buffer with the current time.
    let uniforms = create_uniforms(app.time);
    let std140_uniforms = uniforms.std140();
    let uniforms_bytes = std140_uniforms.as_raw();
    let uniforms_size = uniforms_bytes.len();
    let new_uniform_buffer =
        device.create_buffer_with_data(uniforms_bytes, wgpu::BufferUsage::COPY_SRC);

    // The encoder we'll use to encode the compute pass.
    let desc = wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
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
        let mut cpass = encoder.begin_compute_pass();
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

    let texture_view = model.app_texture.view().build();

    {
        let mut render_pass = wgpu::RenderPassBuilder::new()
            .color_attachment(&texture_view, |color| color)
            .begin(&mut encoder);
        render_pass.set_bind_group(0, &model.bind_group, &[]);
        render_pass.set_pipeline(&model.render_pipeline);
        render_pass.set_vertex_buffer(0, &model.vertex_buffer, 0, 0);
        let vertex_range = 0..VERTICES.len() as u32;
        let instance_range = 0..1;
        render_pass.draw(vertex_range, instance_range);
    }

    // copy_texture(
    //     &device,
    //     &mut encoder,
    //     &model.app_texture,
    //     &model.uniform_texture,
    // );

    // copy app texture to uniform texture
    println!("copying app texture to buffer");
    copy_texture(&mut encoder, &model.app_texture, &model.uniform_texture);

    // submit encoded command buffer
    println!("submitting encoded command buffer");
    window.swap_chain_queue().submit(&[encoder.finish()]);

    // Spawn a future that reads the result of the compute pass.
    let positions = model.positions.clone();
    let buffer_size = model.compute.buffer_size;
    let read_positions_future = async move {
        let result = read_position_buffer.map_read(0, buffer_size).await;
        if let Ok(mapping) = result {
            if let Ok(mut positions) = positions.lock() {
                let bytes = mapping.as_slice();
                // "Cast" the slice of bytes to a slice of Vector2 as required.
                let slice = {
                    let len = bytes.len() / std::mem::size_of::<Vector2>();
                    let ptr = bytes.as_ptr() as *const Vector2;
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

pub fn vectors_as_byte_vec(data: &[Vector2]) -> Vec<u8> {
    let mut bytes = vec![];
    data.iter().for_each(|v| {
        bytes.extend(float_as_bytes(&v.x));
        bytes.extend(float_as_bytes(&v.y));
    });
    bytes
}

fn create_app_texture(device: &wgpu::Device, size: Point2, msaa_samples: u32) -> wgpu::Texture {
    wgpu::TextureBuilder::new()
        .size([size[0] as u32, size[1] as u32])
        .usage(
            wgpu::TextureUsage::OUTPUT_ATTACHMENT
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST,
        )
        .sample_count(msaa_samples)
        .format(Frame::TEXTURE_FORMAT)
        .build(device)
}

fn create_uniform_texture(device: &wgpu::Device, size: Point2) -> wgpu::Texture {
    wgpu::TextureBuilder::new()
        .size([size[0] as u32, size[1] as u32])
        .usage(wgpu::TextureBuilder::REQUIRED_IMAGE_TEXTURE_USAGE | wgpu::TextureUsage::SAMPLED)
        // .sample_count(1)
        .format(Frame::TEXTURE_FORMAT)
        .build(device)
}

fn create_texture_reshaper(
    device: &wgpu::Device,
    texture: &wgpu::Texture,
    msaa_samples: u32,
) -> wgpu::TextureReshaper {
    let texture_view = texture.view().build();
    let texture_component_type = texture.component_type();
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

fn create_bind_group_layout(
    device: &wgpu::Device,
    texture_component_type: wgpu::TextureComponentType,
) -> wgpu::BindGroupLayout {
    let storage_dynamic = false;
    let storage_readonly = false;
    let uniform_dynamic = false;
    wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(
            wgpu::ShaderStage::FRAGMENT,
            storage_dynamic,
            storage_readonly,
        )
        .sampled_texture(
            wgpu::ShaderStage::FRAGMENT,
            false,
            wgpu::TextureViewDimension::D2,
            texture_component_type,
        )
        .sampler(wgpu::ShaderStage::FRAGMENT)
        .uniform_buffer(wgpu::ShaderStage::FRAGMENT, uniform_dynamic)
        .build(device)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    position_buffer: &wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
    texture: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    wgpu::BindGroupBuilder::new()
        .buffer_bytes(position_buffer, 0..buffer_size)
        .texture_view(texture)
        .sampler(sampler)
        .buffer::<Uniforms>(uniform_buffer, 0..1)
        .build(device, layout)
}

fn create_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    let desc = wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    };
    device.create_pipeline_layout(&desc)
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vs_mod: &wgpu::ShaderModule,
    fs_mod: &wgpu::ShaderModule,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    wgpu::RenderPipelineBuilder::from_layout(layout, vs_mod)
        .fragment_shader(fs_mod)
        .color_format(Frame::TEXTURE_FORMAT)
        .add_vertex_buffer::<Vertex>(&wgpu::vertex_attr_array![0 => Float2])
        .sample_count(sample_count)
        .primitive_topology(wgpu::PrimitiveTopology::TriangleStrip)
        .build(device)
}

// See the `nannou::wgpu::bytes` documentation for why this is necessary.
fn vertices_as_bytes(data: &[Vertex]) -> &[u8] {
    unsafe { wgpu::bytes::from_slice(data) }
}

fn create_compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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

fn create_compute_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    position_buffer: &wgpu::Buffer,
    velocity_buffer: &wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    wgpu::BindGroupBuilder::new()
        .buffer_bytes(position_buffer, 0..buffer_size)
        .buffer_bytes(velocity_buffer, 0..buffer_size)
        .buffer::<Uniforms>(uniform_buffer, 0..1)
        .build(device, layout)
}

fn create_compute_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    cs_mod: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let compute_stage = wgpu::ProgrammableStageDescriptor {
        module: &cs_mod,
        entry_point: "main",
    };
    let desc = wgpu::ComputePipelineDescriptor {
        layout,
        compute_stage,
    };
    device.create_compute_pipeline(&desc)
}

pub fn copy_texture(encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture) {
    let src_copy_view = src.default_copy_view();
    let dst_copy_view = dst.default_copy_view();
    let copy_size = dst.extent();
    encoder.copy_texture_to_texture(src_copy_view, dst_copy_view, copy_size);
}
