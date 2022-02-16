use nannou::prelude::*;

mod capture;
mod compute;
mod particles;
mod render;
mod uniforms;
mod util;

use crate::capture::*;
use crate::particles::*;
use crate::render::*;
use crate::uniforms::*;
use crate::util::*;

struct Model {
    uniform_texture: wgpu::Texture,
    render: CustomRenderer,
    uniforms: UniformBuffer,
    particle_system: ParticleSystem,
    capturer: FrameCapturer,
}

const WIDTH: u32 = 1440;
const HEIGHT: u32 = 810;
const PARTICLE_COUNT: u32 = 3000;

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

    let uniforms = UniformBuffer::new(
        device,
        PARTICLE_COUNT,
        WIDTH as f32,
        HEIGHT as f32,
        app.time,
    );

    let particle_system = ParticleSystem::new(app, device, &uniforms);

    // Create the compute shader module.
    println!("loading shaders");
    let vs_mod = compile_shader(app, device, "shader.vert", shaderc::ShaderKind::Vertex);
    let init_fs_mod = compile_shader(app, device, "init.frag", shaderc::ShaderKind::Fragment);
    let fs_mod = compile_shader(app, device, "shader.frag", shaderc::ShaderKind::Fragment);

    // create our custom texture for rendering
    println!("creating app texure");
    let sample_count = window.msaa_samples();
    let size = pt2(WIDTH as f32, HEIGHT as f32);

    println!("creating uniform texture");
    let uniform_texture = create_uniform_texture(&device, size, sample_count);

    // Create the sampler for sampling from the source texture.
    println!("creating sampler");
    let sampler = wgpu::SamplerBuilder::new().build(device);

    let init = CustomRenderer::new::<Uniforms>(
        device,
        &vs_mod,
        &init_fs_mod,
        None,
        None,
        None,
        None,
        Some(&uniforms.buffer),
        WIDTH,
        HEIGHT,
        sample_count,
    )
    .unwrap();

    let render = CustomRenderer::new::<Uniforms>(
        device,
        &vs_mod,
        &fs_mod,
        Some(&vec![&particle_system.position_buffer]),
        Some(&vec![&particle_system.buffer_size]),
        Some(&vec![&uniform_texture]),
        Some(&sampler),
        Some(&uniforms.buffer),
        WIDTH,
        HEIGHT,
        sample_count,
    )
    .unwrap();

    let mut capturer = FrameCapturer::new(app);

    // Render our drawing to the texture.
    println!("rendering");
    let ce_desc = wgpu::CommandEncoderDescriptor {
        label: Some("texture-renderer"),
    };
    let mut encoder = device.create_command_encoder(&ce_desc);

    init.render(&mut encoder);

    // copy app texture to uniform texture
    println!("copying app texture to buffer");
    init.texture_reshaper
        .encode_render_pass(&uniform_texture.view().build(), &mut encoder);

    capturer.take_snapshot(device, &mut encoder, &render.output_texture);

    // submit encoded command buffer
    println!("submitting encoded command buffer");
    window.swap_chain_queue().submit(&[encoder.finish()]);

    capturer.save_frame(app);

    // Create a thread pool capable of running our GPU buffer read futures.

    Model {
        uniform_texture,
        render,
        uniforms,
        particle_system,
        capturer,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let window = app.main_window();
    let device = window.swap_chain_device();

    // The encoder we'll use to encode the compute pass and render pass.
    let desc = wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    };
    let mut encoder = device.create_command_encoder(&desc);

    model.uniforms.update(device, &mut encoder, app.time);

    model.particle_system.update(&mut encoder);

    model.render.render(&mut encoder);

    // copy app texture to uniform texture
    model
        .render
        .texture_reshaper
        .encode_render_pass(&model.uniform_texture.view().build(), &mut encoder);

    model
        .capturer
        .take_snapshot(device, &mut encoder, &model.render.output_texture);

    // submit encoded command buffer
    window.swap_chain_queue().submit(&[encoder.finish()]);

    model.capturer.save_frame(app);
}

fn view(_app: &App, model: &Model, frame: Frame) {
    // Sample the texture and write it to the frame.
    let mut encoder = frame.command_encoder();
    model
        .render
        .texture_reshaper
        .encode_render_pass(frame.texture_view(), &mut *encoder);
}

fn create_uniform_texture(device: &wgpu::Device, size: Point2, msaa_samples: u32) -> wgpu::Texture {
    wgpu::TextureBuilder::new()
        .size([size[0] as u32, size[1] as u32])
        .usage(
            wgpu::TextureBuilder::REQUIRED_IMAGE_TEXTURE_USAGE
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        )
        .sample_count(msaa_samples)
        .format(Frame::TEXTURE_FORMAT)
        .build(device)
}
