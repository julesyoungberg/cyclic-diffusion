use nannou::prelude::*;

#[derive(Debug)]
pub enum ComputeError {
    MissingBufferSizes,
    BufferCountAndBufferSizeCountMismatch,
}

pub struct Compute {
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::ComputePipeline,
}

/// Builds and manages a Compute pipeline with any number of buffers,
/// and one optional uniform buffer
impl Compute {
    pub fn new<T>(
        device: &wgpu::Device,
        buffers: Option<Vec<&wgpu::Buffer>>,
        buffer_sizes: Option<Vec<wgpu::BufferAddress>>,
        uniform_buffer: Option<&wgpu::Buffer>,
        cs_mod: &wgpu::ShaderModule,
    ) -> Result<Self, ComputeError>
    where
        T: std::marker::Copy,
    {
        let mut bind_group_layout_builder = wgpu::BindGroupLayoutBuilder::new();
        let mut bind_group_builder = wgpu::BindGroupBuilder::new();

        // add buffers to bind group
        if let Some(b) = buffers.as_ref() {
            if let Some(s) = buffer_sizes.as_ref() {
                if b.len() != s.len() {
                    return Err(ComputeError::BufferCountAndBufferSizeCountMismatch);
                }

                let storage_dynamic = false;
                let storage_readonly = false;

                for (i, buffer) in b.iter().enumerate() {
                    let buffer_size = s[i];

                    bind_group_layout_builder = bind_group_layout_builder.storage_buffer(
                        wgpu::ShaderStage::COMPUTE,
                        storage_dynamic,
                        storage_readonly,
                    );

                    bind_group_builder = bind_group_builder.buffer_bytes(buffer, 0..buffer_size);
                }
            } else {
                return Err(ComputeError::MissingBufferSizes);
            }
        }

        // add uniform buffer to bind group
        if let Some(u) = uniform_buffer {
            let uniform_dynamic = false;
            bind_group_layout_builder = bind_group_layout_builder
                .uniform_buffer(wgpu::ShaderStage::COMPUTE, uniform_dynamic);

            bind_group_builder = bind_group_builder.buffer::<T>(u, 0..1);
        }

        let bind_group_layout = bind_group_layout_builder.build(device);
        let bind_group = bind_group_builder.build(device, &bind_group_layout);

        let pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
        let pipeline = create_compute_pipeline(device, &pipeline_layout, cs_mod);

        Ok(Self {
            bind_group,
            pipeline,
        })
    }

    pub fn compute(&self, encoder: &mut wgpu::CommandEncoder, num_groups: u32) {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch(num_groups, 1, 1);
    }
}

fn create_pipeline_layout(
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
