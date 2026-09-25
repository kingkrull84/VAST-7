use std::borrow::Cow;
use crate::sim::grid::DoubleBufferGrid;

pub struct ComputePipelineManager {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
}

impl ComputePipelineManager {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("V.A.S.T. 7 Gather Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("sim_gather.wgsl"))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sim Compute Bind Group Layout"),
            entries: &[
                // Uniform SimParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Read Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Write Buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sim Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sim Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            bind_group_layout,
            pipeline,
        }
    }

    pub fn setup_bind_groups(&self, device: &wgpu::Device, grid: &mut DoubleBufferGrid) {
        let bind_group_a_read = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sim Bind Group A Read -> B Write"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid.param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid.buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid.buffer_b.as_entire_binding(),
                },
            ],
        });

        let bind_group_b_read = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sim Bind Group B Read -> A Write"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid.param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid.buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid.buffer_a.as_entire_binding(),
                },
            ],
        });

        grid.bind_group_a_read = Some(bind_group_a_read);
        grid.bind_group_b_read = Some(bind_group_b_read);
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, grid: &mut DoubleBufferGrid) {
        let bind_group = if grid.current_read_a {
            grid.bind_group_a_read.as_ref().unwrap()
        } else {
            grid.bind_group_b_read.as_ref().unwrap()
        };

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Lattice Compton Tick Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);

            let wg_x = (grid.config.dim_x + 7) / 8;
            let wg_y = (grid.config.dim_y + 7) / 8;
            let wg_z = (grid.config.dim_z + 7) / 8;

            compute_pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        grid.swap();
    }
}
