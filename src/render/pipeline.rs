use std::borrow::Cow;
use wgpu::util::DeviceExt;
use crate::render::camera::Camera3D;
use crate::sim::grid::DoubleBufferGrid;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LatticeVertex {
    pub position: [f32; 3],
    pub grid_coord: [u32; 3],
}

impl LatticeVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LatticeVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Uint32x3,
                },
            ],
        }
    }
}

pub struct RenderPipelineManager {
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
    pub camera_bind_group_a: wgpu::BindGroup,
    pub camera_bind_group_b: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub depth_texture: wgpu::Texture,
    pub depth_texture_view: wgpu::TextureView,
}

impl RenderPipelineManager {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

    pub fn new(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
        camera: &Camera3D,
        grid: &DoubleBufferGrid,
    ) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let dim_x = grid.config.dim_x;
        let dim_y = grid.config.dim_y;
        let dim_z = grid.config.dim_z;

        let step = 4;
        let offset = [
            (dim_x as f32) * 0.5,
            (dim_y as f32) * 0.5,
            (dim_z as f32) * 0.5,
        ];

        let mut coord_to_idx = std::collections::HashMap::new();

        for z in (0..dim_z).step_by(step as usize) {
            for y in (0..dim_y).step_by(step as usize) {
                for x in (0..dim_x).step_by(step as usize) {
                    let idx = vertices.len() as u32;
                    coord_to_idx.insert((x, y, z), idx);

                    let pos = [
                        (x as f32) - offset[0],
                        (y as f32) - offset[1],
                        (z as f32) - offset[2],
                    ];

                    vertices.push(LatticeVertex {
                        position: pos,
                        grid_coord: [x, y, z],
                    });
                }
            }
        }

        for z in (0..dim_z).step_by(step as usize) {
            for y in (0..dim_y).step_by(step as usize) {
                for x in (0..dim_x).step_by(step as usize) {
                    if let Some(&curr_idx) = coord_to_idx.get(&(x, y, z)) {
                        if x + step < dim_x {
                            if let Some(&next_idx) = coord_to_idx.get(&(x + step, y, z)) {
                                indices.push(curr_idx);
                                indices.push(next_idx);
                            }
                        }
                        if y + step < dim_y {
                            if let Some(&next_idx) = coord_to_idx.get(&(x, y + step, z)) {
                                indices.push(curr_idx);
                                indices.push(next_idx);
                            }
                        }
                        if z + step < dim_z {
                            if let Some(&next_idx) = coord_to_idx.get(&(x, y, z + step)) {
                                indices.push(curr_idx);
                                indices.push(next_idx);
                            }
                        }
                    }
                }
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lattice Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lattice Mesh Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let (depth_texture, depth_texture_view) = Self::create_depth_texture(device, surface_config);

        let camera_uniform = camera.to_uniform();
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lattice Render Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("lattice.wgsl"))),
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lattice Render Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let camera_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group Buffer A"),
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid.param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid.buffer_a.as_entire_binding(),
                },
            ],
        });

        let camera_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group Buffer B"),
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid.param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid.buffer_b.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lattice Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lattice Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[LatticeVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group_a,
            camera_bind_group_b,
            pipeline,
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            depth_texture,
            depth_texture_view,
        }
    }

    pub fn create_depth_texture(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: surface_config.width.max(1),
            height: surface_config.height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Lattice Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub fn resize(&mut self, device: &wgpu::Device, surface_config: &wgpu::SurfaceConfiguration) {
        let (depth_texture, depth_texture_view) = Self::create_depth_texture(device, surface_config);
        self.depth_texture = depth_texture;
        self.depth_texture_view = depth_texture_view;
    }

    pub fn update_camera(&self, queue: &wgpu::Queue, camera: &Camera3D) {
        let uniform = camera.to_uniform();
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[uniform]));
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        grid: &DoubleBufferGrid,
    ) {
        let bind_group = if grid.current_read_a {
            &self.camera_bind_group_a
        } else {
            &self.camera_bind_group_b
        };

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Lattice Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.04,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}
