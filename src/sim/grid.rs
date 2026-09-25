use wgpu::util::DeviceExt;
use crate::sim::voxel::{GpuVoxel, BASELINE_PRESSURE, CEILING_PRESSURE};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub grid_dim: [u32; 4], // [dim_x, dim_y, dim_z, padding]
    pub baseline_pressure: u32,
    pub ceiling_pressure: u32,
    pub padding: [u32; 2],
}

pub struct GridConfig {
    pub dim_x: u32,
    pub dim_y: u32,
    pub dim_z: u32,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            dim_x: 64,
            dim_y: 64,
            dim_z: 64,
        }
    }
}

impl GridConfig {
    pub fn total_voxels(&self) -> usize {
        (self.dim_x * self.dim_y * self.dim_z) as usize
    }

    pub fn flat_index(&self, x: u32, y: u32, z: u32) -> usize {
        (x + y * self.dim_x + z * self.dim_x * self.dim_y) as usize
    }
}

pub struct DoubleBufferGrid {
    pub config: GridConfig,
    pub buffer_a: wgpu::Buffer,
    pub buffer_b: wgpu::Buffer,
    pub param_buffer: wgpu::Buffer,
    pub bind_group_a_read: Option<wgpu::BindGroup>,
    pub bind_group_b_read: Option<wgpu::BindGroup>,
    pub current_read_a: bool,
}

impl DoubleBufferGrid {
    pub fn new(device: &wgpu::Device, config: GridConfig) -> Self {
        let total = config.total_voxels();
        let mut initial_data = vec![GpuVoxel::ambient(); total];

        // Place central Sink and Source for initial gravity well simulation
        let cx = config.dim_x / 2;
        let cy = config.dim_y / 2;
        let cz = config.dim_z / 2;

        let sink_idx = config.flat_index(cx, cy, cz);
        initial_data[sink_idx] = GpuVoxel::sink();

        if cx + 10 < config.dim_x {
            let source_idx = config.flat_index(cx + 10, cy, cz);
            initial_data[source_idx] = GpuVoxel::source();
        }

        let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lattice Storage Buffer A"),
            contents: bytemuck::cast_slice(&initial_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lattice Storage Buffer B"),
            contents: bytemuck::cast_slice(&initial_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let params = SimParams {
            grid_dim: [config.dim_x, config.dim_y, config.dim_z, 0],
            baseline_pressure: BASELINE_PRESSURE,
            ceiling_pressure: CEILING_PRESSURE,
            padding: [0, 0],
        };

        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SimParams Uniform Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            config,
            buffer_a,
            buffer_b,
            param_buffer,
            bind_group_a_read: None,
            bind_group_b_read: None,
            current_read_a: true,
        }
    }

    pub fn swap(&mut self) {
        self.current_read_a = !self.current_read_a;
    }
}
