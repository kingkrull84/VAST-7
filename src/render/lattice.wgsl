struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct SimParams {
    grid_dim: vec4<u32>,
    baseline_pressure: u32,
    ceiling_pressure: u32,
    padding: vec2<u32>,
};

struct Voxel {
    pressure: u32,
    packed_state: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> sim_params: SimParams;
@group(0) @binding(2) var<storage, read> lattice_buffer: array<Voxel>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) grid_coord: vec3<u32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vpu_pressure: f32,
    @location(1) world_position: vec3<f32>,
};

fn get_flat_index(coord: vec3<u32>, dim: vec3<u32>) -> u32 {
    return coord.x + (coord.y * dim.x) + (coord.z * dim.x * dim.y);
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let dim = sim_params.grid_dim.xyz;
    let center_idx = get_flat_index(model.grid_coord, dim);
    let voxel = lattice_buffer[center_idx];

    let baseline = f32(sim_params.baseline_pressure);
    let pressure = f32(voxel.pressure);

    // Normalized pressure offset from baseline [-1.0, 1.0]
    let delta_p = (pressure - baseline) / baseline;

    // Calculate displacement towards world center (0,0,0) based on local pressure gradient / strain
    let grid_center_world = vec3<f32>(0.0);
    let dir = normalize(grid_center_world - model.position + vec3<f32>(0.0001));

    // Curvilinear displacement strain with clamping at event horizon
    let displacement_scale = 3.0;
    let raw_displacement = dir * delta_p * displacement_scale;
    let clamped_displacement = clamp(raw_displacement, vec3<f32>(-5.0), vec3<f32>(5.0));

    let world_pos = model.position + clamped_displacement;

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.vpu_pressure = pressure / f32(sim_params.ceiling_pressure);
    out.world_position = world_pos;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = clamp(in.vpu_pressure, 0.0, 1.0);

    var color = vec3<f32>(0.0);
    if (p < 0.5) {
        let t = p * 2.0;
        color = mix(vec3<f32>(0.0, 0.4, 1.0), vec3<f32>(0.0, 0.9, 0.2), t);
    } else {
        let t = (p - 0.5) * 2.0;
        color = mix(vec3<f32>(0.0, 0.9, 0.2), vec3<f32>(1.0, 0.1, 0.1), t);
    }

    return vec4<f32>(color, 0.85);
}
