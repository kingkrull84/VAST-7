struct Voxel {
    pressure: u32,
    packed_state: u32,
};

struct SimParams {
    grid_dim: vec4<u32>,        // xyz = dimensions, w = padding
    baseline_pressure: u32,     // 1,073,741,824 (2^30)
    ceiling_pressure: u32,      // 2,147,483,648 (2^31)
    padding: vec2<u32>,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> buffer_read: array<Voxel>;
@group(0) @binding(2) var<storage, read_write> buffer_write: array<Voxel>;

fn get_flat_index(coord: vec3<u32>, dim: vec3<u32>) -> u32 {
    return coord.x + (coord.y * dim.x) + (coord.z * dim.x * dim.y);
}

fn unpack_id_state(packed_data: u32) -> i32 {
    let mapped_id = packed_data & 0x3u;
    return i32(mapped_id) - 1;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim = params.grid_dim.xyz;

    if (global_id.x >= dim.x || global_id.y >= dim.y || global_id.z >= dim.z) {
        return;
    }

    let center_idx = get_flat_index(global_id, dim);
    let center_voxel = buffer_read[center_idx];
    let id_state = unpack_id_state(center_voxel.packed_state);

    // 1. Permanent Aperture Anchors
    if (id_state == -1) {
        buffer_write[center_idx].pressure = 0u;
        buffer_write[center_idx].packed_state = center_voxel.packed_state;
        return;
    }
    if (id_state == 1) {
        buffer_write[center_idx].pressure = params.ceiling_pressure;
        buffer_write[center_idx].packed_state = center_voxel.packed_state;
        return;
    }

    // 2. 26-Neighbor Stencil Gather Loop
    var net_pressure_delta: i32 = 0;
    var active_neighbors: i32 = 0;

    let pos = vec3<i32>(global_id);
    let center_p = i32(center_voxel.pressure);

    for (var dz = -1; dz <= 1; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }

                let neighbor_pos = pos + vec3<i32>(dx, dy, dz);

                if (neighbor_pos.x >= 0 && neighbor_pos.x < i32(dim.x) &&
                    neighbor_pos.y >= 0 && neighbor_pos.y < i32(dim.y) &&
                    neighbor_pos.z >= 0 && neighbor_pos.z < i32(dim.z)) {

                    let n_idx = get_flat_index(vec3<u32>(neighbor_pos), dim);
                    let neighbor_p = i32(buffer_read[n_idx].pressure);

                    net_pressure_delta += (neighbor_p - center_p);
                    active_neighbors += 1;
                }
            }
        }
    }

    // 3. Pressure Equalization Step
    var new_p = center_p;
    if (active_neighbors > 0) {
        let equalization_rate = 26;
        new_p += (net_pressure_delta / equalization_rate);
    }

    let clamped_p = u32(clamp(new_p, 0, i32(params.ceiling_pressure)));

    buffer_write[center_idx].pressure = clamped_p;
    buffer_write[center_idx].packed_state = center_voxel.packed_state;
}
