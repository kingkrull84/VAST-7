use crate::sim::voxel::{
    pack_voxel_state, unpack_voxel_state, GpuVoxel, IdState, BASELINE_PRESSURE, CEILING_PRESSURE, SINK_PRESSURE,
};
use crate::sim::grid::GridConfig;

#[test]
fn test_voxel_bit_packing() {
    let states = [IdState::Sink, IdState::Ambient, IdState::Source];
    let vectors = [0u8, 12u8, 26u8];
    let momentums = [0u32, 1000u32, 33554431u32];

    for id in states {
        for vec in vectors {
            for mom in momentums {
                let packed = pack_voxel_state(id.to_i32(), vec as u32, mom);
                let (un_id, un_vec, un_mom) = unpack_voxel_state(packed);

                assert_eq!(IdState::from_i32(un_id), id);
                assert_eq!(un_vec, vec as u32);
                assert_eq!(un_mom, mom);
            }
        }
    }
}

#[test]
fn test_voxel_constructor_and_unpack() {
    let ambient = GpuVoxel::ambient();
    let (p, id, vec, mom) = ambient.unpack();
    assert_eq!(p, BASELINE_PRESSURE);
    assert_eq!(id, IdState::Ambient);
    assert_eq!(vec, 0);
    assert_eq!(mom, 0);

    let sink = GpuVoxel::sink();
    let (p_s, id_s, _, _) = sink.unpack();
    assert_eq!(p_s, SINK_PRESSURE);
    assert_eq!(id_s, IdState::Sink);

    let source = GpuVoxel::source();
    let (p_src, id_src, _, _) = source.unpack();
    assert_eq!(p_src, CEILING_PRESSURE);
    assert_eq!(id_src, IdState::Source);
}

#[test]
fn test_grid_flat_indexing() {
    let config = GridConfig { dim_x: 64, dim_y: 64, dim_z: 64 };
    assert_eq!(config.total_voxels(), 64 * 64 * 64);

    let idx1 = config.flat_index(0, 0, 0);
    assert_eq!(idx1, 0);

    let idx2 = config.flat_index(63, 63, 63);
    assert_eq!(idx2, 64 * 64 * 64 - 1);
}
