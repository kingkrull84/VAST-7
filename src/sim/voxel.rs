use bytemuck::{Pod, Zeroable};

pub const BASELINE_PRESSURE: u32 = 1_073_741_824; // 2^30
pub const CEILING_PRESSURE: u32 = 2_147_483_648;  // 2^31
pub const SINK_PRESSURE: u32 = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdState {
    Sink = -1,
    Ambient = 0,
    Source = 1,
}

impl IdState {
    pub fn from_i32(val: i32) -> Self {
        match val {
            -1 => IdState::Sink,
            1 => IdState::Source,
            _ => IdState::Ambient,
        }
    }

    pub fn to_i32(self) -> i32 {
        self as i32
    }
}

/// 8-byte GPU Voxel State representation
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuVoxel {
    pub pressure: u32,
    pub packed_state: u32,
}

impl GpuVoxel {
    pub fn new(pressure: u32, id_state: IdState, vector: u8, momentum: u32) -> Self {
        let packed_state = pack_voxel_state(id_state.to_i32(), vector as u32, momentum);
        Self {
            pressure,
            packed_state,
        }
    }

    pub fn ambient() -> Self {
        Self::new(BASELINE_PRESSURE, IdState::Ambient, 0, 0)
    }

    pub fn sink() -> Self {
        Self::new(SINK_PRESSURE, IdState::Sink, 0, 0)
    }

    pub fn source() -> Self {
        Self::new(CEILING_PRESSURE, IdState::Source, 0, 0)
    }

    pub fn unpack(&self) -> (u32, IdState, u8, u32) {
        let (id, vec, mom) = unpack_voxel_state(self.packed_state);
        (self.pressure, IdState::from_i32(id), vec as u8, mom)
    }
}

/// Bit-packs ID_State (bits 0-1), Vector (bits 2-6), and Momentum (bits 7-31) into a single u32.
pub fn pack_voxel_state(id_state: i32, vector: u32, momentum: u32) -> u32 {
    let mapped_id = ((id_state + 1) as u32) & 0x3;
    let vec_bits = (vector & 0x1F) << 2;
    let mom_bits = (momentum & 0x1FF_FFFF) << 7;
    mom_bits | vec_bits | mapped_id
}

/// Unpacks (id_state, vector, momentum) from a packed u32.
pub fn unpack_voxel_state(packed: u32) -> (i32, u32, u32) {
    let mapped_id = packed & 0x3;
    let id_state = (mapped_id as i32) - 1;
    let vector = (packed >> 2) & 0x1F;
    let momentum = packed >> 7;
    (id_state, vector, momentum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_packing_unpacking() {
        let test_cases = [
            (IdState::Sink, 0u8, 0u32),
            (IdState::Ambient, 14u8, 100u32),
            (IdState::Source, 26u8, 33554431u32),
        ];

        for (id, vec, mom) in test_cases {
            let packed = pack_voxel_state(id.to_i32(), vec as u32, mom);
            let (un_id, un_vec, un_mom) = unpack_voxel_state(packed);
            assert_eq!(IdState::from_i32(un_id), id);
            assert_eq!(un_vec, vec as u32);
            assert_eq!(un_mom, mom);
        }
    }
}
