use glam::{Mat4, Vec3};

pub struct Camera3D {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

impl Camera3D {
    pub fn new(aspect: f32) -> Self {
        let distance = 100.0;
        let yaw = std::f32::consts::FRAC_PI_4;
        let pitch = std::f32::consts::FRAC_PI_6;
        let target = Vec3::ZERO;
        let eye = Self::calculate_eye(target, distance, yaw, pitch);

        Self {
            eye,
            target,
            up: Vec3::Y,
            aspect,
            fovy: 45.0f32.to_radians(),
            znear: 0.1,
            zfar: 1000.0,
            yaw,
            pitch,
            distance,
        }
    }

    fn calculate_eye(target: Vec3, distance: f32, yaw: f32, pitch: f32) -> Vec3 {
        let x = distance * pitch.cos() * yaw.sin();
        let y = distance * pitch.sin();
        let z = distance * pitch.cos() * yaw.cos();
        target + Vec3::new(x, y, z)
    }

    pub fn update_eye(&mut self) {
        self.eye = Self::calculate_eye(self.target, self.distance, self.yaw, self.pitch);
    }

    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }

    pub fn to_uniform(&self) -> CameraUniform {
        let view_proj = self.build_view_projection_matrix().to_cols_array_2d();
        CameraUniform {
            view_proj,
            camera_pos: [self.eye.x, self.eye.y, self.eye.z, 1.0],
        }
    }

    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw += delta_x * 0.005;
        self.pitch = (self.pitch + delta_y * 0.005).clamp(-1.5, 1.5);
        self.update_eye();
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta * 2.0).clamp(5.0, 500.0);
        self.update_eye();
    }

    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let right = (self.target - self.eye).cross(self.up).normalize();
        let up = right.cross(self.target - self.eye).normalize();
        let pan_speed = self.distance * 0.001;
        self.target += (-right * delta_x + up * delta_y) * pan_speed;
        self.update_eye();
    }
}
