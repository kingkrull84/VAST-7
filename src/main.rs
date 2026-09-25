use std::cell::RefCell;
use std::rc::Rc;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::Window,
};
use vast7_engine::{
    compute::pipeline::ComputePipelineManager,
    gui::ui::{GuiOverlay, SimulationState},
    render::{camera::Camera3D, pipeline::RenderPipelineManager},
    sim::grid::{DoubleBufferGrid, GridConfig},
};

fn main() {
    env_logger::init();
    println!("Launching V.A.S.T. 7 Curvilinear Lattice Engine...");

    let event_loop = EventLoop::new().unwrap();
    let window = std::sync::Arc::new(
        event_loop.create_window(
            Window::default_attributes()
                .with_title("V.A.S.T. 7 Curvilinear Lattice Physics Engine")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        ).unwrap()
    );

    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window.clone()).unwrap();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("VAST-7 Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .unwrap();

    let size = window.inner_size();
    let surface_config = surface.get_default_config(&adapter, size.width, size.height).unwrap();
    surface.configure(&device, &surface_config);

    // Initialize Simulation Grid & Compute Pipeline
    let config = GridConfig { dim_x: 64, dim_y: 64, dim_z: 64 };
    let mut grid = DoubleBufferGrid::new(&device, config);
    let compute_manager = ComputePipelineManager::new(&device);
    compute_manager.setup_bind_groups(&device, &mut grid);

    // Initialize Render Pipeline & Camera
    let mut camera = Camera3D::new((size.width as f32) / (size.height as f32));
    let mut render_manager = RenderPipelineManager::new(&device, &surface_config, &camera, &grid);

    // Initialize State & egui Overlay
    let sim_state = Rc::new(RefCell::new(SimulationState::default()));
    let _egui_overlay = GuiOverlay::new(sim_state.clone());

    let mut mouse_pressed_left = false;
    let mut mouse_pressed_right = false;
    let mut last_cursor_pos = (0.0, 0.0);

    let _ = event_loop.run(move |event, window_target| {
        match event {
            Event::WindowEvent { ref event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => window_target.exit(),
                    WindowEvent::Resized(physical_size) => {
                        if physical_size.width > 0 && physical_size.height > 0 {
                            let mut config = surface_config.clone();
                            config.width = physical_size.width;
                            config.height = physical_size.height;
                            surface.configure(&device, &config);
                            camera.aspect = (physical_size.width as f32) / (physical_size.height as f32);
                            render_manager.resize(&device, &config);
                        }
                    }
                    WindowEvent::MouseInput { button, state, .. } => {
                        match button {
                            MouseButton::Left => mouse_pressed_left = *state == ElementState::Pressed,
                            MouseButton::Right => mouse_pressed_right = *state == ElementState::Pressed,
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let dx = position.x - last_cursor_pos.0;
                        let dy = position.y - last_cursor_pos.1;
                        last_cursor_pos = (position.x, position.y);

                        if mouse_pressed_left {
                            camera.rotate(dx as f32, dy as f32);
                        } else if mouse_pressed_right {
                            camera.pan(dx as f32, dy as f32);
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let zoom_delta = match delta {
                            MouseScrollDelta::LineDelta(_, y) => *y,
                            MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.1,
                        };
                        camera.zoom(zoom_delta);
                    }
                    WindowEvent::RedrawRequested => {
                        // Dispatch GPU Compute Simulation Step
                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Sim & Render Encoder"),
                        });

                        {
                            let mut state = sim_state.borrow_mut();
                            if state.is_running || state.single_step {
                                let ticks = if state.single_step { 1 } else { state.ticks_per_frame };
                                for _ in 0..ticks {
                                    compute_manager.dispatch(&mut encoder, &mut grid);
                                    state.total_ticks += 1;
                                }
                                state.single_step = false;
                            }
                        }

                        render_manager.update_camera(&queue, &camera);

                        let output = surface.get_current_texture().unwrap();
                        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

                        render_manager.render(&mut encoder, &view, &grid);

                        queue.submit(std::iter::once(encoder.finish()));
                        output.present();
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}
