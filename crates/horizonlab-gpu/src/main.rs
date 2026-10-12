//! HorizonLab interactive GPU viewer (Schwarzschild, JMN-1).
//!
//! Single-precision preview of the CPU reference renderer, for exploring
//! parameters in real time. `--headless` renders one frame to PNG and
//! `--compare` measures agreement with a CPU (`horizonlab`) render made with
//! the same parameters.
//!
//! Controls: left-drag orbit · wheel zoom · M metric · ↑/↓ R_b · D disk ·
//! N Novikov–Thorne/Newtonian · +/- exposure · [ ] field of view · Esc quit.

mod gpu;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use gpu::{Gpu, MetricChoice, Target, View};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetricKind {
    Schwarzschild,
    Jmn1,
}

/// Interactive GPU preview. Lengths in units of M.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(long, value_enum, default_value = "schwarzschild")]
    metric: MetricKind,
    #[arg(long, default_value_t = 2.5)]
    rb: f64,
    #[arg(long, default_value_t = 30.0)]
    distance: f64,
    #[arg(long, default_value_t = 80.0)]
    inclination: f64,
    #[arg(long, default_value_t = 30.0)]
    fov: f64,
    #[arg(long, default_value_t = 6.0)]
    disk_in: f64,
    #[arg(long, default_value_t = 20.0)]
    disk_out: f64,
    #[arg(long)]
    no_disk: bool,
    /// Use the Newtonian flux profile instead of Novikov–Thorne.
    #[arg(long)]
    newtonian: bool,
    #[arg(long, default_value_t = 1.5)]
    exposure: f64,
    #[arg(long, default_value_t = 1280)]
    width: u32,
    #[arg(long, default_value_t = 720)]
    height: u32,
    /// Render one frame to this PNG and exit (no window).
    #[arg(long)]
    headless: Option<PathBuf>,
    /// With --headless: compare against a CPU render of identical settings.
    #[arg(long, requires = "headless")]
    compare: Option<PathBuf>,
}

impl Args {
    fn view(&self) -> View {
        View {
            metric: match self.metric {
                MetricKind::Schwarzschild => MetricChoice::Schwarzschild,
                MetricKind::Jmn1 => MetricChoice::Jmn1 { rb: self.rb },
            },
            distance: self.distance,
            inclination_deg: self.inclination,
            azimuth_deg: 0.0,
            fov_deg: self.fov,
            disk: !self.no_disk,
            novikov_thorne: !self.newtonian,
            disk_in: self.disk_in,
            disk_out: self.disk_out,
            exposure: self.exposure,
        }
    }
}

fn headless(args: &Args, out: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::default();
    let (mut gpu, adapter) = pollster::block_on(Gpu::new(&instance, None))?;
    println!("GPU: {}", adapter.get_info().name);
    let target = gpu.create_target(args.width, args.height);
    let start = Instant::now();
    gpu.trace(&target, &args.view())?;
    let rgba = gpu.read(&target);
    println!(
        "Traced {}x{} in {:.2?}",
        args.width,
        args.height,
        start.elapsed()
    );
    let img = image::RgbaImage::from_raw(args.width, args.height, rgba)
        .ok_or("readback size mismatch")?;
    img.save(out)?;
    println!("-> {}", out.display());

    if let Some(cpu_path) = &args.compare {
        let cpu = image::open(cpu_path)?.to_rgb8();
        if cpu.dimensions() != img.dimensions() {
            return Err("CPU image has different dimensions".into());
        }
        let (mut sum, mut big) = (0.0f64, 0usize);
        let n = (args.width * args.height) as usize;
        for (g, c) in img.pixels().zip(cpu.pixels()) {
            let d = (0..3)
                .map(|i| (g[i] as i32 - c[i] as i32).abs())
                .max()
                .unwrap_or(0);
            sum += d as f64;
            if d > 16 {
                big += 1;
            }
        }
        println!(
            "Versus CPU: mean max-channel difference {:.2}/255, {:.2}% of pixels differ by > 16/255",
            sum / n as f64,
            100.0 * big as f64 / n as f64
        );
    }
    Ok(())
}

struct Viewer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    target: Target,
    blit: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    blit_group: wgpu::BindGroup,
    decode_flag: wgpu::Buffer,
    view: View,
    dirty: bool,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    status: String,
}

impl Viewer {
    async fn new(window: Arc<Window>, view: View) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| e.to_string())?;
        let (gpu, adapter) = Gpu::new(&instance, Some(&surface)).await?;
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        // Prefer a non-sRGB format: the compute shader already encodes sRGB.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("blit"),
                source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
            });
        let blit_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blit"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("blit"),
                bind_group_layouts: &[&blit_layout],
                push_constant_ranges: &[],
            });
        let blit = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blit"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: "vs_blit",
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: "fs_blit",
                    targets: &[Some(format.into())],
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });
        let decode_flag = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decode flag"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let flag = if format.is_srgb() { 1.0f32 } else { 0.0 };
        gpu.queue.write_buffer(
            &decode_flag,
            0,
            bytemuck::cast_slice(&[flag, 0.0, 0.0, 0.0]),
        );

        let target = gpu.create_target(config.width, config.height);
        let blit_group = Self::blit_group(&gpu, &blit_layout, &target, &decode_flag);
        Ok(Self {
            window,
            surface,
            config,
            gpu,
            target,
            blit,
            blit_layout,
            blit_group,
            decode_flag,
            view,
            dirty: true,
            dragging: false,
            last_cursor: None,
            status: String::new(),
        })
    }

    fn blit_group(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        target: &Target,
        flag: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let view = target.texture.create_view(&Default::default());
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flag.as_entire_binding(),
                },
            ],
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.target = self.gpu.create_target(width, height);
        self.blit_group = Self::blit_group(
            &self.gpu,
            &self.blit_layout,
            &self.target,
            &self.decode_flag,
        );
        self.dirty = true;
    }

    fn title(&self) -> String {
        let metric = match self.view.metric {
            MetricChoice::Schwarzschild => "Schwarzschild".to_string(),
            MetricChoice::Jmn1 { rb } => format!("JMN-1 R_b={rb:.2}M"),
        };
        format!(
            "HorizonLab — {metric} · r={:.1}M · i={:.0}° · disk {} ({}) · exp {:.2} · fov {:.0}° {}",
            self.view.distance,
            self.view.inclination_deg,
            if self.view.disk { "on" } else { "off" },
            if self.view.novikov_thorne { "NT" } else { "Newtonian" },
            self.view.exposure,
            self.view.fov_deg,
            self.status
        )
    }

    fn redraw(&mut self) {
        if self.dirty {
            let start = Instant::now();
            self.status = match self.gpu.trace(&self.target, &self.view) {
                Ok(()) => {
                    self.gpu.device.poll(wgpu::Maintain::Wait);
                    format!("· {:.0} ms", start.elapsed().as_secs_f64() * 1e3)
                }
                Err(e) => format!("· {e}"),
            };
            self.dirty = false;
            self.window.set_title(&self.title());
        }
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.gpu.device, &self.config);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.blit);
            pass.set_bind_group(0, &self.blit_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.gpu.queue.submit([encoder.finish()]);
        frame.present();
    }

    fn key(&mut self, key: &Key) {
        let v = &mut self.view;
        match key {
            Key::Character(c) => match c.as_str() {
                "m" | "M" => {
                    v.metric = match v.metric {
                        MetricChoice::Schwarzschild => MetricChoice::Jmn1 { rb: 2.5 },
                        MetricChoice::Jmn1 { .. } => MetricChoice::Schwarzschild,
                    }
                }
                "d" | "D" => v.disk = !v.disk,
                "n" | "N" => v.novikov_thorne = !v.novikov_thorne,
                "+" | "=" => v.exposure *= 1.25,
                "-" => v.exposure /= 1.25,
                "[" => v.fov_deg = (v.fov_deg / 1.1).max(1.0),
                "]" => v.fov_deg = (v.fov_deg * 1.1).min(120.0),
                _ => return,
            },
            Key::Named(NamedKey::ArrowUp) | Key::Named(NamedKey::ArrowDown) => {
                if let MetricChoice::Jmn1 { rb } = v.metric {
                    let step = if matches!(key, Key::Named(NamedKey::ArrowUp)) {
                        0.05
                    } else {
                        -0.05
                    };
                    v.metric = MetricChoice::Jmn1 {
                        rb: (rb + step).max(2.05),
                    };
                } else {
                    return;
                }
            }
            _ => return,
        }
        self.dirty = true;
        self.window.request_redraw();
    }
}

struct App {
    initial: View,
    viewer: Option<Viewer>,
    error: Option<String>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.viewer.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("HorizonLab")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.error = Some(e.to_string());
                event_loop.exit();
                return;
            }
        };
        match pollster::block_on(Viewer::new(window, self.initial)) {
            Ok(v) => {
                v.window.request_redraw();
                self.viewer = Some(v);
            }
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(v) = self.viewer.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                v.resize(size.width, size.height);
                v.window.request_redraw();
            }
            WindowEvent::RedrawRequested => v.redraw(),
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                v.dragging = state == ElementState::Pressed;
                v.last_cursor = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                if v.dragging {
                    if let Some((x0, y0)) = v.last_cursor {
                        v.view.azimuth_deg -= 0.3 * (position.x - x0);
                        v.view.inclination_deg =
                            (v.view.inclination_deg - 0.3 * (position.y - y0)).clamp(1.0, 179.0);
                        v.dirty = true;
                        v.window.request_redraw();
                    }
                    v.last_cursor = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 50.0,
                };
                v.view.distance = (v.view.distance * 1.1f64.powf(-steps)).clamp(4.0, 2000.0);
                v.dirty = true;
                v.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if event.logical_key == Key::Named(NamedKey::Escape) {
                    event_loop.exit();
                } else {
                    v.key(&event.logical_key);
                }
            }
            _ => {}
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if let Some(out) = &args.headless {
        return headless(&args, out);
    }
    println!(
        "Controls: left-drag orbit · wheel zoom · M metric · ↑/↓ R_b · D disk · \
         N NT/Newtonian · +/- exposure · [ ] fov · Esc quit"
    );
    let event_loop = EventLoop::new()?;
    let mut app = App {
        initial: args.view(),
        viewer: None,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(e) => Err(e.into()),
        None => Ok(()),
    }
}
