//! wgpu device, compute pipeline (geodesic tracer) and output handling.

use bytemuck::{Pod, Zeroable};
use horizonlab_core::{Jmn1, NovikovThorne, Schwarzschild, SphericalMetric, SphericalOrbits};

/// Samples of the Novikov–Thorne table uploaded to the GPU.
pub const NT_SAMPLES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetricChoice {
    Schwarzschild,
    Jmn1 { rb: f64 },
}

impl MetricChoice {
    pub fn build(self) -> Box<dyn SphericalMetric> {
        match self {
            MetricChoice::Schwarzschild => Box::new(Schwarzschild::new(1.0)),
            MetricChoice::Jmn1 { rb } => Box::new(Jmn1::new(1.0, rb)),
        }
    }
}

/// Everything that defines a frame.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub metric: MetricChoice,
    pub distance: f64,
    pub inclination_deg: f64,
    pub azimuth_deg: f64,
    pub fov_deg: f64,
    pub disk: bool,
    pub novikov_thorne: bool,
    pub disk_in: f64,
    pub disk_out: f64,
    pub exposure: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    pos_hat: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    view: [f32; 4],
    metric: [f32; 4],
    disk: [f32; 4],
    frame: [f32; 4],
    table: [f32; 4],
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    a.map(|v| v / n)
}

fn v4(a: [f64; 3]) -> [f32; 4] {
    [a[0] as f32, a[1] as f32, a[2] as f32, 0.0]
}

/// Peak of the Newtonian profile from r_in = 6M (CPU normalisation).
fn newtonian_reference_peak() -> f64 {
    let r: f64 = 49.0 / 36.0 * 6.0;
    r.powi(-3) * (1.0 - (6.0 / r).sqrt())
}

/// Novikov–Thorne flux table on a log grid, normalised like the CPU
/// renderer (peak of the Schwarzschild r_in = 6M disk = 1). Returns
/// `(ln r₀, Δ ln r, values)` or `None` if no circular orbit at `r_in`.
fn nt_table(metric: &dyn SphericalMetric, r_in: f64, r_out: f64) -> Option<(f64, f64, Vec<f32>)> {
    let schw = Schwarzschild::new(1.0);
    let reference = NovikovThorne::new(&SphericalOrbits(&schw), 6.0, 100.0)
        .ok()?
        .peak();
    let nt = NovikovThorne::new(&SphericalOrbits(metric), r_in, r_out).ok()?;
    let (l0, l1) = (r_in.ln(), r_out.ln());
    let dl = (l1 - l0) / (NT_SAMPLES - 1) as f64;
    let values = (0..NT_SAMPLES)
        .map(|i| (nt.flux((l0 + dl * i as f64).exp().min(r_out)) / reference) as f32)
        .collect();
    Some((l0, dl, values))
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    compute: wgpu::ComputePipeline,
    compute_layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    nt_buffer: wgpu::Buffer,
    /// Disk inner radius and metric the NT table was built for.
    nt_key: Option<(MetricChoice, u64, u64)>,
    nt_meta: (f32, f32),
}

/// Output image of the compute pass.
pub struct Target {
    pub width: u32,
    pub height: u32,
    pub texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

impl Gpu {
    pub async fn new(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<(Self, wgpu::Adapter), String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("no suitable GPU adapter")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("horizonlab"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| e.to_string())?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute"),
            bind_group_layouts: &[&compute_layout],
            push_constant_ranges: &[],
        });
        let compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("geodesics"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nt_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nt flux"),
            size: (NT_SAMPLES * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok((
            Self {
                device,
                queue,
                compute,
                compute_layout,
                params,
                nt_buffer,
                nt_key: None,
                nt_meta: (0.0, 1.0),
            },
            adapter,
        ))
    }

    pub fn create_target(&self, width: u32, height: u32) -> Target {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute"),
            layout: &self.compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.nt_buffer.as_entire_binding(),
                },
            ],
        });
        Target {
            width,
            height,
            texture,
            bind_group,
        }
    }

    /// Upload parameters (and the NT table if the disk changed) and run the
    /// tracer into `target`. Returns an error if NT is requested where no
    /// circular orbit exists at `disk_in`.
    pub fn trace(&mut self, target: &Target, v: &View) -> Result<(), String> {
        let metric = v.metric.build();
        let key = (v.metric, v.disk_in.to_bits(), v.disk_out.to_bits());
        if v.disk && v.novikov_thorne && self.nt_key != Some(key) {
            let (l0, dl, values) = nt_table(metric.as_ref(), v.disk_in, v.disk_out)
                .ok_or_else(|| format!("no circular orbit at r_in = {}M", v.disk_in))?;
            self.queue
                .write_buffer(&self.nt_buffer, 0, bytemuck::cast_slice(&values));
            self.nt_meta = (l0 as f32, dl as f32);
            self.nt_key = Some(key);
        }

        let (inc, az) = (
            v.inclination_deg.clamp(0.01, 179.99).to_radians(),
            v.azimuth_deg.to_radians(),
        );
        let pos_hat = [inc.sin() * az.cos(), inc.sin() * az.sin(), inc.cos()];
        let forward = pos_hat.map(|c| -c);
        let fz = forward[2];
        let up = normalize([-fz * forward[0], -fz * forward[1], 1.0 - fz * forward[2]]);
        let right = cross(forward, up);
        let (kind, m0, alpha, rb) = match v.metric {
            MetricChoice::Schwarzschild => (0.0, 0.0, 0.0, 0.0),
            MetricChoice::Jmn1 { rb } => {
                let m0 = 2.0 / rb;
                (1.0, m0, m0 / (1.0 - m0), rb)
            }
        };
        let params = Params {
            pos_hat: v4(pos_hat),
            forward: v4(forward),
            right: v4(right),
            up: v4(up),
            view: [
                v.distance as f32,
                (v.fov_deg.to_radians() * 0.5).tan() as f32,
                target.width as f32 / target.height as f32,
                v.exposure as f32,
            ],
            metric: [kind, m0 as f32, alpha as f32, rb as f32],
            disk: [
                v.disk_in as f32,
                v.disk_out as f32,
                if v.disk { 1.0 } else { 0.0 },
                if v.novikov_thorne { 1.0 } else { 0.0 },
            ],
            frame: [
                target.width as f32,
                target.height as f32,
                (10.0 * v.distance).max(1000.0) as f32,
                4000.0,
            ],
            table: [
                self.nt_meta.0,
                self.nt_meta.1,
                NT_SAMPLES as f32,
                newtonian_reference_peak() as f32,
            ],
        };
        self.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.compute);
            pass.set_bind_group(0, &target.bind_group, &[]);
            pass.dispatch_workgroups(target.width.div_ceil(8), target.height.div_ceil(8), 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Copy the target back to the CPU as tightly packed RGBA8.
    pub fn read(&self, target: &Target) -> Vec<u8> {
        let (w, h) = (target.width, target.height);
        let padded = (4 * w).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            target.texture.as_image_copy(),
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((4 * w * h) as usize);
        for row in 0..h as usize {
            let start = row * padded as usize;
            out.extend_from_slice(&data[start..start + 4 * w as usize]);
        }
        out
    }
}
