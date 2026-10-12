// Full-screen blit of the compute output to the window.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> blit_decode: vec4<f32>;

@vertex
fn vs_blit(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(i & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn decode(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@fragment
fn fs_blit(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(src_tex, vec2<i32>(pos.xy), 0);
    if (blit_decode.x > 0.5) {
        // sRGB swapchain re-encodes: undo the shader's encoding.
        return vec4<f32>(decode(c.x), decode(c.y), decode(c.z), 1.0);
    }
    return vec4<f32>(c.xyz, 1.0);
}
