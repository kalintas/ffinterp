#![no_std]
use cuda_std::prelude::*;
use cuda_std::vek::num_traits::Float;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IFSMap<T> {
    pub a: T,
    pub d: T,
    pub e: T,
    pub q: [T; 4],
    pub end_x: T,
}

pub trait GpuFloat: 
    Float + 
    Copy + 
    PartialOrd + 
    cuda_std::FloatExt 
{}

impl GpuFloat for f32 {}
impl GpuFloat for f64 {}

/// A generic implementation for the interpolation functions. 
unsafe fn interpolate_impl<T: GpuFloat>(
    maps: &[IFSMap<T>],
    inputs: &[T],
    outputs: *mut T,
    iterations: usize,
    p0_x: T, p0_y: T,
    pn_x: T, pn_y: T,
    epsilon: T,
) {
    let i = thread::index_1d() as usize;
    if i >= inputs.len() { return; }

    let mut x = inputs[i];

    if x <= p0_x { unsafe { *outputs.add(i) = p0_y; } return; }
    if x >= pn_x { unsafe { *outputs.add(i) = pn_y; } return; }

    let mut y_accum = T::zero();
    let mut d_prod = T::one();

    for _ in 0..iterations {
        let mut left = 0;
        let mut right = maps.len();
        // Find the correct map.
        while left < right {
            let mid = left + (right - left) / 2;
            if maps[mid].end_x <= x { left = mid + 1; } else { right = mid; }
        }
        
        let idx = if left >= maps.len() { maps.len() - 1 } else { left };
        let map = &maps[idx];
        let start_x = if idx == 0 { p0_x } else { maps[idx - 1].end_x };
        
        let u = (x - start_x) / (map.end_x - start_x);

        let mut q_val = T::zero();
        q_val = q_val * u + map.q[3];
        q_val = q_val * u + map.q[2];
        q_val = q_val * u + map.q[1];
        q_val = q_val * u + map.q[0];

        y_accum = y_accum + (d_prod * q_val);
        d_prod = d_prod * map.d;

        let mut x_prev = (x - map.e) / map.a;
        if x_prev < p0_x { x_prev = p0_x; }
        if x_prev > pn_x { x_prev = pn_x; }
        x = x_prev;

        if d_prod.abs() < epsilon { break; }
    }
    unsafe { *outputs.add(i) = y_accum; }
}

#[kernel]
pub unsafe fn interpolate_f32(
    maps_ptr: *const IFSMap<f32>,
    maps_len: usize,
    inputs_ptr: *const f32,
    inputs_len: usize,
    outputs: *mut f32,
    iters: usize,
    p0x: f32, p0y: f32,
    pnx: f32, pny: f32
) {
    unsafe {
        // Reconstruct slices safely for the logic function
        let maps = core::slice::from_raw_parts(maps_ptr, maps_len);
        let inputs = core::slice::from_raw_parts(inputs_ptr, inputs_len);
        interpolate_impl(maps, inputs, outputs, iters, p0x, p0y, pnx, pny, 1e-7);
    } 
}

#[kernel]
pub unsafe fn interpolate_f64(
    maps_ptr: *const IFSMap<f64>,
    maps_len: usize,
    inputs_ptr: *const f64,
    inputs_len: usize,
    outputs: *mut f64,
    iters: usize,
    p0x: f64, p0y: f64,
    pnx: f64, pny: f64
) {
    unsafe {
        let maps = core::slice::from_raw_parts(maps_ptr, maps_len);
        let inputs = core::slice::from_raw_parts(inputs_ptr, inputs_len);
        
        interpolate_impl(maps, inputs, outputs, iters, p0x, p0y, pnx, pny, 1e-14);
    }
}
