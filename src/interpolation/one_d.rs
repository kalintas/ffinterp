use std::{
    fmt::Debug,
    iter::Sum,
    ops::{AddAssign, MulAssign},
};

use nalgebra::Point2;
use num::{Float, Zero};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::interpolation::{FreeVariables, IFSMap, Interpolant};

#[cfg(feature = "cuda")]
use cust::{memory::DeviceCopy, prelude::*};

// TODO: Find a better way to conditionally add this trait.
#[cfg(not(feature = "cuda"))]
pub trait DeviceCopy {}
#[cfg(not(feature = "cuda"))]
impl<T> DeviceCopy for T {}

#[derive(Debug, Clone)]
pub struct Interpolant1D<T: Float + Debug + 'static + DeviceCopy> {
    maps: Vec<IFSMap<T>>,
    iterations: usize,
    /// First point in the given interpolated points.
    first_point: Point2<T>,
    /// Last point in the given interpolated points.
    last_point: Point2<T>,
}

impl<T> Interpolant1D<T>
where
    T: Debug + Float + 'static + Send + Sync + AddAssign + MulAssign + Sum + DeviceCopy,
{
    /// A helper function that solves the cubic hermite spline for the given parameters.
    /// Returns the resulting coefficients p(t) = at^3 + bt^2 + ct + d in order.
    fn solve_cubic_hermite(y0: T, yn: T, m0: T, mn: T) -> [T; 4] {
        let two = T::from(2.0).unwrap();
        let three = T::from(3.0).unwrap();

        // Coefficients for p(t) = at^3 + bt^2 + ct + d on t in [0, 1]
        let d = y0;
        let c = m0;
        let b = three * (yn - y0) - (two * m0 + mn);
        let a = (two * y0 - two * yn) + (m0 + mn);

        [d, c, b, a]
    }

    /// Creates a new Interpolant for the given points and the free variables.
    /// This will create points.len() - 1 IFSMaps internally to map every point to other.
    /// Currently by default it uses a mixture between fractal interpolation and cubic hermite
    /// spline.
    /// # Arguments
    /// * `points` - Strictly sorted 2d points. If not passed sorted created interpolant will not
    /// give correct results.
    /// * 'free_variables' - Free variables or vertical scaling factors for the IFSMaps. They
    /// should be either a FreeVariables::Scalar or a FreeVariables::Array containing an array with
    /// the size points.len() - 1. Each variable should be in the range (-1, 1). A scalar 0 for the
    /// free variable would make the Interpolant a cubic hermite spline negating all the fractal
    /// interpolation.
    /// * 'iterations' - The maximum iteration count used in the evaluation methods.
    pub fn new(points: &[Point2<T>], free_variables: FreeVariables<T>, iterations: usize) -> Self {
        let n = points.len();

        if n <= 1 {
            panic!("More than one point is required to create the Interpolant.");
        }

        if let FreeVariables::Array(array) = &free_variables {
            if array.len() != points.len() - 1 {
                panic!(
                    "Invalid array size for the free_variables. It should be an array with the size points.len() - 1."
                );
            }
        }

        let first_point = points.first().unwrap();
        let last_point = points.last().unwrap();
        let total_x_range = last_point.x - first_point.x;

        // Find the derivatives for the points.
        let mut ks = Vec::with_capacity(n);
        // k0 using finite difference (forward)
        ks.push((points[1].y - points[0].y) / (points[1].x - points[0].x));
        // Interior points using central difference
        for i in 1..n - 1 {
            ks.push((points[i + 1].y - points[i - 1].y) / (points[i + 1].x - points[i - 1].x));
        }
        // kn using finite difference (backward)
        ks.push((points[n - 1].y - points[n - 2].y) / (points[n - 1].x - points[n - 2].x));

        // Build maps in parallel.
        let maps = (0..(n - 1))
            .into_par_iter()
            .map(|i| {
                let p = points[i];
                let p_next = points[i + 1];

                let di = match &free_variables {
                    FreeVariables::Scalar(variable) => *variable,
                    FreeVariables::Array(array) => array[i],
                };

                // Fractal affine map parameters.
                let a = (p_next.x - p.x) / total_x_range;
                let e = (last_point.x * p.x - first_point.x * p_next.x) / total_x_range;

                // Values for the polynomial q in the fractal-spline formulation (these are
                // the function values after subtracting the d_i * endpoint contributions).
                let y_start = p.y - di * first_point.y;
                let y_end = p_next.y - di * last_point.y;

                let k_start_global = a * ks[i] - di * ks[0];
                let k_end_global = a * ks[i + 1] - di * ks[n - 1];

                let k_start_local = k_start_global * a;
                let k_end_local = k_end_global * a;

                // Find the q which is the coefficients for the hermite spline.
                let q = Self::solve_cubic_hermite(y_start, y_end, k_start_local, k_end_local);

                IFSMap {
                    a,
                    d: di,
                    e,
                    q,
                    end_x: p_next.x,
                }
            })
            .collect();

        Self {
            maps,
            iterations,
            first_point: *first_point,
            last_point: *last_point,
        }
    }

    /// Returns the MSE(Mean Sqaure Error) of the Interpolant calculated from the given test
    /// points.
    pub fn get_mse(&self, test_points: &[Point2<T>]) -> T {
        let n = T::from(test_points.len()).unwrap();

        test_points
            .par_iter()
            .map(|p: &Point2<T>| -> T {
                let y_pred = self.evaluate(p.x);
                let y_true = p.y;
                let diff = y_pred - y_true;
                diff * diff
            })
            .sum::<T>()
            / n
    }
}

impl<T> Interpolant for Interpolant1D<T>
where
    T: Float + Debug + AddAssign + MulAssign + Send + Sync + DeviceCopy,
{
    type Scalar = T;
    
    /// Calculates the definite integral of the fractal function from the first
    /// point to the last point.
    fn integrate(&self) -> Self::Scalar {
        let mut sum_q_scaled = T::zero();
        let mut sum_ad = T::zero();

        for map in &self.maps {
            // Integral of the cubic polynomial q.
            let q_int = map.q[0] 
                + map.q[1] / T::from(2.0).unwrap() 
                + map.q[2] / T::from(3.0).unwrap() 
                + map.q[3] / T::from(4.0).unwrap();

            // Summing up the scaled integrals and the d * a products.
            sum_q_scaled += q_int * map.a; 
            sum_ad += map.a * map.d;
        }

        sum_q_scaled / (T::one() - sum_ad)
    }

    /// Evaluates a single point and returns the result.
    /// The point should be in the range of the Interpolant. It will get clamped if its not.
    fn evaluate(&self, mut x: Self::Scalar) -> Self::Scalar {
        let p0_x = self.first_point.x;
        let pn_x = self.last_point.x;

        // Boundary checks
        if x <= p0_x {
            return self.first_point.y;
        }
        if x >= pn_x {
            return self.last_point.y;
        }

        let mut y_accumulated = T::zero();
        let mut d_product = T::one();

        for _ in 0..self.iterations {
            let map_idx = self.maps.partition_point(|m| m.end_x <= x);
            let map = &self.maps[map_idx];

            let start_x = if map_idx == 0 {
                p0_x
            } else {
                self.maps[map_idx - 1].end_x
            };
            // Find the normalized local coordinate.
            let u = (x - start_x) / (map.end_x - start_x);

            // Evaluate q(u) using Horner's Method
            let mut q_val = T::zero();
            for &coeff in map.q.iter().rev() {
                q_val = q_val * u + coeff;
            }

            y_accumulated += d_product * q_val;
            d_product *= map.d;

            // Find the x for the next iteration. Clamp the values so it doesn't explode.
            let mut x_prev = (x - map.e) / map.a;
            if x_prev < p0_x {
                x_prev = p0_x;
            }
            if x_prev > pn_x {
                x_prev = pn_x;
            }

            x = x_prev;

            // After d_product gets this small there is no need to continue the iterations.
            // Because it doesn't affect the result.
            if d_product.abs() < T::from(f64::EPSILON * 100.0).unwrap() {
                break;
            }
        }
        y_accumulated
    }

    /// Evaluates given points in parallel. And returns the result in a Vec.
    /// Basicaly calls evaluate function for each point.
    fn evaluate_many(&self, points: &[Self::Scalar]) -> Vec<Self::Scalar> {
        points.par_iter().map(|&x| self.evaluate(x)).collect()
    }

    /// Evaluates given points in the GPU using CUDA.
    /// It launches the kernel for the GPU and copies the points.
    /// So there is a big overhead to this function every time it gets called.
    /// It might only makes sense using it for very large amounts of data.
    #[cfg(feature = "cuda")]
    fn evaluate_gpu(&self, points: &[T]) -> Result<Vec<T>, Box<dyn std::error::Error>> {
        // Initialize CUDA with default flags
        cust::init(cust::CudaFlags::empty())?;
        let device = Device::get_device(0)?;

        let _ctx = Context::new(device)?;

        // Load the ptx created in the build.
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
        let module = Module::from_ptx(ptx, &[])?;

        // Select the function.
        let kernel_name = if std::mem::size_of::<T>() == 4 {
            "interpolate_f32"
        } else {
            "interpolate_f64"
        };
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let kernel = module.get_function(kernel_name)?;

        // Allocate buffers.
        let gpu_maps = DeviceBuffer::from_slice(&self.maps)?;
        let gpu_inputs = DeviceBuffer::from_slice(points)?;
        let gpu_outputs = unsafe { DeviceBuffer::uninitialized(points.len())? };

        // Select  (hard coded for now).
        let threads_per_block = 256;
        // Use .len() on the slice passed into the function
        let blocks_per_grid = (points.len() + threads_per_block - 1) / threads_per_block;

        // Launch kernel
        unsafe {
            launch!(
                kernel<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                    gpu_maps.as_device_ptr(),
                    gpu_maps.len(),
                    gpu_inputs.as_device_ptr(),
                    gpu_inputs.len(),
                    gpu_outputs.as_device_ptr(),
                    self.iterations,
                    self.first_point.x,
                    self.first_point.y,
                    self.last_point.x,
                    self.last_point.y
                )
            )?;
        }

        // Retrieve the results.
        stream.synchronize()?;
        let mut results = vec![T::zero(); points.len()];
        gpu_outputs.copy_to(&mut results)?;

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::f64;

    use assert_approx_eq::assert_approx_eq;

    use super::*;

    fn test_with_function<T: Fn(f64) -> f64>(func: T) {
        let n = 10000;
        let mut points = Vec::<Point2<f64>>::with_capacity(n);
        for i in 0..n {
            let x = i as f64 / n as f64;
            points.push(Point2::new(x, func(x)));
        }

        let interpolant = Interpolant1D::new(&points, FreeVariables::Scalar(0.01), 100);

        points.iter().for_each(|point| {
            let value = interpolant.evaluate(point.x);
            assert_approx_eq!(value, point.y);
        });

        let test_points_n = n * 10;

        let mut mse = 0.0;
        for i in 0..test_points_n {
            let x = i as f64 / test_points_n as f64;
            let value = interpolant.evaluate(x);

            let diff = value - func(x);
            mse += diff * diff;
            //assert_approx_eq!(value, func(x), f64::EPSILON * 1000.0);
        }
        mse /= test_points_n as f64;
        assert_approx_eq!(mse, f64::EPSILON * 100.0);
    }

    #[test]
    fn interpolant1d_sine_wave_evaluate_works() {
        test_with_function(|x| x.sin())
    }

    #[test]
    fn interpolant1d_irregular_data_evaluate_works() {
        test_with_function(|x| {
            let mut product = 1.0;

            for n in 1..=1000 {
                let magnitude = 0.5f64.powi(n);

                if 1.0 + magnitude == 1.0 {
                    break;
                }

                let angle = 6.0f64.powi(n) * f64::consts::PI * x;
                let term = 1.0 + magnitude * angle.sin();

                product *= term;
            }

            product
        })
    }
}
