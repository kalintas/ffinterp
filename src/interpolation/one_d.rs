use std::{
    fmt::Debug,
    iter::Sum,
    ops::{AddAssign, MulAssign},
};

use nalgebra::Point2;
use num::Float;
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

        // Add debug assertions since we don't want to add extra overhead
        // to the library checking the parameters in the release mode.
        #[cfg(debug_assertions)]
        {
            // Check free variables. They should be in the range (-1, 1).
            let one = T::one();
            let neg_one = -one;
            match &free_variables {
                FreeVariables::Scalar(s) => {
                    debug_assert!(
                        *s > neg_one && *s < one,
                        "Free variable must be in range (-1, 1), found {:?}",
                        s
                    );
                }
                FreeVariables::Array(arr) => {
                    for (i, &val) in arr.iter().enumerate() {
                        debug_assert!(
                            val > neg_one && val < one,
                            "Free variable at index {} must be in range (-1, 1), found {:?}",
                            i,
                            val
                        );
                    }
                }
            }

            // Check if the given points are strictly sorted.
            // By taking the points as sorted we leave the decision of needing to
            // sort the data to caller.
            for pair in points.windows(2) {
                debug_assert!(
                    pair[1].x > pair[0].x,
                    "Points must be strictly sorted by x-coordinate. Found x={:?} followed by x={:?}",
                    pair[0].x,
                    pair[1].x
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

    /// Returns the MSE (Mean Square Error) of the Interpolant calculated from the given test
    /// points.
    pub fn get_mse(&self, test_points: &[Point2<T>]) -> T {
        let interp_points: Vec<Point2<T>> = test_points
            .iter()
            .map(|p| Point2::new(p.x, self.evaluate(p.x)))
            .collect();

        crate::metrics::mse(&interp_points, test_points)
    }

    /// Returns the symmetric Hausdorff distance between the interpolant curve and the given test points.
    /// It is better suited for fractal interpolation than MSE as it tries to match the shape of the
    /// curve rather than the values of the curve.
    pub fn get_hausdorff(&self, test_points: &[Point2<T>]) -> T {
        let interp_points: Vec<Point2<T>> = test_points
            .iter()
            .map(|p| Point2::new(p.x, self.evaluate(p.x)))
            .collect();

        crate::metrics::hausdorff(&interp_points, test_points)
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
            let map_idx = map_idx.min(self.maps.len() - 1);
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

    use crate::test_functions::{weierstrass, weierstrass_integral, wen};

    use super::*;

    /// Interpolant creation tests
    #[test]
    #[should_panic(expected = "More than one point is required")]
    fn test_new_with_insufficient_points() {
        let points = vec![Point2::new(0.0, 0.0)];
        let _ = Interpolant1D::new(&points, FreeVariables::Scalar(0.0), 10);
    }

    #[test]
    #[should_panic(expected = "Invalid array size for the free_variables")]
    fn test_new_with_mismatched_free_variables() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
        ];
        let invalid_vars = FreeVariables::Array(vec![0.1, 0.2, 0.3]);
        let _ = Interpolant1D::new(&points, invalid_vars, 10);
    }

    #[test]
    fn test_evaluate_clamping() {
        let points = vec![Point2::new(0.0, 10.0), Point2::new(1.0, 20.0)];
        let interpolant = Interpolant1D::new(&points, FreeVariables::Scalar(0.0), 10);

        assert_eq!(interpolant.evaluate(-1.0), 10.0);
        assert_eq!(interpolant.evaluate(2.0), 20.0);
    }

    #[test]
    fn test_valid_array_free_variables() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 0.0),
        ];
        // 2 intervals, 2 variables. This should NOT panic.
        let valid_vars = FreeVariables::Array(vec![0.1, 0.5]);
        let _ = Interpolant1D::new(&points, valid_vars, 10);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Points must be strictly sorted")]
    fn test_unsorted_points_panic() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 1.0), // x=2.0
            Point2::new(1.0, 0.5), // x=1.0
        ];
        let _ = Interpolant1D::new(&points, FreeVariables::Scalar(0.0), 10);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Points must be strictly sorted")]
    fn test_duplicate_x_panic() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 2.0), // Duplicate x
        ];
        let _ = Interpolant1D::new(&points, FreeVariables::Scalar(0.0), 10);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Free variable must be in range (-1, 1)")]
    fn test_invalid_scalar_range_panic() {
        let points = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)];
        let _ = Interpolant1D::new(&points, FreeVariables::Scalar(1.0), 10); // Boundary 1.0
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Free variable at index 1 must be in range (-1, 1)")]
    fn test_invalid_array_range_panic() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
        ];
        let invalid_vars = FreeVariables::Array(vec![0.5, -1.2]); // Second value out of range
        let _ = Interpolant1D::new(&points, invalid_vars, 10);
    }

    // Interpolation evaluation tests
    #[test]
    fn test_interpolation_at_original_points_large_set() {
        let points: Vec<Point2<f64>> = (0..100)
            .map(|i| i as f64 * 0.1)
            .map(|x| Point2::new(x, x.sin()))
            .collect();

        let interpolant = Interpolant1D::new(&points, FreeVariables::Scalar(0.0), 50);

        // Verify that the interpolant hits EVERY input point exactly
        for p in points.iter() {
            let evaluated = interpolant.evaluate(p.x);

            assert_approx_eq!(evaluated, p.y, f64::EPSILON * 100.0);
        }
    }

    // Integral tests
    fn test_integral_against_curve(
        points: &[Point2<f64>],
        free_variable: f64,
        expected_integral: f64,
        epsilon: f64,
    ) {
        let iterations = 50;
        let interpolant =
            Interpolant1D::new(points, FreeVariables::Scalar(free_variable), iterations);

        let calculated = interpolant.integrate();

        assert_approx_eq!(calculated, expected_integral, epsilon);
    }

    fn add_koch_points(
        p1: Point2<f64>,
        p2: Point2<f64>,
        depth: usize,
        points: &mut Vec<Point2<f64>>,
    ) {
        if depth == 0 {
            points.push(p1);
        } else {
            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;

            let s = Point2::new(p1.x + dx / 3.0, p1.y + dy / 3.0);
            let e = Point2::new(p1.x + 2.0 * dx / 3.0, p1.y + 2.0 * dy / 3.0);

            let h = 3.0f64.sqrt() / 6.0;
            let v = Point2::new((p1.x + p2.x) / 2.0 - h * dy, (p1.y + p2.y) / 2.0 + h * dx);

            add_koch_points(p1, s, depth - 1, points);
            add_koch_points(s, v, depth - 1, points);
            add_koch_points(v, e, depth - 1, points);
            add_koch_points(e, p2, depth - 1, points);
        }
    }

    #[test]
    fn test_integral_koch_snowflake() {
        let mut points = Vec::new();
        let start = Point2::new(0.0, 0.0);
        let end = Point2::new(1.0, 0.0);
        add_koch_points(start, end, 5, &mut points);
        points.push(end);

        points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        points.dedup_by(|a, b| (a.x - b.x).abs() < 1e-12);

        // Theoretical Area: sqrt(3) / 20
        let expected = 3.0f64.sqrt() / 20.0;

        test_integral_against_curve(&points, 0.1, expected, 1e-4);
    }

    #[test]
    fn test_integral_against_weierstrass() {
        let n = 2000;
        let (x_start, x_end) = (0.0, 1.0);

        let points: Vec<Point2<f64>> = (0..=n)
            .map(|i| {
                let x = x_start + (x_end - x_start) * (i as f64 / n as f64);
                Point2::new(x, weierstrass(x))
            })
            .collect();

        let expected = weierstrass_integral(x_end) - weierstrass_integral(x_start);
        test_integral_against_curve(&points, 0.1, expected, 1e-4);
    }

    #[test]
    fn test_integral_against_wen() {
        let n = 5000;
        let (x_start, x_end) = (0.0, 2.0);

        let points: Vec<Point2<f64>> = (0..=n)
            .map(|i| {
                let x = x_start + (x_end - x_start) * (i as f64 / n as f64);
                Point2::new(x, wen(x))
            })
            .collect();

        test_integral_against_curve(&points, 0.1, 2.0, 1e-3);
    }
}
