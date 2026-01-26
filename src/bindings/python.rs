use std::sync::Arc;

use nalgebra::Point2;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::interpolation::one_d::Interpolant1D;
use crate::interpolation::{FreeVariables, Interpolant};

/// Internal enum to hold either f32 or f64 interpolant
enum InterpolantInner {
    F32(Arc<dyn Interpolant<Scalar = f32> + Send + Sync>),
    F64(Arc<dyn Interpolant<Scalar = f64> + Send + Sync>),
}

/// Wrapper around the Interpolant trait for Python access
#[pyclass(name = "Interpolant")]
pub struct PyInterpolant {
    inner: InterpolantInner,
}

/*
* TODO: Currenlty we use dynamic enum dispatch for separating single and double precision
* implementations. But the implicit conversions in the evaluation_many function should be avoided.
* */
#[pymethods]
impl PyInterpolant {
    /// Evaluate the interpolant at a single point.
    /// Accepts f32 or f64, returns the same type.
    fn evaluate(&self, py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
        // Try f32 first, then f64
        if let Ok(x_f32) = x.extract::<f32>(py) {
            let result = match &self.inner {
                InterpolantInner::F32(interp) => interp.evaluate(x_f32),
                InterpolantInner::F64(interp) => interp.evaluate(x_f32 as f64) as f32,
            };
            Ok(result.into_pyobject(py)?.into_any().unbind())
        } else if let Ok(x_f64) = x.extract::<f64>(py) {
            let result = match &self.inner {
                InterpolantInner::F32(interp) => interp.evaluate(x_f64 as f32) as f64,
                InterpolantInner::F64(interp) => interp.evaluate(x_f64),
            };
            Ok(result.into_pyobject(py)?.into_any().unbind())
        } else {
            Err(PyValueError::new_err("x must be a float (f32 or f64)"))
        }
    }

    /// Evaluate the interpolant at multiple points.
    /// Accepts numpy array of f32 or f64, returns the same type.
    fn evaluate_many(&self, py: Python<'_>, points: Py<PyAny>) -> PyResult<Py<PyAny>> {
        // Try f32 array first, then f64
        if let Ok(arr) = points.extract::<PyReadonlyArray1<f32>>(py) {
            let points_slice = arr.as_slice().unwrap();
            let result: Vec<f32> = match &self.inner {
                InterpolantInner::F32(interp) => interp.evaluate_many(points_slice),
                InterpolantInner::F64(interp) => {
                    let f64_points: Vec<f64> = points_slice.iter().map(|&x| x as f64).collect();
                    interp
                        .evaluate_many(&f64_points)
                        .into_iter()
                        .map(|y| y as f32)
                        .collect()
                }
            };
            Ok(result.into_pyarray(py).into_any().unbind())
        } else if let Ok(arr) = points.extract::<PyReadonlyArray1<f64>>(py) {
            let points_slice = arr.as_slice().unwrap();
            let result: Vec<f64> = match &self.inner {
                InterpolantInner::F32(interp) => {
                    let f32_points: Vec<f32> = points_slice.iter().map(|&x| x as f32).collect();
                    interp
                        .evaluate_many(&f32_points)
                        .into_iter()
                        .map(|y| y as f64)
                        .collect()
                }
                InterpolantInner::F64(interp) => interp.evaluate_many(points_slice),
            };
            Ok(result.into_pyarray(py).into_any().unbind())
        } else {
            Err(PyValueError::new_err(
                "points must be a numpy array of f32 or f64",
            ))
        }
    }

    /// Calculate Mean Squared Error against test points.
    /// Points array format: [x0, y0, x1, y1, ...]
    fn get_mse(&self, py: Python<'_>, points: Py<PyAny>) -> PyResult<f64> {
        // Try f32 array first, then f64
        if let Ok(arr) = points.extract::<PyReadonlyArray1<f32>>(py) {
            let points_slice = arr.as_slice().unwrap();
            if points_slice.len() % 2 != 0 {
                return Err(PyValueError::new_err(
                    "Points array must have an even number of elements (x, y pairs)",
                ));
            }
            let test_points: Vec<Point2<f32>> = points_slice
                .chunks(2)
                .map(|chunk| Point2::new(chunk[0], chunk[1]))
                .collect();

            let mse: f64 = match &self.inner {
                InterpolantInner::F32(interp) => test_points
                    .iter()
                    .map(|p| {
                        let diff = interp.evaluate(p.x) - p.y;
                        (diff * diff) as f64
                    })
                    .sum::<f64>()
                    / test_points.len() as f64,
                InterpolantInner::F64(interp) => test_points
                    .iter()
                    .map(|p| {
                        let diff = interp.evaluate(p.x as f64) - p.y as f64;
                        diff * diff
                    })
                    .sum::<f64>()
                    / test_points.len() as f64,
            };
            Ok(mse)
        } else if let Ok(arr) = points.extract::<PyReadonlyArray1<f64>>(py) {
            let points_slice = arr.as_slice().unwrap();
            if points_slice.len() % 2 != 0 {
                return Err(PyValueError::new_err(
                    "Points array must have an even number of elements (x, y pairs)",
                ));
            }
            let test_points: Vec<Point2<f64>> = points_slice
                .chunks(2)
                .map(|chunk| Point2::new(chunk[0], chunk[1]))
                .collect();

            let mse: f64 = match &self.inner {
                InterpolantInner::F32(interp) => test_points
                    .iter()
                    .map(|p| {
                        let diff = interp.evaluate(p.x as f32) as f64 - p.y;
                        diff * diff
                    })
                    .sum::<f64>()
                    / test_points.len() as f64,
                InterpolantInner::F64(interp) => test_points
                    .iter()
                    .map(|p| {
                        let diff = interp.evaluate(p.x) - p.y;
                        diff * diff
                    })
                    .sum::<f64>()
                    / test_points.len() as f64,
            };
            Ok(mse)
        } else {
            Err(PyValueError::new_err(
                "points must be a numpy array of f32 or f64",
            ))
        }
    }

    fn __call__(&self, py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.evaluate(py, x)
    }
}

/// Creates a 1D fractal interpolant from the given points.
///
/// # Arguments
/// * `points` - A flat numpy array of (x, y) pairs, e.g., [x0, y0, x1, y1, ...]
///   Can be f32 or f64.
/// * `free_variable` - Either a scalar or an array of free variables.
///   - Scalar: applies the same free variable to all segments
///   - Array: must have n-1 elements for n points (one per segment)
///   Should be in the range (-1, 1). A value of 0 makes it a pure cubic Hermite spline.
/// * `iterations` - The maximum iteration count used in evaluation.
///
/// # Returns
/// A PyInterpolant object that can be called or used to evaluate points.
#[pyfunction]
fn interpolate(
    py: Python<'_>,
    points: Py<PyAny>,
    free_variable: Py<PyAny>,
    iterations: usize,
) -> PyResult<PyInterpolant> {
    // Try to extract as f32 array first, then f64
    if let Ok(arr) = points.extract::<PyReadonlyArray1<f32>>(py) {
        let points_slice = arr.as_slice().unwrap();

        if points_slice.len() % 2 != 0 {
            return Err(PyValueError::new_err(
                "Points array must have an even number of elements (x, y pairs)",
            ));
        }

        let num_points = points_slice.len() / 2;
        if num_points <= 1 {
            return Err(PyValueError::new_err(
                "More than one point is required to create the Interpolant.",
            ));
        }

        // Parse free_variable
        let free_vars: FreeVariables<f32> = if let Ok(scalar) = free_variable.extract::<f32>(py) {
            FreeVariables::Scalar(scalar)
        } else if let Ok(fv_arr) = free_variable.extract::<PyReadonlyArray1<f32>>(py) {
            let fv_slice = fv_arr.as_slice().unwrap();
            if fv_slice.len() != num_points - 1 {
                return Err(PyValueError::new_err(format!(
                    "Free variables array must have {} elements (one per segment), got {}",
                    num_points - 1,
                    fv_slice.len()
                )));
            }
            FreeVariables::Array(fv_slice.to_vec())
        } else {
            return Err(PyValueError::new_err(
                "free_variable must be a float or a numpy array matching points dtype",
            ));
        };

        let point_vec: Vec<Point2<f32>> = points_slice
            .chunks(2)
            .map(|chunk| Point2::new(chunk[0], chunk[1]))
            .collect();

        let interpolant = Interpolant1D::new(&point_vec, free_vars, iterations);

        Ok(PyInterpolant {
            inner: InterpolantInner::F32(Arc::new(interpolant)),
        })
    } else if let Ok(arr) = points.extract::<PyReadonlyArray1<f64>>(py) {
        let points_slice = arr.as_slice().unwrap();

        if points_slice.len() % 2 != 0 {
            return Err(PyValueError::new_err(
                "Points array must have an even number of elements (x, y pairs)",
            ));
        }

        let num_points = points_slice.len() / 2;
        if num_points <= 1 {
            return Err(PyValueError::new_err(
                "More than one point is required to create the Interpolant.",
            ));
        }

        // Parse free_variable
        let free_vars: FreeVariables<f64> = if let Ok(scalar) = free_variable.extract::<f64>(py) {
            FreeVariables::Scalar(scalar)
        } else if let Ok(fv_arr) = free_variable.extract::<PyReadonlyArray1<f64>>(py) {
            let fv_slice = fv_arr.as_slice().unwrap();
            if fv_slice.len() != num_points - 1 {
                return Err(PyValueError::new_err(format!(
                    "Free variables array must have {} elements (one per segment), got {}",
                    num_points - 1,
                    fv_slice.len()
                )));
            }
            FreeVariables::Array(fv_slice.to_vec())
        } else {
            return Err(PyValueError::new_err(
                "free_variable must be a float or a numpy array matching points dtype",
            ));
        };

        let point_vec: Vec<Point2<f64>> = points_slice
            .chunks(2)
            .map(|chunk| Point2::new(chunk[0], chunk[1]))
            .collect();

        let interpolant = Interpolant1D::new(&point_vec, free_vars, iterations);

        Ok(PyInterpolant {
            inner: InterpolantInner::F64(Arc::new(interpolant)),
        })
    } else {
        Err(PyValueError::new_err(
            "points must be a numpy array of f32 or f64",
        ))
    }
}

#[pymodule]
fn ffinterp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interpolate, m)?)?;
    m.add_class::<PyInterpolant>()?;

    Ok(())
}
