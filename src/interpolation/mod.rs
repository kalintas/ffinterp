use std::fmt::Debug;

use num::Float;

#[cfg(feature = "cuda")]
use cust::DeviceCopy;

pub mod one_d;

#[derive(Clone, Debug)]
pub enum FreeVariables<T> {
    Scalar(T),
    Array(Vec<T>),
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "cuda", derive(DeviceCopy))]
struct IFSMap<T> {
    a: T,
    d: T,
    e: T,
    q: [T; 4],
    end_x: T,
}

pub trait Interpolant {
    type Scalar: Float + Clone + Debug + 'static;

    fn evaluate(&self, x: Self::Scalar) -> Self::Scalar;
    fn evaluate_many(&self, points: &[Self::Scalar]) -> Vec<Self::Scalar>;

    #[cfg(feature = "cuda")]
    fn evaluate_gpu(
        &self,
        points: &[Self::Scalar],
    ) -> Result<Vec<Self::Scalar>, Box<dyn std::error::Error>>;
}
