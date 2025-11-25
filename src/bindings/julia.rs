use core::panic;
use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign};
use std::sync::Arc;

use jlrs::convert::unbox::Unbox;
use jlrs::data::layout::is_bits::IsBits;
use jlrs::data::layout::valid_layout::{ValidField, ValidLayout};
use jlrs::data::managed::array::TypedArray;
use jlrs::data::managed::value::typed::{TypedValue, TypedValueRet};
use jlrs::data::types::abstract_type::Integer;
use jlrs::data::types::construct_type::ConstructType;
use jlrs::data::types::foreign_type::OpaqueType;
use jlrs::{prelude::*, weak_handle};
use nalgebra::Point2;
use num::Float;

use crate::interpolation::one_d::Interpolant1D;
use crate::interpolation::{FreeVariables, Interpolant};
use jlrs::data::managed::array::{TypedVector, TypedVectorRet};
#[derive(OpaqueType, Clone)]
#[allow(dead_code)]
#[jlrs(bounds = "T <: ::jlrs::data::types::abstract_type::AbstractFloat")]
struct InterpolantHandle<T> {
    interpolant: Arc<dyn Interpolant<Scalar = T> + Send + Sync>,
}

impl<T> InterpolantHandle<T>
where
    T: 'static
        + Send
        + Sync
        + Debug
        + Clone
        + Float
        + ValidLayout
        + IsBits
        + ValidField
        + ConstructType,
{
    fn evaluate(&self, x: T) -> T {
        self.interpolant.evaluate(x)
    }

    fn evaluate_many(&self, points: TypedVector<T>) -> TypedVectorRet<T> {
        let points_slice = unsafe {
            let raw_ptr = points.data_ptr().cast::<T>();
            let length = points.length();

            std::slice::from_raw_parts(raw_ptr, length)
        };

        let result = self.interpolant.evaluate_many(points_slice);
        let length = result.len();

        weak_handle!().unwrap().local_scope::<_, 1>(|mut frame| {
            let data_array = TypedVector::from_vec(&mut frame, result, length)
                .unwrap()
                .unwrap();

            data_array.leak()
        })
    }
}

trait InterpolationMethod<T> {
    fn new(free_variables: FreeVariables<T>) -> Self;
}

#[derive(OpaqueType, Clone)]
#[allow(dead_code)]
struct Interp1D<T> {
    free_variables: FreeVariables<T>,
}
impl<T> InterpolationMethod<T> for Interp1D<T> {
    fn new(free_variables: FreeVariables<T>) -> Self {
        Self { free_variables }
    }
}

#[derive(OpaqueType)]
#[allow(dead_code)]
struct HInterp1D<T> {
    free_variables: FreeVariables<T>,
}

impl<T> InterpolationMethod<T> for HInterp1D<T> {
    fn new(free_variables: FreeVariables<T>) -> Self {
        Self { free_variables }
    }
}

#[derive(OpaqueType)]
#[allow(dead_code)]
struct Interp2D<T> {
    free_variables: FreeVariables<T>,
}

impl<T> InterpolationMethod<T> for Interp2D<T> {
    fn new(free_variables: FreeVariables<T>) -> Self {
        Self { free_variables }
    }
}

#[derive(OpaqueType)]
#[allow(dead_code)]
struct HInterp2D<T> {
    free_variables: FreeVariables<T>,
}

impl<T> InterpolationMethod<T> for HInterp2D<T> {
    fn new(free_variables: FreeVariables<T>) -> Self {
        Self { free_variables }
    }
}

fn create_interp<T, I>(fv: FreeVariables<T>) -> TypedValueRet<I>
where
    T: Send + Sync + 'static + ConstructType,
    I: OpaqueType + InterpolationMethod<T>,
{
    weak_handle!().unwrap().local_scope::<_, 1>(|mut frame| {
        let data = I::new(fv);
        TypedValue::new(&mut frame, data).leak()
    })
}

fn new_interp_scalar<T, I>(input: T) -> TypedValueRet<I>
where
    T: Send + Sync + 'static + Clone + ConstructType + Unbox<Output = T>,
    I: OpaqueType + InterpolationMethod<T>,
{
    create_interp::<T, I>(FreeVariables::Scalar(input))
}

fn new_interp_array<T, I>(input: TypedArray<T>) -> TypedValueRet<I>
where
    T: Send + Sync + 'static + Clone + ConstructType + ValidField,
    I: OpaqueType + InterpolationMethod<T>,
{
    unsafe {
        let data = input.inline_data();
        create_interp::<T, I>(FreeVariables::Array(data.as_slice().to_vec()))
    }
}

julia_module! {
    become ffinterp_init;

    for T in [f32, f64] {
        struct Interp1D<T>;
        struct HInterp1D<T>;
        struct Interp2D<T>;
        struct HInterp2D<T>;

        fn new_interp_scalar<T, Interp1D>(input: T) -> TypedValueRet<Interp1D<T>> as Interp1D;
        fn new_interp_array<T, Interp1D>(input: TypedArray<T>) -> TypedValueRet<Interp1D<T>> as Interp1D;

        fn new_interp_scalar<T, HInterp1D>(input: T) -> TypedValueRet<HInterp1D<T>> as HInterp1D;
        fn new_interp_array<T, HInterp1D>(input: TypedArray<T>) -> TypedValueRet<HInterp1D<T>> as HInterp1D;

        fn new_interp_scalar<T, Interp2D>(input: T) -> TypedValueRet<Interp2D<T>> as Interp2D;
        fn new_interp_array<T, Interp2D>(input: TypedArray<T>) -> TypedValueRet<Interp2D<T>> as Interp2D;

        fn new_interp_scalar<T, HInterp2D>(input: T) -> TypedValueRet<HInterp2D<T>> as HInterp2D;
        fn new_interp_array<T, HInterp2D>(input: TypedArray<T>) -> TypedValueRet<HInterp2D<T>> as HInterp2D;

        struct InterpolantHandle<T> as Interpolant;
        in InterpolantHandle<T> fn evaluate(&self, x: T) -> T;
        in InterpolantHandle<T> fn evaluate_many(&self, points: TypedVector<T>) -> TypedVectorRet<T>;

        fn interpolate<T>(
            points: TypedVector<T>,
            method: Value,
            f0: Value,
            niter: TypedValue<Integer>
        ) -> TypedValueRet<InterpolantHandle<T>>;
    };
}

fn interpolate<T>(
    pts: TypedVector<T>,
    method: Value,
    f0: Value,
    niter: TypedValue<Integer>,
) -> TypedValueRet<InterpolantHandle<T>>
where
    InterpolantHandle<T>: OpaqueType,
    T: 'static
        + Send
        + Sync
        + Debug
        + ConstructType
        + Unbox<Output = T>
        + Clone
        + Float
        + AddAssign
        + MulAssign,
{
    let points_slice = unsafe {
        let raw_ptr = pts.data_ptr().cast::<Point2<T>>();
        let len = pts.length() / 2;

        std::slice::from_raw_parts(raw_ptr, len)
    };

    let iterations = niter.unbox::<isize>().unwrap() as usize;

    if method.is::<Interp1D<T>>() {
        let interp1d = method.unbox::<Interp1D<T>>().unwrap();

        let interpolant = Interpolant1D::new(points_slice, interp1d.free_variables, iterations);

        let interpolant_handle = InterpolantHandle {
            interpolant: Arc::new(interpolant),
        };

        weak_handle!()
            .unwrap()
            .local_scope::<_, 1>(|mut frame| TypedValue::new(&mut frame, interpolant_handle).leak())
    } else if method.is::<HInterp1D<T>>() {
        unimplemented!()
    } else if method.is::<Interp2D<T>>() {
        unimplemented!()
    } else if method.is::<HInterp2D<T>>() {
        unimplemented!()
    } else {
        panic!()
    }
}
