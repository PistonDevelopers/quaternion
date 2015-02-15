#![deny(missing_docs)]

//! A simple and type agnostic quaternion math library designed for reexporting

extern crate vecmath;

use std::num::Float;

/// Quaternion type alias.
pub type Quaternion<T> = (T, [T; 3]);

/// Adds two quaternions.
#[inline(always)]
pub fn add<T: Float>(
    a: Quaternion<T>,
    b: Quaternion<T>
) -> Quaternion<T> {
    use vecmath::vec3_add as add;
    (a.0 + b.0, add(a.1, b.1))
}

/// Multiplies two quaternions.
#[inline(always)]
pub fn mul<T: Float + Copy>(
    a: Quaternion<T>,
    b: Quaternion<T>
) -> Quaternion<T> {
    use vecmath::vec3_cross as cross;
    use vecmath::vec3_add as add;
    use vecmath::vec3_dot as dot;
    use vecmath::vec3_scale as scale;

    (
        a.0 * b.0 - dot(a.1, b.1),
        add(
            add(scale(b.1, a.0), scale(a.1, b.0)),
            cross(a.1, b.1)
        )
    )
}

/// Takes the quaternion conjugate.
#[inline(always)]
pub fn conj<T: Float>(a: Quaternion<T>) -> Quaternion<T> {
    use vecmath::vec3_neg as neg;

    (a.0, neg(a.1))
}

