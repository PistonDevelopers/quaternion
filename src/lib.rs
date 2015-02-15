#![deny(missing_docs)]

//! A simple and type agnostic quaternion math designed for reexporting

extern crate vecmath;

use std::num::Float;

/// Quaternion type alias.
pub type Quaternion<T> = [T; 4];

pub use vecmath::vec4_add as add;

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

    let r1 = a[0];
    let r2 = b[0];
    let v1 = [a[1], a[2], a[3]];
    let v2 = [b[1], b[2], b[3]];
    let scalar = r1 * r2 - dot(v1, v2);
    let vector = add(
        add(scale(v2, r1), scale(v1, r2)),
        cross(v1, v2)
    );
    [scalar, vector[0], vector[1], vector[2]]
}

/// Takes the quaternion conjugate.
#[inline(always)]
pub fn conj<T: Float>(a: Quaternion<T>) -> Quaternion<T> {
    [a[0], -a[1], -a[2], -a[3]]
}

