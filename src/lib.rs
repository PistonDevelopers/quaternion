#![deny(missing_docs)]

//! A simple and type agnostic quaternion math library designed for reexporting

extern crate vecmath;

use vecmath::Vector3;
use std::num::{Float, FromPrimitive};

/// Quaternion type alias.
pub type Quaternion<T> = (T, [T; 3]);

/// Constructs identity quaternion.
#[inline(always)]
pub fn id<T: Float + Copy>() -> Quaternion<T> {
    let one = Float::one();
    let zero = Float::zero();
    (one, [zero, zero, zero])
}

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

/// Computes the square length of a quaternion.
#[inline(always)]
pub fn square_len<T: Float>(q: Quaternion<T>) -> T {
    use vecmath::vec3_square_len as square_len;
    q.0 * q.0 + square_len(q.1)
}

/// Computes the length of a quaternion.
#[inline(always)]
pub fn len<T: Float>(q: Quaternion<T>) -> T {
    square_len(q).sqrt()
}

/// Rotate the given vector using the given quaternion
#[inline(always)]
pub fn rotate_vector<T: Float>(q: Quaternion<T>, v: Vector3<T>) -> Vector3<T> {
    let zero = Float::zero();
    let v_as_q : Quaternion<T> = (zero, v);
    let q_conj = conj(q);
    mul(mul(q, v_as_q), q_conj).1
}

/// Construct a quaternion representing the given euler angle rotations (in radians)
#[inline(always)]
pub fn euler_angles<T: Float + FromPrimitive>(x: T, y: T, z: T) -> Quaternion<T> {
    let two: T = FromPrimitive::from_int(2).unwrap();

    let half_x = x / two;
    let half_y = y / two;
    let half_z = z / two;

    let cos_x_2 = half_x.cos();
    let cos_y_2 = half_y.cos();
    let cos_z_2 = half_z.cos();

    let sin_x_2 = half_x.sin();
    let sin_y_2 = half_y.sin();
    let sin_z_2 = half_z.sin();

    (
        cos_x_2 * cos_y_2 * cos_z_2 + sin_x_2 * sin_y_2 * sin_z_2,
        [
            sin_x_2 * cos_y_2 * cos_z_2 + cos_x_2 * sin_y_2 * sin_z_2,
            cos_x_2 * sin_y_2 * cos_z_2 + sin_x_2 * cos_y_2 * sin_z_2,
            cos_x_2 * cos_y_2 * sin_z_2 + sin_x_2 * sin_y_2 * cos_z_2
        ]
    )
}

/// Construct a quaternion for the given angle (in radians)
/// about the given axis.
/// Axis must be a unit vector.
#[inline(always)]
pub fn axis_angle<T: Float + FromPrimitive>(axis: Vector3<T>, angle: T) -> Quaternion<T> {
    use vecmath::vec3_scale as scale;
    let two: T = FromPrimitive::from_int(2).unwrap();
    let half_angle = angle / two;
    (half_angle.cos(), scale(axis, half_angle.sin()))
}


/// Tests
#[cfg(test)]
mod test {
    use super::*;
    use vecmath::Vector3;
    use std::f32::consts::PI;
    use std::num::Float;

    /// Fudge factor for float equality checks
    static EPSILON: f32 = 0.000001;

    #[test]
    fn test_axis_angle() {
        use vecmath::vec3_normalized as normalized;
        let axis: Vector3<f32> = [1.0, 1.0, 1.0];
        let q: Quaternion<f32> = axis_angle(
            normalized(axis),
            PI
        );

        // Should be a unit quaternion
        assert!((square_len(q) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_euler_angle() {
        let q: Quaternion<f32> = euler_angles(PI, PI, PI);
        // Should be a unit quaternion
        assert!((square_len(q) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_rotate_vector_axis_angle() {
        let v: Vector3<f32> = [1.0, 1.0, 1.0];
        let q: Quaternion<f32> = axis_angle([0.0, 1.0, 0.0], PI);
        let rotated = rotate_vector(q, v);
        assert!((rotated[0] - -1.0).abs() < EPSILON);
        assert!((rotated[1] - 1.0).abs() < EPSILON);
        assert!((rotated[2] - -1.0).abs() < EPSILON);
    }

    #[test]
    fn test_rotate_vector_euler_angle() {
        let v: Vector3<f32> = [1.0, 1.0, 1.0];
        let q: Quaternion<f32> = euler_angles(0.0, PI, 0.0);
        let rotated = rotate_vector(q, v);
        assert!((rotated[0] - -1.0).abs() < EPSILON);
        assert!((rotated[1] - 1.0).abs() < EPSILON);
        assert!((rotated[2] - -1.0).abs() < EPSILON);
    }

    /// Rotation on axis parallel to vector direction should have no effect
    #[test]
    fn test_rotate_vector_axis_angle_same_axis() {
        use vecmath::vec3_normalized as normalized;

        let v: Vector3<f32> = [1.0, 1.0, 1.0];
        let arbitrary_angle = 32.12f32;
        let axis: Vector3<f32> = [1.0, 1.0, 1.0];
        let q: Quaternion<f32> = axis_angle(
            normalized(axis),
            arbitrary_angle
        );
        let rotated = rotate_vector(q, v);

        assert!((rotated[0] - 1.0).abs() < EPSILON);
        assert!((rotated[1] - 1.0).abs() < EPSILON);
        assert!((rotated[2] - 1.0).abs() < EPSILON);
    }
}
