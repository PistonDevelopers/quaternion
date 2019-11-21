#![deny(missing_docs)]

//! A simple and type agnostic quaternion math library designed for reexporting

extern crate vecmath;

use vecmath::traits::Float;
use vecmath::Vector3;

/// Quaternion type alias.
pub type Quaternion<T> = (T, [T; 3]);

/// Constructs identity quaternion.
#[inline(always)]
pub fn id<T>() -> Quaternion<T>
where
    T: Float,
{
    let one = T::one();
    let zero = T::zero();
    (one, [zero, zero, zero])
}

/// Adds two quaternions.
#[inline(always)]
pub fn add<T>(a: Quaternion<T>, b: Quaternion<T>) -> Quaternion<T>
where
    T: Float,
{
    use vecmath::vec3_add as add;
    (a.0 + b.0, add(a.1, b.1))
}

/// Scales a quaternion (element-wise) by a scalar
#[inline(always)]
pub fn scale<T>(q: Quaternion<T>, t: T) -> Quaternion<T>
where
    T: Float,
{
    use vecmath::vec3_scale as scale;
    (q.0 * t, scale(q.1, t))
}

/// Dot product of two quaternions
#[inline(always)]
pub fn dot<T>(a: Quaternion<T>, b: Quaternion<T>) -> T
where
    T: Float,
{
    a.0 * b.0 + vecmath::vec3_dot(a.1, b.1)
}

/// Multiplies two quaternions.
#[inline(always)]
pub fn mul<T>(a: Quaternion<T>, b: Quaternion<T>) -> Quaternion<T>
where
    T: Float,
{
    use vecmath::vec3_add as add;
    use vecmath::vec3_cross as cross;
    use vecmath::vec3_dot as dot;
    use vecmath::vec3_scale as scale;

    (
        a.0 * b.0 - dot(a.1, b.1),
        add(add(scale(b.1, a.0), scale(a.1, b.0)), cross(a.1, b.1)),
    )
}

/// Takes the quaternion conjugate.
#[inline(always)]
pub fn conj<T>(a: Quaternion<T>) -> Quaternion<T>
where
    T: Float,
{
    use vecmath::vec3_neg as neg;

    (a.0, neg(a.1))
}

/// Computes the square length of a quaternion.
#[inline(always)]
pub fn square_len<T>(q: Quaternion<T>) -> T
where
    T: Float,
{
    use vecmath::vec3_square_len as square_len;
    q.0 * q.0 + square_len(q.1)
}

/// Computes the length of a quaternion.
#[inline(always)]
pub fn len<T>(q: Quaternion<T>) -> T
where
    T: Float,
{
    square_len(q).sqrt()
}

/// Rotate the given vector using the given quaternion
#[inline(always)]
pub fn rotate_vector<T>(q: Quaternion<T>, v: Vector3<T>) -> Vector3<T>
where
    T: Float,
{
    use vecmath::{vec3_add as add, vec3_cross as cross, vec3_scale as scale};
    let two = T::one() + T::one();
    let t: Vector3<T> = scale(cross(q.1, v), two);
    add(add(v, scale(t, q.0)), cross(q.1, t))
}

/// Construct a quaternion representing the rotation from a to b
#[inline(always)]
pub fn rotation_from_to<T>(a: Vector3<T>, b: Vector3<T>) -> Quaternion<T>
where
    T: Float,
{
    use vecmath::{vec3_cross, vec3_dot, vec3_normalized, vec3_square_len};

    const PI: f64 = 3.14159265358979323846264338327950288_f64;

    let _1 = T::one();
    let _0 = T::zero();

    let a = vec3_normalized(a);
    let b = vec3_normalized(b);
    let dot = vec3_dot(a, b);

    if dot >= _1 {
        // a, b are parallel
        return id();
    }

    if dot < T::from_f64(-0.999999) {
        // a, b are anti-parallel
        let mut axis = vec3_cross([_1, _0, _0], a);
        if vec3_square_len(axis) == _0 {
            axis = vec3_cross([_0, _1, _0], a);
        }
        axis = vec3_normalized(axis);
        axis_angle(axis, T::from_f64(PI))
    } else {
        let q = (_1 + dot, vec3_cross(a, b));
        scale(q, _1 / len(q))
    }
}

/// Construct a quaternion representing the given euler angle rotations (in radians)
#[inline(always)]
pub fn euler_angles<T>(x: T, y: T, z: T) -> Quaternion<T>
where
    T: Float,
{
    let two: T = T::one() + T::one();

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
            cos_x_2 * cos_y_2 * sin_z_2 + sin_x_2 * sin_y_2 * cos_z_2,
        ],
    )
}
/// Construct a quaternion for the given angle (in radians)
/// about the given axis.
/// Axis must be a unit vector.
#[inline(always)]
pub fn axis_angle<T>(axis: Vector3<T>, angle: T) -> Quaternion<T>
where
    T: Float,
{
    use vecmath::vec3_scale as scale;
    let two: T = T::one() + T::one();
    let half_angle = angle / two;
    (half_angle.cos(), scale(axis, half_angle.sin()))
}

/// Tests
#[cfg(test)]
mod test {
    use super::*;
    use vecmath::Vector3;

    /// Fudge factor for float equality checks
    static EPSILON: f32 = 0.000001;
    static PI: f32 = 3.14159265358979323846264338327950288_f32;

    #[test]
    fn test_axis_angle() {
        use vecmath::vec3_normalized as normalized;
        let axis: Vector3<f32> = [1.0, 1.0, 1.0];
        let q: Quaternion<f32> = axis_angle(normalized(axis), PI);

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
        let q: Quaternion<f32> = axis_angle(normalized(axis), arbitrary_angle);
        let rotated = rotate_vector(q, v);

        assert!((rotated[0] - 1.0).abs() < EPSILON);
        assert!((rotated[1] - 1.0).abs() < EPSILON);
        assert!((rotated[2] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_rotation_from_to_1() {
        let a: Vector3<f32> = [1.0, 1.0, 1.0];
        let b: Vector3<f32> = [-1.0, -1.0, -1.0];

        let q = super::rotation_from_to(a, b);

        let a_prime = super::rotate_vector(q, a);

        println!("a_prime = {:?}", a_prime);

        assert!((a_prime[0] + 1.0).abs() < EPSILON);
        assert!((a_prime[1] + 1.0).abs() < EPSILON);
        assert!((a_prime[2] + 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_rotation_from_to_2() {
        use vecmath::vec3_normalized as normalized;

        let a: Vector3<f32> = normalized([1.0, 1.0, 0.0]);
        let b: Vector3<f32> = [0.0, 1.0, 0.0];

        let q = super::rotation_from_to(a, b);
        let a_prime = super::rotate_vector(q, a);

        println!("a_prime = {:?}", a_prime);

        assert!((a_prime[0] - 0.0).abs() < EPSILON);
        assert!((a_prime[1] - 1.0).abs() < EPSILON);
        assert!((a_prime[2] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_rotation_from_to_3() {
        let a: Vector3<f32> = [1.0, 0.0, 0.0];
        let b: Vector3<f32> = [0.0, -1.0, 0.0];

        let q = super::rotation_from_to(a, b);

        let a_prime = super::rotate_vector(q, a);

        println!("a_prime = {:?}", a_prime);

        assert!((a_prime[0] - 0.0).abs() < EPSILON);
        assert!((a_prime[1] - -1.0).abs() < EPSILON);
        assert!((a_prime[2] - 0.0).abs() < EPSILON);
    }
}
