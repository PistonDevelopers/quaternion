#![deny(missing_docs)]

//! A simple and type agnostic quaternion math designed for reexporting

extern crate vecmath;

/// Quaternion type alias.
pub type Quaternion<T> = [T; 4];

pub use vecmath::vec4_add as add;

