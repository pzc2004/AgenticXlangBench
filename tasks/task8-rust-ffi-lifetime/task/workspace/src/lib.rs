//! rustops - High-performance data processing library for Python
//!
//! This module provides text processing, hash computation, and data transformation
//! functions implemented in Rust and exposed to Python via PyO3.

use pyo3::prelude::*;

mod ffi;

/// The main Python module definition.
#[pymodule]
fn rustops(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all FFI functions and classes
    ffi::register(m)?;

    // Module-level constants
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__doc__", "High-performance Rust data processing extension for Python")?;

    Ok(())
}
