//! FFI functions exposed to Python via PyO3.
//!
//! This module provides text processing, hash computation, and data transformation
//! functions callable from Python. All functions use proper Rust lifetime management
//! to ensure returned data remains valid after the function returns.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Process input text: converts to uppercase and appends a summary.
///
/// Returns a Python string. The underlying CString is properly transferred
/// to Python via PyO3's automatic conversion.
#[pyfunction]
pub fn process_text(py: Python<'_>, input: &str) -> PyResult<String> {
    // Build the processed result on the heap via String
    let processed = input.to_uppercase();
    let summary = format!(" [len={}]", processed.len());
    let result = format!("{}{}", processed, summary);

    // BUG_LOCATION_1: Return owned String transferred to Python.
    // Correct: return the owned String directly (PyO3 handles conversion).
    // The String is heap-allocated and outlives this function call.
    Ok(result)
}

/// Compute a hash over the input byte slice.
///
/// Returns Python bytes containing the hash digest.
/// The digest bytes are owned by the Python bytes object.
#[pyfunction]
pub fn compute_hash<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    // Compute hash into a heap-allocated Vec<u8>
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let hash_val = hasher.finish();

    // Convert hash to byte representation on the heap
    let hash_bytes: Vec<u8> = hash_val.to_le_bytes().to_vec();

    // BUG_LOCATION_2: Return bytes owned by the PyBytes object.
    // Correct: create PyBytes from the owned Vec, which copies data into Python-managed memory.
    // The Vec is dropped after this line, but the data is safely copied.
    Ok(PyBytes::new(py, &hash_bytes))
}

/// Transform a slice of f64 values: scale, offset, and clamp.
///
/// Returns a Python list of f64 values.
#[pyfunction]
pub fn transform_data(py: Python<'_>, input: Vec<f64>) -> PyResult<Vec<f64>> {
    // Process into a new heap-allocated Vec
    let result: Vec<f64> = input
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let scaled = v * 2.5;
            let offset = scaled + (i as f64) * 0.1;
            offset.clamp(-1e6, 1e6)
        })
        .collect();

    // BUG_LOCATION_3: Return owned Vec<f64>.
    // Correct: return the heap-allocated Vec directly. PyO3 converts it to a Python list,
    // copying each element. The Vec is dropped after conversion.
    Ok(result)
}

/// A pipeline struct for chaining data processing operations.
#[pyclass]
pub struct Pipeline {
    operations: Vec<String>,
    scale_factor: f64,
}

#[pymethods]
impl Pipeline {
    #[new]
    fn new(scale_factor: Option<f64>) -> Self {
        Pipeline {
            operations: Vec::new(),
            scale_factor: scale_factor.unwrap_or(1.0),
        }
    }

    /// Add an operation to the pipeline.
    fn add_operation(&mut self, op: &str) {
        self.operations.push(op.to_string());
    }

    /// Execute the pipeline on input data and return transformed results.
    fn execute(&self, py: Python<'_>, input: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut data: Vec<f64> = input;

        for op in &self.operations {
            match op.as_str() {
                "scale" => {
                    for v in data.iter_mut() {
                        *v *= self.scale_factor;
                    }
                }
                "offset" => {
                    for (i, v) in data.iter_mut().enumerate() {
                        *v += (i as f64) * 0.01;
                    }
                }
                "clamp" => {
                    for v in data.iter_mut() {
                        *v = v.clamp(-1e6, 1e6);
                    }
                }
                "abs" => {
                    for v in data.iter_mut() {
                        *v = v.abs();
                    }
                }
                _ => {
                    // Unknown op, skip
                }
            }
        }

        // BUG_LOCATION_3 (Pipeline::execute): Return owned Vec.
        // Correct: return heap-allocated Vec directly.
        Ok(data)
    }

    /// Get the number of operations in the pipeline.
    fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Get a description of the pipeline.
    fn describe(&self) -> String {
        if self.operations.is_empty() {
            return "Empty pipeline".to_string();
        }
        format!(
            "Pipeline[{} ops, scale={:.2}]: {}",
            self.operations.len(),
            self.scale_factor,
            self.operations.join(" -> ")
        )
    }
}

/// Module registration
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_text, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hash, m)?)?;
    m.add_function(wrap_pyfunction!(transform_data, m)?)?;
    m.add_class::<Pipeline>()?;
    Ok(())
}
