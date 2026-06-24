#!/usr/bin/env python3
"""
Inject 3 compound real bugs + 20 decoys into Rust FFI code.

Bug 1: process_text returns pointer to stack-allocated CString buffer
  - Clean: returns Ok(result) where result is an owned String
  - Bug: converts String to CString, gets raw pointer, drops CString, returns pointer as String
  - Effect: dangling pointer to freed stack memory → segfault or corrupted data

Bug 2: compute_hash returns pointer to stack-allocated array
  - Clean: creates PyBytes from heap-allocated Vec<u8>
  - Bug: stores hash bytes in a stack array, gets raw pointer, returns PyBytes from dangling ptr
  - Effect: reading freed stack memory → garbage hash values or segfault

Bug 3: transform_data returns pointer to temporary Vec
  - Clean: returns Ok(result) where result is an owned Vec<f64>
  - Bug: creates Vec, gets raw pointer, drops Vec, reconstructs from raw pointer, returns
  - Effect: use-after-free on Vec's heap buffer → corrupted data or segfault
"""
import os
import sys

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")


def inject_real_bug():
    """Inject 3 compound real bugs into ffi.rs."""
    success = True
    filepath = os.path.join(WORKSPACE_DIR, "src", "ffi.rs")
    if not os.path.exists(filepath):
        print(f"  FAIL: Cannot find {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # ========================================================================
    # Bug 1: process_text — return dangling pointer to stack CString buffer
    # ========================================================================
    # Clean code:
    #     Ok(result)
    #
    # Buggy code:
    #     let c_string = std::ffi::CString::new(result).unwrap();
    #     let ptr = c_string.into_raw();
    #     unsafe {
    #         let slice = std::ffi::CStr::from_ptr(ptr);
    #         let owned = slice.to_str().unwrap().to_string();
    #         libc::free(ptr as *mut libc::c_void);
    #         Ok(owned)
    #     }
    #
    # Wait, that would actually work since we copy before free.
    # Better approach: use the pointer AFTER the CString is dropped.
    #
    # Clean:
    #     Ok(result)
    #
    # Bug:
    #     let ptr = {
    #         let c_str = std::ffi::CString::new(result.clone()).unwrap();
    #         c_str.as_ptr()
    #     };  // c_str dropped here, ptr is now dangling
    #     unsafe {
    #         let slice = std::ffi::CStr::from_ptr(ptr);
    #         Ok(slice.to_str().unwrap_or("").to_string())
    #     }

    clean1 = """    // BUG_LOCATION_1: Return owned String transferred to Python.
    // Correct: return the owned String directly (PyO3 handles conversion).
    // The String is heap-allocated and outlives this function call.
    Ok(result)"""

    buggy1 = """    // BUG_LOCATION_1: Return owned String transferred to Python.
    // Correct: return the owned String directly (PyO3 handles conversion).
    // The String is heap-allocated and outlives this function call.
    let ptr = {
        let c_str = std::ffi::CString::new(result.clone()).unwrap();
        c_str.as_ptr()
    };  // c_str dropped here, ptr is now dangling
    unsafe {
        let slice = std::ffi::CStr::from_ptr(ptr);
        Ok(slice.to_str().unwrap_or("").to_string())
    }"""

    if clean1 in content:
        content = content.replace(clean1, buggy1, 1)
        print("  [+] Bug 1: process_text returns dangling pointer to dropped CString")
    elif buggy1 in content:
        print("  [~] Bug 1: already injected")
    else:
        print("  [-] Bug 1: target code not found")
        success = False

    # ========================================================================
    # Bug 2: compute_hash — return PyBytes from stack-allocated array
    # ========================================================================
    # Clean code:
    #     Ok(PyBytes::new(py, &hash_bytes))
    #
    # Bug: store hash in a stack array, drop the Vec, use dangling pointer
    #     let hash_array = {
    #         let tmp = hash_val.to_le_bytes();
    #         tmp
    #     };  // tmp is Copy so this is fine — need a different approach
    #
    # Better: use a stack buffer that goes out of scope
    #     let stack_buf;
    #     {
    #         let mut tmp = hash_val.to_le_bytes().to_vec();
    #         tmp.extend_from_slice(&[0u8; 8]); // pad
    #         stack_buf = tmp.as_ptr();
    #     }
    #     unsafe { Ok(PyBytes::new(py, std::slice::from_raw_parts(stack_buf, 8))) }

    clean2 = """    // BUG_LOCATION_2: Return bytes owned by the PyBytes object.
    // Correct: create PyBytes from the owned Vec, which copies data into Python-managed memory.
    // The Vec is dropped after this line, but the data is safely copied.
    Ok(PyBytes::new(py, &hash_bytes))"""

    buggy2 = """    // BUG_LOCATION_2: Return bytes owned by the PyBytes object.
    // Correct: create PyBytes from the owned Vec, which copies data into Python-managed memory.
    // The Vec is dropped after this line, but the data is safely copied.
    let dangling_ptr = {
        let tmp = hash_val.to_le_bytes().to_vec();
        tmp.as_ptr()
    };  // tmp dropped here, dangling_ptr is now invalid
    unsafe {
        Ok(PyBytes::new(py, std::slice::from_raw_parts(dangling_ptr, 8)))
    }"""

    if clean2 in content:
        content = content.replace(clean2, buggy2, 1)
        print("  [+] Bug 2: compute_hash returns PyBytes from dangling stack pointer")
    elif buggy2 in content:
        print("  [~] Bug 2: already injected")
    else:
        print("  [-] Bug 2: target code not found")
        success = False

    # ========================================================================
    # Bug 3: transform_data — return list from dropped Vec's raw pointer
    # ========================================================================
    # Clean code:
    #     Ok(result)
    #
    # Bug: destructure Vec into raw parts, drop it, reconstruct from dangling pointer
    #     let (ptr, len, cap) = result.into_raw_parts();
    #     drop(unsafe { Vec::from_raw_parts(ptr, len, cap) }); // free the buffer
    #     let dangling_result = unsafe { Vec::from_raw_parts(ptr, len, cap) }; // use-after-free
    #     Ok(dangling_result)

    clean3 = """    // BUG_LOCATION_3: Return owned Vec<f64>.
    // Correct: return the heap-allocated Vec directly. PyO3 converts it to a Python list,
    // copying each element. The Vec is dropped after conversion.
    Ok(result)"""

    buggy3 = """    // BUG_LOCATION_3: Return owned Vec<f64>.
    // Correct: return the heap-allocated Vec directly. PyO3 converts it to a Python list,
    // copying each element. The Vec is dropped after conversion.
    let (ptr, len, cap) = result.into_raw_parts();
    drop(unsafe { Vec::from_raw_parts(ptr, len, cap) });  // free the buffer
    let result = unsafe { Vec::from_raw_parts(ptr, len, cap) };  // use-after-free!
    Ok(result)"""

    if clean3 in content:
        content = content.replace(clean3, buggy3, 1)
        print("  [+] Bug 3: transform_data returns Vec from use-after-free raw pointer")
    elif buggy3 in content:
        print("  [~] Bug 3: already injected")
    else:
        print("  [-] Bug 3: target code not found")
        success = False

    # Save changes
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"\n  File updated: {filepath}")
    else:
        print("\n  [!] File unchanged")

    return success


def inject_decoys():
    """Inject 20 decoys (misleading comments and minor changes)."""
    filepath = os.path.join(WORKSPACE_DIR, "src", "ffi.rs")
    if not os.path.exists(filepath):
        return 0

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    decoy_count = 0

    decoys = [
        # 1. Add comment about potential UB in process_text
        (
            "    let processed = input.to_uppercase();",
            "    let processed = input.to_uppercase();  /* PERF: consider in-place conversion */",
            "decoy: process_text perf comment"
        ),
        # 2. Add comment about summary format
        (
            '    let summary = format!(" [len={}]", processed.len());',
            '    let summary = format!(" [len={}]", processed.len());  /* TODO: use Cow for efficiency */',
            "decoy: summary Cow comment"
        ),
        # 3. Add comment about hasher choice
        (
            "    let mut hasher = DefaultHasher::new();",
            "    let mut hasher = DefaultHasher::new();  /* NOTE: DefaultHasher is not cryptographically secure */",
            "decoy: hasher security comment"
        ),
        # 4. Add comment about hash_val
        (
            "    let hash_val = hasher.finish();",
            "    let hash_val = hasher.finish();  /* BUG_CANDIDATE: hash collision possible */",
            "decoy: hash collision comment"
        ),
        # 5. Add comment about hash_bytes
        (
            "    let hash_bytes: Vec<u8> = hash_val.to_le_bytes().to_vec();",
            "    let hash_bytes: Vec<u8> = hash_val.to_le_bytes().to_vec();  /* ALLOC: consider stack array */",
            "decoy: stack array comment"
        ),
        # 6. Add comment about transform enumerate
        (
            "        .enumerate()",
            "        .enumerate()  /* BUG_CANDIDATE: enumerate overhead on large inputs */",
            "decoy: enumerate comment"
        ),
        # 7. Add comment about scale factor
        (
            "            let scaled = v * 2.5;",
            "            let scaled = v * 2.5;  /* MAGIC_NUMBER: 2.5 should be configurable */",
            "decoy: magic number comment"
        ),
        # 8. Add comment about offset calculation
        (
            "            let offset = scaled + (i as f64) * 0.1;",
            "            let offset = scaled + (i as f64) * 0.1;  /* PRECISION: potential float drift */",
            "decoy: float precision comment"
        ),
        # 9. Add comment about clamp
        (
            "            offset.clamp(-1e6, 1e6)",
            "            offset.clamp(-1e6, 1e6)  /* TODO: make bounds configurable */",
            "decoy: clamp bounds comment"
        ),
        # 10. Add comment about Pipeline::new
        (
            "    fn new(scale_factor: Option<f64>) -> Self {",
            "    fn new(scale_factor: Option<f64>) -> Self {  /* BUG_CANDIDATE: no validation on scale_factor */",
            "decoy: scale_factor validation comment"
        ),
        # 11. Add comment about operations vec
        (
            "            operations: Vec::new(),",
            "            operations: Vec::new(),  /* ALLOC: consider SmallVec for inline storage */",
            "decoy: SmallVec comment"
        ),
        # 12. Add comment about add_operation
        (
            "        self.operations.push(op.to_string());",
            "        self.operations.push(op.to_string());  /* ALLOC: consider interning strings */",
            "decoy: string interning comment"
        ),
        # 13. Add comment about execute match
        (
            '            match op.as_str() {',
            '            match op.as_str() {  /* BUG_CANDIDATE: string comparison is slow, use enum */',
            "decoy: enum optimization comment"
        ),
        # 14. Add comment about scale operation
        (
            '                "scale" => {',
            '                "scale" => {  /* PERF: use SIMD for vectorized scaling */',
            "decoy: SIMD comment"
        ),
        # 15. Add comment about offset operation
        (
            '                    for (i, v) in data.iter_mut().enumerate() {',
            '                    for (i, v) in data.iter_mut().enumerate() {  /* PARALLEL: use rayon par_iter_mut */',
            "decoy: rayon comment"
        ),
        # 16. Add comment about clamp operation
        (
            '                "clamp" => {',
            '                "clamp" => {  /* BUG_CANDIDATE: NaN not handled */',
            "decoy: NaN handling comment"
        ),
        # 17. Add comment about unknown op handling
        (
            '                _ => {',
            '                _ => {  /* TODO: return error for unknown operations */',
            "decoy: unknown op error comment"
        ),
        # 18. Add comment about operation_count
        (
            "    fn operation_count(&self) -> usize {",
            "    fn operation_count(&self) -> usize {  /* PERF: cache this value */",
            "decoy: cache count comment"
        ),
        # 19. Add comment about describe
        (
            "    fn describe(&self) -> String {",
            "    fn describe(&self) -> String {  /* ALLOC: consider returning &str with lifetime */",
            "decoy: lifetime comment"
        ),
        # 20. Add comment about register function
        (
            "    m.add_function(wrap_pyfunction!(process_text, m)?)?;",
            "    m.add_function(wrap_pyfunction!(process_text, m)?)?;  /* BUG_CANDIDATE: error propagation */",
            "decoy: register error propagation comment"
        ),
    ]

    for old, new, desc in decoys:
        if old in content and new not in content:
            content = content.replace(old, new, 1)
            decoy_count += 1
            print(f"  [+] Decoy {decoy_count}: {desc}")
        elif new in content:
            decoy_count += 1
            print(f"  [~] Decoy {decoy_count}: already exists ({desc})")

    # Save changes
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)

    return decoy_count


def main():
    print("=" * 60)
    print("Rust FFI Lifetime Bug Injection")
    print("=" * 60)

    print("\n>>> Real bugs (3 compound):")
    if not inject_real_bug():
        print("\nFAIL: Bug injection failed!")
        sys.exit(1)

    print(f"\n>>> Decoys (20):")
    decoy_count = inject_decoys()
    print(f"\nTotal: 3 real bugs + {decoy_count} decoys = {3 + decoy_count} changes")

    if decoy_count < 15:
        print("WARNING: Fewer than 15 decoys injected, check code")
        sys.exit(1)

    print("\nInjection complete!")


if __name__ == "__main__":
    main()
