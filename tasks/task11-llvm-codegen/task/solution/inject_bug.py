#!/usr/bin/env python3
"""
Inject 3 real bugs + 20 decoys into LLVM x86 instruction selection code.

Bug 1: Wrong instruction for sext+trunc pattern
        - In X86ISelDAGToDAG.cpp: selects MOVSX (sign-extend) where MOVZX (zero-extend) is needed
        - Effect: negative values get wrong sign extension under -O2

Bug 2: Incorrect operand size for i16 sext
        - Selects 16-bit zero-extend instead of 32-bit sign-extend
        - Effect: short sign-extension produces wrong results

Bug 3: Wrong condition code for conditional move with sext operands
        - Selects signed comparison where unsigned is needed (or vice versa)
        - Effect: conditional moves with sign-extended values give wrong results

Decoys: 20 comments in other LLVM x86 backend files
"""

import os
import sys
import re

LLVM_DIR = os.environ.get("LLVM_DIR", "/build/llvm-project")
X86_DIR = os.path.join(LLVM_DIR, "llvm/lib/Target/X86")


def inject_real_bugs():
    """Inject 3 compound real bugs into X86ISelDAGToDAG.cpp"""
    success = True

    filepath = os.path.join(X86_DIR, "X86ISelDAGToDAG.cpp")
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # === Bug 1: Wrong instruction for sext pattern ===
    # Pattern: When selecting sign-extend from i8 to i32, use MOVZX instead of MOVSX
    # Look for: X86::MOVSX32rr8 or similar pattern
    # Change: MOVSX32rr8 → MOVZX32rr8 (zero-extend instead of sign-extend)
    pattern1 = r'(X86::MOVSX32rr8)'
    replacement1 = 'X86::MOVZX32rr8'
    new_content = re.sub(pattern1, replacement1, content, count=1)
    if new_content != content:
        content = new_content
        print("  Bug 1: MOVSX32rr8 → MOVZX32rr8 (wrong extend for i8→i32)")
    else:
        # Try alternative pattern
        pattern1_alt = r'(MOVSX32rr8)'
        new_content = re.sub(pattern1_alt, 'MOVZX32rr8', content, count=1)
        if new_content != content:
            content = new_content
            print("  Bug 1: MOVSX32rr8 → MOVZX32rr8 (alt pattern)")
        else:
            print("  Could not find Bug 1 pattern")
            success = False

    # === Bug 2: Incorrect operand size for i16 sext ===
    # Pattern: MOVSX32rr16 → MOVZX32rr16
    # Effect: 16-bit sign-extend becomes zero-extend
    pattern2 = r'(X86::MOVSX32rr16)'
    replacement2 = 'X86::MOVZX32rr16'
    new_content = re.sub(pattern2, replacement2, content, count=1)
    if new_content != content:
        content = new_content
        print("  Bug 2: MOVSX32rr16 → MOVZX32rr16 (wrong extend for i16→i32)")
    else:
        pattern2_alt = r'(MOVSX32rr16)'
        new_content = re.sub(pattern2_alt, 'MOVZX32rr16', content, count=1)
        if new_content != content:
            content = new_content
            print("  Bug 2: MOVSX32rr16 → MOVZX32rr16 (alt pattern)")
        else:
            print("  Could not find Bug 2 pattern")
            success = False

    # === Bug 3: Wrong condition code for comparison ===
    # Pattern: X86::COND_E (equal) → X86::COND_NE (not equal) in select_cc
    # Or: change SETG → SETL (signed greater → signed less)
    pattern3 = r'(X86::SETG)'
    replacement3 = 'X86::SETL'
    new_content = re.sub(pattern3, replacement3, content, count=1)
    if new_content != content:
        content = new_content
        print("  Bug 3: SETG → SETL (wrong condition for signed comparison)")
    else:
        pattern3_alt = r'(SETG\b)'
        new_content = re.sub(pattern3_alt, 'SETL', content, count=1)
        if new_content != content:
            content = new_content
            print("  Bug 3: SETG → SETL (alt pattern)")
        else:
            # Try a different condition pair
            pattern3_alt2 = r'(X86::COND_GE)'
            new_content = re.sub(pattern3_alt2, 'X86::COND_LE', content, count=1)
            if new_content != content:
                content = new_content
                print("  Bug 3: COND_GE → COND_LE (wrong signed comparison)")
            else:
                print("  Could not find Bug 3 pattern")
                success = False

    if content == original:
        print("  No bugs could be injected!")
        return False

    with open(filepath, 'w') as f:
        f.write(content)

    return success


def inject_decoys():
    """Inject 20 decoy comments into other LLVM x86 backend files."""
    decoys = [
        ("X86ISelLowering.cpp", "/* float branch_probability = 0.5;  FIXME: branch prediction */"),
        ("X86ISelLowering.cpp", "/* TODO: verify lowering of i64 shift on 32-bit */"),
        ("X86InstrInfo.cpp", "/* WARNING: instruction encoding changed for VEX */"),
        ("X86InstrInfo.cpp", "/* FIXME: AVX-512 register pressure */"),
        ("X86RegisterInfo.cpp", "/* int stack_alignment = 16;  TODO: stack layout */"),
        ("X86RegisterInfo.cpp", "/* float spill_weight = 1.0;  FIXME: register allocation */"),
        ("X86FrameLowering.cpp", "/* WARNING: frame pointer elimination order */"),
        ("X86FrameLowering.cpp", "/* TODO: verify prologue/epilogue emission */"),
        ("X86Subtarget.cpp", "/* int feature_bits = 0;  FIXME: CPU feature detection */"),
        ("X86Subtarget.cpp", "/* float tune_factor = 1.0;  TODO: scheduling */"),
        ("X86TargetTransformInfo.cpp", "/* WARNING: cost model for vector operations */"),
        ("X86TargetTransformInfo.cpp", "/* FIXME: instruction latency table */"),
        ("X86MachineFunctionInfo.cpp", "/* int varargs_offset = 0;  TODO: calling convention */"),
        ("X86MachineFunctionInfo.cpp", "/* float stack_growth = -1.0;  FIXME */"),
        ("X86CallingConv.cpp", "/* WARNING: SysV ABI register order */"),
        ("X86CallingConv.cpp", "/* TODO: verify varargs handling */"),
        ("X86InstrFormats.td", "/* int encoding_size = 4;  FIXME: instruction size */"),
        ("X86InstrFormats.td", "/* WARNING: prefix encoding */"),
        ("X86ScheduleBtVer2.td", "/* float issue_rate = 2.0;  TODO: pipeline scheduling */"),
        ("X86ScheduleBtVer2.td", "/* FIXME: resource conflict detection */"),
    ]

    count = 0
    for filename, comment in decoys:
        filepath = os.path.join(X86_DIR, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Insert after the license/header comment block
        insert_idx = 0
        in_comment = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('/*'):
                in_comment = True
            if in_comment:
                if '*/' in stripped:
                    in_comment = False
                insert_idx = i + 1
                continue
            if stripped.startswith('//'):
                insert_idx = i + 1
                continue
            if stripped.startswith('#include') or stripped.startswith('#define'):
                insert_idx = i + 1
                continue
            if stripped:
                break

        lines.insert(insert_idx, ' ' + comment + '\n')
        with open(filepath, 'w') as f:
            f.writelines(lines)
        count += 1

    return count


def main():
    print("=" * 60)
    print("LLVM x86 Instruction Selection Bug Injection")
    print("=" * 60)

    print(f"\nLLVM directory: {LLVM_DIR}")
    print(f"X86 backend: {X86_DIR}")

    print(f"\n>>> Injecting real bugs into X86ISelDAGToDAG.cpp:")
    if not inject_real_bugs():
        print("WARNING: Bug injection may be incomplete")

    print(f"\n>>> Injecting decoys:")
    decoy_count = inject_decoys()
    print(f"  Injected {decoy_count} decoy comments")

    print(f"\nTotal: 3 bugs + {decoy_count} decoys")


if __name__ == "__main__":
    main()
