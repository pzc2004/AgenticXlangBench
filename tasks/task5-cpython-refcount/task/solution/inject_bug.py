#!/usr/bin/env python3
"""
注入 3 个复合真 bug + 20 个诱饵

Bug 1: vector_push 中删除 Py_INCREF(item) → list 持有借用引用 → use-after-free
Bug 2: vector_get 中删除 Py_INCREF(result) → 调用者获得借用引用 → use-after-free
Bug 3: vector_pop 中删除 Py_INCREF(result) + NULL 检查 → use-after-free + 潜在崩溃
"""
import os
import sys
import re

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")


def inject_real_bug():
    """注入 3 个复合真 bug"""
    success = True
    filepath = os.path.join(WORKSPACE_DIR, "vector.c")
    if not os.path.exists(filepath):
        print(f"  ❌ 找不到 {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # ========================================
    # Bug 1: vector_push 中删除 Py_INCREF(item)
    # ========================================
    # 目标: 在 self->items[self->size] = item 之前的 Py_INCREF(item)
    # 干净代码:
    #     Py_INCREF(item);  /* BUG_LOCATION_1: This INCREF will be removed by inject_bug.py */
    #     self->items[self->size] = item;
    # 修改为:
    #     /* Py_INCREF(item); */  /* BUG_LOCATION_1: INCREF removed - BUG */
    #     self->items[self->size] = item;

    clean1 = "    Py_INCREF(item);  /* BUG_LOCATION_1: This INCREF will be removed by inject_bug.py */"
    buggy1 = "    /* Py_INCREF(item); */  /* BUG_LOCATION_1: INCREF removed - BUG */"

    if clean1 in content:
        content = content.replace(clean1, buggy1, 1)
        print("  ✅ Bug 1: 删除 vector_push 中的 Py_INCREF(item)")
    elif buggy1 in content:
        print("  ⚠️ Bug 1: 已经注入过")
    else:
        print("  ❌ 找不到 Bug 1 目标代码")
        success = False

    # ========================================
    # Bug 2: vector_get 中删除 Py_INCREF(result)
    # ========================================
    # 干净代码:
    #     PyObject *result = self->items[index];
    #     Py_INCREF(result);  /* BUG_LOCATION_2: This INCREF will be removed by inject_bug.py */
    #     return result;
    # 修改为:
    #     PyObject *result = self->items[index];
    #     /* Py_INCREF(result); */  /* BUG_LOCATION_2: INCREF removed - BUG */
    #     return result;

    clean2 = "    Py_INCREF(result);  /* BUG_LOCATION_2: This INCREF will be removed by inject_bug.py */"
    buggy2 = "    /* Py_INCREF(result); */  /* BUG_LOCATION_2: INCREF removed - BUG */"

    if clean2 in content:
        content = content.replace(clean2, buggy2, 1)
        print("  ✅ Bug 2: 删除 vector_get 中的 Py_INCREF(result)")
    elif buggy2 in content:
        print("  ⚠️ Bug 2: 已经注入过")
    else:
        print("  ❌ 找不到 Bug 2 目标代码")
        success = False

    # ========================================
    # Bug 3: vector_pop 中删除 Py_INCREF(result)
    # ========================================
    # 干净代码:
    #     if (result == NULL) {
    #         Py_RETURN_NONE;
    #     }
    #
    #     /* BUG_LOCATION_3: INCREF before DECREF to prevent premature free */
    #     Py_INCREF(result);  /* BUG_LOCATION_3: This INCREF will be removed by inject_bug.py */
    #     self->items[self->size] = NULL;
    #     Py_DECREF(result);
    # 修改为:
    #     if (result == NULL) {
    #         Py_RETURN_NONE;
    #     }
    #
    #     /* BUG_LOCATION_3: INCREF removed - causes use-after-free */
    #     /* Py_INCREF(result); */  /* BUG_LOCATION_3: INCREF removed - BUG */
    #     self->items[self->size] = NULL;
    #     Py_DECREF(result);

    clean3 = "    Py_INCREF(result);  /* BUG_LOCATION_3: This INCREF will be removed by inject_bug.py */"
    buggy3 = "    /* Py_INCREF(result); */  /* BUG_LOCATION_3: INCREF removed - BUG */"

    if clean3 in content:
        content = content.replace(clean3, buggy3, 1)
        print("  ✅ Bug 3: 删除 vector_pop 中的 Py_INCREF(result)")
    elif buggy3 in content:
        print("  ⚠️ Bug 3: 已经注入过")
    else:
        print("  ❌ 找不到 Bug 3 目标代码")
        success = False

    # 保存修改
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"\n  文件已更新: {filepath}")
    else:
        print("\n  ⚠️ 文件未修改")

    return success


def inject_decoys():
    """注入 20 个诱饵 (注释形式的误导性修改)"""
    filepath = os.path.join(WORKSPACE_DIR, "vector.c")
    if not os.path.exists(filepath):
        return 0

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    decoy_count = 0

    # 诱饵列表: (目标代码, 替换后代码, 描述)
    decoys = [
        # 在 Vector_dealloc 中添加注释
        (
            "    /* Release all stored objects */",
            "    /* Release all stored objects */\n    /* TODO: consider using PyObject_GC_Del for tracked objects */",
            "decoy: dealloc TODO 注释"
        ),
        # 在 Vector_new 中添加注释
        (
            "    self = (VectorObject *)type->tp_alloc(type, 0);",
            "    self = (VectorObject *)type->tp_alloc(type, 0);  /* BUG_CANDIDATE: tp_alloc may fail silently */",
            "decoy: tp_alloc 注释"
        ),
        # 在 Vector_init 中添加注释
        (
            "    self->capacity = 16;",
            "    self->capacity = 16;  /* MAGIC_NUMBER: should be configurable */",
            "decoy: capacity 注释"
        ),
        # 在 Vector_init 中添加注释
        (
            "    self->items = (PyObject **)PyMem_Malloc(self->capacity * sizeof(PyObject *));",
            "    self->items = (PyObject **)PyMem_Malloc(self->capacity * sizeof(PyObject *));  /* BUG_CANDIDATE: missing overflow check */",
            "decoy: malloc overflow 注释"
        ),
        # 在 Vector_init 中添加注释
        (
            '        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vector storage");',
            '        PyErr_SetString(PyExc_MemoryError, "Failed to allocate vector storage");  /* TODO: use PyErr_NoMemory() */',
            "decoy: PyErr 注释"
        ),
        # 在 vector_push 中添加注释
        (
            "    if (!PyArg_ParseTuple(args, \"O\", &item))",
            "    if (!PyArg_ParseTuple(args, \"O\", &item))  /* BUG_CANDIDATE: should check for NULL */",
            "decoy: ParseTuple 注释"
        ),
        # 在 vector_push 中添加注释
        (
            "        Py_ssize_t new_capacity = self->capacity * 2;",
            "        Py_ssize_t new_capacity = self->capacity * 2;  /* BUG_CANDIDATE: potential integer overflow */",
            "decoy: integer overflow 注释"
        ),
        # 在 vector_push 中添加注释
        (
            "        PyObject **new_items = (PyObject **)PyMem_Realloc(",
            "        PyObject **new_items = (PyObject **)PyMem_Realloc(  /* BUG_CANDIDATE: old pointer not freed on failure */",
            "decoy: realloc 注释"
        ),
        # 在 vector_push 中添加注释
        (
            '            PyErr_SetString(PyExc_MemoryError, "Failed to grow vector");',
            '            PyErr_SetString(PyExc_MemoryError, "Failed to grow vector");  /* TODO: implement graceful degradation */',
            "decoy: grow error 注释"
        ),
        # 在 vector_push 中添加注释
        (
            "    self->items[self->size] = item;",
            "    self->items[self->size] = item;  /* BUG_CANDIDATE: should check for duplicate refs */",
            "decoy: duplicate refs 注释"
        ),
        # 在 vector_get 中添加注释
        (
            "    if (!PyArg_ParseTuple(args, \"n\", &index))",
            "    if (!PyArg_ParseTuple(args, \"n\", &index))  /* BUG_CANDIDATE: 'n' format may not work on all platforms */",
            "decoy: format 注释"
        ),
        # 在 vector_get 中添加注释
        (
            '        PyErr_SetString(PyExc_IndexError, "vector index out of range");',
            '        PyErr_SetString(PyExc_IndexError, "vector index out of range");  /* TODO: support negative indexing */',
            "decoy: negative indexing 注释"
        ),
        # 在 vector_get 中添加注释
        (
            "    return result;",
            "    return result;  /* BUG_CANDIDATE: should we return a copy instead? */",
            "decoy: return copy 注释"
        ),
        # 在 vector_pop 中添加注释
        (
            '        PyErr_SetString(PyExc_IndexError, "pop from empty vector");',
            '        PyErr_SetString(PyExc_IndexError, "pop from empty vector");  /* TODO: support silent empty return */',
            "decoy: silent empty 注释"
        ),
        # 在 vector_pop 中添加注释
        (
            "    self->size--;",
            "    self->size--;  /* BUG_CANDIDATE: no underflow protection */",
            "decoy: underflow 注释"
        ),
        # 在 vector_pop 中添加注释
        (
            "    self->items[self->size] = NULL;",
            "    self->items[self->size] = NULL;  /* BUG_CANDIDATE: should memset to NULL? */",
            "decoy: memset 注释"
        ),
        # 在 vector_size 中添加注释
        (
            "    return PyLong_FromSsize_t(self->size);",
            "    return PyLong_FromSsize_t(self->size);  /* BUG_CANDIDATE: cache size as Python int? */",
            "decoy: cache size 注释"
        ),
        # 在 vector_clear 中添加注释
        (
            "        Py_XDECREF(self->items[i]);",
            "        Py_XDECREF(self->items[i]);  /* BUG_CANDIDATE: XDECREF vs DECREF debate */",
            "decoy: XDECREF 注释"
        ),
        # 在 VectorType 定义中添加注释
        (
            "    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,",
            "    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* TODO: add Py_TPFLAGS_HAVE_GC */",
            "decoy: GC flag 注释"
        ),
        # 在 PyInit_pyvector 中添加注释
        (
            "    if (PyType_Ready(&VectorType) < 0)",
            "    if (PyType_Ready(&VectorType) < 0)  /* BUG_CANDIDATE: error not propagated */",
            "decoy: type ready 注释"
        ),
    ]

    for old, new, desc in decoys:
        if old in content and new not in content:
            content = content.replace(old, new, 1)
            decoy_count += 1
            print(f"  ✅ 诱饵 {decoy_count}: {desc}")
        elif new in content:
            decoy_count += 1
            print(f"  ⚠️ 诱饵 {decoy_count}: 已存在 ({desc})")

    # 保存修改
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)

    return decoy_count


def main():
    print("=" * 60)
    print("CPython C 扩展 Refcount Bug 注入")
    print("=" * 60)

    print("\n>>> 真 bug (3 个复合):")
    if not inject_real_bug():
        print("\n❌ Bug 注入失败!")
        sys.exit(1)

    print(f"\n>>> 诱饵 (20 个):")
    decoy_count = inject_decoys()
    print(f"\n总计: 3 真 bug + {decoy_count} 诱饵 = {3 + decoy_count} 个修改")

    if decoy_count < 15:
        print("⚠️ 诱饵数量不足 15 个，请检查代码")
        sys.exit(1)

    print("\n✅ 注入完成!")


if __name__ == "__main__":
    main()
