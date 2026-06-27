#!/usr/bin/env python3
"""
[STUB] 真实测试脚本已移动到 /task/tests/test_vmap.py

agent 不应依赖此文件；评测时由 /task/tests/test.sh 调用隐藏版本。
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print("accuracy 0 0")
    print("skipped 0")
    print("final_accuracy 0.0%")
    print("nan_detected False")
    print("FAIL")


if __name__ == "__main__":
    main()
