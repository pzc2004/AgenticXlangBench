#!/bin/bash
# oracle_per_bug.sh — 针对每个 bug 单独测试
# 要求：每个 bug 单独注入时，test.sh 不能满分

set -e

python3 /task/solution/oracle_per_bug.py
