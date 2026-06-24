#!/bin/bash
# build_op.sh -- Build the tensor_ops C extension in-place.
set -e
cd "$(dirname "$0")"
pip install -e . 2>&1
echo "tensor_ops extension built successfully."
