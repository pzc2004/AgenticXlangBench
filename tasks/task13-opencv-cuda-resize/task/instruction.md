# 任务：修复 OpenCV CUDA 图像缩放质量下降问题

## 背景

我们使用了一个从源码编译的 OpenCV 4.10.0(源码在 `/build/opencv/`)。
在图像处理管线中，使用 CUDA 加速的 `cv2.cuda.resize()` 进行图像缩放后，
后续的特征检测和匹配步骤的精度明显下降。

## Bug 现象

运行图像处理测试脚本：

```bash
cd /workspace
python test_resize.py
```

预期：GPU 缩放后的图像与 CPU 缩放结果的 PSNR 应 > 40dB，特征匹配 inlier 率 > 90%。
实际：PSNR 仅 ~25-30dB，特征匹配 inlier 率 ~60-70%。

注意：用 `cv2.resize()` (CPU 版本)时 **结果正常** (PSNR > 40dB)。问题仅出现在 CUDA 版本。

## 已知信息

- OpenCV 是从源码编译的，源码在 `/build/opencv/`
- **CUDA resize 结果与 CPU 结果存在系统性偏差**
- 你需要找到并修复 OpenCV CUDA 源码中导致缩放质量下降的 bug
- Bug 位于 `modules/cudaimgproc/src/cuda/resize.cu` 中的双线性插值逻辑

## 你的任务

1. **理解 bug**：分析为什么 CUDA resize 结果与 CPU 结果不一致
2. **定位 bug**：在 `/build/opencv/` 的 CUDA 源码中找到有问题的代码
3. **修复 bug**：修改 CUDA 源码(只允许修改 `.cu` / `.cpp` / `.h` 文件)
4. **重新编译**：修复后需要重新编译 OpenCV
   ```bash
   cd /build/opencv/build
   make -j$(nproc)
   ```
5. **验证修复**：运行测试确认修复成功
   ```bash
   bash /task/tests/test.sh
   ```
6. **检查分数**：测试会输出 0-1 的分数。**分数 >= 0.6 才算通过。如果分数 < 0.6，说明修复不正确，请继续排查。**

## 约束条件

- **只允许修改 C++/CUDA 文件**(`.cu` / `.cpp` / `.h`)，不允许修改 Python 文件
- **不允许**：
  - 用 CPU resize 替代 GPU resize
  - 用 PIL/Pillow 替代 OpenCV
  - 在 Python 层加后处理校正像素值
  - 修改 test_resize.py

## 文件说明

- `/workspace/test_resize.py` — 图像处理测试脚本
- `/task/tests/test.sh` — 测试脚本(跑完后输出 0-1 分数)
- `/build/opencv/` — OpenCV 源码

## 运行环境

- OpenCV 4.10.0(从源码编译，含 CUDA 支持)
- CUDA 12.8
- 编译工具：nvcc， gcc， make， cmake

## 验收标准

运行 `/task/tests/test.sh`，分数 >= 0.6 视为通过。
**如果分数 < 0.6，说明修复不正确，请继续排查 OpenCV CUDA 源码。**
