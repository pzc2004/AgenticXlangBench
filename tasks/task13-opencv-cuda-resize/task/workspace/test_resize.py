#!/usr/bin/env python3
"""
OpenCV CUDA Resize 测试脚本
用法: python test_resize.py

测试 GPU resize 与 CPU resize 的一致性,
以及缩放后图像在下游处理(特征检测+匹配)中的质量。
"""

import cv2
import numpy as np
import sys
import os


def create_test_image(width=640, height=480, seed=42):
    """生成测试图像(含丰富纹理,适合特征检测)"""
    np.random.seed(seed)
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # 添加一些结构化内容(使特征检测更有意义)
    for i in range(10):
        x1, y1 = np.random.randint(0, width-100), np.random.randint(0, height-100)
        x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(50, 100)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    for i in range(5):
        center = (np.random.randint(50, width-50), np.random.randint(50, height-50))
        radius = np.random.randint(20, 50)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.circle(img, center, radius, color, -1)

    return img


def test_resize_psnr(scale_factor=0.5, interpolation=cv2.INTER_LINEAR):
    """测试 GPU vs CPU resize 的 PSNR"""
    img = create_test_image()
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    dsize = (new_w, new_h)

    # CPU resize
    cpu_result = cv2.resize(img, dsize, interpolation=interpolation)

    # GPU resize
    try:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_result = cv2.cuda.resize(gpu_img, dsize, interpolation=interpolation)
        gpu_result = gpu_result.download()
    except Exception as e:
        print(f"GPU resize 失败: {e}")
        return 0.0

    # 计算 PSNR
    mse = np.mean((cpu_result.astype(float) - gpu_result.astype(float)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)

    return psnr


def test_resize_mse(scale_factor=0.5, interpolation=cv2.INTER_LINEAR):
    """测试 GPU vs CPU resize 的 MSE"""
    img = create_test_image()
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    dsize = (new_w, new_h)

    cpu_result = cv2.resize(img, dsize, interpolation=interpolation)

    try:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_result = cv2.cuda.resize(gpu_img, dsize, interpolation=interpolation)
        gpu_result = gpu_result.download()
    except Exception as e:
        print(f"GPU resize 失败: {e}")
        return float('inf')

    mse = np.mean((cpu_result.astype(float) - gpu_result.astype(float)) ** 2)
    return mse


def test_feature_matching(scale_factor=0.5):
    """测试缩放后图像的特征匹配质量"""
    img = create_test_image()
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    dsize = (new_w, new_h)

    # CPU resize
    cpu_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)

    # GPU resize
    try:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_resized = cv2.cuda.resize(gpu_img, dsize, interpolation=cv2.INTER_LINEAR)
        gpu_resized = gpu_resized.download()
    except Exception as e:
        print(f"GPU resize 失败: {e}")
        return 0.0

    # 特征检测
    orb = cv2.ORB_create(nfeatures=500)

    # CPU 图像特征
    kp1, des1 = orb.detectAndCompute(cpu_resized, None)
    # GPU 缩放图像特征
    kp2, des2 = orb.detectAndCompute(gpu_resized, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0

    # 特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return 0.0

    # 计算 inlier 率(距离 < 阈值的匹配)
    good_matches = [m for m in matches if m.distance < 50]
    inlier_rate = len(good_matches) / len(matches) if matches else 0

    return inlier_rate


def test_multi_scale():
    """测试多种缩放比例"""
    results = {}
    for scale in [0.25, 0.5, 0.75, 1.5, 2.0]:
        psnr = test_resize_psnr(scale_factor=scale)
        results[scale] = psnr
        print(f"  scale={scale}: PSNR={psnr:.2f} dB")
    return results


def main():
    print("=" * 60)
    print("OpenCV CUDA Resize 测试")
    print("=" * 60)

    # 检查 CUDA 可用性
    if not hasattr(cv2, 'cuda') or cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("❌ CUDA 不可用")
        sys.exit(1)

    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"CUDA 设备数: {cv2.cuda.getCudaEnabledDeviceCount()}")
    print()

    # 测试 1: 单次 resize PSNR
    print(">>> 测试 1: 单次 resize PSNR (scale=0.5, INTER_LINEAR)")
    psnr = test_resize_psnr(scale_factor=0.5, interpolation=cv2.INTER_LINEAR)
    print(f"  PSNR = {psnr:.2f} dB")
    print()

    # 测试 2: MSE 检查
    print(">>> 测试 2: MSE 检查 (scale=0.5)")
    mse = test_resize_mse(scale_factor=0.5, interpolation=cv2.INTER_LINEAR)
    print(f"  MSE = {mse:.4f}")
    print()

    # 测试 3: 多缩放比例
    print(">>> 测试 3: 多缩放比例 PSNR")
    multi_results = test_multi_scale()
    print()

    # 测试 4: 特征匹配
    print(">>> 测试 4: 特征匹配质量")
    inlier_rate = test_feature_matching(scale_factor=0.5)
    print(f"  Inlier 率 = {inlier_rate*100:.1f}%")
    print()

    # 汇总
    print("=" * 60)
    print("汇总:")
    print(f"  PSNR (scale=0.5): {psnr:.2f} dB")
    print(f"  MSE (scale=0.5): {mse:.4f}")
    print(f"  特征匹配 Inlier 率: {inlier_rate*100:.1f}%")
    avg_psnr = np.mean(list(multi_results.values()))
    print(f"  平均 PSNR (多比例): {avg_psnr:.2f} dB")

    # 输出结构化结果供 test.sh 解析
    print(f"\nRESULT_PSNR {psnr:.2f}")
    print(f"RESULT_MSE {mse:.4f}")
    print(f"RESULT_INLIER {inlier_rate:.4f}")
    print(f"RESULT_AVG_PSNR {avg_psnr:.2f}")


if __name__ == "__main__":
    main()
