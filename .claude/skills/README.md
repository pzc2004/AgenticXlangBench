# Skills

Claude Code skills for the AgenticXlangBench project.

## /generate-task

生成一道跨语言 bug-fix 评测题。

用法：
```
/generate-task --cve CVE-2024-xxxx --framework pytorch
/generate-task --bug-file kernel.cu --bug-line 246 --bug-desc "rsqrt missing eps"
```

包含 9 个 Phase 的完整流程，以及 **Bug 构造策略**：
- 删除关键代码(最难)
- 条件触发(次难)
- 跨函数依赖(较难)
- 诱饵淹没(干扰)
- 数值精度陷阱(隐蔽)

详细说明见 `generate-task.md`。

## /calibrate-task

校准一道评测题的难度。

用法：
```
/calibrate-task --task-dir tasks/taskN-xxx --runs 3 --budget 10
```

详细说明见 `calibrate-task.md`。

## /select-tasks

从 N 道题里选最优 M 道。

详细说明见 `select-tasks.md`。

## /anti-hack

配置反 hack 措施，防止 agent 绕过调试过程。

8 种措施：
1. 禁止上网搜索 (WebSearch/WebFetch deny)
2. 禁止 git 查看历史 (删除 .git)
3. 禁止修改 Python 文件 (test.sh 检查)
4. 禁止 CPU 回退 (性能测试)
5. 禁止 NaN 处理掩盖 (静态分析)
6. 禁止绕过 vmap (API 检查)
7. 统一文件修改时间 (Dockerfile touch)
8. 判分逻辑防读 (setuid grade + 非 root agent)

详细说明见 `anti-hack.md`。

## 参考实现(canonical 样例)

skill 不内置脚本模板，出题时直接引用并 cp 已跑通的真实任务：
- 编译型/底层(CUDA、C/C++) → `tasks/task1-pytorch-cuda-index/`
- 纯 Python/JIT(JAX、TF) → `tasks/task4-jax-vmap-batch/`

引用索引表(抄哪个文件、改哪几处)见 `generate-task.md` 的「参考实现」章节。
