# Skills

Claude Code skills for the AgenticXlangBench project.

## /generate-task

生成一道跨语言 bug-fix 评测题。

用法:
```
/generate-task --cve CVE-2024-xxxx --framework pytorch
/generate-task --bug-file kernel.cu --bug-line 246 --bug-desc "rsqrt missing eps"
```

详细说明见 `generate-task.md`。
