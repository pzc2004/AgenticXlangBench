/*
 * grade.c — setuid-root 评分入口
 *
 * agent（非 root）执行此程序即可拿到当前修复的最终分数，但读不到判分逻辑：
 *   - 以 root 跑受保护的 /opt/judge/test.sh（root:root 0700，agent 无读权限）
 *   - 丢弃 test.sh 的全部 stdout/stderr（不泄漏测了哪些 kernel / 路径）
 *   - 只回显一行 score=X.XX（读自 root 写的 /logs/verifier/reward.txt）
 *
 * 安全要点：
 *   - execle 使用固定 PATH/HOME，杜绝 agent 通过环境变量劫持 bash/python
 *   - setuid 进程默认 dumpable=0，agent（同非 root uid）无法 ptrace 或读其 fd
 *
 * 编译：gcc -O2 -o /usr/local/bin/grade grade.c && chmod 4755 /usr/local/bin/grade
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>

static const char *JUDGE = "/opt/judge/test.sh";
static const char *REWARD = "/logs/verifier/reward.txt";

int main(void) {
    /* 提权到 root（二进制为 setuid root 时成功） */
    if (setgid(0) != 0 || setuid(0) != 0) {
        fprintf(stderr, "grade: 无法提权（请确认 setuid 位）\n");
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return 1;
    }
    if (pid == 0) {
        /* 子进程：丢弃 test.sh 的输出，避免泄漏判分细节 */
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
        }
        char *env[] = {
            "PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME=/root",
            NULL
        };
        execle("/bin/bash", "bash", JUDGE, (char *)NULL, env);
        _exit(127); /* execle 失败 */
    }

    int status = 0;
    waitpid(pid, &status, 0);

    /* 读回分数并回显（只给总分，不给分项） */
    char buf[64] = {0};
    FILE *f = fopen(REWARD, "r");
    if (f) {
        if (fgets(buf, sizeof(buf) - 1, f)) {
            char *nl = strchr(buf, '\n');
            if (nl) *nl = '\0';
        }
        fclose(f);
    }
    printf("score=%s\n", buf[0] ? buf : "0.0");
    return 0;
}
