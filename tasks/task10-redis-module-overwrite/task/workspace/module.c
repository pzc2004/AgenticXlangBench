/*
 * Redis module with StringDMA usage.
 * This module stores strings using direct memory access (DMA).
 *
 * Contains 3 intentional bugs (injected):
 *   Bug 1: Off-by-one write in RedisModule_StringDMA (writes len+1 bytes into len-byte buffer)
 *   Bug 2: Incorrect DMA size calculation (includes NULL terminator in buffer size)
 *   Bug 3: Missing length validation before DMA write
 */

#include "redismodule.h"
#include <string.h>
#include <stdlib.h>

/* Command: BUGGY.SET key value
 * Stores a string value using StringDMA.
 */
int BuggySet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 3) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_ERR;
    }

    RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ | REDISMODULE_WRITE);
    if (!key) {
        RedisModule_ReplyWithError(ctx, "ERR failed to open key");
        return REDISMODULE_ERR;
    }

    size_t len;
    const char *value = RedisModule_StringPtrLen(argv[2], &len);

    /* Create or resize the string */
    RedisModule_StringTruncate(key, len);

    /* Get DMA access to the string buffer */
    size_t dma_len;
    char *dma_buf = RedisModule_StringDMA(key, &dma_len, REDISMODULE_WRITE);

    if (!dma_buf) {
        RedisModule_ReplyWithError(ctx, "ERR failed to get DMA buffer");
        RedisModule_CloseKey(key);
        return REDISMODULE_ERR;
    }

    /*
     * BUG 1: Off-by-one write
     * Correct: memcpy(dma_buf, value, len);
     * Buggy:   memcpy(dma_buf, value, len + 1);   <-- writes 1 byte past buffer end
     */
    memcpy(dma_buf, value, len + 1);  /* BUG: off-by-one, writes past buffer */

    RedisModule_CloseKey(key);
    RedisModule_ReplyWithSimpleString(ctx, "OK");
    return REDISMODULE_OK;
}

/* Command: BUGGY.SETNX key value
 * Like BUGGY.SET but with different DMA size calculation.
 */
int BuggySetNX_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 3) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_ERR;
    }

    RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ | REDISMODULE_WRITE);
    if (!key) {
        RedisModule_ReplyWithError(ctx, "ERR failed to open key");
        return REDISMODULE_ERR;
    }

    /* Check if key already exists */
    if (RedisModule_KeyType(key) != REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_ReplyWithLongLong(ctx, 0);
        RedisModule_CloseKey(key);
        return REDISMODULE_OK;
    }

    size_t len;
    const char *value = RedisModule_StringPtrLen(argv[2], &len);

    /*
     * BUG 2: Incorrect size calculation
     * Correct: RedisModule_StringTruncate(key, len);
     * Buggy:   RedisModule_StringTruncate(key, len + 1);  <-- allocates extra byte
     *          then copies len+1 bytes (including potential garbage)
     */
    RedisModule_StringTruncate(key, len + 1);  /* BUG: allocates 1 extra byte */

    size_t dma_len;
    char *dma_buf = RedisModule_StringDMA(key, &dma_len, REDISMODULE_WRITE);

    if (!dma_buf) {
        RedisModule_ReplyWithError(ctx, "ERR failed to get DMA buffer");
        RedisModule_CloseKey(key);
        return REDISMODULE_ERR;
    }

    memcpy(dma_buf, value, len + 1);  /* BUG: copies 1 extra byte */

    RedisModule_CloseKey(key);
    RedisModule_ReplyWithLongLong(ctx, 1);
    return REDISMODULE_OK;
}

/* Command: BUGGY.APPEND key value
 * Appends to a string using DMA.
 */
int BuggyAppend_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 3) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_ERR;
    }

    RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ | REDISMODULE_WRITE);
    if (!key) {
        RedisModule_ReplyWithError(ctx, "ERR failed to open key");
        return REDISMODULE_ERR;
    }

    size_t old_len = 0;
    size_t append_len;
    const char *append_val = RedisModule_StringPtrLen(argv[2], &append_len);

    if (RedisModule_KeyType(key) != REDISMODULE_KEYTYPE_EMPTY) {
        size_t cur_len;
        char *cur_buf = RedisModule_StringDMA(key, &cur_len, REDISMODULE_READ);
        if (cur_buf) old_len = cur_len;
    }

    size_t new_len = old_len + append_len;
    RedisModule_StringTruncate(key, new_len);

    size_t dma_len;
    char *dma_buf = RedisModule_StringDMA(key, &dma_len, REDISMODULE_WRITE);

    if (!dma_buf) {
        RedisModule_ReplyWithError(ctx, "ERR failed to get DMA buffer");
        RedisModule_CloseKey(key);
        return REDISMODULE_ERR;
    }

    /*
     * BUG 3: Missing bounds check
     * Correct: if (old_len + append_len <= dma_len) { memcpy(dma_buf + old_len, append_val, append_len); }
     * Buggy:   memcpy(dma_buf + old_len, append_val, append_len + 1);  <-- no bounds check, off-by-one
     */
    memcpy(dma_buf + old_len, append_val, append_len + 1);  /* BUG: no bounds check, off-by-one */

    RedisModule_CloseKey(key);
    RedisModule_ReplyWithSimpleString(ctx, "OK");
    return REDISMODULE_OK;
}

/* Command: BUGGY.GET key
 * Retrieves a string value.
 */
int BuggyGet_RedisCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 2) {
        RedisModule_WrongArity(ctx);
        return REDISMODULE_ERR;
    }

    RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], REDISMODULE_READ);
    if (!key) {
        RedisModule_ReplyWithNull(ctx);
        return REDISMODULE_OK;
    }

    if (RedisModule_KeyType(key) == REDISMODULE_KEYTYPE_EMPTY) {
        RedisModule_ReplyWithNull(ctx);
        RedisModule_CloseKey(key);
        return REDISMODULE_OK;
    }

    size_t len;
    char *dma_buf = RedisModule_StringDMA(key, &len, REDISMODULE_READ);

    if (!dma_buf) {
        RedisModule_ReplyWithNull(ctx);
        RedisModule_CloseKey(key);
        return REDISMODULE_OK;
    }

    /* Use HGET style - the hash field is embedded in the key name */
    RedisModule_ReplyWithStringBuffer(ctx, dma_buf, len);
    RedisModule_CloseKey(key);
    return REDISMODULE_OK;
}

int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    REDISMODULE_NOT_USED(argv);
    REDISMODULE_NOT_USED(argc);

    if (RedisModule_Init(ctx, "buggy", 1, REDISMODULE_APIVER_1) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "buggy.set", BuggySet_RedisCommand,
            "write", 1, 1, 1) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "buggy.setnx", BuggySetNX_RedisCommand,
            "write", 1, 1, 1) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "buggy.append", BuggyAppend_RedisCommand,
            "write", 1, 1, 1) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    if (RedisModule_CreateCommand(ctx, "buggy.get", BuggyGet_RedisCommand,
            "read", 1, 1, 1) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }

    return REDISMODULE_OK;
}
