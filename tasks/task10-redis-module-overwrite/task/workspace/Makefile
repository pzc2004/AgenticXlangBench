# Makefile for the buggy Redis module
# Usage: make

CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC
LDFLAGS = -shared

# Redis module API headers (adjust path if needed)
REDIS_INCLUDE ?= /usr/local/include/redis
REDIS_SERVER_INCLUDE ?= /build/redis/src

TARGET = module.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): module.c
	$(CC) $(CFLAGS) -I$(REDIS_INCLUDE) -I$(REDIS_SERVER_INCLUDE) \
		$(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
