# Task 9: PostgreSQL Executor NULL Handling Bug
#
# Build:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task9 -f tasks/task9-postgres-executor-null/task/environment/Dockerfile .

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PG_MAJOR=15

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential libreadline-dev zlib1g-dev libssl-dev \
    libxml2-dev libxslt-dev libsystemd-dev libicu-dev \
    python3 python3-pip git wget ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Build PostgreSQL 15 from source so the source tree is available at /usr/src/postgresql
RUN wget -q https://ftp.postgresql.org/pub/source/v${PG_MAJOR}.13/postgresql-${PG_MAJOR}.13.tar.gz -O /tmp/pg.tar.gz && \
    mkdir -p /usr/src && \
    tar xzf /tmp/pg.tar.gz -C /usr/src && \
    mv /usr/src/postgresql-${PG_MAJOR}.13 /usr/src/postgresql && \
    rm /tmp/pg.tar.gz

WORKDIR /usr/src/postgresql
RUN ./configure --prefix=/usr/local \
    --with-openssl --with-icu --with-libxml --with-libxslt 2>&1 | tail -5 && \
    make -j$(nproc) 2>&1 | tail -5 && \
    make install 2>&1 | tail -5

# Initialize PostgreSQL data directory and create test database
RUN useradd -m postgres 2>/dev/null || true && \
    mkdir -p /var/lib/postgresql/data /var/run/postgresql && \
    chown -R postgres:postgres /var/lib/postgresql /var/run/postgresql && \
    su postgres -c "/usr/local/bin/initdb -D /var/lib/postgresql/data --locale=en_US.UTF-8" && \
    echo "unix_socket_directories = '/var/run/postgresql'" >> /var/lib/postgresql/data/postgresql.conf && \
    echo "listen_addresses = 'localhost'" >> /var/lib/postgresql/data/postgresql.conf && \
    su postgres -c "/usr/local/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/logfile start" && \
    sleep 2 && \
    su postgres -c "/usr/local/bin/createdb testdb" && \
    su postgres -c "/usr/local/bin/pg_ctl -D /var/lib/postgresql/data stop"

# Install Node.js and Kimi Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Install Python packages
RUN pip3 install psycopg2-binary

# Inject bugs into PostgreSQL executor source
COPY tasks/task9-postgres-executor-null/task/solution/inject_bug.py /tmp/inject_bug.py
ENV PG_SRC=/usr/src/postgresql
RUN python3 /tmp/inject_bug.py 2>&1 && rm /tmp/inject_bug.py

# Recompile PostgreSQL with bugs injected
RUN cd /usr/src/postgresql && \
    make -j$(nproc) -C src/backend/executor 2>&1 | tail -3 && \
    make -j$(nproc) 2>&1 | tail -3 && \
    make install 2>&1 | tail -3

# Restart PostgreSQL with buggy binaries
RUN su postgres -c "/usr/local/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/logfile restart" && \
    sleep 2

# Copy workspace and test files
COPY tasks/task9-postgres-executor-null/task/workspace/ /workspace/
COPY tasks/task9-postgres-executor-null/task/tests/ /task/tests/
COPY tasks/task9-postgres-executor-null/task/solution/ /task/solution/
COPY tasks/task9-postgres-executor-null/task/instruction.md /task/instruction.md

# Create logs directory
RUN mkdir -p /logs/verifier

WORKDIR /workspace

CMD ["/bin/bash"]
