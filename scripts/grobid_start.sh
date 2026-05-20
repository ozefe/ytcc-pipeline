#!/usr/bin/env bash
#
# Start the GROBID server in the background, tuned for reference parsing.
#
# Usage:
#     scripts/grobid_start.sh        # default paths, port 8070
#     GROBID_PORT=9090 scripts/grobid_start.sh
#
# Environment overrides (all optional):
#     GROBID_HOME      Source / install directory containing `gradlew` and
#                      `grobid-home/`. Default: ~/grobid.
#     JAVA_HOME        OpenJDK 21 install. Default: ~/.conda/envs/grobid-env
#                      (the conda env documented in docs/references.md).
#     GROBID_PORT      Service port. Default: 8070 (matches `cfg.grobid_url`).
#     GROBID_LOG_DIR   Server-log directory. Default: $GROBID_HOME/logs.
#     GROBID_PID_FILE  Pid-file path. Default: $GROBID_HOME/grobid.pid.
#     GROBID_READY_TIMEOUT  Seconds to wait for /api/isalive. Default: 90.
#
# Why a custom config:
#     The pipeline only ever calls /api/processCitationList. Forcing
#     `modelPreload: false` skips warm-loading the ~10 unrelated CRF
#     models (fulltext, header, figure, table,...) -- the citation model
#     loads lazily on the first request instead. Startup drops from
#     ~10 s to ~3 s and idle RSS shrinks by ~1 GiB.
#
# Idempotent: starting an already-running instance is a no-op + exit 0.

set -euo pipefail

GROBID_HOME="${GROBID_HOME:-$HOME/grobid}"
JAVA_HOME="${JAVA_HOME:-$HOME/.conda/envs/grobid-env}"
GROBID_PORT="${GROBID_PORT:-8070}"
GROBID_LOG_DIR="${GROBID_LOG_DIR:-$GROBID_HOME/logs}"
GROBID_PID_FILE="${GROBID_PID_FILE:-$GROBID_HOME/grobid.pid}"
GROBID_READY_TIMEOUT="${GROBID_READY_TIMEOUT:-90}"

die() { echo "grobid_start: $*" >&2; exit 1; }

[[ -d "$GROBID_HOME" ]] || die "GROBID_HOME=$GROBID_HOME does not exist. Clone https://github.com/kermitt2/grobid and rebuild (see docs/references.md)."
[[ -x "$GROBID_HOME/gradlew" ]] || die "missing $GROBID_HOME/gradlew -- is GROBID built?"
[[ -x "$JAVA_HOME/bin/java" ]] || die "JAVA_HOME=$JAVA_HOME has no bin/java"

# If the recorded pid is still alive AND owns the port, treat as up.
if [[ -f "$GROBID_PID_FILE" ]] && kill -0 "$(cat "$GROBID_PID_FILE")" 2>/dev/null; then
    if curl -fsS "http://localhost:$GROBID_PORT/api/isalive" >/dev/null 2>&1; then
        echo "grobid_start: already running, pid=$(cat "$GROBID_PID_FILE") port=$GROBID_PORT"
        exit 0
    fi
    echo "grobid_start: stale pid file ($(cat "$GROBID_PID_FILE")); removing" >&2
    rm -f "$GROBID_PID_FILE"
fi

mkdir -p "$GROBID_LOG_DIR"

# Generate a citation-only config snippet next to the canonical one. The
# Grobid YAML loader applies overrides last-write-wins inside the same
# file; we copy + patch rather than symlink so user-side tweaks survive.
SRC_CONFIG="$GROBID_HOME/grobid-home/config/grobid.yaml"
[[ -f "$SRC_CONFIG" ]] || die "missing $SRC_CONFIG -- has assemble completed?"
TUNED_CONFIG="$GROBID_HOME/grobid-home/config/grobid-reference-only.yaml"

# Idempotent: regenerate every start so changes to the upstream config
# carry through after a `git pull` in the GROBID checkout.
sed -E 's/^(\s*modelPreload:\s*).*/\1false/' "$SRC_CONFIG" > "$TUNED_CONFIG"
cp "$TUNED_CONFIG" "$SRC_CONFIG"

# `setsid` puts the wrapper in its own session, so its PID equals its
# PGID. `grobid_stop.sh` later signals -PID to take down the whole
# JVM-wrapper-JVM chain in one go without needing pstree.
LOG_FILE="$GROBID_LOG_DIR/grobid.log"
echo "grobid_start: starting at port $GROBID_PORT, log=$LOG_FILE"
(
    cd "$GROBID_HOME"
    setsid env \
        JAVA_HOME="$JAVA_HOME" PATH="$JAVA_HOME/bin:$PATH" \
        ./gradlew run --no-daemon \
        >"$LOG_FILE" 2>&1 \
        </dev/null &
    echo $! > "$GROBID_PID_FILE"
)
PID=$(cat "$GROBID_PID_FILE")
echo "grobid_start: pid=$PID (session leader)"

# Wait for /api/isalive. The first cold start downloads no models
# (everything ships in grobid-home/) but JVM warmup + Jetty bring-up
# takes ~10 s on an idle host.
for ((i = 1; i <= GROBID_READY_TIMEOUT; i++)); do
    if curl -fsS "http://localhost:$GROBID_PORT/api/isalive" 2>/dev/null | grep -q '^true$'; then
        echo "grobid_start: ready in ${i}s -- http://localhost:$GROBID_PORT/api/isalive returned true"
        exit 0
    fi
    sleep 1
done

echo "grobid_start: not ready after ${GROBID_READY_TIMEOUT}s; check $LOG_FILE" >&2
exit 1
