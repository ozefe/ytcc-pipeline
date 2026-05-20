#!/usr/bin/env bash
#
# Stop the GROBID server started by `grobid_start.sh`.
#
# Usage:
#     scripts/grobid_stop.sh                # default paths, port 8070
#     GROBID_PORT=9090 scripts/grobid_stop.sh
#
# Environment overrides (all optional):
#     GROBID_HOME      Default: ~/grobid. Only used to derive GROBID_PID_FILE.
#     GROBID_PORT      Service port. Default: 8070. Used to confirm shutdown.
#     GROBID_PID_FILE  Pid-file path. Default: $GROBID_HOME/grobid.pid.
#     GROBID_STOP_TIMEOUT  Seconds to wait for graceful exit before SIGKILL.
#                          Default: 10.
#
# Stop is a single SIGTERM to the session leader's process group -- the
# `setsid` in grobid_start.sh ensured PID == PGID, so `kill -- -PID`
# takes down the wrapper, the gradle JVM, and the GROBID JVM in one go
# without needing pstree / lsof / psutil.
#
# Idempotent: stopping when nothing is running is a no-op + exit 0.

set -euo pipefail

GROBID_HOME="${GROBID_HOME:-$HOME/grobid}"
GROBID_PORT="${GROBID_PORT:-8070}"
GROBID_PID_FILE="${GROBID_PID_FILE:-$GROBID_HOME/grobid.pid}"
GROBID_STOP_TIMEOUT="${GROBID_STOP_TIMEOUT:-10}"

if [[ ! -f "$GROBID_PID_FILE" ]]; then
    echo "grobid_stop: no pid file at $GROBID_PID_FILE; nothing to stop"
    exit 0
fi

PID=$(cat "$GROBID_PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
    echo "grobid_stop: pid $PID not running; removing stale pid file"
    rm -f "$GROBID_PID_FILE"
    exit 0
fi

echo "grobid_stop: SIGTERM pgid=$PID"
# `--` prevents the leading dash on the negative PID from being parsed
# as a kill flag. Signalling the process group (`-PID`) hits every
# descendant the wrapper spawned.
kill -TERM -- -"$PID" 2>/dev/null || true

# Poll the leader pid until it exits (children typically die together).
for ((i = 1; i <= GROBID_STOP_TIMEOUT; i++)); do
    if ! kill -0 "$PID" 2>/dev/null; then
        rm -f "$GROBID_PID_FILE"
        echo "grobid_stop: stopped in ${i}s"
        exit 0
    fi
    sleep 1
done

echo "grobid_stop: graceful timeout -- SIGKILL pgid=$PID" >&2
kill -KILL -- -"$PID" 2>/dev/null || true
sleep 1
rm -f "$GROBID_PID_FILE"

# Final confirmation: nobody listening on the port anymore.
if curl -fsS "http://localhost:$GROBID_PORT/api/isalive" >/dev/null 2>&1; then
    echo "grobid_stop: WARNING -- port $GROBID_PORT still responding; another instance?" >&2
    exit 1
fi
echo "grobid_stop: stopped (forced)"
