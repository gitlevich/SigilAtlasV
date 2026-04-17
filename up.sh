#!/bin/bash
# Start SigilAtlas: sidecar + Tauri app.
# Kills any orphan processes (sidecar, vite, tauri) from this directory first.
# Ctrl-C stops everything gracefully.
PORT=8321
VITE_PORT=1522
HMR_PORT=1421
DIR="$(cd "$(dirname "$0")" && pwd)"

SIDECAR_PID=""
cleanup() {
    [ -n "$SIDECAR_PID" ] && kill "$SIDECAR_PID" 2>/dev/null
    exit 0
}
trap cleanup INT TERM

# Kill any process listening on a port if its cwd is under $DIR.
# Does not touch processes from other projects that happen to use the port.
kill_ours_on_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null)
    for pid in $pids; do
        local cwd
        cwd=$(lsof -a -d cwd -p "$pid" -Fn 2>/dev/null | awk '/^n/{print substr($0,2)}')
        if [ -n "$cwd" ] && [[ "$cwd" == "$DIR"* ]]; then
            echo "[up] killing pid $pid on :$port (cwd=$cwd)"
            kill "$pid" 2>/dev/null
            sleep 0.2
            kill -9 "$pid" 2>/dev/null
        fi
    done
}

kill_ours_on_port "$PORT"
kill_ours_on_port "$VITE_PORT"
kill_ours_on_port "$HMR_PORT"

# Start sidecar in background
cd "$DIR/python"
.venv/bin/python -m sigil_atlas.serve --workspace "$DIR/workspace" --port $PORT &
SIDECAR_PID=$!

# Wait for sidecar health
for i in $(seq 1 20); do
    curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    sleep 0.2
done

# Start Tauri dev in foreground — blocks until app closes
cd "$DIR/app"
pnpm tauri dev

# App exited, clean up sidecar
cleanup
