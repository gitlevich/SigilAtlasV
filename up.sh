#!/bin/bash
# Start SigilAtlas: sidecar + Tauri app.
# Ctrl-C stops everything gracefully.
PORT=8321
DIR="$(cd "$(dirname "$0")" && pwd)"

SIDECAR_PID=""
cleanup() {
    [ -n "$SIDECAR_PID" ] && kill $SIDECAR_PID 2>/dev/null
    exit 0
}
trap cleanup INT TERM

# Kill existing sidecar
kill $(lsof -ti:$PORT) 2>/dev/null

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
