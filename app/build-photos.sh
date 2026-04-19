#!/bin/bash
# Build sa-photos and stage it under src-tauri/binaries/ with the
# target-triple suffix Tauri's externalBin mechanism expects.
set -euo pipefail

cd "$(dirname "$0")/src-photos"

CONFIG="${1:-release}"
swift build -c "$CONFIG" >&2

TRIPLE="$(rustc -vV | sed -n 's|host: ||p')"
SRC=".build/$(uname -m)-apple-macosx/$CONFIG/sa-photos"
DEST_DIR="../src-tauri/binaries"
DEST="$DEST_DIR/sa-photos-$TRIPLE"

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"
echo "[build-photos] staged $DEST" >&2
