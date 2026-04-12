#!/usr/bin/env bash
# Compile-check the SigilAtlas sigil spec.
# Invokes the sigil compiler from the Sigil app repo without copying it.
set -euo pipefail

SIGIL_APP="/Users/vlad/Attention Lab/sigil-specs/sigil"
SIGIL_ROOT="${1:-/Users/vlad/SigilAtlas/sigil_atlas.sigil}"

cd "$SIGIL_APP"
exec npx tsx scripts/compile-check.ts "$SIGIL_ROOT"
