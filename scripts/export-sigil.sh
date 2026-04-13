#!/usr/bin/env bash
# Export a .sigil directory tree into a single flat file for review.
# Usage: ./export-sigil.sh <path-to.sigil> [output-file]
#
# Output format per file:
#   ══ relative/path/to/file.md ══
#   <file contents>
#   <blank line>
#
# Skips .private/ and .sigil/ directories, .DS_Store files.
# Includes: *.md, *.order, *.folded

set -euo pipefail

SIGIL_DIR="${1:?Usage: export-sigil.sh <sigil-dir> [output-file]}"
OUTPUT="${2:-$(dirname "$SIGIL_DIR")/$(basename "$SIGIL_DIR").exported.txt}"

# Resolve to absolute
SIGIL_DIR="$(cd "$SIGIL_DIR" && pwd)"

{
  echo "# Sigil Export: $(basename "$SIGIL_DIR")"
  echo "# Generated: $(date -Iseconds)"
  echo "# Source: $SIGIL_DIR"
  echo ""

  find "$SIGIL_DIR" \( -name '.private' -o -name '.sigil' \) -prune -o \
    \( -name '*.md' -o -name '*.order' -o -name '*.folded' \) \
    -not -name '.DS_Store' -print \
    | sort \
    | while IFS= read -r filepath; do
        rel="${filepath#"$SIGIL_DIR"/}"
        echo "══ $rel ══"
        cat "$filepath"
        echo ""
      done
} > "$OUTPUT"

wc -l < "$OUTPUT" | xargs -I{} echo "Wrote {} lines to $OUTPUT"
