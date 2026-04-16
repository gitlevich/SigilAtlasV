#!/usr/bin/env bash
set -euo pipefail

# Release script — bumps VERSION, tags, pushes. The GitHub Actions
# workflow (.github/workflows/release.yml) builds the DMG.
#
# Usage:
#   ./scripts/release.sh           # patch bump (0.1.0 → 0.1.1)
#   ./scripts/release.sh minor     # minor bump (0.1.1 → 0.2.0)
#   ./scripts/release.sh major     # major bump (0.2.0 → 1.0.0)

cd "$(git rev-parse --show-toplevel)"

BUMP="${1:-patch}"

# Read current version
CURRENT=$(cat VERSION)
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "$BUMP" in
  patch) PATCH=$((PATCH + 1)) ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  *) echo "Usage: $0 [patch|minor|major]"; exit 1 ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
TAG="v$NEW_VERSION"

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is dirty. Commit or stash first."
  exit 1
fi

# Check tag doesn't already exist
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "Tag $TAG already exists."
  exit 1
fi

echo "$CURRENT -> $NEW_VERSION"

# Write version
echo "$NEW_VERSION" > VERSION

# Commit and tag
git add VERSION
git commit -m "Release $TAG"
git tag "$TAG"

# Push both
git push origin master "$TAG"

# Print link to watch the build
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo ""
echo "Tagged $TAG and pushed. Watch the build:"
echo "  https://github.com/$REPO/actions"
echo ""
echo "When done, the DMG will be at:"
echo "  https://github.com/$REPO/releases/tag/$TAG"
