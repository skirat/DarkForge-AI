#!/usr/bin/env bash
# Commit and push using this repo's Git user (personal: see .git/config) and origin.
#
# Usage:
#   ./scripts/git-commit-push-personal.sh "Your commit message"
#   ./scripts/git-commit-push-personal.sh --force-with-lease "Message after amend / history rewrite"
#   ./scripts/git-commit-push-personal.sh --push-only   # push only (no commit)
#
# Requires origin to point at your GitHub repo (e.g. git@github.com-personal:skirat/DarkForge-AI.git).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FORCE=()
PUSH_ONLY=0
MSG_PARTS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force-with-lease|-f)
      FORCE=(--force-with-lease)
      shift
      ;;
    --push-only)
      PUSH_ONLY=1
      shift
      ;;
    *)
      MSG_PARTS+=("$1")
      shift
      ;;
  esac
done

MSG="${MSG_PARTS[*]:-}"
if [[ -z "$MSG" && "$PUSH_ONLY" -eq 0 ]]; then
  MSG="Update $(date -u +%Y-%m-%dT%H:%MZ)"
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
ORIGIN="$(git remote get-url origin 2>/dev/null || true)"
if [[ -z "$ORIGIN" ]]; then
  echo "error: no remote named origin" >&2
  exit 1
fi

if [[ "$PUSH_ONLY" -eq 1 ]]; then
  git push "${FORCE[@]}" origin "$BRANCH"
  echo "Pushed $BRANCH to origin ($ORIGIN)"
  exit 0
fi

git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit (staged empty). Use --push-only to push existing commits."
  exit 0
fi

git commit -m "$MSG"
git push "${FORCE[@]}" origin "$BRANCH"
echo "Committed and pushed $BRANCH to origin ($ORIGIN)"
