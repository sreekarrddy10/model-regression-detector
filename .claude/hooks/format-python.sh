#!/usr/bin/env bash
# PostToolUse hook: format and check a Python file after it is written.
#
# Reads the tool payload on stdin and exits immediately for anything that is not
# a .py file, so editing markdown costs nothing. Measured on this repo: black
# 0.25s, isort 0.01s, ruff 0.01s, mypy 0.22s warm - under a second total, which
# is the budget where a save-time hook stays useful rather than annoying.
#
# Never fails the tool call. A formatter that blocks an edit is worse than an
# unformatted file; `make lint` and CI are the enforcement points.
set -uo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PY="$ROOT/.venv/bin"
[ -x "$PY/black" ] || exit 0

# -c, not a heredoc: `python - <<EOF` would take the program from stdin and leave
# nothing for json.load to read, which fails silently and gives you a hook that
# looks installed and does nothing.
FILE=$("$PY/python" -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)
path = (payload.get("tool_input") or {}).get("file_path", "")
print(path if path.endswith((".py", ".pyi")) else "")
' 2>/dev/null)

[ -n "$FILE" ] || exit 0
[ -f "$FILE" ] || exit 0

cd "$ROOT" || exit 0
"$PY/black" -q "$FILE" 2>/dev/null
"$PY/isort" -q "$FILE" 2>/dev/null
"$PY/ruff" check --fix -q "$FILE" 2>/dev/null

# Type-check the package, not the single file: mypy needs the whole import graph
# to be meaningful, and the warm cache makes it affordable.
if ! "$PY/mypy" 2>/dev/null | tail -20 | grep -q "^Success"; then
    echo "mypy reports errors - run: make lint" >&2
fi

exit 0
