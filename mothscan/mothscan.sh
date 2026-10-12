#!/usr/bin/env bash
#
# mothscan - convenience wrapper.
#
# Run this from anywhere; it resolves its own location and works in place.
#
#   ./mothscan.sh setup            one-time: create the venv, install deps, check
#   ./mothscan.sh start            start the viewer in the background
#   ./mothscan.sh stop             stop it
#   ./mothscan.sh status           is it running?
#   ./mothscan.sh restart          stop then start
#   ./mothscan.sh logs             tail the viewer log
#   ./mothscan.sh open             start (if needed), print the URL, open a browser
#
#   ./mothscan.sh serve            run the viewer in the FOREGROUND (ctrl-c to stop)
#   ./mothscan.sh build            scan -> preview -> export, end to end
#   ./mothscan.sh test             unit tests (no data or server needed)
#   ./mothscan.sh verify           headless behavioural test of the viewer
#
#   ./mothscan.sh scan [--limit N]        analyse recordings
#   ./mothscan.sh stats                   coverage and gate rates
#   ./mothscan.sh preview --gap -0.25     render the per-site streams
#   ./mothscan.sh export --cut 0.5        write the Max/MSP coll files
#   ./mothscan.sh retune novelty=1.2      re-score from cache with new weights
#   ./mothscan.sh inspect FILE | pick FILE | diagnose FILE
#
# Configuration (all optional):
#   MOTHSCAN_SITE_ROOT   where the recordings are (default /Volumes/Solardisk/Harpefoss)
#   MOTHSCAN_SITES       comma-separated site names
#   MOTHSCAN_DATA_DIR    where caches/index/renders go (default <repo>/data)
#   MOTHSCAN_VENV        python venv to use (default: first of .venv, ../.venv)
#   MOTHSCAN_PORT        viewer port (default 8770)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PORT="${MOTHSCAN_PORT:-8770}"
HOST="${MOTHSCAN_HOST:-127.0.0.1}"
URL="http://$HOST:$PORT"

# --------------------------------------------------------------------------
# find a python that has the dependencies
# --------------------------------------------------------------------------
# A fresh clone has no venv, while the original checkout kept it one level up.
# Try both, plus an explicit override, and only then fall back to the system
# interpreter so an already-provisioned machine still works.
find_python() {
  local candidates=()
  [[ -n "${MOTHSCAN_VENV:-}" ]] && candidates+=("${MOTHSCAN_VENV%/}/bin/python")
  candidates+=("$HERE/.venv/bin/python" "$HERE/../.venv/bin/python")

  local c
  for c in "${candidates[@]}"; do
    if [[ -x "$c" ]] && "$c" -c "import numpy, scipy, soundfile" >/dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1 \
     && python3 -c "import numpy, scipy, soundfile" >/dev/null 2>&1; then
    command -v python3; return 0
  fi
  return 1
}

die_no_python() {
  cat >&2 <<EOF
No usable Python found.

mothscan needs numpy, scipy and soundfile. None of these had them:
  - ${MOTHSCAN_VENV:-<MOTHSCAN_VENV unset>}/bin/python
  - $HERE/.venv/bin/python
  - $HERE/../.venv/bin/python
  - python3 on PATH

Fix it with:

  cd "$HERE"
  ./mothscan.sh setup

or point at an existing environment:

  MOTHSCAN_VENV=/path/to/venv ./mothscan.sh start
EOF
  exit 1
}

PY="$(find_python)" || die_no_python

# matplotlib and fontconfig both want a writable cache directory, and the
# default ~/.matplotlib is not writable on every machine.
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HERE/../data/.mplcache}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

needs_web() {
  if ! "$PY" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    echo "the viewer needs fastapi and uvicorn, which this python lacks:" >&2
    echo "  $PY" >&2
    echo "install them with:  $PY -m pip install fastapi 'uvicorn[standard]'" >&2
    exit 1
  fi
}

cmd="${1:-}"; shift || true

case "$cmd" in
  # ---------------------------------------------------------------- setup ---
  setup)
    VENV="${MOTHSCAN_VENV:-$HERE/.venv}"
    echo "creating venv at $VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r "$HERE/requirements.txt"
    echo
    echo "checking the install"
    "$VENV/bin/python" - <<'PYCHECK'
import importlib, sys
mods = ["numpy", "scipy", "soundfile", "matplotlib", "fastapi", "uvicorn"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("  MISSING:", ", ".join(missing))
    sys.exit(1)
print("  all dependencies import:", ", ".join(mods))
PYCHECK
    echo
    echo "setup complete.  next:"
    echo "  export MOTHSCAN_SITE_ROOT=/path/to/recordings"
    echo "  ./mothscan.sh scan --limit 200      # try a sample first"
    echo "  ./mothscan.sh start                 # viewer on $URL"
    ;;

  # -------------------------------------------------------- start / stop ----
  start|bg)
    needs_web
    exec "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --bg "$@"
    ;;

  stop)
    exec "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --stop
    ;;

  status)
    "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --status
    ;;

  restart)
    "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --stop || true
    exec "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --bg "$@"
    ;;

  logs|log)
    LOG="${MOTHSCAN_DATA_DIR:-$HERE/../data}/viewer.log"
    if [[ ! -f "$LOG" ]]; then
      echo "no log at $LOG (has the viewer been started?)" >&2
      exit 1
    fi
    echo "tailing $LOG  (ctrl-c to stop)"
    exec tail -n 40 -f "$LOG"
    ;;

  open|url)
    needs_web
    if ! "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --status >/dev/null 2>&1; then
      "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --bg >/dev/null
    fi
    echo "$URL"
    if command -v open >/dev/null 2>&1 && [[ -z "${MOTHSCAN_NO_BROWSER:-}" ]]; then
      open "$URL"
    fi
    ;;

  serve|fg)
    needs_web
    exec "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" "$@"
    ;;

  # ------------------------------------------------------------ pipeline ----
  build)
    # Full pipeline in the order that matters: points, then the stream that
    # makes them audible, then the files Max reads.  No `exec` here, or the
    # chain would stop after the first stage.
    "$PY" -m mothscan.cli scan "$@"
    "$PY" -m mothscan.cli preview --cut 0.5 --gap 0.05 --rate 16000 --name scan
    "$PY" -m mothscan.cli export --cut 0.5 --gap 0.05
    echo
    echo "build complete:"
    "$PY" -m mothscan.cli stats | head -6
    ;;

  scan|stats|preview|export|inspect|pick|retune)
    exec "$PY" -m mothscan.cli "$cmd" "$@"
    ;;

  diagnose)
    exec "$PY" scripts/diagnose.py "$@"
    ;;

  test|tests)
    exec "$PY" tests/test_preview.py "$@"
    ;;

  verify)
    command -v node >/dev/null || { echo "node is required for verify" >&2; exit 1; }
    started_here=0
    if ! "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --status >/dev/null 2>&1; then
      echo "starting a viewer for the test..."
      "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --bg >/dev/null
      started_here=1
    fi
    status=0
    node scripts/verify_ui.mjs "$URL" || status=$?
    if [[ $started_here == 1 ]]; then
      "$PY" -m mothscan.cli serve --host "$HOST" --port "$PORT" --stop >/dev/null || true
    fi
    exit $status
    ;;

  *)
    # Print the header comment block only: start at the first comment line,
    # stop at the first line that is not one.
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
    if [[ -n "$cmd" ]]; then
      echo >&2
      echo "unknown command: $cmd" >&2
      exit 2
    fi
    ;;
esac
