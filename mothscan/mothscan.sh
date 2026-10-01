#!/usr/bin/env bash
# mothscan convenience wrapper.
#
#   ./mothscan.sh setup                  create the venv and install deps
#   ./mothscan.sh build [--workers N]    scan -> preview -> export
#   ./mothscan.sh serve                  browser viewer on :8770
#   ./mothscan.sh stats | inspect FILE | diagnose
#   ./mothscan.sh retune novelty=1.2 tonality=1.0
#   ./mothscan.sh preview --gap 0.2 --cut 0.4 --rate 22050
#   ./mothscan.sh preview --missing skip          # shorter, gapless per-site streams
#   ./mothscan.sh export --only-gate
#   ./mothscan.sh verify                 headless behavioural test of the viewer
#
# Everything lands under data/; the external drive is only read.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$HERE/../.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "no venv at $PY — run: $0 setup" >&2
  exit 1
fi

# matplotlib and fontconfig both want a writable cache directory; the default
# ~/.matplotlib is not writable on this machine.
export MPLCONFIGDIR="$HERE/../data/.mplcache"
mkdir -p "$MPLCONFIGDIR"

cmd="${1:-}"; shift || true
case "$cmd" in
  setup)
    python3 -m venv "$HERE/../.venv"
    "$HERE/../.venv/bin/pip" install --upgrade pip
    "$HERE/../.venv/bin/pip" install numpy scipy soundfile matplotlib fastapi "uvicorn[standard]"
    ;;

  build)
    # Full pipeline in the order that matters: points, then the stream that
    # makes them audible, then the files Max reads.  Must not use `exec` here,
    # or the chain would stop after the first stage.
    "$PY" -m mothscan.cli scan "$@"
    "$PY" -m mothscan.cli preview --cut 0.5 --gap 0.05 --rate 16000 --name scan
    "$PY" -m mothscan.cli export --cut 0.5 --gap 0.05
    echo
    echo "build complete:"
    "$PY" -m mothscan.cli stats | head -6
    ;;

  serve)     exec "$PY" -m mothscan.cli serve --port 8770 "$@" ;;
  stats)     exec "$PY" -m mothscan.cli stats "$@" ;;
  inspect)   exec "$PY" -m mothscan.cli inspect "$@" ;;
  pick)      exec "$PY" -m mothscan.cli pick "$@" ;;
  preview)   exec "$PY" -m mothscan.cli preview "$@" ;;
  export)    exec "$PY" -m mothscan.cli export "$@" ;;
  retune)    exec "$PY" -m mothscan.cli retune --weights "$@" ;;

  diagnose)
    exec "$PY" scripts/diagnose.py "$@"
    ;;

  verify)
    command -v node >/dev/null || { echo "node is required for verify" >&2; exit 1; }
    url="http://127.0.0.1:8770"
    if ! curl -sf -o /dev/null "$url/api/state"; then
      echo "starting a temporary viewer for the test…"
      "$PY" -m mothscan.cli serve --port 8770 >/tmp/mothscan-verify-server.log 2>&1 &
      srv=$!
      trap 'kill $srv 2>/dev/null || true' EXIT
      for _ in $(seq 1 30); do curl -sf -o /dev/null "$url/api/state" && break; sleep 0.5; done
    fi
    exec node scripts/verify_ui.mjs "$url"
    ;;

  *)
    # Print the header comment block only (stop at the first non-comment line).
    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
    if [[ -n "$cmd" ]]; then echo >&2; echo "unknown command: $cmd" >&2; exit 2; fi
    ;;
esac
