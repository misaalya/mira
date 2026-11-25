#!/bin/sh
# Run a full, isolated Mira session inside Xephyr (:9) + dbus-run-session
# without touching ~/.config, ~/.local/share, ~/.cache or any live X11 (:0)
# daemon.
set -eu

ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
DISP_NUM="${MIRA_TEST_DISPLAY:-:9}"
SCREEN_SIZE="${MIRA_TEST_SCREEN:-1920x1080}"

[ -d "$ROOT/build/bin" ] || "$ROOT/tools/build.sh"

SANDBOX=$(mktemp -d /tmp/mira-dev-sandbox.XXXXXX)
XPID=""
SESSION_PID=""

kill_display_procs() {
    for p in $(pgrep -u "$(id -u)" 2>/dev/null); do
        [ "$p" = "$$" ] && continue
        [ -r "/proc/$p/environ" ] || continue
        d=$(tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null | sed -n 's/^DISPLAY=//p' | head -n 1)
        [ "$d" = "$DISP_NUM" ] && kill "$p" 2>/dev/null || true
    done
}

cleanup() {
    [ -n "$SESSION_PID" ] && kill "$SESSION_PID" 2>/dev/null || true
    kill_display_procs
    sleep 0.2
    [ -n "$XPID" ] && kill "$XPID" 2>/dev/null || true
    rm -rf "$SANDBOX"
}
trap cleanup EXIT INT TERM

mkdir -p "$SANDBOX/run" "$SANDBOX/cache" "$SANDBOX/walls" "$SANDBOX/config/bspwm" \
    "$SANDBOX/data" "$SANDBOX/state"
chmod 700 "$SANDBOX/run"

# Daemons keep user data under XDG_DATA_HOME (misa-clip: clipboard history),
# so the sandbox gets its own. Fonts, icons and desktop entries are only read:
# link them in so the session still looks and launches like the real one.
HOST_DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
for d in fonts icons applications; do
    [ -d "$HOST_DATA/$d" ] && ln -s "$HOST_DATA/$d" "$SANDBOX/data/$d"
done

cp -a "$ROOT/session/." "$SANDBOX/config/bspwm/"
cp -a "$ROOT/build/bin/." "$SANDBOX/config/bspwm/bin/"
printf 'mira\n' >"$SANDBOX/config/bspwm/.theme"
chmod 755 "$SANDBOX/config/bspwm/bspwmrc" "$SANDBOX/config/bspwm/bin/"* \
    "$SANDBOX/config/bspwm/bars/"*/scripts/* 2>/dev/null || true

Xephyr "$DISP_NUM" -screen "$SCREEN_SIZE" -noreset -ac >/dev/null 2>&1 &
XPID=$!
sleep 0.6

export DISPLAY="$DISP_NUM"
export MIRA_CONFIG_DIR="$SANDBOX/config/bspwm"
export MIRA_SANDBOX=1
export MIRA_USER_WALL_DIR="$SANDBOX/walls"
export XDG_RUNTIME_DIR="$SANDBOX/run"
export XDG_CACHE_HOME="$SANDBOX/cache"
export XDG_DATA_HOME="$SANDBOX/data"
export XDG_STATE_HOME="$SANDBOX/state"
export BSPWM_SOCKET="$SANDBOX/bspwm.sock"
export PATH="$SANDBOX/config/bspwm/bin:$PATH"

# Give Xephyr's "default" output a normal monitor name (rofi rejects "-m default").
W=${SCREEN_SIZE%x*}
H=${SCREEN_SIZE#*x}
xrandr --setmonitor eDP-1 "${W}/508x${H}/286+0+0" default >/dev/null 2>&1 || true

dbus-run-session -- sh -c '
    bspwm -c "$MIRA_CONFIG_DIR/bspwmrc"
' >"$SANDBOX/session.log" 2>&1 &
SESSION_PID=$!

echo "Mira isolated session running on DISPLAY=$DISPLAY (Xephyr PID=$XPID)."
echo "Config: $MIRA_CONFIG_DIR  |  Log: $SANDBOX/session.log"
echo "Press Ctrl+C to stop and clean up."
wait "$XPID"
