#!/bin/sh
# =============================================================
# Mira — BSPWM X11 session entry point (launched by SDDM).
# =============================================================

MIRA_DIR="${MIRA_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/bspwm}"
export MIRA_CONFIG_DIR="$MIRA_DIR"
export PATH="$HOME/.local/bin:$MIRA_DIR/bin:$PATH"
export XDG_CURRENT_DESKTOP='bspwm'
export XDG_SESSION_TYPE='x11'
export XDG_SESSION_DESKTOP='bspwm'
export DESKTOP_SESSION='bspwm'
export XCURSOR_THEME='everforest-cursors'
export XCURSOR_SIZE='24'
export XCURSOR_PATH="$HOME/.local/share/icons:$HOME/.icons:/usr/share/icons:/usr/share/pixmaps"

setxkbmap -layout us -option '' 2>/dev/null || true

systemctl --user import-environment DISPLAY XDG_CURRENT_DESKTOP \
    2>/dev/null || true

exec bspwm
