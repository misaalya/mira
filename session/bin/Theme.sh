#!/bin/sh
# =============================================================
# Theme.sh — Apply the active Mira theme and launch its bar.
# Adapted from gh0stzk/dotfiles (GPL-3.0) for Mira.
# Original work: https://github.com/gh0stzk/dotfiles
# Copyright (C) 2021-2026 gh0stzk <z0mbi3.zk@protonmail.com>
# Copyright (C) 2026 Misaalya
# Modified 2026-11-23 by Misaalya (Debian 13)
# SPDX-License-Identifier: GPL-3.0-only
# =============================================================

MIRA_DIR="${MIRA_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/bspwm}"
export MIRA_CONFIG_DIR="$MIRA_DIR"

# Serialize theme switches so rapid calls never leave duplicate bars.
_lock_dir="${XDG_RUNTIME_DIR:-/tmp}/mira-theme.lock.d"
_i=0
while ! mkdir "$_lock_dir" 2>/dev/null; do
    _owner=$(cat "$_lock_dir/pid" 2>/dev/null || true)
    if [ -z "$_owner" ]; then
        _age=$(($(date +%s) - $(stat -c %Y "$_lock_dir" 2>/dev/null || echo 0)))
        if [ "$_age" -ge 2 ]; then
            rm -rf "$_lock_dir"
            continue
        fi
    elif ! kill -0 "$_owner" 2>/dev/null; then
        rm -rf "$_lock_dir"
        continue
    fi
    _i=$((_i + 1))
    [ "$_i" -lt 600 ] || {
        echo "Theme.sh: another theme switch is still running" >&2
        exit 1
    }
    sleep 0.1
done
printf '%s\n' "$$" >"$_lock_dir/pid"
trap 'rm -rf "$_lock_dir"' EXIT
trap 'exit 1' HUP INT TERM
unset _i _owner _age

# Active theme (default: mira)
THEME="mira"
[ -r "$MIRA_DIR/.theme" ] && read -r THEME < "$MIRA_DIR/.theme"
[ -f "$MIRA_DIR/themes/$THEME/theme.conf" ] || THEME="mira"
# shellcheck disable=SC1090
. "$MIRA_DIR/themes/$THEME/theme.conf"

MODULE_DIR="$MIRA_DIR/config/modules"
DISPLAY="${DISPLAY:-:0}"
export DISPLAY

# Stop running bars on this X display cleanly before rebuilding.
"$MIRA_DIR/bin/StopBars" >/dev/null 2>&1 || true

# Atomic writer shared with config/modules/*.sh
_write() {
    target=$1
    dir=$(dirname "$target")
    mkdir -p "$dir"
    tmp=$(mktemp "$dir/.tmp.XXXXXX") || return 1
    cat >"$tmp" || { rm -f "$tmp"; return 1; }
    mv -f "$tmp" "$target" || { rm -f "$tmp"; return 1; }
}

# Run all theme modules
for module in "$MODULE_DIR"/*.sh; do
    [ -e "$module" ] || continue
    # shellcheck disable=SC1090
    . "$module"
done
unset module

# Reload native notification panel palette if running
command -v NotifPanel >/dev/null 2>&1 && NotifPanel reload >/dev/null 2>&1 || true

# Launch the selected bar
if [ -f "$MIRA_DIR/bars/$main_bar/Bar.bash" ]; then
    # shellcheck disable=SC1090
    . "$MIRA_DIR/bars/$main_bar/Bar.bash"
fi
