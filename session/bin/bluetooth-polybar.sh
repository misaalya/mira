#!/bin/sh
# =============================================================
# bluetooth-polybar.sh — Polybar Bluetooth status icon for Mira.
# Adapted from gh0stzk/dotfiles (GPL-3.0).
# Original work: https://github.com/gh0stzk/dotfiles
# Copyright (C) 2021-2026 gh0stzk <z0mbi3.zk@protonmail.com>
# Copyright (C) 2026 Misaalya
# Modified 2026-11-23 by Misaalya (Debian 13)
# SPDX-License-Identifier: GPL-3.0-only
# =============================================================

MIRA_DIR="${MIRA_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/bspwm}"
COLORS="$MIRA_DIR/bars/colors.ini"

get_color() {
    name=$1
    [ -r "$COLORS" ] && awk -F'=' -v k="$name" '
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" { gsub(/[[:space:]]/, "", $2); print $2; exit }
    ' "$COLORS"
}

POWER_ON=$(get_color "${1:-fg}")
POWER_OFF=$(get_color "${2:-grey}")
POWER_ON=${POWER_ON:-#e6e6e6}
POWER_OFF=${POWER_OFF:-#7a7a7a}

get_bt_status() {
    if command -v busctl >/dev/null 2>&1; then
        val=$(busctl get-property org.bluez /org/bluez/hci0 org.bluez.Adapter1 Powered 2>/dev/null)
        case "$val" in
            "b true")  return 0 ;;
            "b false") return 2 ;;
        esac
    fi
    systemctl is-active bluetooth.service >/dev/null 2>&1 || return 1
    bluetoothctl show 2>/dev/null | grep -q "Powered: yes" && return 0 || return 2
}

get_bt_status
case $? in
    0) printf '%%{F%s}󰂯%%{F-}\n' "$POWER_ON" ;;
    2) printf '%%{F%s}󰂲%%{F-}\n' "$POWER_OFF" ;;
esac
exit 0
