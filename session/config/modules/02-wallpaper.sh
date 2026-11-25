#!/bin/sh
# Wallpaper module for Mira (uses native misa-wall SIMD engine, fallback feh).
# Original work: https://github.com/gh0stzk/dotfiles
# Copyright (C) 2021-2026 gh0stzk <z0mbi3.zk@protonmail.com>
# Copyright (C) 2026 Misaalya
# Modified 2026-11-23 by Misaalya (Debian 13)
# SPDX-License-Identifier: GPL-3.0-only
# Adapted from gh0stzk/dotfiles (GPL-3.0).

pick_random_wall() {
    find "$1" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) 2>/dev/null | shuf -n 1
}

set_wall() {
    [ -n "$1" ] && [ -f "$1" ] || return 0
    if command -v misa-wall >/dev/null 2>&1; then
        misa-wall "$1"
    elif command -v feh >/dev/null 2>&1; then
        feh --bg-fill "$1"
    fi
    # Blurred copy for the login screen (no-op without the SDDM theme).
    command -v SddmBackground >/dev/null 2>&1 && SddmBackground "$1" >/dev/null 2>&1 &
}

case "${ENGINE:-Default}" in
    Random)
        set_wall "$(pick_random_wall "$MIRA_DIR/themes/$THEME/walls")"
        ;;
    CustomDir)
        if [ -n "${CUSTOM_DIR:-}" ] && [ -d "$CUSTOM_DIR" ]; then
            set_wall "$(pick_random_wall "$CUSTOM_DIR")"
        else
            set_wall "$(pick_random_wall "$MIRA_DIR/themes/$THEME/walls")"
        fi
        ;;
    Default|*)
        if [ -n "${DEFAULT_WALL:-}" ] && [ -f "$DEFAULT_WALL" ]; then
            set_wall "$DEFAULT_WALL"
        else
            set_wall "$(pick_random_wall "$MIRA_DIR/themes/$THEME/walls")"
        fi
        ;;
esac
