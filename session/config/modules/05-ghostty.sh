#!/bin/sh
# Ghostty theme generated from the active Mira theme.
# Original work: https://github.com/gh0stzk/dotfiles
# Copyright (C) 2021-2026 gh0stzk <z0mbi3.zk@protonmail.com>
# Copyright (C) 2026 Misaalya
# Modified 2026-11-23 by Misaalya (Debian 13)
# SPDX-License-Identifier: GPL-3.0-only
# Adapted from gh0stzk/dotfiles (GPL-3.0). Skipped in isolated sandboxes.

[ -z "${MIRA_SANDBOX:-}" ] || return 0

cfg="${XDG_CONFIG_HOME:-$HOME/.config}/ghostty/config"
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/ghostty/themes"

_write "${XDG_CONFIG_HOME:-$HOME/.config}/ghostty/themes/mira" << EOF
# Colors generated for the ${THEME} theme.

background = ${bg}
foreground = ${fg}
cursor-color = ${fg}
selection-background = ${fg}
selection-foreground = ${bg}

palette = 0=${black}
palette = 1=${red}
palette = 2=${green}
palette = 3=${yellow}
palette = 4=${blue}
palette = 5=${magenta}
palette = 6=${cyan}
palette = 7=${white}
palette = 8=${blackb}
palette = 9=${redb}
palette = 10=${greenb}
palette = 11=${yellowb}
palette = 12=${blueb}
palette = 13=${magentab}
palette = 14=${cyanb}
palette = 15=${whiteb}
EOF

if [ -f "$cfg" ]; then
    if grep -q '^theme = ' "$cfg"; then
        sed -i "s|^theme = .*|theme = mira|" "$cfg"
    else
        printf '\ntheme = mira\n' >> "$cfg"
    fi

    if grep -q '^background-opacity = ' "$cfg"; then
        sed -i "s|^background-opacity = .*|background-opacity = ${P_TERM_OPACITY:-1.0}|" "$cfg"
    else
        printf 'background-opacity = %s\n' "${P_TERM_OPACITY:-1.0}" >> "$cfg"
    fi
else
    printf 'theme = mira\nbackground-opacity = %s\n' "${P_TERM_OPACITY:-1.0}" > "$cfg"
fi

pkill -USR2 -x ghostty 2>/dev/null || true
