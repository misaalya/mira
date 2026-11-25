#!/bin/sh
# Rofi & native misa-ui shared palette generator (adapted from gh0stzk/dotfiles, GPL-3.0).
# Original work: https://github.com/gh0stzk/dotfiles
# Copyright (C) 2021-2026 gh0stzk <z0mbi3.zk@protonmail.com>
# Copyright (C) 2026 Misaalya
# Modified 2026-11-23 by Misaalya (Debian 13)
# SPDX-License-Identifier: GPL-3.0-only

_write "$MIRA_DIR/config/rofi/shared.rasi" <<-EOF
configuration {
    icon-theme: "${gtk_icons:-Papirus-Dark}";
    kb-move-front: "Control+Home";
    kb-clear-line: "Control+a,Control+w";
}
* {
    font: "${rofi_font:-Inter Medium 10}";
    background: ${rofi_background:-$bg};
    bg-alt: ${rofi_bg_alt:-$black};
    background-alt: ${rofi_background_alt:-${bg}E0};
    foreground: ${rofi_fg:-$fg};
    selected: ${rofi_selected:-$fg};
    active: ${rofi_active:-$fg};
    urgent: ${rofi_urgent:-$red};
    selected-soft: ${rofi_selected:-$fg}2E;
    foreground-muted: ${rofi_fg:-$fg}8C;
    border-soft: ${rofi_fg:-$fg}1F;
    surface-soft: ${rofi_fg:-$fg}0F;
    backdrop: #00000066;
    font-key: "${display_font:-${ui_font:-Inter} Bold} 9";
}
EOF
