#!/bin/sh
# Eww widget palette generator (adapted from gh0stzk/dotfiles, GPL-3.0).
# Original work: https://github.com/gh0stzk/dotfiles
# Copyright (C) 2021-2026 gh0stzk <z0mbi3.zk@protonmail.com>
# Copyright (C) 2026 Misaalya
# Modified 2026-11-23 by Misaalya (Debian 13)
# SPDX-License-Identifier: GPL-3.0-only

_write "$MIRA_DIR/eww/colors.scss" << EOF
\$bg: ${bg};
\$bg-alt: ${accent_color};
\$fg: ${fg};
\$black: ${blackb};
\$red: ${red};
\$green: ${green};
\$yellow: ${yellow};
\$blue: ${blue};
\$magenta: ${magenta};
\$cyan: ${cyan};
\$font: "${ui_font:-Inter}";
EOF
