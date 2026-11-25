#!/bin/sh
# Topbar color palette generation (adapted from gh0stzk/dotfiles, GPL-3.0).
# Original work: https://github.com/gh0stzk/dotfiles
# Copyright (C) 2021-2026 gh0stzk <z0mbi3.zk@protonmail.com>
# Copyright (C) 2026 Misaalya
# Modified 2026-11-23 by Misaalya (Debian 13)
# SPDX-License-Identifier: GPL-3.0-only

_write "$MIRA_DIR/bars/colors.ini" << EOF
;; Colors generated for the ${THEME} theme.
[color]
bg = ${bg_alpha:-$bg}
bg-alt = $bg_alt
fg = $fg
pill = $accent_color
trans = $trans

black = $black
red = $red
green = $green
yellow = $yellow
blue = $blue
magenta = $magenta
cyan = $cyan
white = $white
black-bright = $blackb
red-bright = $redb
green-bright = $greenb
yellow-bright = $yellowb
blue-bright = $blueb
magenta-bright = $magentab
cyan-bright = $cyanb
white-bright = $whiteb

purple = $purple
pink = $pink
lime = $lime
amber = $amber
orange = $orange
grey = $grey
indigo = $indigo
blue-gray = $bluegray
EOF

_write "$MIRA_DIR/bars/colors.scss" << EOF
\$bg: ${bg_alpha_css:-$bg};
\$bg-alt: $bg_alt;
\$fg: $fg;
\$pill: $accent_color;
\$trans: $trans;

\$black: $black;
\$red: $red;
\$green: $green;
\$yellow: $yellow;
\$blue: $blue;
\$magenta: $magenta;
\$cyan: $cyan;
\$white: $white;
\$black-bright: $blackb;
\$red-bright: $redb;
\$green-bright: $greenb;
\$yellow-bright: $yellowb;
\$blue-bright: $blueb;
\$magenta-bright: $magentab;
\$cyan-bright: $cyanb;
\$white-bright: $whiteb;

\$purple: $purple;
\$pink: $pink;
\$lime: $lime;
\$amber: $amber;
\$orange: $orange;
\$grey: $grey;
\$indigo: $indigo;
\$blue-gray: $bluegray;
EOF
