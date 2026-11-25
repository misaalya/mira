#!/bin/sh
# BSPWM border and padding settings per theme (adapted from gh0stzk/dotfiles, GPL-3.0).
# Original work: https://github.com/gh0stzk/dotfiles
# Copyright (C) 2021-2026 gh0stzk <z0mbi3.zk@protonmail.com>
# Copyright (C) 2026 Misaalya
# Modified 2026-11-23 by Misaalya (Debian 13)
# SPDX-License-Identifier: GPL-3.0-only

bspc config border_width "${BORDER_WIDTH:-0}"
bspc config top_padding "${TOP_PADDING:-1}"
bspc config bottom_padding "${BOTTOM_PADDING:-1}"
bspc config left_padding "${LEFT_PADDING:-1}"
bspc config right_padding "${RIGHT_PADDING:-1}"
bspc config normal_border_color "${NORMAL_BC:-$black}"
bspc config focused_border_color "${FOCUSED_BC:-$grey}"
bspc config presel_feedback_color "${blue:-$fg}"
