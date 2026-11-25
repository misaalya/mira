#!/bin/sh
# Colors for x-shot (screenshot overlay) from the active Mira theme.

_write "$MIRA_DIR/config/x-shot.conf" << EOF
# Generated for the ${THEME} theme by config/modules/06-x-shot.sh.
background = ${bg}
foreground = ${fg}
accent = ${blue}
save_dir = \$XDG_PICTURES_DIR/ScreenShots
EOF
