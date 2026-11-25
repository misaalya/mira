#!/bin/sh
# =============================================================
# Install (or remove) the Mira SDDM login theme. Run with sudo.
# SDDM is system-wide: the theme is shown before every session it
# offers, so it carries no rice-specific branding. Its background is a
# blurred copy of the last wallpaper Mira set (session/bin/SddmBackground).
#
#   sudo tools/sddm-theme.sh            install and make it current
#   sudo tools/sddm-theme.sh --remove   back to the default theme
# =============================================================
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
DEST=/usr/share/sddm/themes/mira
CONF=/etc/sddm.conf.d/10-mira-theme.conf
# Blurred wallpaper for the theme, written by session/bin/SddmBackground
# as the desktop user (SDDM itself cannot read $HOME).
BGDIR=/var/lib/mira-sddm
USER_NAME="${SUDO_USER:-}"

if [ "${1:-}" = "--remove" ]; then
    rm -rf "$DEST" "$CONF" "$BGDIR"
    echo "Removed the Mira SDDM theme."
    exit 0
fi

rm -rf "$DEST"
install -d "$DEST"
cp -R "$ROOT/sddm/mira/." "$DEST/"

install -d /etc/sddm.conf.d
cat >"$CONF" <<'CONF'
# Written by mira/tools/sddm-theme.sh; delete it to go back to the default theme.
[Theme]
Current=mira

[General]
# The theme has no on-screen keyboard. Without this, SDDM 0.21 opens Qt's
# virtual keyboard whenever the password field has focus.
InputMethod=
CONF

if [ -n "$USER_NAME" ] && [ "$USER_NAME" != root ]; then
    install -d -o "$USER_NAME" -g "$(id -gn "$USER_NAME")" -m 755 "$BGDIR"
else
    echo "Run through sudo from your user to enable the wallpaper background." >&2
fi

echo "Installed the Mira SDDM theme ($DEST). It is used from the next login screen."
