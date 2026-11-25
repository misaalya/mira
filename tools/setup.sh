#!/bin/sh
# =============================================================
# One-time system setup for Mira on Debian 13 (trixie).
# Installs required X11/BSPWM packages, configures libinput
# touchpad defaults, and registers the "Mira (BSPWM)" SDDM entry.
# =============================================================
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

USER_NAME="${SUDO_USER:-$(id -un)}"
USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)
[ -n "$USER_HOME" ] || { echo "Cannot resolve home for $USER_NAME" >&2; exit 1; }

PACKAGES="bspwm sxhkd picom polybar rofi feh maim xclip slop xxhash wmctrl
i3lock xserver-xephyr pavucontrol libnotify-bin x11-xserver-utils x11-utils
xdotool xsettingsd xserver-xorg-input-libinput lxpolkit webp-pixbuf-loader
imagemagick jq playerctl pulseaudio-utils brightnessctl network-manager bluez
qrencode fonts-inter fonts-jetbrains-mono libglib2.0-bin cargo build-essential pkg-config
libgtk-3-dev libx11-dev libxext-dev libxfixes-dev libxrender-dev libxrandr-dev
libxft-dev zlib1g-dev"

echo "Installing Debian 13 packages for Mira..."
# shellcheck disable=SC2086
apt-get install -y $PACKAGES

echo "Writing libinput touchpad configuration..."
mkdir -p /etc/X11/xorg.conf.d
cat >/etc/X11/xorg.conf.d/90-mira-touchpad.conf <<'EOF'
Section "InputClass"
    Identifier "mira-touchpad"
    MatchIsTouchpad "on"
    Driver "libinput"
    Option "Tapping" "on"
    Option "TappingButtonMap" "lrm"
    Option "TappingDrag" "on"
    Option "NaturalScrolling" "false"
    Option "ScrollMethod" "twofinger"
    Option "DisableWhileTyping" "true"
EndSection
EOF

echo "Installing SDDM session entry..."
install -d /usr/local/share/xsessions
cat >/usr/local/bin/mira-bspwm-session <<'EOF'
#!/bin/sh
exec "$HOME/.config/bspwm/session.sh"
EOF
chmod 755 /usr/local/bin/mira-bspwm-session

cat >/usr/local/share/xsessions/mira-bspwm.desktop <<'EOF'
[Desktop Entry]
Name=Mira (BSPWM)
Comment=Mira monochrome BSPWM desktop
Exec=/usr/local/bin/mira-bspwm-session
Type=Application
DesktopNames=bspwm
EOF

if [ -f /usr/share/xsessions/bspwm.desktop ] &&
    [ ! -f /usr/share/xsessions/bspwm.desktop.distrib ]; then
    dpkg-divert --local --rename --add /usr/share/xsessions/bspwm.desktop >/dev/null
fi

echo
echo "Done. Deploy Mira as $USER_NAME with tools/deploy.sh, then pick 'Mira (BSPWM)' at SDDM."
