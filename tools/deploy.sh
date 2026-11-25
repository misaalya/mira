#!/bin/sh
# =============================================================
# Deploy Mira into ~/.config/bspwm (safe & repeatable).
# Validates picom config first, builds native binaries, preserves
# user theme/wallpaper/hardware choices, and installs icon fonts
# and xdg-desktop-portal termfilechooser config.
# =============================================================
set -eu

ROOT=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/session"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
DEST="${MIRA_CONFIG_DIR:-$CONFIG_HOME/bspwm}"

[ -d "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }

# Validate picom config before touching the live session.
if [ -n "${DISPLAY:-}" ] && command -v picom >/dev/null 2>&1; then
    check_dir=$(mktemp -d)
    cp "$SRC"/config/picom/*.conf "$check_dir"/
    if picom --config "$check_dir/picom.conf" --diagnostics 2>&1 |
        grep -q 'parse_config.*FATAL\|Failed to locate included'; then
        picom --config "$check_dir/picom.conf" --diagnostics 2>&1 |
            grep 'FATAL' | sed "s|$check_dir|session/config/picom|" >&2
        rm -rf "$check_dir"
        echo "deploy: picom config does not parse; nothing deployed." >&2
        exit 1
    fi
    rm -rf "$check_dir"
fi

# Build native Rust + C binaries into ./build/bin first.
"$ROOT/tools/build.sh"

mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST" ]; then
    cp -a "$SRC" "$DEST"
else
    rsync -a --delete \
        --exclude='.theme' \
        --exclude='config/system.ini' \
        --exclude='config/.sys' \
        --exclude='themes/*/theme.conf' \
        --exclude='themes/*/walls/' \
        --exclude='bars/colors.ini' \
        --exclude='bars/colors.scss' \
        --exclude='eww/colors.scss' \
        --exclude='config/x-shot.conf' \
        --exclude='config/rofi/shared.rasi' \
        --exclude='config/xsettingsd.conf' \
        --exclude='bin/misa-clip' \
        --exclude='bin/misa-backdrop' \
        --exclude='bin/misa-wall' \
        --exclude='bin/Scroll' \
        --exclude='bin/ClipPanel' \
        --exclude='bin/WifiPanel' \
        --exclude='bin/BluetoothPanel' \
        --exclude='bin/NotifPanel' \
        --exclude='bin/WindowSwitch' \
        --exclude='bin/MediaPillUI' \
        --exclude='bin/CardPicker' \
        --exclude='bin/WaitCompositor' \
        --exclude='bin/x-shot' \
        "$SRC/" "$DEST/"
fi

# Copy any newly added template files without clobbering user choices.
find "$SRC" -type f | while IFS= read -r f; do
    rel=${f#"$SRC"/}
    [ -e "$DEST/$rel" ] && continue
    mkdir -p "$(dirname "$DEST/$rel")"
    cp -a "$f" "$DEST/$rel"
done

# Always sync built-in theme wallpapers & sidecar GLSL shaders (never touches user wallpapers).
for theme_dir in "$SRC"/themes/*; do
    [ -d "$theme_dir/walls" ] || continue
    tname=${theme_dir##*/}
    mkdir -p "$DEST/themes/$tname/walls"
    for asset in "$theme_dir"/walls/wall-*.webp "$theme_dir"/walls/wall-*.frag; do
        [ -f "$asset" ] || continue
        cp -f "$asset" "$DEST/themes/$tname/walls/${asset##*/}"
    done
done

[ -f "$DEST/.theme" ] || printf 'mira\n' > "$DEST/.theme"

# Atomically install built native binaries (new inode keeps running daemons safe).
for bin in "$ROOT"/build/bin/*; do
    [ -f "$bin" ] && [ -x "$bin" ] || continue
    name=${bin##*/}
    cp -f "$bin" "$DEST/bin/.$name.new"
    chmod 755 "$DEST/bin/.$name.new"
    mv -f "$DEST/bin/.$name.new" "$DEST/bin/$name"
done

find "$DEST/bin" -type f -exec chmod 755 {} +
find "$DEST/bars" -path '*/scripts/*' -type f -exec chmod 755 {} +
chmod 755 "$DEST/bspwmrc" "$DEST/session.sh"

# Re-apply picom & x-shot theme modules so template defaults match active theme.
theme=$(cat "$DEST/.theme" 2>/dev/null || echo mira)
theme_cfg="$DEST/themes/$theme/theme.conf"
if [ -f "$theme_cfg" ]; then
    (
        set +eu
        MIRA_DIR="$DEST"
        THEME="$theme"
        export MIRA_DIR THEME
        # shellcheck disable=SC1090
        . "$theme_cfg"
        _write() { cat >"$1"; }
        for module in 00-picom 06-x-shot; do
            # shellcheck disable=SC1090
            [ -f "$DEST/config/modules/$module.sh" ] && . "$DEST/config/modules/$module.sh"
        done
    ) >/dev/null 2>&1 || true
    for p in $(pgrep -x picom 2>/dev/null); do
        d=$(tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null | sed -n 's/^DISPLAY=//p' | head -n 1)
        [ "$d" = "${DISPLAY:-:0}" ] && kill -USR1 "$p" 2>/dev/null || true
    done
fi

# Install bundled icon font locally.
FONTS="$ROOT/assets/fonts"
if [ -d "$FONTS" ]; then
    font_dest="${XDG_DATA_HOME:-$HOME/.local/share}/fonts/mira"
    mkdir -p "$font_dest"
    rsync -a --delete "$FONTS/" "$font_dest/"
    command -v fc-cache >/dev/null 2>&1 && fc-cache -f "$font_dest" >/dev/null 2>&1 || true
fi

# Reload running polybars on this display so bar & font updates apply immediately.
if [ -z "${MIRA_SANDBOX:-}" ] && command -v polybar-msg >/dev/null 2>&1; then
    for p in $(pgrep -x polybar 2>/dev/null); do
        d=$(tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null | sed -n 's/^DISPLAY=//p' | head -n 1)
        [ "$d" = "${DISPLAY:-:0}" ] && polybar-msg -p "$p" cmd restart >/dev/null 2>&1 || true
    done
fi

# Reload running live wallpaper daemon on this display (if any) so updated
# misa-wall binary & .frag shaders take effect immediately without logout.
if [ -z "${MIRA_SANDBOX:-}" ] && [ -n "${DISPLAY:-}" ] && [ -x "$DEST/bin/misa-wall" ]; then
    disp_safe=$(printf '%s' "$DISPLAY" | tr -c 'A-Za-z0-9' '_')
    live_state="${XDG_RUNTIME_DIR:-/tmp}/misa-wall-live-${disp_safe}.state"
    if [ -r "$live_state" ]; then
        live_img=$(sed -n '1p' "$live_state")
        if [ -f "$live_img" ]; then
            "$DEST/bin/misa-wall" --stop-live >/dev/null 2>&1 || true
            "$DEST/bin/misa-wall" --no-transition "$live_img" >/dev/null 2>&1 || true
        fi
    fi
fi

# Ensure XDG screenshot & cache dirs exist.
pics=$(xdg-user-dir PICTURES 2>/dev/null || true)
[ -n "$pics" ] || pics="$HOME/Pictures"
mkdir -p "$pics/ScreenShots" "${XDG_CACHE_HOME:-$HOME/.cache}"

# Deploy xdg-desktop-portal + yazi termfilechooser config.
if [ -f "$ROOT/portal/bspwm-portals.conf" ]; then
    mkdir -p "$CONFIG_HOME/xdg-desktop-portal"
    cp "$ROOT/portal/bspwm-portals.conf" "$CONFIG_HOME/xdg-desktop-portal/bspwm-portals.conf"
fi
if [ -f "$ROOT/portal/termfilechooser/config" ]; then
    mkdir -p "$CONFIG_HOME/xdg-desktop-portal-termfilechooser"
    cp "$ROOT/portal/termfilechooser/config" "$CONFIG_HOME/xdg-desktop-portal-termfilechooser/config"
    cp "$ROOT/portal/termfilechooser/yazi-wrapper.sh" "$CONFIG_HOME/xdg-desktop-portal-termfilechooser/yazi-wrapper.sh"
    chmod 755 "$CONFIG_HOME/xdg-desktop-portal-termfilechooser/yazi-wrapper.sh"
fi

cat <<EOF
Deployed Mira to $DEST

Next:
  1. One-time Debian packages + SDDM entry:  sudo $ROOT/tools/setup.sh
  2. Test isolated in Xephyr:                $ROOT/tools/dev-session.sh
  3. Log out and select "Mira (BSPWM)" at SDDM
EOF
