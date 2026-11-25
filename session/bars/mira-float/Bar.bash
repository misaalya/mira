# Launch the Mira Floating polybar on every connected monitor.
MIRA_DIR="${MIRA_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/bspwm}"
export MIRA_CONFIG_DIR="$MIRA_DIR"

# Polybar 3.7 needs an 8-digit ARGB background when combining radius +
# transparent borders + internal/tray; derive it from the active theme's bg.
_mira_bg=""
if [ -r "${MIRA_DIR}/bars/colors.ini" ]; then
    _mira_bg=$(sed -n 's/^[[:space:]]*bg[[:space:]]*=[[:space:]]*//p' "${MIRA_DIR}/bars/colors.ini" | head -n 1)
fi
case "$_mira_bg" in
    \#[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F])
        MIRA_BG_ARGB="#ff${_mira_bg#\#}"
        ;;
    \#[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F])
        MIRA_BG_ARGB="$_mira_bg"
        ;;
    *)
        MIRA_BG_ARGB="#ff000000"
        ;;
esac
export MIRA_BG_ARGB
unset _mira_bg

for mon in $(polybar --list-monitors | cut -d":" -f1); do
    MONITOR=$mon setsid -f polybar -q mira-float -c "${MIRA_DIR}/bars/${main_bar}/config.ini" </dev/null >/dev/null 2>&1
done
