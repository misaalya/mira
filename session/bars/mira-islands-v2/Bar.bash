# Launch the Mira Islands v2 (Dynamic Island) polybar on every connected monitor.
MIRA_DIR="${MIRA_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/bspwm}"
export MIRA_CONFIG_DIR="$MIRA_DIR"
for mon in $(polybar --list-monitors | cut -d":" -f1); do
    MONITOR=$mon setsid -f polybar -q mira-islands-v2 -c "${MIRA_DIR}/bars/${main_bar}/config.ini" </dev/null >/dev/null 2>&1
done
