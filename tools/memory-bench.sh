#!/bin/sh
# Memory used by the running Mira session, per component.
#
#   tools/memory-bench.sh [-n SAMPLES] [-i SECONDS] [-d DISPLAY]
#
# Counts PSS (proportional set size: shared pages split between the
# processes that map them, so the numbers add up) from
# /proc/<pid>/smaps_rollup. A process belongs to Mira when it runs from the
# config dir ($MIRA_CONFIG_DIR, default ~/.config/bspwm) or is one of the
# session's core programs (bspwm, sxhkd, picom, polybar, eww, xsettingsd),
# plus everything those start (bspc subscribe, playerctl, xev, ...), except
# apps launched from keybindings (children of sxhkd / bspwm). Only processes
# on DISPLAY (default: this one) are counted.
#
# With -n > 1 it takes that many samples -i seconds apart and reports the
# median of each component. GPU memory (picom's textures) is not in PSS.
set -eu

samples=1
interval=10
display=${DISPLAY:-:0}
while getopts n:i:d: o; do
    case $o in
        n) samples=$OPTARG ;;
        i) interval=$OPTARG ;;
        d) display=$OPTARG ;;
        *) echo "usage: $0 [-n samples] [-i seconds] [-d display]" >&2; exit 2 ;;
    esac
done

MIRA_DIR="${MIRA_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/bspwm}"
uid=$(id -u)

# One sample: "<component> <pss_kB>" per Mira process.
sample() {
    for st in /proc/[0-9]*/stat; do
        p=${st#/proc/}
        p=${p%/stat}
        [ -O "/proc/$p" ] || continue
        d=$(tr '\0' '\n' 2>/dev/null <"/proc/$p/environ" | sed -n 's/^DISPLAY=//p' | head -n 1) || continue
        pss=$(awk '/^Pss:/ {print $2; exit}' "/proc/$p/smaps_rollup" 2>/dev/null) || continue
        [ -n "$pss" ] || continue
        ppid=$(sed 's/^.*) //' "/proc/$p/stat" 2>/dev/null | awk '{print $2}') || continue
        comm=$(cat "/proc/$p/comm" 2>/dev/null) || continue
        args=$(tr '\0' ' ' 2>/dev/null <"/proc/$p/cmdline") || continue
        # pid ppid display pss comm <TAB> args
        printf '%s %s %s %s %s\t%s\n' "$p" "$ppid" "${d:--}" "$pss" "$comm" "$args"
    done | awk -F'\t' -v dir="$MIRA_DIR" -v disp="$display" -v self="$$" '
        {
            split($1, f, " ")
            pid = f[1]; ppid[pid] = f[2]; dsp[pid] = f[3]; pss[pid] = f[4]; comm[pid] = f[5]
            args = $2
            core = (comm[pid] ~ /^(bspwm|sxhkd|picom|polybar|eww|xsettingsd)$/)
            mira[pid] = (core || index(args, dir "/") > 0) && pid != self
            launcher[pid] = (comm[pid] == "sxhkd" || comm[pid] == "bspwm")
        }
        function belongs(p,   q, n) {
            if (mira[p]) return 1
            for (q = ppid[p]; q != "" && q > 1 && n < 64; q = ppid[q]) {
                n++
                if (launcher[q]) return 0
                if (mira[q]) return 1
            }
            return 0
        }
        END {
            for (p in pss) if (dsp[p] == disp && p != self && belongs(p)) print comm[p], pss[p]
        }'
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
i=1
while [ "$i" -le "$samples" ]; do
    # Per sample: component -> summed PSS and process count.
    sample | awk '{ k[$1] += $2; c[$1]++ } END { for (n in k) print n, k[n], c[n] }' >"$tmp/s$i"
    awk '/^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2} END {print t, t - a}' /proc/meminfo >"$tmp/m$i"
    [ "$i" -lt "$samples" ] && sleep "$interval"
    i=$((i + 1))
done

echo "Mira memory (PSS) on $display, $(date '+%Y-%m-%d %H:%M:%S'), $samples sample(s)"
echo "load average: $(cut -d' ' -f1-3 /proc/loadavg)"
echo
cat "$tmp"/s* | awk -v n="$samples" '
    function median(list,   a, m, i, j, t) {
        m = split(list, a, " ")
        for (i = 1; i <= m; i++) for (j = i + 1; j <= m; j++) if (a[j] < a[i]) { t = a[i]; a[i] = a[j]; a[j] = t }
        return m % 2 ? a[(m + 1) / 2] : (a[m / 2] + a[m / 2 + 1]) / 2
    }
    { v[$1] = v[$1] " " $2; c[$1] = $3; seen[$1]++ }
    END {
        for (k in v) { while (seen[k] < n) { v[k] = v[k] " 0"; seen[k]++ } med[k] = median(v[k]) }
        # sort by median, largest first
        cnt = 0; for (k in med) keys[++cnt] = k
        for (i = 1; i <= cnt; i++) for (j = i + 1; j <= cnt; j++) if (med[keys[j]] > med[keys[i]]) { t = keys[i]; keys[i] = keys[j]; keys[j] = t }
        printf "%-16s %5s %10s\n", "component", "procs", "PSS (MiB)"
        for (i = 1; i <= cnt; i++) { k = keys[i]; tot += med[k]; printf "%-16s %5d %10.1f\n", k, c[k], med[k] / 1024 }
        printf "%-16s %5s %10.1f\n", "TOTAL", "", tot / 1024
    }'
echo
cat "$tmp"/m* | sort -k2,2n | awk -v n="$samples" '
    { t = $1; u[NR] = $2 }
    END { m = u[int((NR + 1) / 2)]; printf "system: %.0f MiB used of %.0f MiB (MemTotal - MemAvailable, median)\n", m / 1024, t / 1024 }'
