/* SoundPanel - native GTK3 sound/volume popup for the top bar
 * (bin/SoundPanel [--pointer] [--side left|right]).
 *
 * Replaces the former Eww volume panel with zero idle memory (one-shot
 * process like BluetoothPanel): title + pavucontrol gear, two channel blocks
 * (Output and Microphone) with mute toggle buttons and thick pill sliders,
 * and the active output device description in the footer.
 *
 * Placement, backdrop, seat grab, click-outside/Esc dismissal, and
 * single-instance flock toggle follow BluetoothPanel.
 * picom: class SoundPanel (the "Bar panels" rule).
 */
#include "common.h"

#include <glib-unix.h>

#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#define WIDTH 354
#define GAP 6

/* Material Design Icons code points. */
enum {
    IC_GEAR = 0xF0493,
    IC_VOL_MUTE = 0xF075F,
    IC_VOL_LOW = 0xF057F,
    IC_VOL_MED = 0xF0580,
    IC_VOL_HIGH = 0xF057E,
    IC_MIC = 0xF036C,
    IC_MIC_MUTE = 0xF036D,
    IC_SPEAKER = 0xF04C3,
};

typedef struct {
    const char *base_name;
    GtkWidget *block;
    GtkWidget *icon_btn;
    GtkWidget *name_lbl;
    GtkWidget *num_lbl;
    GtkWidget *scale;
    int value;
    gboolean muted;
    gboolean dragging;
    gboolean is_mic;
    guint send_timer;
    int pending_val;
    gboolean has_pending;
} Channel;

static struct {
    GtkWidget *win, *box;
    GtkWidget *dev_lbl;
    Channel out, mic;
    gboolean pointer, closing, syncing;
    char side[8];
    Rect mon;
    char edge[8];
    int off, px, shade;
    guint poll_id;
    guint dev_poll_id;
    gint64 suppress_until_us;
    gboolean state_inflight;
    gboolean dev_inflight;
} S;

static void panel_close(void);
static void poll_state_async(void);
static void poll_device_async(void);

/* -- helpers --------------------------------------------------------------- */
static void add_class(GtkWidget *w, const char *c) {
    gtk_style_context_add_class(gtk_widget_get_style_context(w), c);
}

static void set_class(GtkWidget *w, const char *c, gboolean on) {
    GtkStyleContext *sc = gtk_widget_get_style_context(w);
    if (on) gtk_style_context_add_class(sc, c);
    else gtk_style_context_remove_class(sc, c);
}

static char *u8(gunichar c) {
    char *s = g_malloc0(8);
    g_unichar_to_utf8(c, s);
    return s;
}

static void set_pointer_cursor(GtkWidget *w) {
    GdkWindow *gw = gtk_widget_get_window(w);
    if (!gw) return;
    GdkCursor *cur = gdk_cursor_new_from_name(gdk_window_get_display(gw), "pointer");
    if (cur) {
        gdk_window_set_cursor(gw, cur);
        g_object_unref(cur);
    }
}

static void on_clickable_realize(GtkWidget *w, gpointer _d) {
    set_pointer_cursor(w);
}

static void make_clickable(GtkWidget *w) {
    g_signal_connect(w, "realize", G_CALLBACK(on_clickable_realize), NULL);
}

static void spawn_fire(const char *const *argv) {
    g_spawn_async(NULL, (char **)argv, NULL,
                  G_SPAWN_SEARCH_PATH | G_SPAWN_STDOUT_TO_DEV_NULL | G_SPAWN_STDERR_TO_DEV_NULL,
                  NULL, NULL, NULL, NULL);
}

/* -- placement (as BluetoothPanel) ----------------------------------------- */
static void anchor(gboolean pointer) {
    S.mon = (Rect){0, 0, 1920, 1080};
    strcpy(S.edge, "top");
    S.off = 0;
    const char *argv[] = {"BarAnchor", pointer ? "--pointer" : NULL, NULL};
    char *out = run_out(argv);
    gboolean ok = FALSE;
    if (out) {
        char **f = g_strsplit_set(g_strstrip(out), " \t\n", -1);
        if (g_strv_length(f) >= 3 && monitor_rect(f[0], &S.mon)) {
            g_strlcpy(S.edge, f[1], sizeof S.edge);
            S.off = atoi(f[2]);
            ok = TRUE;
        }
        g_strfreev(f);
        g_free(out);
    }
    if (!ok) S.mon = focused_monitor();
    int y = 0;
    gdk_device_get_position(gdk_seat_get_pointer(gdk_display_get_default_seat(gdk_display_get_default())),
                            NULL, &S.px, &y);
}

static void place(int h) {
    if (h <= 0) h = gtk_widget_get_allocated_height(S.win);
    if (h <= 0) h = 100;
    int x;
    if (S.pointer) {
        x = S.px - WIDTH / 2;
    } else if (strcmp(S.side, "left") == 0) {
        x = S.mon.x + 8;
    } else {
        x = S.mon.x + S.mon.w - WIDTH - 8;
    }
    x = MAX(S.mon.x + 8, MIN(x, S.mon.x + S.mon.w - WIDTH - 8));
    int y = strcmp(S.edge, "top") == 0 ? S.mon.y + S.off + GAP : S.mon.y + S.mon.h - S.off - GAP - h;
    int cx, cy;
    gtk_window_get_position(GTK_WINDOW(S.win), &cx, &cy);
    if (cx != x || cy != y) gtk_window_move(GTK_WINDOW(S.win), x, y);
}

static void fit(void) {
    GtkRequisition nat;
    gtk_widget_get_preferred_size(S.box, NULL, &nat);
    gtk_window_resize(GTK_WINDOW(S.win), WIDTH, MAX(1, nat.height));
    place(nat.height);
}

/* -- UI updates ------------------------------------------------------------ */
static gunichar channel_icon(const Channel *ch) {
    if (ch->is_mic) return ch->muted ? IC_MIC_MUTE : IC_MIC;
    if (ch->muted) return IC_VOL_MUTE;
    if (ch->value > 66) return IC_VOL_HIGH;
    if (ch->value > 33) return IC_VOL_MED;
    return IC_VOL_LOW;
}

static void update_channel_ui(Channel *ch, gboolean sync_scale) {
    char *ic = u8(channel_icon(ch));
    gtk_button_set_label(GTK_BUTTON(ch->icon_btn), ic);
    g_free(ic);

    char name_buf[64];
    if (ch->muted) g_snprintf(name_buf, sizeof name_buf, "%s \xc2\xb7 Muted", ch->base_name);
    else g_strlcpy(name_buf, ch->base_name, sizeof name_buf);
    gtk_label_set_text(GTK_LABEL(ch->name_lbl), name_buf);

    char pct_buf[16];
    g_snprintf(pct_buf, sizeof pct_buf, "%d%%", ch->value);
    gtk_label_set_text(GTK_LABEL(ch->num_lbl), pct_buf);

    set_class(ch->block, "muted", ch->muted);
    set_class(ch->scale, "zero", ch->value <= 0);

    if (sync_scale && !ch->dragging) {
        S.syncing = TRUE;
        gtk_range_set_value(GTK_RANGE(ch->scale), (double)ch->value);
        S.syncing = FALSE;
    }
}

/* -- PulseAudio / PipeWire async backend ----------------------------------- */
static void send_channel_value(Channel *ch, int pct) {
    pct = CLAMP(pct, 0, 100);
    char cmd[128];
    if (ch->is_mic) {
        g_snprintf(cmd, sizeof cmd,
                   "pactl set-source-mute @DEFAULT_SOURCE@ 0; pactl set-source-volume @DEFAULT_SOURCE@ %d%%",
                   pct);
    } else {
        g_snprintf(cmd, sizeof cmd,
                   "pactl set-sink-mute @DEFAULT_SINK@ 0; pactl set-sink-volume @DEFAULT_SINK@ %d%%",
                   pct);
    }
    const char *argv[] = {"sh", "-c", cmd, NULL};
    spawn_fire(argv);
}

static gboolean flush_channel_timer(gpointer data) {
    Channel *ch = data;
    ch->send_timer = 0;
    if (ch->has_pending && !S.closing) {
        ch->has_pending = FALSE;
        send_channel_value(ch, ch->pending_val);
    }
    return G_SOURCE_REMOVE;
}

static void schedule_channel_send(Channel *ch, int pct) {
    ch->pending_val = pct;
    ch->has_pending = TRUE;
    S.suppress_until_us = g_get_monotonic_time() + 500000;
    if (!ch->send_timer) {
        ch->send_timer = g_timeout_add(45, flush_channel_timer, ch);
    }
}

static void on_scale_changed(GtkRange *range, gpointer data) {
    if (S.syncing || S.closing) return;
    Channel *ch = data;
    int pct = CLAMP((int)lround(gtk_range_get_value(range)), 0, 100);
    if (pct == ch->value && !ch->muted) return;
    ch->value = pct;
    ch->muted = FALSE;
    update_channel_ui(ch, FALSE);
    schedule_channel_send(ch, pct);
}

static gboolean on_scale_press(GtkWidget *w, GdkEventButton *ev, gpointer data) {
    if (ev->button == 1) {
        Channel *ch = data;
        ch->dragging = TRUE;
        S.suppress_until_us = g_get_monotonic_time() + 1000000;
    }
    return FALSE;
}

static gboolean on_scale_release(GtkWidget *w, GdkEventButton *ev, gpointer data) {
    Channel *ch = data;
    if (ch->dragging) {
        ch->dragging = FALSE;
        S.suppress_until_us = g_get_monotonic_time() + 450000;
        if (ch->send_timer) {
            g_source_remove(ch->send_timer);
            ch->send_timer = 0;
        }
        if (ch->has_pending) {
            ch->has_pending = FALSE;
            send_channel_value(ch, ch->pending_val);
        }
    }
    return FALSE;
}

static void on_mute_click(GtkButton *_b, gpointer data) {
    Channel *ch = data;
    ch->muted = !ch->muted;
    S.suppress_until_us = g_get_monotonic_time() + 500000;
    update_channel_ui(ch, FALSE);
    const char *argv_out[] = {"pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle", NULL};
    const char *argv_mic[] = {"pactl", "set-source-mute", "@DEFAULT_SOURCE@", "toggle", NULL};
    spawn_fire(ch->is_mic ? argv_mic : argv_out);
}

static void on_gear_click(GtkButton *_b, gpointer _d) {
    panel_close();
    const char *argv[] = {"pavucontrol", NULL};
    spawn_fire(argv);
}

static void state_done(GObject *src, GAsyncResult *res, gpointer _d) {
    S.state_inflight = FALSE;
    char *out = NULL;
    GSubprocess *p = G_SUBPROCESS(src);
    gboolean ok = g_subprocess_communicate_utf8_finish(p, res, &out, NULL, NULL);
    g_object_unref(p);
    if (!ok || !out || S.closing) {
        g_free(out);
        return;
    }
    if (g_get_monotonic_time() < S.suppress_until_us || S.out.dragging || S.mic.dragging) {
        g_free(out);
        return;
    }
    int ov = 0, om = 0, mv = 0, mm = 0;
    if (sscanf(out, "%d %d %d %d", &ov, &om, &mv, &mm) == 4) {
        S.out.value = CLAMP(ov, 0, 100);
        S.out.muted = om != 0;
        S.mic.value = CLAMP(mv, 0, 100);
        S.mic.muted = mm != 0;
        update_channel_ui(&S.out, TRUE);
        update_channel_ui(&S.mic, TRUE);
    }
    g_free(out);
}

static void poll_state_async(void) {
    if (S.closing || S.state_inflight) return;
    if (g_get_monotonic_time() < S.suppress_until_us || S.out.dragging || S.mic.dragging) return;
    static const char *script =
        "ov=$(pactl get-sink-volume @DEFAULT_SINK@ 2>/dev/null | grep -o '[0-9]*%' | head -n1 | tr -d %);"
        "pactl get-sink-mute @DEFAULT_SINK@ 2>/dev/null | grep -q yes && om=1 || om=0;"
        "mv=$(pactl get-source-volume @DEFAULT_SOURCE@ 2>/dev/null | grep -o '[0-9]*%' | head -n1 | tr -d %);"
        "pactl get-source-mute @DEFAULT_SOURCE@ 2>/dev/null | grep -q yes && mm=1 || mm=0;"
        "printf '%d %d %d %d\\n' \"${ov:-0}\" \"$om\" \"${mv:-0}\" \"$mm\"";
    const char *argv[] = {"sh", "-c", script, NULL};
    GSubprocess *p = g_subprocess_newv(argv, G_SUBPROCESS_FLAGS_STDOUT_PIPE |
                                       G_SUBPROCESS_FLAGS_STDERR_SILENCE, NULL);
    if (!p) return;
    S.state_inflight = TRUE;
    g_subprocess_communicate_utf8_async(p, NULL, NULL, state_done, NULL);
}

static void device_done(GObject *src, GAsyncResult *res, gpointer _d) {
    S.dev_inflight = FALSE;
    char *out = NULL;
    GSubprocess *p = G_SUBPROCESS(src);
    gboolean ok = g_subprocess_communicate_utf8_finish(p, res, &out, NULL, NULL);
    g_object_unref(p);
    if (!ok || !out || S.closing) {
        g_free(out);
        return;
    }
    g_strstrip(out);
    gtk_label_set_text(GTK_LABEL(S.dev_lbl), *out ? out : "No output device");
    g_free(out);
}

static void poll_device_async(void) {
    if (S.closing || S.dev_inflight) return;
    /* Parse active default sink description without requiring jq. */
    static const char *script =
        "sink=$(pactl get-default-sink 2>/dev/null); "
        "[ -n \"$sink\" ] || exit 0; "
        "pactl list sinks 2>/dev/null | awk -v s=\"$sink\" '"
        "/^[[:space:]]*Name:/ { match_sink = ($2 == s) } "
        "match_sink && /^[[:space:]]*Description:/ { sub(/^[[:space:]]*Description:[[:space:]]*/, \"\"); print; exit }'";
    const char *argv[] = {"sh", "-c", script, NULL};
    GSubprocess *p = g_subprocess_newv(argv, G_SUBPROCESS_FLAGS_STDOUT_PIPE |
                                       G_SUBPROCESS_FLAGS_STDERR_SILENCE, NULL);
    if (!p) return;
    S.dev_inflight = TRUE;
    g_subprocess_communicate_utf8_async(p, NULL, NULL, device_done, NULL);
}

static gboolean on_poll_state(gpointer _d) {
    if (S.closing) return G_SOURCE_REMOVE;
    poll_state_async();
    return G_SOURCE_CONTINUE;
}

static gboolean on_poll_device(gpointer _d) {
    if (S.closing) return G_SOURCE_REMOVE;
    poll_device_async();
    return G_SOURCE_CONTINUE;
}

/* -- open / close & input -------------------------------------------------- */
static void panel_close(void) {
    if (S.closing) return;
    S.closing = TRUE;
    if (S.poll_id) { g_source_remove(S.poll_id); S.poll_id = 0; }
    if (S.dev_poll_id) { g_source_remove(S.dev_poll_id); S.dev_poll_id = 0; }
    if (S.out.send_timer) { g_source_remove(S.out.send_timer); S.out.send_timer = 0; }
    if (S.mic.send_timer) { g_source_remove(S.mic.send_timer); S.mic.send_timer = 0; }
    ungrab();
    gtk_widget_hide(S.win);
    drop_backdrop(S.shade);
    gtk_main_quit();
}

static void nudge_output(int delta) {
    int next = CLAMP(S.out.value + delta, 0, 100);
    S.out.value = next;
    S.out.muted = FALSE;
    update_channel_ui(&S.out, TRUE);
    schedule_channel_send(&S.out, next);
}

static gboolean on_key(GtkWidget *_w, GdkEventKey *ev, gpointer _d) {
    guint k = ev->keyval;
    int step = (ev->state & GDK_SHIFT_MASK) ? 1 : 5;
    if (k == GDK_KEY_Escape) { panel_close(); return TRUE; }
    if (k == GDK_KEY_Right || k == GDK_KEY_Up || k == GDK_KEY_plus || k == GDK_KEY_equal) {
        nudge_output(step);
        return TRUE;
    }
    if (k == GDK_KEY_Left || k == GDK_KEY_Down || k == GDK_KEY_minus) {
        nudge_output(-step);
        return TRUE;
    }
    if (k == GDK_KEY_m || k == GDK_KEY_M) {
        on_mute_click(NULL, &S.out);
        return TRUE;
    }
    return FALSE;
}

static gboolean on_outside(GtkWidget *_w, GdkEventButton *ev, gpointer _d) {
    int x, y, w, h;
    gtk_window_get_position(GTK_WINDOW(S.win), &x, &y);
    gtk_window_get_size(GTK_WINDOW(S.win), &w, &h);
    if (!(ev->x_root >= x && ev->x_root < x + w && ev->y_root >= y && ev->y_root < y + h)) {
        panel_close();
        return TRUE;
    }
    return FALSE;
}

static void on_alloc(GtkWidget *_w, GdkRectangle *_a, gpointer _d) { place(0); }
static gboolean on_term(gpointer _d) { panel_close(); return G_SOURCE_REMOVE; }

/* -- UI construction ------------------------------------------------------- */
static char *build_css(const Palette *p) {
    GString *c = g_string_new(NULL);
    g_string_append_printf(c, "@define-color fg %s;\n@define-color bg %s;\n", p->fg, p->bg);
    g_string_append_printf(c, "* { font-family: \"%s\"; }\n", p->font);
    g_string_append(c,
        "window { background: transparent; }\n"
        ".vol-panel {\n"
        "  background-color: @bg;\n"
        "  color: @fg;\n"
        "  border: 1px solid alpha(@fg, 0.12);\n"
        "  border-radius: 14px;\n"
        "  padding: 14px 16px 12px 16px;\n"
        "}\n"
        ".vol-title {\n"
        "  font-size: 15px;\n"
        "  font-weight: bold;\n"
        "  color: @fg;\n"
        "  padding-left: 2px;\n"
        "}\n"
        "button {\n"
        "  background-image: none;\n"
        "  box-shadow: none;\n"
        "  text-shadow: none;\n"
        "  -gtk-icon-shadow: none;\n"
        "  border: none;\n"
        "  outline: none;\n"
        "  min-height: 0;\n"
        "  min-width: 0;\n"
        "}\n"
        "button.vol-gear {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 18px;\n"
        "  padding: 2px 6px;\n"
        "  border-radius: 8px;\n"
        "  background: transparent;\n"
        "  color: alpha(@fg, 0.55);\n"
        "}\n"
        "button.vol-gear:hover {\n"
        "  color: @fg;\n"
        "  background-color: alpha(@fg, 0.08);\n"
        "}\n"
        ".vol-block {\n"
        "  padding: 10px 12px 12px 12px;\n"
        "  border-radius: 12px;\n"
        "  background-color: alpha(@fg, 0.06);\n"
        "  transition: opacity 120ms ease-out;\n"
        "}\n"
        ".vol-block.muted { opacity: 0.4; }\n"
        "button.vol-icon {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 18px;\n"
        "  min-width: 26px;\n"
        "  padding: 0px 2px;\n"
        "  border-radius: 8px;\n"
        "  background: transparent;\n"
        "  color: @fg;\n"
        "}\n"
        "button.vol-icon:hover { background-color: alpha(@fg, 0.10); }\n"
        ".vol-name {\n"
        "  font-size: 13px;\n"
        "  font-weight: 600;\n"
        "  color: alpha(@fg, 0.70);\n"
        "}\n"
        ".vol-num {\n"
        "  font-size: 13px;\n"
        "  font-weight: bold;\n"
        "  color: @fg;\n"
        "}\n"
        "scale.vol-pill {\n"
        "  padding: 0;\n"
        "  margin: 0;\n"
        "}\n"
        "scale.vol-pill contents { padding: 0; margin: 0; }\n"
        "scale.vol-pill trough {\n"
        "  min-height: 26px;\n"
        "  border-radius: 13px;\n"
        "  background-color: alpha(@fg, 0.14);\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "  background-image: none;\n"
        "  padding: 0;\n"
        "  margin: 0;\n"
        "}\n"
        "scale.vol-pill highlight {\n"
        "  min-height: 26px;\n"
        "  border-radius: 13px;\n"
        "  background-color: @fg;\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "  background-image: none;\n"
        "  padding: 0;\n"
        "  margin: 0;\n"
        "}\n"
        "scale.vol-pill.zero highlight { background-color: transparent; }\n"
        "scale.vol-pill slider {\n"
        "  min-width: 0px;\n"
        "  min-height: 0px;\n"
        "  margin: 0px;\n"
        "  padding: 0px;\n"
        "  background: transparent;\n"
        "  background-image: none;\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "}\n"
        ".vol-foot {\n"
        "  padding: 0px 4px;\n"
        "  color: alpha(@fg, 0.50);\n"
        "}\n"
        ".vol-foot-icon {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 14px;\n"
        "  color: alpha(@fg, 0.50);\n"
        "}\n"
        ".vol-dev {\n"
        "  font-size: 12px;\n"
        "  color: alpha(@fg, 0.50);\n"
        "}\n"
        "tooltip, tooltip.background {\n"
        "  background-color: @bg;\n"
        "  color: @fg;\n"
        "  border: 1px solid alpha(@fg, 0.14);\n"
        "  border-radius: 8px;\n"
        "}\n");
    return g_string_free(c, FALSE);
}

static void build_channel(Channel *ch, const char *name, const char *tip, gboolean is_mic) {
    ch->base_name = name;
    ch->is_mic = is_mic;

    ch->block = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    add_class(ch->block, "vol-block");

    GtkWidget *top = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    char *ic = u8(is_mic ? IC_MIC : IC_VOL_MED);
    ch->icon_btn = gtk_button_new_with_label(ic);
    g_free(ic);
    add_class(ch->icon_btn, "vol-icon");
    gtk_widget_set_can_focus(ch->icon_btn, FALSE);
    gtk_widget_set_tooltip_text(ch->icon_btn, tip);
    make_clickable(ch->icon_btn);
    g_signal_connect(ch->icon_btn, "clicked", G_CALLBACK(on_mute_click), ch);

    ch->name_lbl = gtk_label_new(name);
    add_class(ch->name_lbl, "vol-name");
    gtk_label_set_xalign(GTK_LABEL(ch->name_lbl), 0.0f);
    gtk_widget_set_valign(ch->name_lbl, GTK_ALIGN_CENTER);
    gtk_widget_set_hexpand(ch->name_lbl, TRUE);

    ch->num_lbl = gtk_label_new("0%");
    add_class(ch->num_lbl, "vol-num");
    gtk_widget_set_valign(ch->num_lbl, GTK_ALIGN_CENTER);

    gtk_box_pack_start(GTK_BOX(top), ch->icon_btn, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(top), ch->name_lbl, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(top), ch->num_lbl, FALSE, FALSE, 0);

    ch->scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 100.0, 1.0);
    add_class(ch->scale, "vol-pill");
    gtk_scale_set_draw_value(GTK_SCALE(ch->scale), FALSE);
    gtk_widget_set_can_focus(ch->scale, FALSE);
    gtk_widget_set_hexpand(ch->scale, TRUE);
    make_clickable(ch->scale);
    g_signal_connect(ch->scale, "value-changed", G_CALLBACK(on_scale_changed), ch);
    g_signal_connect(ch->scale, "button-press-event", G_CALLBACK(on_scale_press), ch);
    g_signal_connect(ch->scale, "button-release-event", G_CALLBACK(on_scale_release), ch);

    gtk_box_pack_start(GTK_BOX(ch->block), top, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(ch->block), ch->scale, FALSE, FALSE, 0);
}

static void build_ui(void) {
    /* Click anywhere on a pill slider jumps immediately to that point. */
    g_object_set(gtk_settings_get_default(), "gtk-primary-button-warps-slider", TRUE, NULL);

    Palette p;
    palette_load(&p);
    char *css = build_css(&p);
    css_load(css);
    g_free(css);

    S.win = overlay_window("SoundPanel", 0, 0, WIDTH, 100);
    gtk_widget_set_size_request(S.win, WIDTH, -1);
    gtk_window_set_default_size(GTK_WINDOW(S.win), -1, -1);
    gtk_window_set_resizable(GTK_WINDOW(S.win), TRUE);
    gtk_widget_add_events(S.win, GDK_KEY_PRESS_MASK | GDK_BUTTON_PRESS_MASK);
    g_signal_connect(S.win, "key-press-event", G_CALLBACK(on_key), NULL);
    g_signal_connect(S.win, "button-press-event", G_CALLBACK(on_outside), NULL);
    g_signal_connect(S.win, "size-allocate", G_CALLBACK(on_alloc), NULL);

    S.box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 14);
    add_class(S.box, "vol-panel");

    /* Header */
    GtkWidget *head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget *title = gtk_label_new("Sound");
    add_class(title, "vol-title");
    gtk_label_set_xalign(GTK_LABEL(title), 0.0f);
    gtk_widget_set_hexpand(title, TRUE);

    char *gear_ic = u8(IC_GEAR);
    GtkWidget *gear = gtk_button_new_with_label(gear_ic);
    g_free(gear_ic);
    add_class(gear, "vol-gear");
    gtk_widget_set_can_focus(gear, FALSE);
    gtk_widget_set_tooltip_text(gear, "Sound settings");
    make_clickable(gear);
    g_signal_connect(gear, "clicked", G_CALLBACK(on_gear_click), NULL);

    gtk_box_pack_start(GTK_BOX(head), title, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(head), gear, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.box), head, FALSE, FALSE, 0);

    /* Output & Microphone blocks */
    build_channel(&S.out, "Output", "Mute output", FALSE);
    build_channel(&S.mic, "Microphone", "Mute microphone", TRUE);
    gtk_box_pack_start(GTK_BOX(S.box), S.out.block, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.box), S.mic.block, FALSE, FALSE, 0);

    /* Footer */
    GtkWidget *foot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    add_class(foot, "vol-foot");
    char *spk_ic = u8(IC_SPEAKER);
    GtkWidget *spk = gtk_label_new(spk_ic);
    g_free(spk_ic);
    add_class(spk, "vol-foot-icon");
    S.dev_lbl = gtk_label_new("");
    add_class(S.dev_lbl, "vol-dev");
    gtk_label_set_xalign(GTK_LABEL(S.dev_lbl), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(S.dev_lbl), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(S.dev_lbl), 38);
    gtk_widget_set_hexpand(S.dev_lbl, TRUE);
    gtk_box_pack_start(GTK_BOX(foot), spk, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(foot), S.dev_lbl, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(S.box), foot, FALSE, FALSE, 0);

    gtk_container_add(GTK_CONTAINER(S.win), S.box);
    gtk_widget_show_all(S.box);
    gtk_widget_realize(S.win);
}

/* -- single-instance toggle ------------------------------------------------ */
static int single(void) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    char *path = g_build_filename(run && *run ? run : "/tmp", "sound-panel.pid", NULL);
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    g_free(path);
    if (fd < 0) return 1;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        char t[32] = {0};
        if (pread(fd, t, sizeof t - 1, 0) > 0 && atoi(t) > 1) kill(atoi(t), SIGTERM);
        return 0;
    }
    if (ftruncate(fd, 0) == 0) {
        char t[32];
        int n = g_snprintf(t, sizeof t, "%d\n", (int)getpid());
        if (pwrite(fd, t, n, 0) < 0) { /* lock holds */ }
    }
    return 1;
}

int main(int argc, char **argv) {
    strcpy(S.side, "right");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pointer")) S.pointer = TRUE;
        else if (!strcmp(argv[i], "--side") && i + 1 < argc)
            g_strlcpy(S.side, argv[++i], sizeof S.side);
    }
    if (!single()) return 0;
    close_bar_popups("sound");
    gtk_init(&argc, &argv);
    g_unix_signal_add(SIGTERM, on_term, NULL);
    build_ui();
    anchor(S.pointer);
    fit();
    S.shade = backdrop(S.mon, 0, NULL);
    gtk_widget_show(S.win);
    gdk_window_raise(gtk_widget_get_window(S.win));
    grab_soon(S.win, G_CALLBACK(panel_close), NULL);

    poll_state_async();
    poll_device_async();
    S.poll_id = g_timeout_add(1000, on_poll_state, NULL);
    S.dev_poll_id = g_timeout_add(5000, on_poll_device, NULL);
    gtk_main();
    return 0;
}
