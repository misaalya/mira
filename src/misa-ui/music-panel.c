/* MusicPanel - native GTK3 MPRIS music player popup for the top bar
 * (bin/MusicPanel [--pointer] [--side left|right]).
 *
 * Replaces the former Eww music panel with zero idle memory (one-shot
 * process like BluetoothPanel): "Now playing" + player name header, 96x96
 * rounded album artwork (or fallback note icon), title & artist, smooth
 * clickable seekbar + timestamps, and shuffle / prev / play-pause / next /
 * repeat controls.
 *
 * Placement, backdrop, seat grab, click-outside/Esc dismissal, and
 * single-instance flock toggle follow BluetoothPanel.
 * picom: class MusicPanel (the "Bar panels" rule).
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

#define WIDTH 410
#define GAP 6
#define ART_SIZE 96
#define ART_RADIUS 10

/* Material Design Icons code points. */
enum {
    IC_MUSIC_BOX = 0xF075A,
    IC_SHUFFLE = 0xF049F,
    IC_PREV = 0xF04AE,
    IC_PLAY = 0xF040A,
    IC_PAUSE = 0xF03E4,
    IC_NEXT = 0xF04AD,
    IC_REPEAT_OFF = 0xF0457,
    IC_REPEAT_ALL = 0xF0456,
    IC_REPEAT_ONE = 0xF0458,
};

static struct {
    GtkWidget *win, *box;
    GtkWidget *player_lbl;
    GtkWidget *art_stack, *art_img, *art_empty;
    GtkWidget *title_lbl, *artist_lbl;
    GtkWidget *scale, *pos_lbl, *len_lbl;
    GtkWidget *controls;
    GtkWidget *btn_shuffle, *btn_prev, *btn_toggle, *btn_next, *btn_loop;

    gboolean pointer, closing, syncing, seeking;
    char side[8];
    Rect mon;
    char edge[8];
    int off, px, shade;

    gboolean has_player;
    gboolean playing;
    gboolean shuffle;
    char loop_mode[16];     /* "None", "Playlist", "Track" */
    double pos_sec, len_sec;
    char *art_url;          /* last rendered artUrl */
    char *cover_path;       /* runtime cache path for remote http(s) covers */

    guint poll_id;
    gint64 suppress_until_us;
    gboolean poll_inflight;
    gboolean art_inflight;
} S;

static void panel_close(void);
static void poll_state_async(void);

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

static void on_clickable_realize(GtkWidget *w, gpointer _d) {
    GdkWindow *gw = gtk_widget_get_window(w);
    if (!gw) return;
    GdkCursor *cur = gdk_cursor_new_from_name(gdk_window_get_display(gw), "pointer");
    if (cur) {
        gdk_window_set_cursor(gw, cur);
        g_object_unref(cur);
    }
}

static void make_clickable(GtkWidget *w) {
    g_signal_connect(w, "realize", G_CALLBACK(on_clickable_realize), NULL);
}

static void fmt_time(double sec, char *out, size_t n) {
    int total = MAX(0, (int)floor(sec));
    int h = total / 3600;
    int m = (total % 3600) / 60;
    int s = total % 60;
    if (h > 0) g_snprintf(out, n, "%d:%02d:%02d", h, m, s);
    else g_snprintf(out, n, "%d:%02d", m, s);
}

static void spawn_fire(const char *const *argv) {
    g_spawn_async(NULL, (char **)argv, NULL,
                  G_SPAWN_SEARCH_PATH | G_SPAWN_STDOUT_TO_DEV_NULL | G_SPAWN_STDERR_TO_DEV_NULL,
                  NULL, NULL, NULL, NULL);
}

static gboolean delayed_poll(gpointer _d) {
    if (!S.closing) poll_state_async();
    return G_SOURCE_REMOVE;
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

/* -- album artwork --------------------------------------------------------- */
static void show_art_empty(void) {
    gtk_image_clear(GTK_IMAGE(S.art_img));
    gtk_stack_set_visible_child_name(GTK_STACK(S.art_stack), "empty");
}

static void apply_art_file(const char *path) {
    if (!path || !*path) {
        show_art_empty();
        return;
    }
    GdkPixbuf *pb = cover_file(path, ART_SIZE, ART_SIZE, ART_RADIUS, FALSE);
    if (pb) {
        gtk_image_set_from_pixbuf(GTK_IMAGE(S.art_img), pb);
        g_object_unref(pb);
        gtk_stack_set_visible_child_name(GTK_STACK(S.art_stack), "img");
    } else {
        show_art_empty();
    }
}

static void art_dl_done(GObject *src, GAsyncResult *res, gpointer _d) {
    S.art_inflight = FALSE;
    GSubprocess *p = G_SUBPROCESS(src);
    gboolean ok = g_subprocess_wait_check_finish(p, res, NULL);
    g_object_unref(p);
    if (S.closing) return;
    if (ok && S.has_player && S.cover_path && g_file_test(S.cover_path, G_FILE_TEST_EXISTS)) {
        apply_art_file(S.cover_path);
    } else {
        show_art_empty();
    }
}

static void update_art(const char *url) {
    if (!S.has_player || !url || !*url) {
        g_clear_pointer(&S.art_url, g_free);
        show_art_empty();
        return;
    }
    if (S.art_url && strcmp(S.art_url, url) == 0) return;
    g_free(S.art_url);
    S.art_url = g_strdup(url);

    if (g_str_has_prefix(url, "file://")) {
        char *path = g_uri_unescape_string(url + 7, NULL);
        apply_art_file(path ? path : (url + 7));
        g_free(path);
        return;
    }
    if (g_str_has_prefix(url, "http://") || g_str_has_prefix(url, "https://")) {
        if (S.art_inflight) return;
        const char *argv[] = {"curl", "-fsSL", "--max-time", "4", "-o", S.cover_path, url, NULL};
        GSubprocess *p = g_subprocess_newv(argv, G_SUBPROCESS_FLAGS_STDOUT_SILENCE |
                                           G_SUBPROCESS_FLAGS_STDERR_SILENCE, NULL);
        if (!p) { show_art_empty(); return; }
        S.art_inflight = TRUE;
        g_subprocess_wait_check_async(p, NULL, art_dl_done, NULL);
        return;
    }
    show_art_empty();
}

/* -- controls & seek ------------------------------------------------------- */
static void update_controls_ui(void) {
    set_class(S.controls, "idle", !S.has_player);

    char *ic_tog = u8(S.playing ? IC_PAUSE : IC_PLAY);
    gtk_button_set_label(GTK_BUTTON(S.btn_toggle), ic_tog);
    g_free(ic_tog);

    set_class(S.btn_shuffle, "dim", !S.shuffle);

    gunichar loop_ic = IC_REPEAT_OFF;
    gboolean loop_on = FALSE;
    if (g_ascii_strcasecmp(S.loop_mode, "Track") == 0) {
        loop_ic = IC_REPEAT_ONE;
        loop_on = TRUE;
    } else if (g_ascii_strcasecmp(S.loop_mode, "Playlist") == 0) {
        loop_ic = IC_REPEAT_ALL;
        loop_on = TRUE;
    }
    char *ic_lp = u8(loop_ic);
    gtk_button_set_label(GTK_BUTTON(S.btn_loop), ic_lp);
    g_free(ic_lp);
    set_class(S.btn_loop, "dim", !loop_on);
}

static void on_shuffle_click(GtkButton *_b, gpointer _d) {
    if (!S.has_player) return;
    S.shuffle = !S.shuffle;
    update_controls_ui();
    const char *argv[] = {"playerctl", "shuffle", "toggle", NULL};
    spawn_fire(argv);
    g_timeout_add(160, delayed_poll, NULL);
}

static void on_prev_click(GtkButton *_b, gpointer _d) {
    if (!S.has_player) return;
    const char *argv[] = {"playerctl", "previous", NULL};
    spawn_fire(argv);
    g_timeout_add(220, delayed_poll, NULL);
}

static void on_toggle_click(GtkButton *_b, gpointer _d) {
    if (!S.has_player) return;
    S.playing = !S.playing;
    update_controls_ui();
    const char *argv[] = {"playerctl", "play-pause", NULL};
    spawn_fire(argv);
    g_timeout_add(180, delayed_poll, NULL);
}

static void on_next_click(GtkButton *_b, gpointer _d) {
    if (!S.has_player) return;
    const char *argv[] = {"playerctl", "next", NULL};
    spawn_fire(argv);
    g_timeout_add(220, delayed_poll, NULL);
}

static void on_loop_click(GtkButton *_b, gpointer _d) {
    if (!S.has_player) return;
    const char *next = "Playlist";
    if (g_ascii_strcasecmp(S.loop_mode, "Playlist") == 0) next = "Track";
    else if (g_ascii_strcasecmp(S.loop_mode, "Track") == 0) next = "None";
    g_strlcpy(S.loop_mode, next, sizeof S.loop_mode);
    update_controls_ui();
    const char *argv[] = {"playerctl", "loop", next, NULL};
    spawn_fire(argv);
    g_timeout_add(160, delayed_poll, NULL);
}

static void commit_seek(double sec) {
    if (!S.has_player || S.len_sec <= 0) return;
    sec = CLAMP(sec, 0.0, S.len_sec);
    S.pos_sec = sec;
    S.suppress_until_us = g_get_monotonic_time() + 500000;
    char sbuf[32];
    g_snprintf(sbuf, sizeof sbuf, "%.1f", sec);
    const char *argv[] = {"playerctl", "position", sbuf, NULL};
    spawn_fire(argv);
}

static void on_scale_changed(GtkRange *range, gpointer _d) {
    if (S.syncing || S.closing || !S.has_player) return;
    double v = gtk_range_get_value(range);
    char tbuf[24];
    fmt_time(v, tbuf, sizeof tbuf);
    gtk_label_set_text(GTK_LABEL(S.pos_lbl), tbuf);
    if (!S.seeking) commit_seek(v);
}

static gboolean on_scale_press(GtkWidget *_w, GdkEventButton *ev, gpointer _d) {
    if (ev->button == 1 && S.has_player && S.len_sec > 0) {
        S.seeking = TRUE;
        S.suppress_until_us = g_get_monotonic_time() + 1500000;
    }
    return FALSE;
}

static gboolean on_scale_release(GtkWidget *_w, GdkEventButton *_ev, gpointer _d) {
    if (S.seeking) {
        S.seeking = FALSE;
        commit_seek(gtk_range_get_value(GTK_RANGE(S.scale)));
    }
    return FALSE;
}

/* -- async MPRIS poll ------------------------------------------------------ */
static void apply_empty_state(void) {
    S.has_player = FALSE;
    S.playing = FALSE;
    S.shuffle = FALSE;
    S.pos_sec = 0;
    S.len_sec = 0;
    strcpy(S.loop_mode, "None");

    gtk_label_set_text(GTK_LABEL(S.player_lbl), "");
    gtk_label_set_text(GTK_LABEL(S.title_lbl), "Nothing playing");
    gtk_label_set_text(GTK_LABEL(S.artist_lbl), "Start a player to see it here");
    gtk_label_set_text(GTK_LABEL(S.pos_lbl), "0:00");
    gtk_label_set_text(GTK_LABEL(S.len_lbl), "0:00");

    S.syncing = TRUE;
    gtk_range_set_range(GTK_RANGE(S.scale), 0.0, 1.0);
    gtk_range_set_value(GTK_RANGE(S.scale), 0.0);
    gtk_widget_set_sensitive(S.scale, FALSE);
    set_class(S.scale, "zero", TRUE);
    S.syncing = FALSE;

    update_art(NULL);
    update_controls_ui();
}

static void state_done(GObject *src, GAsyncResult *res, gpointer _d) {
    S.poll_inflight = FALSE;
    char *out = NULL;
    GSubprocess *p = G_SUBPROCESS(src);
    gboolean fine = g_subprocess_communicate_utf8_finish(p, res, &out, NULL, NULL);
    int exited_ok = fine && g_subprocess_get_if_exited(p) && g_subprocess_get_exit_status(p) == 0;
    g_object_unref(p);
    if (S.closing) { g_free(out); return; }

    if (!exited_ok || !out) {
        g_free(out);
        apply_empty_state();
        return;
    }
    /* Take first line only (active player). */
    char *nl = strchr(out, '\n');
    if (nl) *nl = '\0';
    if (!*out) {
        g_free(out);
        apply_empty_state();
        return;
    }

    /* Fields: 0:player 1:status 2:title 3:artist 4:pos_us 5:len_us 6:shuffle 7:loop 8:artUrl */
    char *f[9] = {0};
    char *cur = out;
    for (int i = 0; i < 9; i++) {
        f[i] = cur;
        char *tab = strchr(cur, '\t');
        if (!tab) break;
        *tab = '\0';
        cur = tab + 1;
    }
    const char *player = f[0] ? f[0] : "";
    if (!*player) {
        g_free(out);
        apply_empty_state();
        return;
    }
    const char *status = f[1] ? f[1] : "Stopped";
    const char *title = f[2] ? f[2] : "";
    const char *artist = f[3] ? f[3] : "";
    double pos_sec = f[4] && *f[4] ? (g_ascii_strtod(f[4], NULL) / 1000000.0) : 0.0;
    double len_sec = f[5] && *f[5] ? (g_ascii_strtod(f[5], NULL) / 1000000.0) : 0.0;
    const char *shuf = f[6] ? f[6] : "";
    const char *loop = f[7] ? f[7] : "None";
    const char *art = f[8] ? f[8] : "";

    S.has_player = TRUE;
    S.playing = (g_ascii_strcasecmp(status, "Playing") == 0);
    S.shuffle = (g_ascii_strcasecmp(shuf, "true") == 0 || g_ascii_strcasecmp(shuf, "On") == 0);
    g_strlcpy(S.loop_mode, *loop ? loop : "None", sizeof S.loop_mode);
    S.len_sec = MAX(0.0, len_sec);

    gtk_label_set_text(GTK_LABEL(S.player_lbl), player);
    gtk_label_set_text(GTK_LABEL(S.title_lbl), *title ? title : "Unknown title");
    gtk_label_set_text(GTK_LABEL(S.artist_lbl), artist);

    if (!S.seeking && g_get_monotonic_time() >= S.suppress_until_us) {
        S.pos_sec = MAX(0.0, pos_sec);
        char pbuf[24], lbuf[24];
        fmt_time(S.pos_sec, pbuf, sizeof pbuf);
        fmt_time(S.len_sec, lbuf, sizeof lbuf);
        gtk_label_set_text(GTK_LABEL(S.pos_lbl), pbuf);
        gtk_label_set_text(GTK_LABEL(S.len_lbl), lbuf);

        S.syncing = TRUE;
        gtk_range_set_range(GTK_RANGE(S.scale), 0.0, S.len_sec > 0.0 ? S.len_sec : 1.0);
        double clamped = S.len_sec > 0.0 ? CLAMP(S.pos_sec, 0.0, S.len_sec) : 0.0;
        gtk_range_set_value(GTK_RANGE(S.scale), clamped);
        gtk_widget_set_sensitive(S.scale, S.len_sec > 0.0);
        set_class(S.scale, "zero", clamped <= 0.1);
        S.syncing = FALSE;
    }

    update_art(art);
    update_controls_ui();
    g_free(out);
}

static void poll_state_async(void) {
    if (S.closing || S.poll_inflight) return;
    static const char *fmt =
        "{{playerName}}\t{{status}}\t{{title}}\t{{artist}}\t{{position}}\t{{mpris:length}}\t{{shuffle}}\t{{loop}}\t{{mpris:artUrl}}";
    const char *argv[] = {"playerctl", "metadata", "--format", fmt, NULL};
    GSubprocess *p = g_subprocess_newv(argv, G_SUBPROCESS_FLAGS_STDOUT_PIPE |
                                       G_SUBPROCESS_FLAGS_STDERR_SILENCE, NULL);
    if (!p) { apply_empty_state(); return; }
    S.poll_inflight = TRUE;
    g_subprocess_communicate_utf8_async(p, NULL, NULL, state_done, NULL);
}

static gboolean on_poll_tick(gpointer _d) {
    if (S.closing) return G_SOURCE_REMOVE;
    poll_state_async();
    return G_SOURCE_CONTINUE;
}

/* -- open / close & input -------------------------------------------------- */
static void panel_close(void) {
    if (S.closing) return;
    S.closing = TRUE;
    if (S.poll_id) { g_source_remove(S.poll_id); S.poll_id = 0; }
    ungrab();
    gtk_widget_hide(S.win);
    drop_backdrop(S.shade);
    gtk_main_quit();
}

static gboolean on_key(GtkWidget *_w, GdkEventKey *ev, gpointer _d) {
    guint k = ev->keyval;
    if (k == GDK_KEY_Escape) { panel_close(); return TRUE; }
    if (k == GDK_KEY_space) { on_toggle_click(NULL, NULL); return TRUE; }
    if (k == GDK_KEY_Right && S.has_player && S.len_sec > 0) {
        double v = CLAMP(S.pos_sec + 5.0, 0.0, S.len_sec);
        S.syncing = TRUE;
        gtk_range_set_value(GTK_RANGE(S.scale), v);
        S.syncing = FALSE;
        char tbuf[24]; fmt_time(v, tbuf, sizeof tbuf);
        gtk_label_set_text(GTK_LABEL(S.pos_lbl), tbuf);
        commit_seek(v);
        return TRUE;
    }
    if (k == GDK_KEY_Left && S.has_player && S.len_sec > 0) {
        double v = CLAMP(S.pos_sec - 5.0, 0.0, S.len_sec);
        S.syncing = TRUE;
        gtk_range_set_value(GTK_RANGE(S.scale), v);
        S.syncing = FALSE;
        char tbuf[24]; fmt_time(v, tbuf, sizeof tbuf);
        gtk_label_set_text(GTK_LABEL(S.pos_lbl), tbuf);
        commit_seek(v);
        return TRUE;
    }
    if (k == GDK_KEY_n || k == GDK_KEY_N) { on_next_click(NULL, NULL); return TRUE; }
    if (k == GDK_KEY_p || k == GDK_KEY_P) { on_prev_click(NULL, NULL); return TRUE; }
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
        "window, stack { background: transparent; }\n"
        ".player {\n"
        "  background-color: @bg;\n"
        "  color: @fg;\n"
        "  border: 1px solid alpha(@fg, 0.12);\n"
        "  border-radius: 14px;\n"
        "  padding: 12px 14px 12px 14px;\n"
        "}\n"
        ".caption {\n"
        "  font-size: 12px;\n"
        "  font-weight: bold;\n"
        "  color: alpha(@fg, 0.50);\n"
        "}\n"
        ".art-box {\n"
        "  background-color: alpha(@fg, 0.08);\n"
        "  border-radius: 10px;\n"
        "  min-width: 96px;\n"
        "  min-height: 96px;\n"
        "}\n"
        ".art-icon {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 40px;\n"
        "  color: alpha(@fg, 0.35);\n"
        "}\n"
        ".title {\n"
        "  font-size: 15px;\n"
        "  font-weight: bold;\n"
        "  color: @fg;\n"
        "}\n"
        ".artist {\n"
        "  font-size: 12px;\n"
        "  color: alpha(@fg, 0.60);\n"
        "}\n"
        ".time {\n"
        "  font-size: 11px;\n"
        "  color: alpha(@fg, 0.50);\n"
        "}\n"
        ".progress { margin-top: 6px; }\n"
        "scale.seek { padding: 4px 0px; margin: 0; }\n"
        "scale.seek contents { padding: 0; margin: 0; }\n"
        "scale.seek trough {\n"
        "  min-height: 4px;\n"
        "  border-radius: 4px;\n"
        "  background-color: alpha(@fg, 0.15);\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "  background-image: none;\n"
        "  padding: 0;\n"
        "  margin: 0;\n"
        "}\n"
        "scale.seek highlight {\n"
        "  min-height: 4px;\n"
        "  border-radius: 4px;\n"
        "  background-color: @fg;\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "  background-image: none;\n"
        "  padding: 0;\n"
        "  margin: 0;\n"
        "}\n"
        "scale.seek.zero highlight { background-color: transparent; }\n"
        "scale.seek slider {\n"
        "  min-width: 0px;\n"
        "  min-height: 0px;\n"
        "  margin: 0px;\n"
        "  padding: 0px;\n"
        "  background: transparent;\n"
        "  background-image: none;\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "}\n"
        ".controls { transition: opacity 120ms ease-out; }\n"
        ".controls.idle { opacity: 0.35; }\n"
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
        "button.ctl {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 20px;\n"
        "  padding: 4px 10px;\n"
        "  border-radius: 999px;\n"
        "  background: transparent;\n"
        "  color: @fg;\n"
        "}\n"
        "button.ctl.dim { opacity: 0.45; }\n"
        "button.ctl:hover { background-color: alpha(@fg, 0.08); }\n"
        "button.ctl.toggle {\n"
        "  background-color: @fg;\n"
        "  color: @bg;\n"
        "  min-width: 26px;\n"
        "  min-height: 26px;\n"
        "  padding: 8px;\n"
        "}\n"
        "button.ctl.toggle:hover { background-color: alpha(@fg, 0.85); }\n"
        "tooltip, tooltip.background {\n"
        "  background-color: @bg;\n"
        "  color: @fg;\n"
        "  border: 1px solid alpha(@fg, 0.14);\n"
        "  border-radius: 8px;\n"
        "}\n");
    return g_string_free(c, FALSE);
}

static GtkWidget *make_ctl(gunichar glyph, const char *extra_cls, const char *tip, GCallback cb) {
    char *t = u8(glyph);
    GtkWidget *b = gtk_button_new_with_label(t);
    g_free(t);
    add_class(b, "ctl");
    if (extra_cls) add_class(b, extra_cls);
    gtk_widget_set_can_focus(b, FALSE);
    if (tip) gtk_widget_set_tooltip_text(b, tip);
    make_clickable(b);
    g_signal_connect(b, "clicked", cb, NULL);
    return b;
}

static void build_ui(void) {
    g_object_set(gtk_settings_get_default(), "gtk-primary-button-warps-slider", TRUE, NULL);

    Palette p;
    palette_load(&p);
    char *css = build_css(&p);
    css_load(css);
    g_free(css);

    S.win = overlay_window("MusicPanel", 0, 0, WIDTH, 100);
    gtk_widget_set_size_request(S.win, WIDTH, -1);
    gtk_window_set_default_size(GTK_WINDOW(S.win), -1, -1);
    gtk_window_set_resizable(GTK_WINDOW(S.win), TRUE);
    gtk_widget_add_events(S.win, GDK_KEY_PRESS_MASK | GDK_BUTTON_PRESS_MASK);
    g_signal_connect(S.win, "key-press-event", G_CALLBACK(on_key), NULL);
    g_signal_connect(S.win, "button-press-event", G_CALLBACK(on_outside), NULL);
    g_signal_connect(S.win, "size-allocate", G_CALLBACK(on_alloc), NULL);

    S.box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    add_class(S.box, "player");

    /* Header */
    GtkWidget *head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    GtkWidget *cap = gtk_label_new("Now playing");
    add_class(cap, "caption");
    gtk_label_set_xalign(GTK_LABEL(cap), 0.0f);
    gtk_widget_set_hexpand(cap, TRUE);

    S.player_lbl = gtk_label_new("");
    add_class(S.player_lbl, "caption");
    gtk_label_set_xalign(GTK_LABEL(S.player_lbl), 1.0f);
    gtk_label_set_ellipsize(GTK_LABEL(S.player_lbl), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(S.player_lbl), 18);

    gtk_box_pack_start(GTK_BOX(head), cap, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(head), S.player_lbl, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.box), head, FALSE, FALSE, 0);

    /* Main row: art + metadata & progress */
    GtkWidget *main_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 14);

    S.art_stack = gtk_stack_new();
    gtk_stack_set_transition_type(GTK_STACK(S.art_stack), GTK_STACK_TRANSITION_TYPE_CROSSFADE);
    gtk_stack_set_transition_duration(GTK_STACK(S.art_stack), 120);
    gtk_widget_set_size_request(S.art_stack, ART_SIZE, ART_SIZE);
    gtk_widget_set_valign(S.art_stack, GTK_ALIGN_START);

    GtkWidget *empty_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    add_class(empty_box, "art-box");
    char *ic_note = u8(IC_MUSIC_BOX);
    S.art_empty = gtk_label_new(ic_note);
    g_free(ic_note);
    add_class(S.art_empty, "art-icon");
    gtk_widget_set_halign(S.art_empty, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(S.art_empty, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(empty_box), S.art_empty, TRUE, TRUE, 0);

    S.art_img = gtk_image_new();
    gtk_widget_set_size_request(S.art_img, ART_SIZE, ART_SIZE);

    gtk_stack_add_named(GTK_STACK(S.art_stack), empty_box, "empty");
    gtk_stack_add_named(GTK_STACK(S.art_stack), S.art_img, "img");
    gtk_stack_set_visible_child_name(GTK_STACK(S.art_stack), "empty");

    GtkWidget *body = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_set_hexpand(body, TRUE);
    gtk_widget_set_valign(body, GTK_ALIGN_CENTER);

    S.title_lbl = gtk_label_new("Nothing playing");
    add_class(S.title_lbl, "title");
    gtk_label_set_xalign(GTK_LABEL(S.title_lbl), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(S.title_lbl), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(S.title_lbl), 26);

    S.artist_lbl = gtk_label_new("Start a player to see it here");
    add_class(S.artist_lbl, "artist");
    gtk_label_set_xalign(GTK_LABEL(S.artist_lbl), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(S.artist_lbl), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(S.artist_lbl), 32);

    GtkWidget *prog = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    add_class(prog, "progress");

    S.scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 1.0, 1.0);
    add_class(S.scale, "seek");
    gtk_scale_set_draw_value(GTK_SCALE(S.scale), FALSE);
    gtk_widget_set_can_focus(S.scale, FALSE);
    gtk_widget_set_hexpand(S.scale, TRUE);
    make_clickable(S.scale);
    g_signal_connect(S.scale, "value-changed", G_CALLBACK(on_scale_changed), NULL);
    g_signal_connect(S.scale, "button-press-event", G_CALLBACK(on_scale_press), NULL);
    g_signal_connect(S.scale, "button-release-event", G_CALLBACK(on_scale_release), NULL);

    GtkWidget *times = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    S.pos_lbl = gtk_label_new("0:00");
    add_class(S.pos_lbl, "time");
    gtk_label_set_xalign(GTK_LABEL(S.pos_lbl), 0.0f);
    gtk_widget_set_hexpand(S.pos_lbl, TRUE);
    S.len_lbl = gtk_label_new("0:00");
    add_class(S.len_lbl, "time");
    gtk_label_set_xalign(GTK_LABEL(S.len_lbl), 1.0f);
    gtk_box_pack_start(GTK_BOX(times), S.pos_lbl, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(times), S.len_lbl, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(prog), S.scale, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(prog), times, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(body), S.title_lbl, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(body), S.artist_lbl, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(body), prog, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(main_row), S.art_stack, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(main_row), body, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(S.box), main_row, FALSE, FALSE, 0);

    /* Controls */
    S.controls = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    add_class(S.controls, "controls");
    add_class(S.controls, "idle");
    gtk_widget_set_halign(S.controls, GTK_ALIGN_CENTER);

    S.btn_shuffle = make_ctl(IC_SHUFFLE, "dim", "Shuffle", G_CALLBACK(on_shuffle_click));
    S.btn_prev = make_ctl(IC_PREV, "skip", "Previous", G_CALLBACK(on_prev_click));
    S.btn_toggle = make_ctl(IC_PLAY, "toggle", "Play / Pause", G_CALLBACK(on_toggle_click));
    S.btn_next = make_ctl(IC_NEXT, "skip", "Next", G_CALLBACK(on_next_click));
    S.btn_loop = make_ctl(IC_REPEAT_OFF, "dim", "Repeat", G_CALLBACK(on_loop_click));

    gtk_box_pack_start(GTK_BOX(S.controls), S.btn_shuffle, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.controls), S.btn_prev, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.controls), S.btn_toggle, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.controls), S.btn_next, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.controls), S.btn_loop, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.box), S.controls, FALSE, FALSE, 0);

    gtk_container_add(GTK_CONTAINER(S.win), S.box);
    gtk_widget_show_all(S.box);
    gtk_widget_realize(S.win);
}

/* -- single-instance toggle ------------------------------------------------ */
static int single(void) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    char *path = g_build_filename(run && *run ? run : "/tmp", "music-panel.pid", NULL);
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
    strcpy(S.loop_mode, "None");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pointer")) S.pointer = TRUE;
        else if (!strcmp(argv[i], "--side") && i + 1 < argc)
            g_strlcpy(S.side, argv[++i], sizeof S.side);
    }
    if (!single()) return 0;
    close_bar_popups("music");
    gtk_env_lean();
    gtk_init(&argc, &argv);
    g_unix_signal_add(SIGTERM, on_term, NULL);

    const char *run = g_getenv("XDG_RUNTIME_DIR");
    S.cover_path = g_build_filename(run && *run ? run : "/tmp", "mira-music-cover.img", NULL);

    build_ui();
    anchor(S.pointer);
    fit();
    S.shade = backdrop(S.mon, 0, NULL);
    gtk_widget_show(S.win);
    gdk_window_raise(gtk_widget_get_window(S.win));
    grab_soon(S.win, G_CALLBACK(panel_close), NULL);

    poll_state_async();
    S.poll_id = g_timeout_add(1000, on_poll_tick, NULL);
    gtk_main();

    g_free(S.art_url);
    g_free(S.cover_path);
    return 0;
}
