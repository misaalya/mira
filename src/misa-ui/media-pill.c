/* MediaPillUI - the now-playing bubble (started by bin/MediaPill when
 * something starts playing; exits by itself).
 *
 * Bubble: the album art as a disc that swells with the sound inside a thin
 * ring that fills with the track's progress. Monochrome, in the rice's own
 * foreground: no colour taken from the cover.
 *   - drag it anywhere; let go and it glides into the nearest corner of the
 *     monitor under the pointer (any monitor). The corner is remembered.
 *   - click: it grows from its corner into a "now playing" card: the cover
 *     with title and artist beside it, a full-width play bar with the elapsed
 *     and total time, and prev / play-pause / next.
 *   - a click anywhere else (or on the cover) folds it back into the bubble
 *     (that click is passed on to whatever was under it). Right click: play /
 *     pause.
 *
 * Cost:
 *   - no polling: track and status come from `playerctl -F` (event driven);
 *     the time and the ring advance on a 1 s timer only while playing;
 *   - the sound level comes from `parec` on the output monitor (8 kHz mono)
 *     only while the bubble is shown and playing;
 *   - every picture (16 disc sizes per track, 61 ring frames once) is made
 *     ahead, each in its own Image of a Stack: a level change only flips the
 *     visible child (no new surface, no allocation per frame);
 *   - covers are decoded small; freed heap is handed back (malloc_trim);
 *   - paused for a minute or stopped, the process exits (bin/MediaPill
 *     starts it again).
 *
 * An override-redirect window (bspwm never manages it); picom: class MediaPill.
 */
#include "common.h"

#include <errno.h>
#include <glib-unix.h>
#include <glib/gstdio.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define ART 54                 /* px, the disc */
#define SWELL 0.20             /* the disc grows up to +20% with the sound */
#define GAP 3                  /* px, ring: gap to the fully swollen disc */
#define TH 3                   /* px, ring thickness */
#define R_IN (ART * (1 + SWELL) / 2 + GAP)
#define RING ((int)(2 * (R_IN + TH)) + 2)  /* the disc + ring square (~78) */
#define HOLD (RING + 6)        /* px, the bubble window */
#define LEVELS 16
#define RING_STEPS 60
#define MARGIN 18              /* px from the monitor's edges */
#define CARD_W 300             /* px, the open card */
#define COVER 64               /* px, its cover */
#define OPEN_MS 280
#define PAUSED_EXIT 60
#define RATE 8000
#define CHUNK (RATE / 20 * 2)  /* bytes per 50 ms (s16 mono): 20 updates/s */
#define DRAG_START 6           /* px of motion before a press becomes a drag */

#define FMT "{{status}}\t{{mpris:artUrl}}\t{{title}}\t{{artist}}\t{{mpris:length}}"

#define G_PAUSE "\xf3\xb0\x8f\xa4"   /* U+F03E4 */
#define G_PLAY  "\xf3\xb0\x90\x8a"   /* U+F040A */
#define G_PREV  "\xf3\xb0\x92\xae"   /* U+F04AE */
#define G_NEXT  "\xf3\xb0\x92\xad"   /* U+F04AD */

typedef struct { int x, y, w, h, tp, bp; } Mon;

static struct {
    Palette pal;
    guint8 fg[3];
    char *cache, *dock_file;
    char *status, *title, *artist, *art_url;
    double length, pos_ref, t_ref;
    gboolean have_discs;
    int step, ring_step;
    double level, peak;
    GPid parec;
    int parec_fd;
    guint parec_watch;
    gboolean open;
    guint exit_timer, clock_timer;
    double anim_until;
    gboolean seeking;
    gboolean pressed;
    double press_x, press_y;
    int press_wx, press_wy;
    gboolean dragging;
    gboolean gliding;
    int g_sx, g_sy, g_ex, g_ey;
    double g_t0;
    gboolean have_mon;
    Mon mon;
    char *dock_model, *dock_corner;
    GPid pc;
    int pc_fd;
    GString *buf;
    GdkPixbuf *rings[RING_STEPS + 1];
    GtkWidget *win, *ring, *disc, *hold, *bub, *cover, *l_title, *l_artist;
    GtkWidget *seek, *l_pos, *l_len, *b_play, *b_play_label, *stack, *shell;
} P;

static double now(void) { return g_get_monotonic_time() / 1e6; }

static void set_str(char **slot, const char *v) {
    g_free(*slot);
    *slot = g_strdup(v);
}

static void fmt_time(double s, char out[32]) {
    long v = (long)MAX(0.0, s);
    g_snprintf(out, 32, "%ld:%02ld", v / 60, v % 60);
}

static void hexrgb(const char *h, guint8 out[3]) {
    unsigned long v = strtoul(h + 1, NULL, 16);
    out[0] = (v >> 16) & 255;
    out[1] = (v >> 8) & 255;
    out[2] = v & 255;
}

static void spawn_quiet(const char *const *argv) {
    g_spawn_async(NULL, (char **)argv, NULL,
                  G_SPAWN_SEARCH_PATH | G_SPAWN_STDOUT_TO_DEV_NULL | G_SPAWN_STDERR_TO_DEV_NULL, NULL,
                  NULL, NULL, NULL);
}

static gboolean spawn_pipe(const char *const *argv, GPid *pid, int *fd) {
    return g_spawn_async_with_pipes(NULL, (char **)argv, NULL,
                                    G_SPAWN_SEARCH_PATH | G_SPAWN_DO_NOT_REAP_CHILD |
                                        G_SPAWN_STDERR_TO_DEV_NULL,
                                    NULL, NULL, pid, NULL, fd, NULL, NULL);
}

static void stop_child(GPid *pid, int *fd) {
    if (*pid > 0) {
        kill(*pid, SIGKILL);
        waitpid(*pid, NULL, 0);
    }
    if (*fd >= 0) close(*fd);
    *pid = 0;
    *fd = -1;
}

/* -- pictures made ahead -------------------------------------------------------- */
static void free_pixels(guchar *pixels, gpointer _) { g_free(pixels); }

static GdkPixbuf *rgba_pixbuf(int size, guchar *data) {
    return gdk_pixbuf_new_from_data(data, GDK_COLORSPACE_RGB, TRUE, 8, size, size, size * 4,
                                    free_pixels, NULL);
}

/* RING_STEPS + 1 pictures of the progress ring (0 .. full), clockwise from
 * the top: fg where done, a faint track elsewhere. */
static void ring_frames(void) {
    double c = RING / 2.0, r_in = R_IN, r_out = R_IN + TH;
    int n = 0;
    int *off = g_new(int, RING * RING);
    double *cov = g_new(double, RING * RING), *ang = g_new(double, RING * RING);
    for (int y = 0; y < RING; y++) {
        double dy = y + 0.5 - c;
        for (int x = 0; x < RING; x++) {
            double dx = x + 0.5 - c;
            double r = hypot(dx, dy);
            double a = MIN(r - r_in, r_out - r) + 0.5;
            if (a <= 0) continue;
            double f = atan2(dx, -dy) / (2 * G_PI);
            f -= floor(f);                          /* Python's % 1.0 */
            off[n] = (y * RING + x) * 4;
            cov[n] = MIN(1.0, a);
            ang[n] = f;
            n++;
        }
    }
    for (int k = 0; k <= RING_STEPS; k++) {
        double done = (double)k / RING_STEPS;
        guchar *data = g_malloc0(RING * RING * 4);
        for (int i = 0; i < n; i++) {
            memcpy(data + off[i], P.fg, 3);
            data[off[i] + 3] = (guchar)(255 * cov[i] * (ang[i] < done ? 1.0 : 0.16));
        }
        P.rings[k] = rgba_pixbuf(RING, data);
    }
    g_free(off);
    g_free(cov);
    g_free(ang);
}

static char *css(const Palette *p) {
    const char *fg = p->fg, *bg = p->bg;
    return g_strdup_printf(
        "* { font-family: \"%s\"; }\n"
        "window { background: transparent; }\n"
        ".card {\n"
        "    background-color: %s;\n"
        "    border: 1px solid alpha(%s, 0.10);\n"
        "    border-radius: 22px;\n"
        "    padding: 16px 16px 12px 16px;\n"
        "}\n"
        ".shell.paused .disc { opacity: 0.55; }\n"
        ".disc { transition: opacity 300ms ease-out; }\n"
        ".title { font-size: 15px; font-weight: 700; color: %s; }\n"
        ".artist { font-size: 12px; font-weight: 500; color: alpha(%s, 0.6); }\n"
        ".time { font-size: 10px; font-weight: 600; color: alpha(%s, 0.5); font-feature-settings: \"tnum\"; }\n"
        ".ctl {\n"
        "    font-family: \"Material Design Icons Desktop\"; font-size: 24px;\n"
        "    color: alpha(%s, 0.7); padding: 8px; border-radius: 999px;\n"
        "    transition: background-color 150ms ease-out, color 150ms ease-out;\n"
        "}\n"
        ".ctl:hover { color: %s; background-color: alpha(%s, 0.10); }\n"
        ".play {\n"
        "    font-family: \"Material Design Icons Desktop\"; font-size: 26px;\n"
        "    min-width: 50px; min-height: 50px; border-radius: 999px;\n"
        "    background-color: %s; color: %s;\n"
        "}\n"
        "scale { padding: 0px; margin: 0px; min-height: 10px; background: none; border: none; box-shadow: none; }\n"
        "scale trough { min-height: 4px; border-radius: 4px; border: none; box-shadow: none; outline: none;\n"
        "               background-image: none; background-color: alpha(%s, 0.16); }\n"
        "scale highlight { min-height: 4px; border-radius: 4px; border: none; box-shadow: none;\n"
        "                   background-image: none; background-color: %s; }\n"
        "scale slider { min-width: 0px; min-height: 0px; margin: 0px; background: none; border: none; box-shadow: none; }\n"
        "scale:hover slider { min-width: 10px; min-height: 10px; margin: -4px; border-radius: 999px; background-color: %s; }\n",
        p->font, bg, fg, fg, fg, fg, fg, fg, fg, fg, bg, fg, fg, fg);
}

/* -- forward ---------------------------------------------------------------------- */
static gboolean quit(gpointer _);
static void set_open(gboolean value);
static void make_pictures(const char *path);
static void update_level_source(void);

/* -- dock: which corner of which monitor ---------------------------------------- */
static void load_dock(void) {
    char *text = NULL;
    set_str(&P.dock_model, "");
    set_str(&P.dock_corner, "br");
    if (!g_file_get_contents(P.dock_file, &text, NULL, NULL)) return;
    char **w = g_strsplit_set(g_strstrip(text), " \t\n", -1);
    char *tok[3] = {0};
    int n = 0;
    for (int i = 0; w[i]; i++) {
        if (!*w[i]) continue;
        if (n < 3) tok[n] = w[i];
        n++;
    }
    if (n == 2) {
        set_str(&P.dock_model, tok[0]);
        set_str(&P.dock_corner, tok[1]);
    }
    g_strfreev(w);
    g_free(text);
}

static void save_dock(void) {
    g_mkdir_with_parents(P.cache, 0755);
    char *s = g_strdup_printf("%s %s\n", P.dock_model, P.dock_corner);
    g_file_set_contents(P.dock_file, s, -1, NULL);
    g_free(s);
}

static int bspc_pad(const char *side) {
    const char *argv[] = {"bspc", "config", "-m", *P.dock_model ? P.dock_model : "focused", side, NULL};
    char *out = run_out(argv);
    int v = 0;
    if (out) {
        char *s = g_strstrip(out);
        const char *d = s[0] == '-' ? s + strspn(s, "-") : s;
        if (*d && strspn(d, "0123456789") == strlen(d)) v = atoi(s);
        g_free(out);
    }
    return MAX(0, v);
}

/* The dock's monitor (the focused one when the remembered monitor is not
 * connected) and bspwm's top / bottom padding there. */
static Mon monitor(void) {
    GdkDisplay *d = gdk_display_get_default();
    GdkMonitor *mon = NULL;
    for (int i = 0; i < gdk_display_get_n_monitors(d); i++) {
        GdkMonitor *m = gdk_display_get_monitor(d, i);
        const char *model = gdk_monitor_get_model(m);
        if (*P.dock_model && model && !strcmp(model, P.dock_model)) mon = m;
    }
    if (!mon) {
        Rect r = focused_monitor();
        mon = gdk_display_get_monitor_at_point(d, r.x + r.w / 2, r.y + r.h / 2);
        const char *model = gdk_monitor_get_model(mon);
        set_str(&P.dock_model, model ? model : "");
    }
    GdkRectangle g;
    gdk_monitor_get_geometry(mon, &g);
    Mon m = {g.x, g.y, g.width, g.height, bspc_pad("top_padding"), bspc_pad("bottom_padding")};
    return m;
}

static void target(int w, int h, int *tx, int *ty) {
    const Mon *m = &P.mon;
    *tx = P.dock_corner[1] == 'r' ? m->x + m->w - MARGIN - w : m->x + MARGIN;
    *ty = P.dock_corner[0] == 'b' ? m->y + m->h - MARGIN - m->bp - h : m->y + MARGIN + m->tp;
}

static void size(int *w, int *h) {
    GtkRequisition nat;
    gtk_widget_get_preferred_size(P.shell, NULL, &nat);
    *w = nat.width;
    *h = nat.height;
}

static void place(void) {
    if (P.dragging || P.gliding || !P.have_mon) return;
    int w, h, x, y;
    size(&w, &h);
    gtk_window_resize(GTK_WINDOW(P.win), w, h);
    target(w, h, &x, &y);
    gtk_window_move(GTK_WINDOW(P.win), x, y);
}

/* -- time ---------------------------------------------------------------------------- */
static double position(void) {
    double p = P.pos_ref;
    if (!g_strcmp0(P.status, "Playing")) p += now() - P.t_ref;
    return P.length ? MIN(p, P.length) : p;
}

static void show_time(void) {
    double p = position();
    if (P.length) {
        int k = MIN(RING_STEPS, (int)(p / P.length * RING_STEPS));
        if (k != P.ring_step) {
            P.ring_step = k;
            gtk_image_set_from_pixbuf(GTK_IMAGE(P.ring), P.rings[k]);
        }
    }
    if (P.open) {
        char t[32];
        if (!P.seeking) gtk_range_set_value(GTK_RANGE(P.seek), p);
        fmt_time(p, t);
        gtk_label_set_text(GTK_LABEL(P.l_pos), t);
        if (P.length) fmt_time(P.length, t);
        else t[0] = 0;
        gtk_label_set_text(GTK_LABEL(P.l_len), t);
    }
}

static void sync_position(void) {
    const char *argv[] = {"playerctl", "position", NULL};
    char *out = run_out(argv);
    P.pos_ref = 0.0;
    if (out) {
        char *s = g_strstrip(out), *end = NULL;
        double v = g_ascii_strtod(s, &end);
        if (*s && end && !*end && isfinite(v)) P.pos_ref = v;
        g_free(out);
    }
    P.t_ref = now();
    show_time();
}

static gboolean tick_clock(gpointer _) {
    show_time();
    if (g_strcmp0(P.status, "Playing")) {
        P.clock_timer = 0;
        return G_SOURCE_REMOVE;
    }
    return G_SOURCE_CONTINUE;
}

static void ctl(const char *action) {
    const char *argv[] = {"playerctl", action, NULL};
    spawn_quiet(argv);
}

/* -- player events ---------------------------------------------------------------- */
static void show(void);

static void set_status(const char *status) {
    char *prev = P.status;
    P.status = g_strdup(status);
    gboolean playing = !strcmp(status, "Playing");
    gtk_label_set_text(GTK_LABEL(P.b_play_label), playing ? G_PAUSE : G_PLAY);
    GtkStyleContext *ctx = gtk_widget_get_style_context(P.shell);
    if (playing) gtk_style_context_remove_class(ctx, "paused");
    else gtk_style_context_add_class(ctx, "paused");
    if (g_strcmp0(status, prev)) sync_position();
    g_free(prev);
    if (P.exit_timer) {
        g_source_remove(P.exit_timer);
        P.exit_timer = 0;
    }
    if (playing) {
        if (!gtk_widget_get_visible(P.win)) show();
        if (!P.clock_timer) P.clock_timer = g_timeout_add(1000, tick_clock, NULL);
    } else if (!strcmp(status, "Paused")) {
        P.exit_timer = g_timeout_add_seconds(PAUSED_EXIT, quit, NULL);
    } else {
        set_open(FALSE);
        gtk_widget_hide(P.win);
        P.exit_timer = g_timeout_add_seconds(2, quit, NULL);
    }
    update_level_source();
}

static gboolean all_digits(const char *s) {
    return *s && strspn(s, "0123456789") == strlen(s);
}

static void load_art(const char *url);

static void on_line(const char *line) {
    char **parts = g_strsplit(line, "\t", -1);
    const char *f[5] = {"", "", "", "", ""};
    for (int i = 0; i < 5 && parts[i]; i++) f[i] = parts[i];
    const char *status = f[0], *art = f[1], *title = f[2], *artist = f[3], *length = f[4];
    if (!*status) {
        set_status("Stopped");
        g_strfreev(parts);
        return;
    }
    if (g_strcmp0(title, P.title) || g_strcmp0(artist, P.artist) || g_strcmp0(art, P.art_url)) {
        gboolean new_art = g_strcmp0(art, P.art_url) != 0;
        set_str(&P.title, title);
        set_str(&P.artist, artist);
        set_str(&P.art_url, art);
        P.length = all_digits(length) ? g_ascii_strtod(length, NULL) / 1e6 : 0.0;
        gtk_label_set_text(GTK_LABEL(P.l_title), *title ? title : "Unknown title");
        gtk_label_set_text(GTK_LABEL(P.l_artist), *artist ? artist : "Unknown artist");
        gtk_range_set_range(GTK_RANGE(P.seek), 0, MAX(1.0, P.length));
        if (new_art || !P.have_discs) load_art(art);
        P.ring_step = -1;
        sync_position();
    }
    set_status(status);
    g_strfreev(parts);
}

static gboolean on_meta(int fd, GIOCondition cond, gpointer _) {
    /* Raw reads + own line buffer: a buffered reader could keep a second
     * line hidden in its buffer until the next event. */
    char data[65536];
    ssize_t n = 0;
    if (cond & G_IO_IN) {
        do n = read(fd, data, sizeof data); while (n < 0 && errno == EINTR);
    }
    if (n <= 0) {
        quit(NULL);
        return G_SOURCE_REMOVE;
    }
    g_string_append_len(P.buf, data, n);
    char *nl;
    while ((nl = memchr(P.buf->str, '\n', P.buf->len))) {
        char *line = g_strndup(P.buf->str, nl - P.buf->str);
        g_string_erase(P.buf, 0, nl - P.buf->str + 1);
        if (!g_utf8_validate(line, -1, NULL)) {
            char *ok = g_utf8_make_valid(line, -1);
            g_free(line);
            line = ok;
        }
        on_line(line);
        g_free(line);
    }
    return G_SOURCE_CONTINUE;
}

static void follow(void) {
    P.buf = g_string_new("");
    const char *argv[] = {"playerctl", "-F", "metadata", "--format", FMT, NULL};
    if (!spawn_pipe(argv, &P.pc, &P.pc_fd)) {
        g_idle_add(quit, NULL);
        return;
    }
    g_unix_fd_add(P.pc_fd, G_IO_IN | G_IO_HUP | G_IO_ERR, on_meta, NULL);
}

/* -- art ------------------------------------------------------------------------------ */
typedef struct { char *url, *path, *part; } Fetch;

static void fetched(GPid pid, gint status, gpointer data) {
    Fetch *f = data;
    g_spawn_close_pid(pid);
    if (g_spawn_check_wait_status(status, NULL) && g_rename(f->part, f->path) == 0) {
        if (!g_strcmp0(f->url, P.art_url)) make_pictures(f->path);
    } else {
        g_unlink(f->part);
    }
    g_free(f->url);
    g_free(f->path);
    g_free(f->part);
    g_free(f);
}

static void load_art(const char *url) {
    if (g_str_has_prefix(url, "file://")) {
        char *path = g_uri_unescape_string(url + 7, NULL);
        make_pictures(path ? path : url + 7);
        g_free(path);
    } else if (g_str_has_prefix(url, "http://") || g_str_has_prefix(url, "https://")) {
        char *sum = g_compute_checksum_for_string(G_CHECKSUM_SHA1, url, -1);
        char *path = g_build_filename(P.cache, sum, NULL);
        g_free(sum);
        if (g_file_test(path, G_FILE_TEST_EXISTS)) {
            make_pictures(path);
            g_free(path);
            return;
        }
        make_pictures(NULL);
        g_mkdir_with_parents(P.cache, 0755);
        Fetch *f = g_new0(Fetch, 1);
        f->url = g_strdup(url);
        f->path = path;
        f->part = g_strconcat(path, ".part", NULL);
        const char *argv[] = {"curl", "-fsL", "--max-time", "8", "--max-filesize", "4194304",
                              "-o", f->part, "--", url, NULL};
        GPid pid;
        if (g_spawn_async(NULL, (char **)argv, NULL,
                          G_SPAWN_SEARCH_PATH | G_SPAWN_DO_NOT_REAP_CHILD | G_SPAWN_STDOUT_TO_DEV_NULL |
                              G_SPAWN_STDERR_TO_DEV_NULL,
                          NULL, NULL, &pid, NULL)) {
            g_child_watch_add(pid, fetched, f);
        } else {
            g_free(f->url);
            g_free(f->path);
            g_free(f->part);
            g_free(f);
        }
    } else {
        make_pictures(NULL);
    }
}

static void set_step(int s) {
    if (s == P.step || !P.have_discs) return;
    P.step = s;
    char name[8];
    g_snprintf(name, sizeof name, "%d", s);
    gtk_stack_set_visible_child_name(GTK_STACK(P.disc), name);
}

static void make_pictures(const char *path) {
    int big = (int)(ART * (1 + SWELL));
    GdkPixbuf *pb = NULL;
    if (path) /* decoded small: a big cover never sits in memory */
        pb = gdk_pixbuf_new_from_file_at_scale(path, big * 2, big * 2, TRUE, NULL);
    if (!pb) {
        pb = gdk_pixbuf_new(GDK_COLORSPACE_RGB, TRUE, 8, big, big);
        guint8 bg[3];
        hexrgb(P.pal.bg, bg);
        int mix[3];
        for (int i = 0; i < 3; i++) mix[i] = (int)(bg[i] * 0.8 + P.fg[i] * 0.2);
        gdk_pixbuf_fill(pb, ((guint32)mix[0] << 24) | (mix[1] << 16) | (mix[2] << 8) | 0xFF);
        GtkIconTheme *theme = gtk_icon_theme_get_default();
        static const char *names[] = {"audio-x-generic-symbolic", "audio-x-generic", NULL};
        for (int i = 0; names[i]; i++) {
            if (!gtk_icon_theme_has_icon(theme, names[i])) continue;
            GdkPixbuf *ic = gtk_icon_theme_load_icon(theme, names[i], big / 2, GTK_ICON_LOOKUP_FORCE_SIZE, NULL);
            if (ic) {
                int q = big / 4;
                gdk_pixbuf_composite(ic, pb, q, q, big / 2, big / 2, q, q, 1, 1, GDK_INTERP_BILINEAR, 200);
                g_object_unref(ic);
            }
            break;
        }
    }
    GdkPixbuf *cv = cover_pixbuf(pb, COVER, COVER, 12, FALSE);
    gtk_image_set_from_pixbuf(GTK_IMAGE(P.cover), cv);
    g_object_unref(cv);
    GdkPixbuf *disc = circle_pixbuf(pb, big);
    g_object_unref(pb);
    GList *kids = gtk_container_get_children(GTK_CONTAINER(P.disc));
    for (GList *l = kids; l; l = l->next) gtk_container_remove(GTK_CONTAINER(P.disc), l->data);
    g_list_free(kids);
    for (int i = 0; i < LEVELS; i++) {
        double t = (double)i / (LEVELS - 1);
        int d = (int)nearbyint(ART * (1 + SWELL * t));
        GdkPixbuf *s = gdk_pixbuf_scale_simple(disc, d, d, GDK_INTERP_BILINEAR);
        char name[8];
        g_snprintf(name, sizeof name, "%d", i);
        gtk_stack_add_named(GTK_STACK(P.disc), gtk_image_new_from_pixbuf(s), name);
        g_object_unref(s);
    }
    g_object_unref(disc);
    P.have_discs = TRUE;
    gtk_widget_show_all(P.disc);
    P.step = -1;
    set_step(0);
    heap_trim();
}

/* -- sound level -------------------------------------------------------------------- */
static gboolean on_sound(int fd, GIOCondition cond, gpointer _) {
    guchar data[CHUNK * 4];
    ssize_t n = 0;
    if (cond & G_IO_IN) {
        do n = read(fd, data, sizeof data); while (n < 0 && errno == EINTR);
    }
    if (n <= 0) {
        /* parec went away: forget it (reaped when the next one is wanted) */
        P.parec_watch = 0;
        stop_child(&P.parec, &P.parec_fd);
        return G_SOURCE_REMOVE;
    }
    const guchar *d = data + (n > CHUNK ? n - CHUNK : 0);   /* only the newest 50 ms */
    int len = (int)MIN(n, CHUNK) / 2;
    if (!len) return G_SOURCE_CONTINUE;
    double sum = 0;
    for (int i = 0; i < len; i++) {
        gint16 v = (gint16)(d[2 * i] | d[2 * i + 1] << 8);
        sum += (double)v * v;
    }
    double rms = sqrt(sum / len);
    /* Auto gain: compare with the recent loudest (decays over ~4 s), so beats
     * stand out at any volume; quiet parts sit near zero. */
    P.peak = MAX(MAX(rms, P.peak * 0.992), 400.0);
    double target = pow(MIN(1.0, MAX(0.0, (rms / P.peak - 0.25) / 0.75)), 1.3);
    double k = target > P.level ? 0.75 : 0.22;   /* fast attack, slow release */
    P.level += (target - P.level) * k;
    set_step((int)nearbyint(P.level * (LEVELS - 1)));
    return G_SOURCE_CONTINUE;
}

static void update_level_source(void) {
    gboolean want = !g_strcmp0(P.status, "Playing") && !P.open && gtk_widget_get_visible(P.win);
    if (want && P.parec <= 0) {
        char rate[32];
        g_snprintf(rate, sizeof rate, "--rate=%d", RATE);
        const char *argv[] = {"parec", "-d", "@DEFAULT_MONITOR@", "--format=s16le", rate,
                              "--channels=1", "--latency-msec=50", "--raw", NULL};
        if (!spawn_pipe(argv, &P.parec, &P.parec_fd)) {
            P.parec = 0;
            P.parec_fd = -1;
            return;
        }
        P.parec_watch = g_unix_fd_add(P.parec_fd, G_IO_IN | G_IO_HUP | G_IO_ERR, on_sound, NULL);
    } else if (!want && P.parec > 0) {
        if (P.parec_watch) g_source_remove(P.parec_watch);
        P.parec_watch = 0;
        stop_child(&P.parec, &P.parec_fd);
        P.level = 0.0;
        P.step = -1;
        set_step(0);
    }
}

/* -- drag / click on the bubble ------------------------------------------------------ */
static gboolean glide_tick(GtkWidget *w, GdkFrameClock *c, gpointer _) {
    if (!P.gliding) return G_SOURCE_REMOVE;
    double t = MIN(1.0, (now() - P.g_t0) / 0.34);
    /* ease-out back: arrives with a small settle past the corner */
    double cc = 1.4;
    double e = 1 + (cc + 1) * pow(t - 1, 3) + cc * pow(t - 1, 2);
    gtk_window_move(GTK_WINDOW(P.win), (int)(P.g_sx + (P.g_ex - P.g_sx) * e),
                    (int)(P.g_sy + (P.g_ey - P.g_sy) * e));
    if (t >= 1) {
        P.gliding = FALSE;
        place();
        return G_SOURCE_REMOVE;
    }
    return G_SOURCE_CONTINUE;
}

/* Glide into the nearest corner of the monitor under (x, y). */
static void snap(double x, double y) {
    GdkMonitor *mon = gdk_display_get_monitor_at_point(gdk_display_get_default(), (int)x, (int)y);
    GdkRectangle g;
    gdk_monitor_get_geometry(mon, &g);
    char corner[3] = {y < g.y + g.height / 2.0 ? 't' : 'b', x < g.x + g.width / 2.0 ? 'l' : 'r', 0};
    const char *model = gdk_monitor_get_model(mon);
    set_str(&P.dock_model, model ? model : "");
    set_str(&P.dock_corner, corner);
    save_dock();
    P.mon = monitor();
    P.have_mon = TRUE;
    int w, h;
    size(&w, &h);
    gtk_window_get_position(GTK_WINDOW(P.win), &P.g_sx, &P.g_sy);
    target(w, h, &P.g_ex, &P.g_ey);
    P.g_t0 = now();
    P.gliding = TRUE;
    gtk_widget_add_tick_callback(P.win, glide_tick, NULL, NULL);
}

static gboolean on_press(GtkWidget *_w, GdkEventButton *ev, gpointer _) {
    if (ev->button == 3) {
        ctl("play-pause");
        return TRUE;
    }
    if (ev->button == 1) {
        gtk_window_get_position(GTK_WINDOW(P.win), &P.press_wx, &P.press_wy);
        P.press_x = ev->x_root;
        P.press_y = ev->y_root;
        P.pressed = TRUE;
        P.dragging = FALSE;
    }
    return TRUE;
}

static gboolean on_motion(GtkWidget *_w, GdkEventMotion *ev, gpointer _) {
    if (!P.pressed || P.open) return TRUE;
    double dx = ev->x_root - P.press_x, dy = ev->y_root - P.press_y;
    if (!P.dragging && hypot(dx, dy) < DRAG_START) return TRUE;
    P.dragging = TRUE;
    P.gliding = FALSE;
    gtk_window_move(GTK_WINDOW(P.win), (int)(P.press_wx + dx), (int)(P.press_wy + dy));
    return TRUE;
}

static gboolean on_release(GtkWidget *_w, GdkEventButton *ev, gpointer _) {
    if (ev->button != 1 || !P.pressed) return TRUE;
    P.pressed = FALSE;
    if (P.dragging) {
        P.dragging = FALSE;
        snap(ev->x_root, ev->y_root);
    } else {
        set_open(!P.open);
    }
    return TRUE;
}

/* -- open / fold ---------------------------------------------------------------------- */
static gboolean follow_size(GtkWidget *w, GdkFrameClock *c, gpointer _) {
    place();
    return now() < P.anim_until ? G_SOURCE_CONTINUE : G_SOURCE_REMOVE;
}

static gboolean grab_pointer(gpointer _) {
    GdkWindow *gw = gtk_widget_get_window(P.win);
    if (P.open && gw)
        gdk_seat_grab(gdk_display_get_default_seat(gdk_display_get_default()), gw,
                      GDK_SEAT_CAPABILITY_ALL_POINTING, TRUE, NULL, NULL, NULL, NULL);
    return G_SOURCE_REMOVE;
}

static void set_open(gboolean value) {
    if (value == P.open) return;
    P.open = value;
    if (value) show_time();
    gtk_stack_set_visible_child_name(GTK_STACK(P.stack), value ? "card" : "bubble");
    update_level_source();
    /* Follow the stack's size every frame, anchored at the dock corner. */
    P.anim_until = now() + OPEN_MS / 1000.0 + 0.12;
    gtk_widget_add_tick_callback(P.win, follow_size, NULL, NULL);
    if (value) {
        /* A click anywhere else folds the card: hold the pointer (the click
         * outside is then passed on to what was under it). */
        g_timeout_add(30, grab_pointer, NULL);
    } else {
        ungrab();
    }
}

static gboolean pass_click(gpointer data) {
    char b[8];
    g_snprintf(b, sizeof b, "%d", GPOINTER_TO_INT(data));
    const char *argv[] = {"xdotool", "click", b, NULL};
    spawn_quiet(argv);
    return G_SOURCE_REMOVE;
}

static gboolean on_any_press(GtkWidget *_w, GdkEventButton *ev, gpointer _) {
    if (!P.open) return FALSE;
    int x, y, w, h;
    gtk_window_get_position(GTK_WINDOW(P.win), &x, &y);
    gtk_window_get_size(GTK_WINDOW(P.win), &w, &h);
    if (x <= ev->x_root && ev->x_root < x + w && y <= ev->y_root && ev->y_root < y + h) return FALSE;
    guint button = ev->button;
    set_open(FALSE);
    /* Pass the click on, once the pointer is free. */
    g_timeout_add(40, pass_click, GINT_TO_POINTER(button));
    return TRUE;
}

static gboolean on_seek_press(GtkWidget *_w, GdkEvent *_e, gpointer _) {
    P.seeking = TRUE;
    return FALSE;
}

static gboolean on_seek_release(GtkWidget *_w, GdkEvent *_e, gpointer _) {
    P.seeking = FALSE;
    double v = gtk_range_get_value(GTK_RANGE(P.seek));
    char s[G_ASCII_DTOSTR_BUF_SIZE];
    g_ascii_formatd(s, sizeof s, "%.1f", v);
    const char *argv[] = {"playerctl", "position", s, NULL};
    spawn_quiet(argv);
    P.pos_ref = v;
    P.t_ref = now();
    return FALSE;
}

static gboolean on_cover(GtkWidget *_w, GdkEventButton *ev, gpointer _) {
    if (ev->button == 1) set_open(FALSE);
    return TRUE;
}

static gboolean on_button(GtkWidget *_w, GdkEventButton *ev, gpointer action) {
    if (ev->button == 1) ctl(action);
    return TRUE;
}

/* -- life ---------------------------------------------------------------------------- */
static void show(void) {
    P.mon = monitor();
    P.have_mon = TRUE;
    place();
    gtk_widget_show(P.win);
    gdk_window_raise(gtk_widget_get_window(P.win));
    update_level_source();
}

static gboolean quit(gpointer _) {
    if (P.parec > 0) kill(P.parec, SIGKILL);
    if (P.pc > 0) kill(P.pc, SIGKILL);
    ungrab();
    gtk_main_quit();
    return G_SOURCE_REMOVE;
}

/* -- widgets ------------------------------------------------------------------------- */
static void add_class(GtkWidget *w, const char *c) {
    gtk_style_context_add_class(gtk_widget_get_style_context(w), c);
}

static GtkWidget *label(const char *cls) {
    GtkWidget *lb = gtk_label_new("");
    gtk_label_set_xalign(GTK_LABEL(lb), 0);
    gtk_label_set_ellipsize(GTK_LABEL(lb), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(lb), 1);
    add_class(lb, cls);
    return lb;
}

static GtkWidget *button(const char *glyph, const char *action, const char *cls, GtkWidget **lab) {
    GtkWidget *b = gtk_label_new(glyph);
    add_class(b, cls);
    GtkWidget *ev = gtk_event_box_new();
    gtk_container_add(GTK_CONTAINER(ev), b);
    g_signal_connect(ev, "button-press-event", G_CALLBACK(on_button), (gpointer)action);
    if (lab) *lab = b;
    return ev;
}

static void build(void) {
    palette_load(&P.pal);
    hexrgb(P.pal.fg, P.fg);
    P.peak = 1500.0;
    P.step = P.ring_step = -1;
    P.parec_fd = P.pc_fd = -1;
    P.t_ref = now();
    load_dock();

    P.win = overlay_window("MediaPill", 0, 0, HOLD, HOLD);
    gtk_widget_set_size_request(P.win, -1, -1);
    gtk_window_set_resizable(GTK_WINDOW(P.win), TRUE);
    char *c = css(&P.pal);
    css_load(c);
    g_free(c);

    /* The bubble: ring and disc stacked, centred in a square. The disc: one
     * Image per level in a Stack; a level change only flips the visible
     * child. (Swapping the pixbuf of one Image every frame makes GTK build a
     * new surface each time: CPU, and a heap that keeps growing.) */
    P.ring = gtk_image_new();
    P.disc = gtk_stack_new();
    add_class(P.disc, "disc");
    P.hold = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_size_request(P.hold, HOLD, HOLD);
    GtkWidget *over = gtk_overlay_new();
    gtk_container_add(GTK_CONTAINER(over), P.hold);
    GtkWidget *layers[2] = {P.ring, P.disc};
    for (int i = 0; i < 2; i++) {
        gtk_widget_set_halign(layers[i], GTK_ALIGN_CENTER);
        gtk_widget_set_valign(layers[i], GTK_ALIGN_CENTER);
        gtk_overlay_add_overlay(GTK_OVERLAY(over), layers[i]);
        gtk_overlay_set_overlay_pass_through(GTK_OVERLAY(over), layers[i], TRUE);
    }
    P.bub = gtk_event_box_new();
    gtk_container_add(GTK_CONTAINER(P.bub), over);
    gtk_widget_add_events(P.bub, GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK | GDK_BUTTON1_MOTION_MASK);
    g_signal_connect(P.bub, "button-press-event", G_CALLBACK(on_press), NULL);
    g_signal_connect(P.bub, "motion-notify-event", G_CALLBACK(on_motion), NULL);
    g_signal_connect(P.bub, "button-release-event", G_CALLBACK(on_release), NULL);

    /* The card the bubble grows into (from its corner): the cover and the
     * title / artist beside it, a full-width play bar with the elapsed and
     * total time under its ends, then prev / play / next. */
    P.cover = gtk_image_new();
    GtkWidget *cov = gtk_event_box_new();
    gtk_widget_set_valign(cov, GTK_ALIGN_CENTER);
    gtk_container_add(GTK_CONTAINER(cov), P.cover);
    g_signal_connect(cov, "button-press-event", G_CALLBACK(on_cover), NULL);
    P.l_title = label("title");
    gtk_label_set_line_wrap(GTK_LABEL(P.l_title), TRUE);   /* up to two lines, then "…" */
    gtk_label_set_lines(GTK_LABEL(P.l_title), 2);
    P.l_artist = label("artist");
    GtkWidget *names = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    gtk_widget_set_valign(names, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(names), P.l_title, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(names), P.l_artist, FALSE, FALSE, 0);
    GtkWidget *head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 14);
    gtk_box_pack_start(GTK_BOX(head), cov, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(head), names, TRUE, TRUE, 0);

    P.seek = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0, 1, 1);
    gtk_scale_set_draw_value(GTK_SCALE(P.seek), FALSE);
    g_signal_connect(P.seek, "button-press-event", G_CALLBACK(on_seek_press), NULL);
    g_signal_connect(P.seek, "button-release-event", G_CALLBACK(on_seek_release), NULL);
    P.l_pos = gtk_label_new("0:00");
    gtk_label_set_xalign(GTK_LABEL(P.l_pos), 0);
    P.l_len = gtk_label_new("");
    gtk_label_set_xalign(GTK_LABEL(P.l_len), 1);
    GtkWidget *times = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    add_class(P.l_pos, "time");
    add_class(P.l_len, "time");
    gtk_box_pack_start(GTK_BOX(times), P.l_pos, FALSE, FALSE, 0);
    gtk_box_pack_end(GTK_BOX(times), P.l_len, FALSE, FALSE, 0);

    P.b_play = button(G_PAUSE, "play-pause", "play", &P.b_play_label);
    GtkWidget *ctlbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 26);
    gtk_widget_set_halign(ctlbox, GTK_ALIGN_CENTER);
    GtkWidget *btns[3] = {button(G_PREV, "previous", "ctl", NULL), P.b_play,
                          button(G_NEXT, "next", "ctl", NULL)};
    for (int i = 0; i < 3; i++) {
        gtk_box_pack_start(GTK_BOX(ctlbox), btns[i], FALSE, FALSE, 0);
        gtk_widget_set_valign(btns[i], GTK_ALIGN_CENTER);
    }

    GtkWidget *card = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    add_class(card, "card");
    gtk_widget_set_size_request(card, CARD_W, -1);
    gtk_box_pack_start(GTK_BOX(card), head, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(card), P.seek, FALSE, FALSE, 14);
    gtk_box_pack_start(GTK_BOX(card), times, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(card), ctlbox, FALSE, FALSE, 6);

    /* Bubble <-> card: a cross-fade while the size slides between the two;
     * the window follows it every frame, pinned to its corner. */
    P.stack = gtk_stack_new();
    gtk_stack_set_hhomogeneous(GTK_STACK(P.stack), FALSE);
    gtk_stack_set_vhomogeneous(GTK_STACK(P.stack), FALSE);
    gtk_stack_set_interpolate_size(GTK_STACK(P.stack), TRUE);
    gtk_stack_set_transition_type(GTK_STACK(P.stack), GTK_STACK_TRANSITION_TYPE_CROSSFADE);
    gtk_stack_set_transition_duration(GTK_STACK(P.stack), OPEN_MS);
    gtk_stack_add_named(GTK_STACK(P.stack), P.bub, "bubble");
    gtk_stack_add_named(GTK_STACK(P.stack), card, "card");
    P.shell = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    add_class(P.shell, "shell");
    gtk_box_pack_start(GTK_BOX(P.shell), P.stack, TRUE, TRUE, 0);
    gtk_container_add(GTK_CONTAINER(P.win), P.shell);
    g_signal_connect(P.win, "button-press-event", G_CALLBACK(on_any_press), NULL);
    gtk_widget_show_all(P.shell);
    gtk_stack_set_visible_child_name(GTK_STACK(P.stack), "bubble");

    ring_frames();
    follow();
}

int main(int argc, char **argv) {
    if (!g_getenv("DISPLAY") || !*g_getenv("DISPLAY")) return 1;
    gtk_env_lean();
    gtk_init(&argc, &argv);
    signal(SIGPIPE, SIG_IGN);
    /* bin/MediaPill checks this to run one UI at a time. */
    const char *rt = g_getenv("XDG_RUNTIME_DIR");
    char *pidf = g_build_filename(rt ? rt : "/tmp", "media-pill.pid", NULL);
    char *pid = g_strdup_printf("%d\n", (int)getpid());
    g_file_set_contents(pidf, pid, -1, NULL);
    g_free(pid);
    g_free(pidf);
    const char *xc = g_getenv("XDG_CACHE_HOME");
    P.cache = xc && *xc ? g_build_filename(xc, "misa-media", NULL)
                        : g_build_filename(g_get_home_dir(), ".cache", "misa-media", NULL);
    P.dock_file = g_build_filename(P.cache, "dock", NULL);
    build();
    gtk_main();
    return 0;
}
