/* misa-ui common: the rice's palette (from rofi's shared.rasi), override-
 * redirect windows that grab like rofi (bspwm and the Scroll layout never
 * see them), the popup backdrop, and rounded / circular pictures. */
#include "common.h"

#include <fcntl.h>
#include <malloc.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static char *read_file(const char *path) {
    char *s = NULL;
    g_file_get_contents(path, &s, NULL, NULL);
    return s;
}

/* `key: "#rrggbb"` / `key: "Font Name 10"` lines of shared.rasi. */
static gboolean rasi_value(const char *text, const char *key, char *out, size_t n) {
    char pat[96];
    g_snprintf(pat, sizeof pat, "(?m)^\\s*%s:\\s*\"?([^\";]+?)\"?\\s*;", key);
    GRegex *re = g_regex_new(pat, 0, 0, NULL);
    GMatchInfo *m = NULL;
    gboolean ok = FALSE;
    if (re && g_regex_match(re, text, 0, &m)) {
        char *v = g_match_info_fetch(m, 1);
        g_strlcpy(out, v, n);
        g_free(v);
        ok = TRUE;
    }
    if (m) g_match_info_free(m);
    if (re) g_regex_unref(re);
    return ok;
}

static void strip_font(char *f) {
    /* "Inter Medium 10" -> "Inter"; "Inter Black 9" -> "Inter Black" */
    char *sp = strrchr(f, ' ');
    if (sp && g_ascii_isdigit(sp[1])) *sp = 0;
}

char *mira_config_path(const char *rel) {
    const char *env = g_getenv("MIRA_CONFIG_DIR");
    if (env && *env) return g_build_filename(env, rel, NULL);
    return g_build_filename(g_get_user_config_dir(), "bspwm", rel, NULL);
}

/* Disable unused GTK3 GLX/Mesa, AT-SPI accessibility bridge, and GVFS modules
 * before gtk_init() so pure-Cairo overlays save ~15-20 MiB PSS & avoid GL init. */
void gtk_env_lean(void) {
    g_setenv("GDK_GL", "disable", FALSE);
    g_setenv("NO_AT_BRIDGE", "1", FALSE);
    g_setenv("GIO_USE_VFS", "local", FALSE);
}

void palette_load(Palette *p) {
    strcpy(p->bg, "#101010");
    strcpy(p->fg, "#e6e6e6");
    strcpy(p->font, "Inter");
    strcpy(p->display, "Inter Black");
    char *path = mira_config_path("config/rofi/shared.rasi");
    char *text = read_file(path);
    g_free(path);
    if (!text) return;
    char v[160];
    if (rasi_value(text, "background", v, sizeof v) && v[0] == '#' && strlen(v) >= 7) {
        g_strlcpy(p->bg, v, 8);
    }
    if (rasi_value(text, "foreground", v, sizeof v) && v[0] == '#' && strlen(v) >= 7) {
        g_strlcpy(p->fg, v, 8);
    }
    if (rasi_value(text, "font", v, sizeof v)) {
        strip_font(v);
        /* the family only: drop a trailing weight word */
        static const char *weights[] = {" Medium", " Regular", " Bold", " SemiBold", NULL};
        for (int i = 0; weights[i]; i++) {
            size_t lv = strlen(v), lw = strlen(weights[i]);
            if (lv > lw && strcmp(v + lv - lw, weights[i]) == 0) { v[lv - lw] = 0; break; }
        }
        g_strlcpy(p->font, v, sizeof p->font);
    }
    if (rasi_value(text, "font-key", v, sizeof v)) {
        strip_font(v);
        g_strlcpy(p->display, v, sizeof p->display);
    }
    g_free(text);
}

void css_load(const char *css) {
    GtkCssProvider *pr = gtk_css_provider_new();
    gtk_css_provider_load_from_data(pr, css, -1, NULL);
    gtk_style_context_add_provider_for_screen(gdk_screen_get_default(), GTK_STYLE_PROVIDER(pr),
                                              GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(pr);
}

GtkWidget *overlay_window(const char *wmclass, int x, int y, int w, int h) {
    GtkWidget *win = gtk_window_new(GTK_WINDOW_POPUP);
    gtk_window_set_title(GTK_WINDOW(win), wmclass);
    char *lower = g_ascii_strdown(wmclass, -1);
    G_GNUC_BEGIN_IGNORE_DEPRECATIONS
    gtk_window_set_wmclass(GTK_WINDOW(win), lower, wmclass);   /* picom matches the class */
    G_GNUC_END_IGNORE_DEPRECATIONS
    g_free(lower);
    GdkVisual *vis = gdk_screen_get_rgba_visual(gtk_widget_get_screen(win));
    if (vis) gtk_widget_set_visual(win, vis);
    gtk_widget_set_app_paintable(win, FALSE);
    gtk_window_move(GTK_WINDOW(win), x, y);
    gtk_widget_set_size_request(win, w, h);
    gtk_window_set_default_size(GTK_WINDOW(win), w, h);
    return win;
}

char *run_out(const char *const *argv) {
    char *out = NULL;
    int status = 0;
    if (!g_spawn_sync(NULL, (char **)argv, NULL, G_SPAWN_SEARCH_PATH | G_SPAWN_STDERR_TO_DEV_NULL,
                      NULL, NULL, &out, NULL, &status, NULL)) {
        return NULL;
    }
    return out;
}

/* Direct zero-fork BSPWM IPC over $BSPWM_SOCKET (0.08 ms vs ~15 ms bspc fork). */
static char *bspwm_ipc(const char *const *argv) {
    char sock_path[108];
    const char *env_sock = g_getenv("BSPWM_SOCKET");
    if (env_sock && *env_sock) {
        g_strlcpy(sock_path, env_sock, sizeof sock_path);
    } else {
        const char *disp = g_getenv("DISPLAY");
        int dnum = 0, snum = 0;
        if (disp) {
            const char *col = strchr(disp, ':');
            if (col) {
                dnum = atoi(col + 1);
                const char *dot = strchr(col + 1, '.');
                if (dot) snum = atoi(dot + 1);
            }
        }
        g_snprintf(sock_path, sizeof sock_path, "/tmp/bspwm_0_%d_%d-socket", dnum, snum);
        if (!g_file_test(sock_path, G_FILE_TEST_EXISTS))
            g_snprintf(sock_path, sizeof sock_path, "/tmp/bspwm_%d_%d-socket", dnum, snum);
    }
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return NULL;
    struct sockaddr_un sa;
    memset(&sa, 0, sizeof sa);
    sa.sun_family = AF_UNIX;
    g_strlcpy(sa.sun_path, sock_path, sizeof sa.sun_path);
    if (connect(fd, (struct sockaddr *)&sa, sizeof sa) < 0) {
        close(fd);
        return NULL;
    }
    GString *req = g_string_new("");
    for (int i = 0; argv[i]; i++) {
        g_string_append(req, argv[i]);
        g_string_append_c(req, '\0');
    }
    ssize_t wr = write(fd, req->str, req->len);
    g_string_free(req, TRUE);
    if (wr < 0) { close(fd); return NULL; }
    GString *resp = g_string_sized_new(2048);
    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof buf)) > 0) g_string_append_len(resp, buf, n);
    close(fd);
    if (resp->len == 0 || resp->str[0] == '\x07') {
        g_string_free(resp, TRUE);
        return NULL;
    }
    return g_string_free(resp, FALSE);
}

static gboolean parse_rect_and_pad(const char *json, Rect *r, int *top_pad, int *bot_pad) {
    if (!json) return FALSE;
    const char *rp = strstr(json, "\"rectangle\":{");
    if (!rp) return FALSE;
    int x = 0, y = 0, w = 0, h = 0;
    if (sscanf(rp, "\"rectangle\":{\"x\":%d,\"y\":%d,\"width\":%d,\"height\":%d}", &x, &y, &w, &h) != 4)
        return FALSE;
    r->x = x; r->y = y; r->w = w; r->h = h;
    if (top_pad && bot_pad) {
        *top_pad = 0; *bot_pad = 0;
        const char *pp = strstr(json, "\"padding\":{");
        if (pp && pp < rp) {
            int t = 0, ri = 0, b = 0, l = 0;
            if (sscanf(pp, "\"padding\":{\"top\":%d,\"right\":%d,\"bottom\":%d,\"left\":%d}", &t, &ri, &b, &l) == 4) {
                *top_pad = t; *bot_pad = b;
            }
        }
    }
    return TRUE;
}

static gboolean cached_mon_ok = FALSE;
static Rect cached_mon_rect;
static int cached_top_pad = 0, cached_bot_pad = 0;

gboolean monitor_rect(const char *name, Rect *r) {
    const char *ipc_argv[] = {"query", "-T", "-m", name, NULL};
    char *out = bspwm_ipc(ipc_argv);
    if (!out) {
        const char *argv[] = {"bspc", "query", "-T", "-m", name, NULL};
        out = run_out(argv);
    }
    int tp = 0, bp = 0;
    gboolean ok = parse_rect_and_pad(out, r, &tp, &bp);
    if (ok && strcmp(name, "focused") == 0) {
        cached_mon_ok = TRUE;
        cached_mon_rect = *r;
        cached_top_pad = tp;
        cached_bot_pad = bp;
    }
    g_free(out);
    return ok;
}

gboolean bar_anchor(gboolean pointer, Rect *mon, char edge[8], int *off) {
    *mon = (Rect){0, 0, 1920, 1080};
    g_strlcpy(edge, "top", 8);
    *off = 0;

    const char *sel = pointer ? "pointed" : "focused";
    const char *ipc1[] = {"query", "-T", "-m", sel, NULL};
    char *out = bspwm_ipc(ipc1);
    if (!out && pointer) {
        const char *ipc2[] = {"query", "-T", "-m", "focused", NULL};
        out = bspwm_ipc(ipc2);
    }
    int top = 0, bot = 0;
    gboolean ok = parse_rect_and_pad(out, mon, &top, &bot);
    g_free(out);
    if (ok) {
        if (bot > top) {
            g_strlcpy(edge, "bottom", 8);
            *off = bot;
        } else {
            g_strlcpy(edge, "top", 8);
            *off = top;
        }
        cached_mon_ok = TRUE;
        cached_mon_rect = *mon;
        cached_top_pad = top;
        cached_bot_pad = bot;
        return TRUE;
    }
    *mon = focused_monitor();
    return FALSE;
}

Rect below_bar(Rect r) {
    if (cached_mon_ok && cached_mon_rect.x == r.x && cached_mon_rect.y == r.y
        && cached_mon_rect.w == r.w && cached_mon_rect.h == r.h) {
        int top = cached_top_pad >= BAR_MIN ? cached_top_pad : 0;
        int bottom = cached_bot_pad >= BAR_MIN ? cached_bot_pad : 0;
        r.y += top;
        r.h -= top + bottom;
        return r;
    }
    const char *ipc_argv[] = {"wm", "-d", NULL};
    char *out = bspwm_ipc(ipc_argv);
    if (!out) {
        const char *argv[] = {"bspc", "wm", "-d", NULL};
        out = run_out(argv);
    }
    if (!out) return r;
    GRegex *re = g_regex_new("\"padding\":\\{\"top\":(\\d+),\"right\":\\d+,\"bottom\":(\\d+),\"left\":\\d+\\},"
                             "\"rectangle\":\\{\"x\":(-?\\d+),\"y\":(-?\\d+),\"width\":(\\d+),\"height\":(\\d+)\\}",
                             0, 0, NULL);
    GMatchInfo *m = NULL;
    g_regex_match(re, out, 0, &m);
    while (g_match_info_matches(m)) {
        int v[6];
        for (int i = 0; i < 6; i++) {
            char *s = g_match_info_fetch(m, i + 1);
            v[i] = atoi(s);
            g_free(s);
        }
        if (v[2] == r.x && v[3] == r.y && v[4] == r.w && v[5] == r.h) {
            /* only a bar's strut: the 1 px window paddings stay covered */
            int top = v[0] >= BAR_MIN ? v[0] : 0, bottom = v[1] >= BAR_MIN ? v[1] : 0;
            r.y += top;
            r.h -= top + bottom;
            break;
        }
        g_match_info_next(m, NULL);
    }
    g_match_info_free(m);
    g_regex_unref(re);
    g_free(out);
    return r;
}

Rect focused_monitor(void) {
    Rect r = {0, 0, 1920, 1080};
    if (monitor_rect("focused", &r)) return r;
    GdkDisplay *d = gdk_display_get_default();
    int x = 0, y = 0;
    gdk_device_get_position(gdk_seat_get_pointer(gdk_display_get_default_seat(d)), NULL, &x, &y);
    GdkRectangle g;
    gdk_monitor_get_geometry(gdk_display_get_monitor_at_point(d, x, y), &g);
    r.x = g.x; r.y = g.y; r.w = g.width; r.h = g.height;
    return r;
}

/* -- grab ------------------------------------------------------------------- */
typedef struct {
    GtkWidget *win;
    GCallback on_fail;
    gpointer data;
    int left;
} Grab;

static gboolean grab_try(gpointer p) {
    Grab *g = p;
    GdkWindow *gw = gtk_widget_get_window(g->win);
    GdkSeat *seat = gdk_display_get_default_seat(gdk_display_get_default());
    if (gw && gdk_seat_grab(seat, gw, GDK_SEAT_CAPABILITY_ALL, TRUE, NULL, NULL, NULL, NULL)
                  == GDK_GRAB_SUCCESS) {
        g_free(g);
        return G_SOURCE_REMOVE;
    }
    if (--g->left <= 0) {
        if (g->on_fail) ((void (*)(gpointer))g->on_fail)(g->data);
        g_free(g);
        return G_SOURCE_REMOVE;
    }
    return G_SOURCE_CONTINUE;
}

void grab_soon(GtkWidget *win, GCallback on_fail, gpointer data) {
    Grab *g = g_new0(Grab, 1);
    g->win = win;
    g->on_fail = on_fail;
    g->data = data;
    g->left = 100;
    if (grab_try(g)) g_timeout_add(5, grab_try, g);
}

gboolean grab(GtkWidget *win) {
    GdkSeat *seat = gdk_display_get_default_seat(gdk_display_get_default());
    GdkWindow *gw = gtk_widget_get_window(win);
    for (int i = 0; i < 40; i++) {
        if (gw && gdk_seat_grab(seat, gw, GDK_SEAT_CAPABILITY_ALL, TRUE, NULL, NULL, NULL, NULL)
                      == GDK_GRAB_SUCCESS)
            return TRUE;
        g_usleep(10000);
    }
    return FALSE;
}

void ungrab(void) {
    gdk_seat_ungrab(gdk_display_get_default_seat(gdk_display_get_default()));
}

/* -- backdrop ---------------------------------------------------------------- */
const char *self_dir(void) {
    static char dir[4096];
    if (!dir[0]) {
        ssize_t n = readlink("/proc/self/exe", dir, sizeof dir - 1);
        if (n > 0) {
            dir[n] = 0;
            char *s = strrchr(dir, '/');
            if (s) *s = 0;
        } else {
            strcpy(dir, ".");
        }
    }
    return dir;
}

int backdrop(Rect r, int pid, const char *click) {
    char *bin = g_build_filename(self_dir(), "misa-backdrop", NULL);
    char a[4][16], spid[16];
    g_snprintf(a[0], 16, "%d", r.x);
    g_snprintf(a[1], 16, "%d", r.y);
    g_snprintf(a[2], 16, "%d", r.w);
    g_snprintf(a[3], 16, "%d", r.h);
    g_snprintf(spid, 16, "%d", pid ? pid : (int)getpid());
    const char *argv[10] = {bin, a[0], a[1], a[2], a[3], "--pid", spid, NULL, NULL, NULL};
    if (click) { argv[7] = "--click"; argv[8] = click; }
    char *out = NULL;
    int got = 0;
    if (g_file_test(bin, G_FILE_TEST_IS_EXECUTABLE) &&
        g_spawn_sync(NULL, (char **)argv, NULL, G_SPAWN_STDERR_TO_DEV_NULL, NULL, NULL, &out, NULL,
                     NULL, NULL) && out) {
        got = atoi(out);
    }
    g_free(out);
    g_free(bin);
    return got;
}

void drop_backdrop(int pid) {
    if (pid > 0) kill(pid, SIGTERM);
}

static gboolean fifo_close(const char *run, const char *name) {
    char *path = g_build_filename(run, name, NULL);
    int fd = open(path, O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    g_free(path);
    if (fd < 0) return FALSE;
    if (write(fd, "close\n", 6) < 0) { /* ignore */ }
    close(fd);
    return TRUE;
}

static int kill_pidfile(const char *run, const char *name, const char *expect_comm) {
    char *path = g_build_filename(run, name, NULL);
    char *txt = read_file(path);
    g_free(path);
    if (!txt) return 0;
    int pid = atoi(txt);
    g_free(txt);
    if (pid <= 1 || pid == (int)getpid()) return 0;
    if (expect_comm) {
        char cpath[64];
        g_snprintf(cpath, sizeof cpath, "/proc/%d/comm", pid);
        char *comm = read_file(cpath);
        gboolean ok = comm && strcmp(g_strstrip(comm), expect_comm) == 0;
        g_free(comm);
        if (!ok) return 0;
    }
    return kill(pid, SIGTERM) == 0 ? pid : 0;
}

void close_bar_popups(const char *except) {
    const char *rt = g_getenv("XDG_RUNTIME_DIR");
    const char *run = (rt && *rt) ? rt : "/tmp";

    if (g_strcmp0(except, "clip") != 0)  fifo_close(run, "clip-panel.fifo");
    if (g_strcmp0(except, "notif") != 0) fifo_close(run, "notif-panel.fifo");

    int killed[6];
    int nk = 0;
    int p;
    if (g_strcmp0(except, "wifi") != 0 && !fifo_close(run, "wifi-panel.fifo")) {
        if ((p = kill_pidfile(run, "wifi-panel.pid", "WifiPanel")) > 0)
            killed[nk++] = p;
    }
    if (g_strcmp0(except, "bt") != 0 && (p = kill_pidfile(run, "bt-panel.pid", "BluetoothPanel")) > 0)
        killed[nk++] = p;
    if (g_strcmp0(except, "sound") != 0 && (p = kill_pidfile(run, "sound-panel.pid", "SoundPanel")) > 0)
        killed[nk++] = p;
    if (g_strcmp0(except, "music") != 0 && (p = kill_pidfile(run, "music-panel.pid", "MusicPanel")) > 0)
        killed[nk++] = p;
    if (g_strcmp0(except, "rofi") != 0 && (p = kill_pidfile(run, "rofi.pid", "rofi")) > 0)
        killed[nk++] = p;

    /* Wait briefly for killed grab-holding popups to exit so their X11 grab
     * is released before the caller maps its own popup. */
    for (int i = 0; nk > 0 && i < 20; i++) {
        gboolean alive = FALSE;
        for (int j = 0; j < nk; j++) {
            if (kill(killed[j], 0) == 0) alive = TRUE;
        }
        if (!alive) break;
        g_usleep(5000);
    }
}

/* -- pictures ----------------------------------------------------------------
 * Same arithmetic as the Python tools had (misa_ui.py), so the pictures come
 * out pixel for pixel the same: bilinear scale, centre crop, corners cut with
 * a per-pixel antialiased alpha. */
static GdkPixbuf *rgba_copy(GdkPixbuf *pb) {
    if (gdk_pixbuf_get_has_alpha(pb)) return gdk_pixbuf_copy(pb);
    return gdk_pixbuf_add_alpha(pb, FALSE, 0, 0, 0);
}

GdkPixbuf *round_corners(GdkPixbuf *src, int r, gboolean top_only) {
    GdkPixbuf *pb = rgba_copy(src);
    int w = gdk_pixbuf_get_width(pb), h = gdk_pixbuf_get_height(pb);
    int rs = gdk_pixbuf_get_rowstride(pb);
    guchar *data = gdk_pixbuf_get_pixels(pb);
    for (int y = 0; y < MIN(r, h); y++)
        for (int x = 0; x < MIN(r, w); x++) {
            double d = hypot(r - x - 0.5, r - y - 0.5) - r;      /* >0: outside */
            double a = d <= -0.5 ? 1.0 : d >= 0.5 ? 0.0 : 0.5 - d;
            if (a >= 1.0) continue;
            for (int bottom = 0; bottom <= (top_only ? 0 : 1); bottom++) {
                int yy = bottom ? h - 1 - y : y;
                int xs[2] = {x, w - 1 - x};
                for (int k = 0; k < 2; k++) {
                    guchar *px = data + yy * rs + xs[k] * 4 + 3;
                    *px = (guchar)(*px * a);
                }
            }
        }
    return pb;
}

/* src scaled to cover w x h, cropped to it (centre), corners rounded. */
GdkPixbuf *cover_pixbuf(GdkPixbuf *src, int w, int h, int r, gboolean top_only) {
    int pw = gdk_pixbuf_get_width(src), ph = gdk_pixbuf_get_height(src);
    double s = MAX((double)w / pw, (double)h / ph);
    int sw = MAX(w, (int)ceil(pw * s)), sh = MAX(h, (int)ceil(ph * s));
    GdkPixbuf *big = gdk_pixbuf_scale_simple(src, sw, sh, GDK_INTERP_BILINEAR);
    GdkPixbuf *sub = gdk_pixbuf_new_subpixbuf(big, (sw - w) / 2, (sh - h) / 2, w, h);
    GdkPixbuf *out = r ? round_corners(sub, r, top_only) : gdk_pixbuf_copy(sub);
    g_object_unref(sub);
    g_object_unref(big);
    return out;
}

/* path decoded no bigger than it takes to cover w x h: the size comes from
 * the header first, and the loader scales while decoding (a 4K wallpaper
 * shown 300 px wide never sits in memory at full size). */
GdkPixbuf *load_cover_size(const char *path, int w, int h) {
    int sw = 0, sh = 0;
    if (gdk_pixbuf_get_file_info(path, &sw, &sh) && sw > 0 && sh > 0) {
        double s = MAX((double)w / sw, (double)h / sh);
        if (s < 1.0)
            return gdk_pixbuf_new_from_file_at_scale(path, MAX(w, (int)ceil(sw * s)), MAX(h, (int)ceil(sh * s)),
                                                     FALSE, NULL);
    }
    return gdk_pixbuf_new_from_file(path, NULL);
}

GdkPixbuf *cover_file(const char *path, int w, int h, int r, gboolean top_only) {
    GdkPixbuf *src = load_cover_size(path, w, h);
    if (!src) return NULL;
    GdkPixbuf *out = cover_pixbuf(src, w, h, r, top_only);
    g_object_unref(src);
    return out;
}

/* src covered to d x d and cut to a circle (antialiased edge). */
GdkPixbuf *circle_pixbuf(GdkPixbuf *src, int d) {
    GdkPixbuf *cv = cover_pixbuf(src, d, d, 0, FALSE);
    GdkPixbuf *pb = rgba_copy(cv);
    g_object_unref(cv);
    int rs = gdk_pixbuf_get_rowstride(pb);
    guchar *data = gdk_pixbuf_get_pixels(pb);
    double r = d / 2.0;
    for (int y = 0; y < d; y++) {
        double dy = y + 0.5 - r;
        for (int x = 0; x < d; x++) {
            double e = hypot(x + 0.5 - r, dy) - r + 0.5;       /* >0: outside */
            if (e <= 0) continue;
            guchar *px = data + y * rs + x * 4 + 3;
            *px = e >= 1 ? 0 : (guchar)(*px * (1 - e));
        }
    }
    return pb;
}

/* -- glide -------------------------------------------------------------------- */
static gboolean glide_tick(GtkWidget *w, GdkFrameClock *clock, gpointer data) {
    Glide *g = data;
    if (!g->on) return G_SOURCE_REMOVE;
    double t = MIN(1.0, (double)(gdk_frame_clock_get_frame_time(clock) - g->t0) / g->dur);
    gtk_adjustment_set_value(g->adj, g->from + (g->to - g->from) * (1 - pow(1 - t, 4)));
    if (t >= 1) {
        g->on = FALSE;
        return G_SOURCE_REMOVE;
    }
    return G_SOURCE_CONTINUE;
}

void glide_init(Glide *g, GtkWidget *widget, GtkAdjustment *adj, int ms) {
    *g = (Glide){.widget = widget, .adj = adj, .dur = (gint64)ms * 1000};
}

void glide_to(Glide *g, double target, gboolean animate) {
    GtkAdjustment *a = g->adj;
    target = MAX(gtk_adjustment_get_lower(a),
                 MIN(target, gtk_adjustment_get_upper(a) - gtk_adjustment_get_page_size(a)));
    GdkFrameClock *clock = gtk_widget_get_frame_clock(g->widget);
    if (!animate || !clock || fabs(target - gtk_adjustment_get_value(a)) < 1) {
        g->on = FALSE;
        gtk_adjustment_set_value(a, target);
        return;
    }
    gboolean first = !g->on;
    g->from = gtk_adjustment_get_value(a);
    g->to = target;
    g->t0 = gdk_frame_clock_get_frame_time(clock);
    g->on = TRUE;
    if (first) gtk_widget_add_tick_callback(g->widget, glide_tick, g, NULL);
}

void heap_trim(void) {
    malloc_trim(0);
}
