/* WifiPanel - the Wi-Fi panel of the bar (bin/WifiPanel [--pointer]).
 *
 * The Wi-Fi screen of a phone (Jakob's law: nothing to learn), quiet and
 * in sentence case. Three pages that slide like a phone's settings:
 *
 *   list    "Wi-Fi" + switch; the networks, one line each (the one in use
 *           first, in bold): the name, then a lock (needs a password), the
 *           signal, and an info button for saved networks. A second line
 *           only while something happens ("Connecting…", "Wrong password").
 *           "Other…" at the end joins a hidden network.
 *   detail  (info button, or →) a saved network: Connect / Disconnect,
 *           Forget This Network (asks once more), Auto-Join, the password
 *           (dots; the eye shows it), Share Password (a QR code a phone can
 *           scan, when `qrencode` is installed), then Signal, Security,
 *           Frequency, Link speed and, when connected, IP address, Router,
 *           DNS.
 *   join    a locked new network (or Other…): the password field with an
 *           eye, Join. Errors stay on the page ("Wrong password").
 *
 * A click on a saved or open network connects at once. Esc / ← / Backspace
 * go back a page; Esc on the list (or a click outside) closes.
 *
 * Everything goes through nmcli, asynchronously, so the panel never
 * freezes. Passwords reach nmcli and qrencode on their standard input,
 * never on a command line, and are wiped from memory when hidden.
 *
 * Placement follows the other bar panels (ClipPanel): under the bar icon
 * with --pointer, else at the bar's right end; black panel, backdrop behind
 * it, keyboard + pointer grab. One instance: running it again while it is
 * open closes it (a second click on the icon). picom: class WifiPanel (the
 * "Bar panels" rule).
 */
#include "common.h"

#include <glib-unix.h>

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#define WIDTH 380
#define PAD 14
#define GAP 6
#define LIST_MAX_H 440
#define PAGE_ANIM_MS 180   /* stack slide + height interpolation */
#define QR_ANIM_MS 250     /* GtkRevealer default */

#define GL_LOCK   "\xf3\xb0\x8c\xbe"   /* U+F033E lock */
#define GL_INFO   "\xf3\xb0\x8b\xbd"   /* U+F02FD information-outline */
#define GL_CHECK  "\xf3\xb0\x84\xac"   /* U+F012C check */
#define GL_EYE    "\xf3\xb0\x88\x88"   /* U+F0208 eye */
#define GL_EYEOFF "\xf3\xb0\x88\x89"   /* U+F0209 eye-off */
#define GL_BACK   "\xf3\xb0\x85\x81"   /* U+F0141 chevron-left */
#define DOTS "\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2"
#define ELLIPSIS "\xe2\x80\xa6"

enum { GROUP_ACTIVE, GROUP_SAVED, GROUP_OTHER };

typedef struct {
    char *ssid, *security, *freq, *rate;
    int signal, group;
    gboolean secure, active, saved, busy;
    char *status;                    /* "Connecting…", an error, or NULL */
    GtkWidget *row, *sub;
} Net;

static struct {
    gboolean daemon, is_open, warm, anchor_cached;
    GtkWidget *win, *box, *stack, *listpage;
    GtkWidget *top_group, *active_list;
    GtkWidget *sw, *scroll, *sections, *off_label;
    GtkWidget *saved_list, *other_list, *other_title, *other_scan;
    GtkWidget *detail, *join;        /* current sub pages (rebuilt) */
    GList *nets;
    GPtrArray *rows;                 /* all GtkListBoxRow* in top-to-bottom order */
    char *sig;                       /* signature of rendered list to skip no-op rebuilds */
    guint refresh_gen;
    gboolean radio, scanning, pointer, closing, syncing, selecting, kb_nav;
    char ifname[64];
    char *page_ssid;                 /* network shown by detail / join */
    gboolean join_hidden;
    /* detail page */
    GtkWidget *d_status, *d_pw, *d_qr, *d_qr_rev;
    char *d_secret;
    gboolean d_confirm;
    /* join page */
    GtkWidget *j_ssid, *j_entry, *j_btn, *j_err;
    Rect mon, below;
    char edge[8];
    int off, px, shade;
} S;

static void refresh(void);
static void refresh_async(void);
static void apply_refresh(const char *out_radio, const char *out_saved, const char *out_list);
static void show_list(void);

/* -- helpers --------------------------------------------------------------- */
static void add_class(GtkWidget *w, const char *c) {
    gtk_style_context_add_class(gtk_widget_get_style_context(w), c);
}

static void set_class(GtkWidget *w, const char *c, gboolean on) {
    GtkStyleContext *sc = gtk_widget_get_style_context(w);
    if (on) gtk_style_context_add_class(sc, c);
    else gtk_style_context_remove_class(sc, c);
}

static GtkWidget *glyph(const char *g, const char *cls) {
    GtkWidget *l = gtk_label_new(g);
    add_class(l, "glyph");
    if (cls) add_class(l, cls);
    return l;
}

static GtkWidget *label(const char *t, const char *cls, float xalign) {
    GtkWidget *l = gtk_label_new(t);
    gtk_label_set_xalign(GTK_LABEL(l), xalign);
    gtk_label_set_ellipsize(GTK_LABEL(l), PANGO_ELLIPSIZE_END);
    if (cls) add_class(l, cls);
    return l;
}

static void wipe(char **s) {
    if (*s) { memset(*s, 0, strlen(*s)); g_free(*s); *s = NULL; }
}

static gboolean showing(GtkWidget *page) {
    return page && gtk_stack_get_visible_child(GTK_STACK(S.stack)) == page;
}

/* nmcli -t: fields split on ':', with "\:" and "\\" escaped. */
static char **terse(const char *line, int n) {
    char **f = g_new0(char *, n + 1);
    GString *cur = g_string_new(NULL);
    int i = 0;
    for (const char *p = line; ; p++) {
        if (*p == '\\' && p[1]) { g_string_append_c(cur, p[1]); p++; continue; }
        if (*p == ':' || *p == 0) {
            if (i < n) f[i++] = g_strdup(cur->str);
            g_string_truncate(cur, 0);
            if (*p == 0) break;
            continue;
        }
        g_string_append_c(cur, *p);
    }
    g_string_free(cur, TRUE);
    for (; i < n; i++) f[i] = g_strdup("");
    return f;
}

typedef void (*Done)(int ok, const char *out, const char *err, char *key);
typedef struct { Done cb; char *key; } Job;

static void job_done(GObject *src, GAsyncResult *res, gpointer data) {
    Job *j = data;
    char *out = NULL, *err = NULL;
    GSubprocess *p = G_SUBPROCESS(src);
    gboolean fine = g_subprocess_communicate_utf8_finish(p, res, &out, &err, NULL);
    int ok = fine && g_subprocess_get_if_exited(p) && g_subprocess_get_exit_status(p) == 0;
    if (j->cb && !S.closing) j->cb(ok, out ? out : "", err ? err : "", j->key);
    g_free(out);
    g_free(err);
    g_free(j->key);
    g_free(j);
    g_object_unref(p);
}

/* Run argv with stdin_text (or nothing) on stdin; cb(ok, out, err, key). */
static void run_async(const char *const *argv, const char *stdin_text, Done cb, const char *key) {
    GSubprocessFlags fl = G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDERR_PIPE |
                          (stdin_text ? G_SUBPROCESS_FLAGS_STDIN_PIPE : 0);
    GSubprocess *p = g_subprocess_newv(argv, fl, NULL);
    if (!p) { if (cb) cb(FALSE, "", "Error: nmcli not found", (char *)key); return; }
    Job *j = g_new0(Job, 1);
    j->cb = cb;
    j->key = g_strdup(key);
    g_subprocess_communicate_utf8_async(p, stdin_text, NULL, job_done, j);
}

static Net *net_by_ssid(const char *ssid) {
    for (GList *l = S.nets; l && ssid; l = l->next) {
        Net *n = l->data;
        if (!strcmp(n->ssid, ssid)) return n;
    }
    return NULL;
}

static Net *net_of(GtkWidget *row) {
    return row ? g_object_get_data(G_OBJECT(row), "net") : NULL;
}

static char *short_error(const char *err) {
    if (strstr(err, "Secrets were required") || strstr(err, "802-11-wireless-security.psk"))
        return g_strdup("Wrong password");
    if (strstr(err, "No network with SSID")) return g_strdup("Network not found");
    const char *e = strstr(err, "Error: ");
    char *s = g_strdup(e ? e + 7 : err);
    char *nl = strchr(s, '\n');
    if (nl) *nl = 0;
    g_strstrip(s);
    if (!*s) { g_free(s); return g_strdup("Couldn't connect"); }
    if (g_utf8_strlen(s, -1) > 46) {
        char *cut = g_utf8_substring(s, 0, 45);
        g_free(s);
        s = g_strconcat(cut, ELLIPSIS, NULL);
        g_free(cut);
    }
    return s;
}

static const char *signal_word(int s) {
    return s >= 75 ? "Excellent" : s >= 50 ? "Good" : s >= 25 ? "Fair" : "Weak";
}

static char *band(const char *freq) {
    int mhz = freq ? atoi(freq) : 0;
    if (mhz <= 0) return NULL;
    return g_strdup(mhz < 3000 ? "2.4 GHz" : mhz < 5925 ? "5 GHz" : "6 GHz");
}

/* -- placement (1 bspc query for monitor rect + bar padding + backdrop rect) -- */
static void anchor(gboolean pointer) {
    int py = 0;
    gdk_device_get_position(gdk_seat_get_pointer(gdk_display_get_default_seat(gdk_display_get_default())),
                            NULL, &S.px, &py);
    if (S.anchor_cached && S.px >= S.mon.x && S.px < S.mon.x + S.mon.w &&
        py >= S.mon.y && py < S.mon.y + S.mon.h) {
        return;
    }
    S.mon = (Rect){0, 0, 1920, 1080};
    strcpy(S.edge, "top");
    S.off = 0;
    const char *sel = pointer ? "pointed" : "focused";
    const char *argv[] = {"bspc", "query", "-T", "-m", sel, NULL};
    char *out = run_out(argv);
    if (!out && pointer) {
        const char *fb[] = {"bspc", "query", "-T", "-m", "focused", NULL};
        out = run_out(fb);
    }
    int top = 0, bottom = 0;
    gboolean ok = FALSE;
    if (out) {
        const char *p = strstr(out, "\"padding\":{");
        if (p) sscanf(p, "\"padding\":{\"top\":%d,\"right\":%*d,\"bottom\":%d", &top, &bottom);
        const char *r = strstr(out, "\"rectangle\":{");
        if (r && sscanf(r, "\"rectangle\":{\"x\":%d,\"y\":%d,\"width\":%d,\"height\":%d",
                        &S.mon.x, &S.mon.y, &S.mon.w, &S.mon.h) == 4) {
            ok = TRUE;
        }
        if (bottom > top) { strcpy(S.edge, "bottom"); S.off = bottom; }
        else              { strcpy(S.edge, "top");    S.off = top;    }
        g_free(out);
    }
    if (!ok) S.mon = focused_monitor();
    S.below = S.mon;
    int btop = top >= BAR_MIN ? top : 0, bbot = bottom >= BAR_MIN ? bottom : 0;
    S.below.y += btop;
    S.below.h -= btop + bbot;
    S.anchor_cached = ok;
}

static int spawn_backdrop(Rect r) {
    char *bin = g_build_filename(self_dir(), "misa-backdrop", NULL);
    char a[4][16], spid[16];
    g_snprintf(a[0], 16, "%d", r.x);
    g_snprintf(a[1], 16, "%d", r.y);
    g_snprintf(a[2], 16, "%d", r.w);
    g_snprintf(a[3], 16, "%d", r.h);
    g_snprintf(spid, 16, "%d", (int)getpid());
    const char *argv[8] = {bin, a[0], a[1], a[2], a[3], "--pid", spid, NULL};
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

static void place(int h) {
    if (h <= 0) h = gtk_widget_get_allocated_height(S.win);
    if (h <= 0) h = 100;
    int x = S.pointer ? S.px - WIDTH / 2 : S.mon.x + S.mon.w - WIDTH - 8;
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

static gboolean fit_idle(gpointer _) { fit(); return G_SOURCE_REMOVE; }

/* Size changes that animate (stack page slide with interpolated height, the
 * QR revealer) report an in-between natural height on every frame, but the
 * toplevel only follows when asked: fitting once at the start and end kept
 * the old height through the slide (content clipped) and then snapped. So
 * while something animates, re-fit on every frame clock tick. */
static guint fit_tick_id;
static gint64 fit_until;

static gboolean fit_tick(GtkWidget *_w, GdkFrameClock *clock, gpointer _d) {
    fit();
    if (gdk_frame_clock_get_frame_time(clock) < fit_until) return G_SOURCE_CONTINUE;
    fit_tick_id = 0;
    return G_SOURCE_REMOVE;
}

static void fit_follow(int ms) {
    fit_until = MAX(fit_until, g_get_monotonic_time() + (gint64)ms * 1000);
    if (!fit_tick_id) fit_tick_id = gtk_widget_add_tick_callback(S.box, fit_tick, NULL, NULL);
    fit();
}

/* -- page scaffolding ----------------------------------------------------------- */
static void go_back(GtkWidget *_w, gpointer _d) { show_list(); }

/* "‹  Title" bar of a sub page. */
static GtkWidget *page_head(const char *title) {
    GtkWidget *back = gtk_button_new_with_label(GL_BACK);
    add_class(back, "back");
    gtk_widget_set_can_focus(back, FALSE);
    gtk_widget_set_tooltip_text(back, "Back");
    g_signal_connect(back, "clicked", G_CALLBACK(go_back), NULL);
    GtkWidget *h = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    add_class(h, "head");
    gtk_box_pack_start(GTK_BOX(h), back, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(h), label(title, "page-title", 0), TRUE, TRUE, 0);
    return h;
}

/* A grouped block of rows (an iOS settings group). */
static GtkWidget *group(void) {
    GtkWidget *g = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    add_class(g, "group");
    return g;
}

static GtkWidget *grow(GtkWidget *grp, const char *key, GtkWidget *value) {
    GtkWidget *r = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    add_class(r, "grow");
    gtk_box_pack_start(GTK_BOX(r), label(key, "gkey", 0), FALSE, FALSE, 0);
    if (value) gtk_box_pack_end(GTK_BOX(r), value, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(grp), r, FALSE, FALSE, 0);
    return r;
}

static void info_row(GtkWidget *grp, const char *key, const char *val) {
    if (!val || !*val) return;
    GtkWidget *v = label(val, "gval", 1);
    gtk_label_set_selectable(GTK_LABEL(v), TRUE);
    gtk_widget_set_can_focus(v, FALSE);
    grow(grp, key, v);
}

static GtkWidget *action_row(GtkWidget *grp, const char *text, GCallback cb) {
    GtkWidget *b = gtk_button_new_with_label(text);
    add_class(b, "action");
    gtk_widget_set_halign(gtk_bin_get_child(GTK_BIN(b)), GTK_ALIGN_START);
    g_signal_connect(b, "clicked", cb, NULL);
    gtk_box_pack_start(GTK_BOX(grp), b, FALSE, FALSE, 0);
    return b;
}

static void swap_page(GtkWidget **slot, GtkWidget *page, const char *name) {
    GtkWidget *old = *slot;
    *slot = page;
    gtk_widget_show_all(page);
    gtk_stack_add_named(GTK_STACK(S.stack), page, old ? NULL : name);
    gtk_stack_set_visible_child(GTK_STACK(S.stack), page);
    if (old) gtk_widget_destroy(old);
    g_idle_add(fit_idle, NULL);
}

/* -- header switch ---------------------------------------------------------------- */
static void update_switch(void) {
    S.syncing = TRUE;
    gtk_switch_set_active(GTK_SWITCH(S.sw), S.radio);
    gtk_switch_set_state(GTK_SWITCH(S.sw), S.radio);
    S.syncing = FALSE;
}

static gboolean refresh_later(gpointer _d) { refresh_async(); return G_SOURCE_REMOVE; }

/* The switch keeps the position the user chose until the radio really is
 * there (refresh() reads `pending`), so it never jumps back and forth. */
static gboolean pending, want;
static int settle_tries;
static void scan(void);

static gboolean radio_is_on(void) {
    const char *ra[] = {"nmcli", "radio", "wifi", NULL};
    char *r = run_out(ra);
    gboolean on = r && g_str_has_prefix(g_strstrip(r), "enabled");
    g_free(r);
    return on;
}

static gboolean settle(gpointer _d) {
    if (S.closing) return G_SOURCE_REMOVE;
    if (radio_is_on() != want && ++settle_tries < 25) return G_SOURCE_CONTINUE;
    pending = FALSE;
    refresh_async();
    if (S.radio || want) {
        scan();
        g_timeout_add(2500, refresh_later, NULL);           /* networks come a moment later */
    }
    return G_SOURCE_REMOVE;
}

static void radio_done(int ok, const char *out, const char *err, char *key) {
    settle_tries = ok ? 0 : 99;                             /* failed: show the real state */
    g_timeout_add(300, settle, NULL);
}

static void toggle_radio(gboolean on) {
    pending = TRUE;
    want = on;
    const char *argv[] = {"nmcli", "radio", "wifi", on ? "on" : "off", NULL};
    run_async(argv, NULL, radio_done, NULL);
    if (!on) apply_refresh("disabled", "", "");             /* "Wi-Fi is off" at once, 0 fork */
    else refresh_async();
}

static gboolean on_switch(GtkSwitch *sw, gboolean on, gpointer _d) {
    if (S.syncing) return FALSE;
    gtk_switch_set_state(sw, on);
    if (on != (pending ? want : S.radio)) toggle_radio(on);
    return TRUE;
}

/* -- connecting ---------------------------------------------------------------------- */
static void set_status(Net *n, const char *status, gboolean busy) {
    if (!n) return;
    g_free(n->status);
    n->status = status ? g_strdup(status) : NULL;
    n->busy = busy;
    gtk_label_set_text(GTK_LABEL(n->sub), n->status ? n->status : "");
    gtk_widget_set_visible(n->sub, n->status != NULL);
    set_class(n->row, "busy", busy);
    set_class(n->row, "error", n->status && !busy);
    g_idle_add(fit_idle, NULL);
}

static void page_status(const char *t) {
    GtkWidget *l = showing(S.detail) ? S.d_status : showing(S.join) ? S.j_err : NULL;
    if (!l) return;
    gtk_label_set_text(GTK_LABEL(l), t ? t : "");
    gtk_widget_set_visible(l, t != NULL);
    g_idle_add(fit_idle, NULL);
}

static void connect_done(int ok, const char *out, const char *err, char *key) {
    Net *n = net_by_ssid(key);
    if (ok) {
        if (n) set_status(n, NULL, FALSE);
        show_list();
        return;
    }
    char *e = short_error(err);
    set_status(n, e, FALSE);
    page_status(e);
    g_free(e);
    if (showing(S.join)) {
        gtk_widget_set_sensitive(S.j_entry, TRUE);
        gtk_widget_set_sensitive(S.j_btn, TRUE);
        gtk_button_set_label(GTK_BUTTON(S.j_btn), "Join");
        gtk_widget_grab_focus(S.j_entry);
        gtk_editable_select_region(GTK_EDITABLE(S.j_entry), 0, -1);
    }
}

/* Saved: `connection up`; new: `device wifi connect`, the password on stdin
 * (--ask); a hidden one adds `hidden yes`. */
static void do_connect(const char *ssid, const char *password, gboolean saved, gboolean hidden) {
    Net *n = net_by_ssid(ssid);
    if (n && n->busy) return;
    set_status(n, "Connecting" ELLIPSIS, TRUE);
    page_status("Connecting" ELLIPSIS);
    if (saved) {
        const char *argv[] = {"nmcli", "--wait", "45", "connection", "up", "id", ssid, NULL};
        run_async(argv, NULL, connect_done, ssid);
        return;
    }
    char *in = password && *password ? g_strconcat(password, "\n", NULL) : NULL;
    GPtrArray *a = g_ptr_array_new();
    g_ptr_array_add(a, "nmcli");
    if (in) g_ptr_array_add(a, "--ask");
    const char *rest[] = {"--wait", "45", "device", "wifi", "connect", ssid, NULL};
    for (int i = 0; rest[i]; i++) g_ptr_array_add(a, (gpointer)rest[i]);
    if (*S.ifname) { g_ptr_array_add(a, "ifname"); g_ptr_array_add(a, S.ifname); }
    if (hidden) { g_ptr_array_add(a, "hidden"); g_ptr_array_add(a, "yes"); }
    g_ptr_array_add(a, NULL);
    run_async((const char *const *)a->pdata, in, connect_done, ssid);
    g_ptr_array_free(a, TRUE);
    if (in) { memset(in, 0, strlen(in)); g_free(in); }
}

/* -- join page ---------------------------------------------------------------------- */
static const char *join_ssid(void) {
    return S.j_ssid ? gtk_entry_get_text(GTK_ENTRY(S.j_ssid)) : S.page_ssid;
}

static void join_validate(void) {
    if (!S.j_btn) return;
    const char *ssid = join_ssid();
    int len = g_utf8_strlen(gtk_entry_get_text(GTK_ENTRY(S.j_entry)), -1);
    gboolean ok = ssid && *ssid && (len >= 8 || (S.join_hidden && len == 0));
    gtk_widget_set_sensitive(S.j_btn, ok);
}

static void on_join_changed(GtkEditable *_e, gpointer _d) {
    gtk_widget_set_visible(S.j_err, FALSE);
    join_validate();
}

static void on_join(GtkWidget *_w, gpointer _d) {
    if (!gtk_widget_get_sensitive(S.j_btn)) return;
    char *s = g_strdup(join_ssid());
    g_free(S.page_ssid);
    S.page_ssid = s;
    gtk_widget_set_sensitive(S.j_entry, FALSE);
    gtk_widget_set_sensitive(S.j_btn, FALSE);
    gtk_button_set_label(GTK_BUTTON(S.j_btn), "Joining" ELLIPSIS);
    do_connect(s, gtk_entry_get_text(GTK_ENTRY(S.j_entry)), FALSE, S.join_hidden);
}

static void on_join_eye(GtkWidget *b, gpointer _d) {
    gboolean vis = !gtk_entry_get_visibility(GTK_ENTRY(S.j_entry));
    gtk_entry_set_visibility(GTK_ENTRY(S.j_entry), vis);
    gtk_button_set_label(GTK_BUTTON(b), vis ? GL_EYEOFF : GL_EYE);
    gtk_widget_grab_focus(S.j_entry);
    gtk_editable_set_position(GTK_EDITABLE(S.j_entry), -1);
}

static GtkWidget *field(GtkWidget **entry, const char *placeholder, gboolean secret) {
    GtkWidget *f = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    add_class(f, "field");
    *entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(*entry), placeholder);
    gtk_box_pack_start(GTK_BOX(f), *entry, TRUE, TRUE, 0);
    if (secret) {
        gtk_entry_set_visibility(GTK_ENTRY(*entry), FALSE);
        gtk_entry_set_input_purpose(GTK_ENTRY(*entry), GTK_INPUT_PURPOSE_PASSWORD);
        GtkWidget *eye = gtk_button_new_with_label(GL_EYE);
        add_class(eye, "eye");
        gtk_widget_set_can_focus(eye, FALSE);
        gtk_widget_set_tooltip_text(eye, "Show password");
        g_signal_connect(eye, "clicked", G_CALLBACK(on_join_eye), NULL);
        gtk_box_pack_start(GTK_BOX(f), eye, FALSE, FALSE, 0);
    }
    g_signal_connect(*entry, "changed", G_CALLBACK(on_join_changed), NULL);
    g_signal_connect(*entry, "activate", G_CALLBACK(on_join), NULL);
    return f;
}

static void show_join(const char *ssid) {
    S.join_hidden = ssid == NULL;
    g_free(S.page_ssid);
    S.page_ssid = g_strdup(ssid);
    S.j_ssid = NULL;
    S.j_btn = NULL;
    GtkWidget *p = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start(GTK_BOX(p), page_head(ssid ? ssid : "Other network"), FALSE, FALSE, 0);
    GtkWidget *body = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    add_class(body, "body");
    gtk_box_pack_start(GTK_BOX(body),
                       label(ssid ? "Enter the password" : "Enter the network name and password", "hint", 0),
                       FALSE, FALSE, 0);
    if (!ssid) gtk_box_pack_start(GTK_BOX(body), field(&S.j_ssid, "Network name", FALSE), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(body), field(&S.j_entry, ssid ? "Password" : "Password (empty if open)", TRUE),
                       FALSE, FALSE, 0);
    S.j_err = label("", "err", 0);
    gtk_widget_set_no_show_all(S.j_err, TRUE);
    gtk_box_pack_start(GTK_BOX(body), S.j_err, FALSE, FALSE, 0);
    S.j_btn = gtk_button_new_with_label("Join");
    add_class(S.j_btn, "primary");
    g_signal_connect(S.j_btn, "clicked", G_CALLBACK(on_join), NULL);
    gtk_box_pack_start(GTK_BOX(body), S.j_btn, FALSE, FALSE, 4);
    gtk_box_pack_start(GTK_BOX(p), body, FALSE, FALSE, 0);
    swap_page(&S.join, p, "join");
    join_validate();
    gtk_widget_grab_focus(S.j_ssid ? S.j_ssid : S.j_entry);
}

/* -- detail page --------------------------------------------------------------------- */
static char *fetch_secret(const char *ssid) {
    const char *argv[] = {"nmcli", "-s", "-g", "802-11-wireless-security.psk", "connection", "show",
                          "id", ssid, NULL};
    char *out = run_out(argv);
    if (out) g_strchomp(out);
    return out;
}

static void on_detail_eye(GtkWidget *b, gpointer _d) {
    if (S.d_secret) {
        wipe(&S.d_secret);
        gtk_label_set_text(GTK_LABEL(S.d_pw), DOTS);
        gtk_label_set_selectable(GTK_LABEL(S.d_pw), FALSE);
        gtk_button_set_label(GTK_BUTTON(b), GL_EYE);
        return;
    }
    S.d_secret = fetch_secret(S.page_ssid);
    if (!S.d_secret || !*S.d_secret) {
        wipe(&S.d_secret);
        gtk_label_set_text(GTK_LABEL(S.d_pw), "Not available");
        return;
    }
    gtk_label_set_text(GTK_LABEL(S.d_pw), S.d_secret);
    gtk_label_set_selectable(GTK_LABEL(S.d_pw), TRUE);
    gtk_button_set_label(GTK_BUTTON(b), GL_EYEOFF);
}

static void simple_done(int ok, const char *out, const char *err, char *key) {
    if (!ok) {
        char *e = short_error(err);
        page_status(e);
        g_free(e);
        return;
    }
    show_list();
}

static void activate(Net *n);

static void on_conn(GtkWidget *_b, gpointer _d) {
    Net *n = net_by_ssid(S.page_ssid);
    if (!n) return;
    if (n->active) {
        page_status("Disconnecting" ELLIPSIS);
        const char *argv[] = {"nmcli", "connection", "down", "id", S.page_ssid, NULL};
        run_async(argv, NULL, simple_done, S.page_ssid);
    } else if (n->saved || !n->secure) {
        do_connect(S.page_ssid, NULL, n->saved, FALSE);
    } else {
        show_join(n->ssid);
    }
}

/* Asks once more before forgetting, as a phone does: the first click turns
 * the row into the question, the second forgets. */
static void on_forget(GtkWidget *b, gpointer _d) {
    if (!S.d_confirm) {
        S.d_confirm = TRUE;
        gtk_button_set_label(GTK_BUTTON(b), "Forget? Click again to confirm");
        gtk_widget_set_halign(gtk_bin_get_child(GTK_BIN(b)), GTK_ALIGN_START);
        add_class(b, "confirm");
        return;
    }
    page_status("Forgetting" ELLIPSIS);
    const char *argv[] = {"nmcli", "connection", "delete", "id", S.page_ssid, NULL};
    run_async(argv, NULL, simple_done, S.page_ssid);
}

static gboolean on_auto(GtkSwitch *sw, gboolean on, gpointer _d) {
    if (S.syncing) return FALSE;
    const char *argv[] = {"nmcli", "connection", "modify", "id", S.page_ssid, "connection.autoconnect",
                          on ? "yes" : "no", NULL};
    run_async(argv, NULL, NULL, NULL);
    gtk_switch_set_state(sw, on);
    return TRUE;
}

/* WIFI: payload with \ ; , : " escaped (the format phone cameras read). */
static void qr_escape(GString *s, const char *v) {
    for (const char *p = v; *p; p++) {
        if (strchr("\\;,:\"", *p)) g_string_append_c(s, '\\');
        g_string_append_c(s, *p);
    }
}

static void on_share(GtkWidget *_b, gpointer _d) {
    if (gtk_revealer_get_reveal_child(GTK_REVEALER(S.d_qr_rev))) {
        gtk_revealer_set_reveal_child(GTK_REVEALER(S.d_qr_rev), FALSE);
        gtk_image_clear(GTK_IMAGE(S.d_qr));
        fit_follow(QR_ANIM_MS);
        return;
    }
    Net *n = net_by_ssid(S.page_ssid);
    char *pw = fetch_secret(S.page_ssid);
    gboolean sec = pw && *pw;
    GString *payload = g_string_new("WIFI:T:");
    g_string_append(payload, !sec ? "nopass"
                    : (n && strstr(n->security, "WPA3") && !strstr(n->security, "WPA2")) ? "SAE" : "WPA");
    g_string_append(payload, ";S:");
    qr_escape(payload, S.page_ssid);
    if (sec) { g_string_append(payload, ";P:"); qr_escape(payload, pw); }
    g_string_append(payload, ";;");
    wipe(&pw);
    /* qrencode reads the payload from stdin: the password never hits argv. */
    const char *argv[] = {"qrencode", "-t", "PNG", "-s", "5", "-m", "2", "-o", "-", NULL};
    GSubprocess *p = g_subprocess_newv(argv, G_SUBPROCESS_FLAGS_STDIN_PIPE | G_SUBPROCESS_FLAGS_STDOUT_PIPE, NULL);
    GBytes *in = g_bytes_new(payload->str, payload->len), *out = NULL;
    if (p && g_subprocess_communicate(p, in, NULL, &out, NULL, NULL) && out) {
        GdkPixbufLoader *ld = gdk_pixbuf_loader_new();
        gsize len;
        const guchar *d = g_bytes_get_data(out, &len);
        if (gdk_pixbuf_loader_write(ld, d, len, NULL) && gdk_pixbuf_loader_close(ld, NULL)) {
            gtk_image_set_from_pixbuf(GTK_IMAGE(S.d_qr), gdk_pixbuf_loader_get_pixbuf(ld));
            gtk_revealer_set_reveal_child(GTK_REVEALER(S.d_qr_rev), TRUE);
        }
        g_object_unref(ld);
    }
    memset(payload->str, 0, payload->len);
    g_string_free(payload, TRUE);
    g_bytes_unref(in);
    if (out) g_bytes_unref(out);
    if (p) g_object_unref(p);
    fit_follow(QR_ANIM_MS);
}

static char *dev_field(char **lines, const char *key) {
    GString *v = g_string_new(NULL);
    for (int i = 0; lines[i]; i++) {
        if (!g_str_has_prefix(lines[i], key)) continue;
        const char *c = strchr(lines[i], ':');
        if (!c || !c[1]) continue;
        if (v->len) g_string_append(v, ", ");
        char *val = g_strdup(c + 1);
        char *slash = strcmp(key, "IP4.ADDRESS") == 0 ? strchr(val, '/') : NULL;
        if (slash) *slash = 0;
        g_string_append(v, val);
        g_free(val);
    }
    return g_string_free(v, v->len == 0);
}

static void show_detail(const char *ssid) {
    Net *n = net_by_ssid(ssid);
    if (!n) return;
    char *keep = g_strdup(ssid);
    g_free(S.page_ssid);
    S.page_ssid = keep;
    wipe(&S.d_secret);
    S.d_confirm = FALSE;
    GtkWidget *p = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start(GTK_BOX(p), page_head(n->ssid), FALSE, FALSE, 0);
    GtkWidget *body = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    add_class(body, "body");

    S.d_status = label("", "err", 0);
    gtk_widget_set_no_show_all(S.d_status, TRUE);
    gtk_box_pack_start(GTK_BOX(body), S.d_status, FALSE, FALSE, 0);

    /* Connect / Disconnect, Forget. */
    GtkWidget *g1 = group();
    action_row(g1, n->active ? "Disconnect" : n->saved ? "Connect" : "Join This Network", G_CALLBACK(on_conn));
    if (n->saved) action_row(g1, "Forget This Network", G_CALLBACK(on_forget));
    gtk_box_pack_start(GTK_BOX(body), g1, FALSE, FALSE, 0);

    /* Auto-Join, Password, Share Password (for saved networks). */
    S.d_qr = S.d_qr_rev = NULL;
    if (n->saved) {
        GtkWidget *g2 = group();
        GtkWidget *aj = gtk_switch_new();
        gtk_widget_set_valign(aj, GTK_ALIGN_CENTER);
        gtk_widget_set_can_focus(aj, FALSE);
        const char *aa[] = {"nmcli", "-g", "connection.autoconnect", "connection", "show", "id", S.page_ssid, NULL};
        char *auto_on = run_out(aa);
        S.syncing = TRUE;
        gtk_switch_set_active(GTK_SWITCH(aj), !auto_on || !g_str_has_prefix(auto_on, "no"));
        S.syncing = FALSE;
        g_free(auto_on);
        g_signal_connect(aj, "state-set", G_CALLBACK(on_auto), NULL);
        grow(g2, "Auto-Join", aj);
        if (n->secure) {
            GtkWidget *pv = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
            S.d_pw = label(DOTS, "gval", 1);
            GtkWidget *eye = gtk_button_new_with_label(GL_EYE);
            add_class(eye, "eye");
            gtk_widget_set_can_focus(eye, FALSE);
            gtk_widget_set_tooltip_text(eye, "Show password");
            g_signal_connect(eye, "clicked", G_CALLBACK(on_detail_eye), NULL);
            gtk_box_pack_start(GTK_BOX(pv), S.d_pw, FALSE, FALSE, 0);
            gtk_box_pack_start(GTK_BOX(pv), eye, FALSE, FALSE, 0);
            grow(g2, "Password", pv);
            char *qr = g_find_program_in_path("qrencode");
            if (qr) {
                S.d_qr = gtk_image_new();
                GtkWidget *qbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
                add_class(qbox, "qr");
                gtk_box_pack_start(GTK_BOX(qbox), S.d_qr, FALSE, FALSE, 0);
                gtk_box_pack_start(GTK_BOX(qbox), label("Scan with a phone camera to join", "hint", 0.5),
                                   FALSE, FALSE, 0);
                S.d_qr_rev = gtk_revealer_new();
                gtk_revealer_set_transition_type(GTK_REVEALER(S.d_qr_rev), GTK_REVEALER_TRANSITION_TYPE_SLIDE_DOWN);
                gtk_revealer_set_transition_duration(GTK_REVEALER(S.d_qr_rev), QR_ANIM_MS);
                gtk_container_add(GTK_CONTAINER(S.d_qr_rev), qbox);
                action_row(g2, "Share Password", G_CALLBACK(on_share));
                gtk_box_pack_start(GTK_BOX(g2), S.d_qr_rev, FALSE, FALSE, 0);
                g_free(qr);
            }
        }
        gtk_box_pack_start(GTK_BOX(body), g2, FALSE, FALSE, 0);
    }

    /* Information. */
    GtkWidget *g3 = group();
    info_row(g3, "Signal", signal_word(n->signal));
    info_row(g3, "Security", n->secure ? n->security : "None");
    char *bd = band(n->freq);
    info_row(g3, "Frequency", bd);
    g_free(bd);
    info_row(g3, "Link speed", n->rate);
    if (n->active && *S.ifname) {
        const char *da[] = {"nmcli", "-t", "-f", "IP4.ADDRESS,IP4.GATEWAY,IP4.DNS", "device", "show", S.ifname, NULL};
        char *out = run_out(da);
        char **lines = g_strsplit(out ? out : "", "\n", -1);
        char *ip = dev_field(lines, "IP4.ADDRESS"), *gw = dev_field(lines, "IP4.GATEWAY"),
             *dns = dev_field(lines, "IP4.DNS");
        info_row(g3, "IP address", ip);
        info_row(g3, "Router", gw);
        info_row(g3, "DNS", dns);
        g_free(ip); g_free(gw); g_free(dns);
        g_strfreev(lines);
        g_free(out);
    }
    gtk_box_pack_start(GTK_BOX(body), g3, FALSE, FALSE, 0);

    gtk_box_pack_start(GTK_BOX(p), body, FALSE, FALSE, 0);
    swap_page(&S.detail, p, "detail");
}

/* -- list page (iOS grouped sections: Wi-Fi + active, My networks, Other networks) -- */
static void activate(Net *n) {
    if (!n || n->busy) return;
    if (n->active) show_detail(n->ssid);
    else if (n->saved || !n->secure) do_connect(n->ssid, NULL, n->saved, FALSE);
    else show_join(n->ssid);
}

static void on_activated(GtkListBox *_l, GtkListBoxRow *row, gpointer _d) {
    if (g_object_get_data(G_OBJECT(row), "other")) { show_join(NULL); return; }
    activate(net_of(GTK_WIDGET(row)));
}

static void on_info(GtkWidget *_b, gpointer data) {
    show_detail(((Net *)data)->ssid);
}

static void select_only(GtkListBoxRow *row) {
    if (!row || S.selecting) return;
    S.selecting = TRUE;
    GtkWidget *parent = gtk_widget_get_parent(GTK_WIDGET(row));
    GtkWidget *lists[] = {S.active_list, S.saved_list, S.other_list};
    for (int i = 0; i < 3; i++) {
        if (lists[i] && lists[i] != parent)
            gtk_list_box_unselect_all(GTK_LIST_BOX(lists[i]));
    }
    if (parent && GTK_IS_LIST_BOX(parent))
        gtk_list_box_select_row(GTK_LIST_BOX(parent), row);
    gtk_widget_grab_focus(GTK_WIDGET(row));
    S.selecting = FALSE;
}

static void on_row_selected(GtkListBox *box, GtkListBoxRow *row, gpointer _d) {
    if (!row || S.selecting) return;
    select_only(row);
}

static GtkListBoxRow *selected_row(void) {
    GtkWidget *lists[] = {S.active_list, S.saved_list, S.other_list};
    for (int i = 0; i < 3; i++) {
        if (!lists[i]) continue;
        GtkListBoxRow *r = gtk_list_box_get_selected_row(GTK_LIST_BOX(lists[i]));
        if (r) return r;
    }
    return NULL;
}

static GtkWidget *make_list_group(const char *extra_cls) {
    GtkWidget *lb = gtk_list_box_new();
    add_class(lb, "net-list");
    if (extra_cls) add_class(lb, extra_cls);
    gtk_list_box_set_selection_mode(GTK_LIST_BOX(lb), GTK_SELECTION_SINGLE);
    gtk_list_box_set_activate_on_single_click(GTK_LIST_BOX(lb), TRUE);
    g_signal_connect(lb, "row-activated", G_CALLBACK(on_activated), NULL);
    g_signal_connect(lb, "row-selected", G_CALLBACK(on_row_selected), NULL);
    return lb;
}

static void unselect_all_lists(void) {
    S.selecting = TRUE;
    GtkWidget *lists[] = {S.active_list, S.saved_list, S.other_list};
    for (int i = 0; i < 3; i++) {
        if (lists[i]) gtk_list_box_unselect_all(GTK_LIST_BOX(lists[i]));
    }
    S.selecting = FALSE;
}

/* iOS-style concentric Wi-Fi arcs: lit arcs up to `lvl` (1..3), dim outer arcs. */
static gboolean draw_wifi_icon(GtkWidget *w, cairo_t *cr, gpointer data) {
    int packed = GPOINTER_TO_INT(data);
    int lvl = packed & 3;
    gboolean active = (packed & 4) != 0;
    GdkRGBA fg = {0.9, 0.9, 0.9, 1.0};
    gtk_style_context_get_color(gtk_widget_get_style_context(w), GTK_STATE_FLAG_NORMAL, &fg);
    double lit_a = active ? 0.98 : 0.72;
    double dim_a = 0.20;

    double cx = gtk_widget_get_allocated_width(w) * 0.5;
    double cy = gtk_widget_get_allocated_height(w) - 2.6;
    const double radii[3] = {4.1, 7.1, 10.1};
    const double a1 = -132.0 * (G_PI / 180.0), a2 = -48.0 * (G_PI / 180.0);

    cairo_set_line_width(cr, 1.55);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
    for (int i = 0; i < 3; i++) {
        cairo_set_source_rgba(cr, fg.red, fg.green, fg.blue, (i < lvl) ? lit_a : dim_a);
        cairo_arc(cr, cx, cy, radii[i], a1, a2);
        cairo_stroke(cr);
    }
    cairo_set_source_rgba(cr, fg.red, fg.green, fg.blue, lit_a);
    cairo_arc(cr, cx, cy, 1.3, 0, 2 * G_PI);
    cairo_fill(cr);
    return FALSE;
}

static GtkWidget *wifi_icon(int signal, gboolean active) {
    int lvl = signal >= 70 ? 3 : signal >= 45 ? 2 : signal >= 20 ? 1 : 0;
    int packed = (lvl & 3) | (active ? 4 : 0);
    GtkWidget *da = gtk_drawing_area_new();
    add_class(da, "signal");
    gtk_widget_set_size_request(da, 18, 15);
    gtk_widget_set_valign(da, GTK_ALIGN_CENTER);
    g_signal_connect(da, "draw", G_CALLBACK(draw_wifi_icon), GINT_TO_POINTER(packed));
    return da;
}

static GtkWidget *make_row(Net *n) {
    n->row = gtk_list_box_row_new();
    g_object_set_data(G_OBJECT(n->row), "net", n);
    add_class(n->row, "net");
    if (n->active) add_class(n->row, "active");

    n->sub = label("", "sub", 0);
    gtk_widget_set_no_show_all(n->sub, TRUE);
    GtkWidget *texts = gtk_box_new(GTK_ORIENTATION_VERTICAL, 1);
    gtk_box_pack_start(GTK_BOX(texts), label(n->ssid, "name", 0), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(texts), n->sub, FALSE, FALSE, 0);
    gtk_widget_set_valign(texts, GTK_ALIGN_CENTER);

    GtkWidget *r = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    if (n->active) {
        GtkWidget *chk = glyph(GL_CHECK, "check");
        gtk_box_pack_start(GTK_BOX(r), chk, FALSE, FALSE, 0);
    }
    gtk_box_pack_start(GTK_BOX(r), texts, TRUE, TRUE, 0);
    if (n->secure) gtk_box_pack_start(GTK_BOX(r), glyph(GL_LOCK, "lock"), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(r), wifi_icon(n->signal, n->active), FALSE, FALSE, 0);
    GtkWidget *info = gtk_button_new_with_label(GL_INFO);
    add_class(info, "info");
    gtk_widget_set_can_focus(info, FALSE);
    gtk_widget_set_tooltip_text(info, "Details");
    g_signal_connect(info, "clicked", G_CALLBACK(on_info), n);
    gtk_box_pack_start(GTK_BOX(r), info, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(n->row), r);
    return n->row;
}

static GtkWidget *other_row(void) {
    GtkWidget *row = gtk_list_box_row_new();
    g_object_set_data(G_OBJECT(row), "other", GINT_TO_POINTER(1));
    add_class(row, "net");
    add_class(row, "other");
    GtkWidget *r = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_box_pack_start(GTK_BOX(r), label("Other" ELLIPSIS, "name", 0), TRUE, TRUE, 0);
    gtk_container_add(GTK_CONTAINER(row), r);
    return row;
}

static void free_net(Net *n) {
    g_free(n->ssid);
    g_free(n->security);
    g_free(n->freq);
    g_free(n->rate);
    g_free(n->status);
    g_free(n);
}

static gint by_group(gconstpointer a, gconstpointer b) {
    const Net *x = a, *y = b;
    if (x->group != y->group) return x->group - y->group;
    if (x->signal != y->signal) return y->signal - x->signal;
    return g_utf8_collate(x->ssid, y->ssid);
}

static GHashTable *parse_saved_names(const char *out) {
    GHashTable *h = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    char **lines = g_strsplit(out ? out : "", "\n", -1);
    for (int i = 0; lines[i]; i++) {
        if (!*lines[i]) continue;
        char **f = terse(lines[i], 2);
        if (!strcmp(f[1], "802-11-wireless")) g_hash_table_add(h, g_strdup(f[0]));
        g_strfreev(f);
    }
    g_strfreev(lines);
    return h;
}

static void update_section(void) {
    if (S.other_scan)
        gtk_widget_set_visible(S.other_scan, S.radio && S.scanning);
}

static const char *const CMD_RADIO[] = {"nmcli", "radio", "wifi", NULL};
static const char *const CMD_SAVED[] = {"nmcli", "-t", "-f", "NAME,TYPE", "connection", "show", NULL};
static const char *const CMD_LIST[]  = {"nmcli", "-t", "-f", "IN-USE,SSID,SIGNAL,SECURITY,FREQ,RATE",
                                        "device", "wifi", "list", "--rescan", "no", NULL};

static void apply_refresh(const char *out_radio, const char *out_saved, const char *out_list) {
    if (S.closing || !showing(S.listpage)) return;
    char *rtrim = g_strdup(out_radio ? out_radio : "");
    gboolean r_on = g_str_has_prefix(g_strstrip(rtrim), "enabled");
    g_free(rtrim);
    S.radio = pending ? want : r_on;
    update_switch();

    GHashTable *busy = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
    for (GList *l = S.nets; l; l = l->next) {
        Net *n = l->data;
        if (n->busy && n->status) g_hash_table_insert(busy, g_strdup(n->ssid), g_strdup(n->status));
    }

    GList *new_nets = NULL;
    if (S.radio) {
        GHashTable *saved = parse_saved_names(out_saved);
        GHashTable *seen = g_hash_table_new(g_str_hash, g_str_equal);
        char **lines = g_strsplit(out_list ? out_list : "", "\n", -1);
        for (int i = 0; lines[i]; i++) {
            if (!*lines[i]) continue;
            char **f = terse(lines[i], 6);
            if (*f[1]) {
                Net *n = g_hash_table_lookup(seen, f[1]);
                int sig = atoi(f[2]);
                if (!n) {
                    n = g_new0(Net, 1);
                    n->ssid = g_strdup(f[1]);
                    n->security = g_strdup(f[3]);
                    n->secure = *f[3] && strcmp(f[3], "--") != 0;
                    n->saved = g_hash_table_contains(saved, f[1]);
                    n->signal = -1;
                    g_hash_table_insert(seen, n->ssid, n);
                    new_nets = g_list_prepend(new_nets, n);
                }
                gboolean use = !strcmp(f[0], "*");
                if (use || (!n->active && sig > n->signal)) {   /* the AP in use, else the strongest */
                    n->signal = sig;
                    g_free(n->freq); n->freq = g_strdup(f[4]);
                    g_free(n->rate); n->rate = g_strdup(f[5]);
                }
                if (use) n->active = TRUE;
            }
            g_strfreev(f);
        }
        g_strfreev(lines);
        g_hash_table_destroy(seen);
        g_hash_table_destroy(saved);
    }

    gboolean has_active = FALSE, has_saved = FALSE;
    for (GList *l = new_nets; l; l = l->next) {
        Net *n = l->data;
        n->group = n->active ? GROUP_ACTIVE : n->saved ? GROUP_SAVED : GROUP_OTHER;
        if (n->group == GROUP_ACTIVE) has_active = TRUE;
        else if (n->group == GROUP_SAVED) has_saved = TRUE;
    }
    new_nets = g_list_sort(new_nets, by_group);

    /* Signature check: skip rebuilding widgets if the network list hasn't changed. */
    GString *sig = g_string_new(S.radio ? "1" : "0");
    for (GList *l = new_nets; l; l = l->next) {
        Net *n = l->data;
        int lvl = n->signal >= 70 ? 3 : n->signal >= 45 ? 2 : n->signal >= 20 ? 1 : 0;
        const char *st = g_hash_table_lookup(busy, n->ssid);
        g_string_append_printf(sig, "|%s:%d:%d:%d:%d:%s", n->ssid, n->group, n->secure, lvl,
                               n->signal, st ? st : "");
    }
    if (S.warm && S.sig && strcmp(S.sig, sig->str) == 0) {
        g_string_free(sig, TRUE);
        g_list_free_full(new_nets, (GDestroyNotify)free_net);
        g_hash_table_destroy(busy);
        update_section();
        return;
    }
    g_free(S.sig);
    S.sig = g_string_free(sig, FALSE);
    S.warm = TRUE;

    char *keep = NULL;
    gboolean keep_other = FALSE;
    GtkListBoxRow *selrow = selected_row();
    if (selrow) {
        Net *sel = net_of(GTK_WIDGET(selrow));
        if (sel) keep = g_strdup(sel->ssid);
        else if (g_object_get_data(G_OBJECT(selrow), "other")) keep_other = TRUE;
    }

    g_list_free_full(S.nets, (GDestroyNotify)free_net);
    S.nets = new_nets;

    if (S.active_list) {
        gtk_widget_destroy(S.active_list);
        S.active_list = NULL;
    }
    S.saved_list = S.other_list = S.other_title = S.other_scan = NULL;
    GList *kids = gtk_container_get_children(GTK_CONTAINER(S.sections));
    for (GList *l = kids; l; l = l->next) gtk_widget_destroy(l->data);
    g_list_free(kids);
    g_ptr_array_set_size(S.rows, 0);

    GtkListBoxRow *pick = NULL;
    if (S.radio) {
        if (has_active) {
            S.active_list = make_list_group("active-group");
            gtk_box_pack_start(GTK_BOX(S.top_group), S.active_list, FALSE, FALSE, 0);
        }
        if (has_saved) {
            GtkWidget *sh = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
            add_class(sh, "section-head");
            gtk_box_pack_start(GTK_BOX(sh), label("My networks", "section", 0), TRUE, TRUE, 0);
            gtk_box_pack_start(GTK_BOX(S.sections), sh, FALSE, FALSE, 0);
            S.saved_list = make_list_group("group");
            gtk_box_pack_start(GTK_BOX(S.sections), S.saved_list, FALSE, FALSE, 0);
        }

        GtkWidget *oh = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
        add_class(oh, "section-head");
        S.other_title = label((has_active || has_saved) ? "Other networks" : "Networks", "section", 0);
        S.other_scan = label("Searching" ELLIPSIS, "section-right", 1);
        gtk_widget_set_no_show_all(S.other_scan, TRUE);
        gtk_box_pack_start(GTK_BOX(oh), S.other_title, TRUE, TRUE, 0);
        gtk_box_pack_end(GTK_BOX(oh), S.other_scan, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(S.sections), oh, FALSE, FALSE, 0);

        S.other_list = make_list_group("group");
        gtk_box_pack_start(GTK_BOX(S.sections), S.other_list, FALSE, FALSE, 0);

        for (GList *l = S.nets; l; l = l->next) {
            Net *n = l->data;
            GtkWidget *target = n->group == GROUP_ACTIVE ? S.active_list
                              : n->group == GROUP_SAVED  ? S.saved_list
                                                         : S.other_list;
            GtkWidget *row = make_row(n);
            gtk_container_add(GTK_CONTAINER(target), row);
            g_ptr_array_add(S.rows, row);
            const char *st = g_hash_table_lookup(busy, n->ssid);
            if (st) set_status(n, st, TRUE);
            if (keep && !strcmp(keep, n->ssid)) pick = GTK_LIST_BOX_ROW(row);
        }

        GtkWidget *o = other_row();
        gtk_container_add(GTK_CONTAINER(S.other_list), o);
        g_ptr_array_add(S.rows, o);
        if (keep_other) pick = GTK_LIST_BOX_ROW(o);

        if (S.active_list) gtk_widget_show_all(S.active_list);
        gtk_widget_show_all(S.sections);
    }

    g_hash_table_destroy(busy);
    g_free(keep);

    gtk_widget_set_visible(S.off_label, !S.radio);
    gtk_widget_set_visible(S.scroll, S.radio);
    update_section();

    if (S.kb_nav) {
        if (!pick && S.rows->len > 0) pick = g_ptr_array_index(S.rows, 0);
        if (pick) select_only(pick);
    } else {
        unselect_all_lists();
    }
    fit();
}

/* Parallel async refresh: spawns radio, connection show, and device wifi list concurrently. */
typedef struct {
    int left;
    guint gen;
    char *out[3];
} RefreshBatch;

typedef struct {
    RefreshBatch *batch;
    int idx;
} RefreshPart;

static void refresh_part_done(GObject *src, GAsyncResult *res, gpointer data) {
    RefreshPart *pt = data;
    RefreshBatch *b = pt->batch;
    char *out = NULL, *err = NULL;
    GSubprocess *p = G_SUBPROCESS(src);
    g_subprocess_communicate_utf8_finish(p, res, &out, &err, NULL);
    b->out[pt->idx] = out ? out : g_strdup("");
    g_free(err);
    g_free(pt);
    g_object_unref(p);
    if (--b->left == 0) {
        if (!S.closing && b->gen == S.refresh_gen)
            apply_refresh(b->out[0], b->out[1], b->out[2]);
        for (int i = 0; i < 3; i++) g_free(b->out[i]);
        g_free(b);
    }
}

static void refresh_async(void) {
    if (S.closing) return;
    const char *const *cmds[3] = {CMD_RADIO, CMD_SAVED, CMD_LIST};
    RefreshBatch *b = g_new0(RefreshBatch, 1);
    b->left = 3;
    b->gen = ++S.refresh_gen;
    for (int i = 0; i < 3; i++) {
        GSubprocess *p = g_subprocess_newv(cmds[i],
            G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDERR_SILENCE, NULL);
        if (!p) {
            b->out[i] = g_strdup("");
            if (--b->left == 0) {
                for (int j = 0; j < 3; j++) g_free(b->out[j]);
                g_free(b);
            }
            continue;
        }
        RefreshPart *pt = g_new0(RefreshPart, 1);
        pt->batch = b;
        pt->idx = i;
        g_subprocess_communicate_utf8_async(p, NULL, NULL, refresh_part_done, pt);
    }
}

static GSubprocess *pre_p[3];

static void prespawn_refresh(void) {
    const char *const *cmds[3] = {CMD_RADIO, CMD_SAVED, CMD_LIST};
    for (int i = 0; i < 3; i++) {
        pre_p[i] = g_subprocess_newv(cmds[i],
            G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDERR_SILENCE, NULL);
    }
}

/* Parallel synchronous fetch (collects prespawned processes, or spawns all 3 concurrently). */
static void refresh(void) {
    if (S.closing || !showing(S.listpage)) return;
    const char *const *cmds[3] = {CMD_RADIO, CMD_SAVED, CMD_LIST};
    GSubprocess *p[3] = {NULL, NULL, NULL};
    char *out[3] = {NULL, NULL, NULL};
    for (int i = 0; i < 3; i++) {
        if (pre_p[i]) {
            p[i] = pre_p[i];
            pre_p[i] = NULL;
        } else {
            p[i] = g_subprocess_newv(cmds[i],
                G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDERR_SILENCE, NULL);
        }
    }
    for (int i = 0; i < 3; i++) {
        if (p[i]) {
            g_subprocess_communicate_utf8(p[i], NULL, NULL, &out[i], NULL, NULL);
            g_object_unref(p[i]);
        }
    }
    apply_refresh(out[0], out[1], out[2]);
    for (int i = 0; i < 3; i++) g_free(out[i]);
}

static void show_list(void) {
    wipe(&S.d_secret);
    g_free(S.sig);
    S.sig = NULL;
    gtk_stack_set_visible_child(GTK_STACK(S.stack), S.listpage);
    refresh_async();
}

static void scan_done(int ok, const char *out, const char *err, char *key) {
    S.scanning = FALSE;
    update_section();
    refresh_async();
}

static void scan(void) {
    if (!S.radio || !*S.ifname) return;
    S.scanning = TRUE;
    update_section();
    const char *argv[] = {"nmcli", "device", "wifi", "rescan", "ifname", S.ifname, NULL};
    run_async(argv, NULL, scan_done, NULL);
}

static void find_ifname(void) {
    if (*S.ifname) return;
    char *nm = g_find_program_in_path("nmcli");
    gboolean sys_nm = nm && (g_str_has_prefix(nm, "/usr/") || g_str_has_prefix(nm, "/bin/"));
    g_free(nm);
    if (sys_nm) {
        GDir *d = g_dir_open("/sys/class/net", 0, NULL);
        if (d) {
            const char *name;
            while ((name = g_dir_read_name(d))) {
                char p1[256], p2[256];
                g_snprintf(p1, sizeof p1, "/sys/class/net/%s/wireless", name);
                g_snprintf(p2, sizeof p2, "/sys/class/net/%s/phy80211", name);
                if (g_file_test(p1, G_FILE_TEST_EXISTS) || g_file_test(p2, G_FILE_TEST_EXISTS)) {
                    g_strlcpy(S.ifname, name, sizeof S.ifname);
                    break;
                }
            }
            g_dir_close(d);
        }
        if (*S.ifname) return;
    }
    const char *argv[] = {"nmcli", "-t", "-f", "DEVICE,TYPE", "device", NULL};
    char *out = run_out(argv);
    char **lines = g_strsplit(out ? out : "", "\n", -1);
    for (int i = 0; lines[i]; i++) {
        char **f = terse(lines[i], 2);
        if (!strcmp(f[1], "wifi") && !*S.ifname) g_strlcpy(S.ifname, f[0], sizeof S.ifname);
        g_strfreev(f);
    }
    g_strfreev(lines);
    g_free(out);
}

/* -- open / close ----------------------------------------------------------------- */
static void panel_close(void) {
    if (S.daemon) {
        if (!S.is_open) return;
        S.is_open = FALSE;
        wipe(&S.d_secret);
        ungrab();
        gtk_widget_hide(S.win);
        drop_backdrop(S.shade);
        S.shade = 0;
        if (S.detail) { gtk_widget_destroy(S.detail); S.detail = NULL; }
        if (S.join)   { gtk_widget_destroy(S.join);   S.join = NULL;   }
        gtk_stack_set_visible_child(GTK_STACK(S.stack), S.listpage);
        S.kb_nav = FALSE;
        unselect_all_lists();
        heap_trim();
        return;
    }
    if (S.closing) return;
    S.closing = TRUE;
    wipe(&S.d_secret);
    ungrab();
    gtk_widget_hide(S.win);
    drop_backdrop(S.shade);
    gtk_main_quit();
}

static void panel_open(gboolean pointer) {
    if (S.is_open) {
        panel_close();
        return;
    }
    S.pointer = pointer;
    S.is_open = TRUE;
    S.kb_nav = FALSE;
    close_bar_popups("wifi");
    wipe(&S.d_secret);
    if (S.detail) { gtk_widget_destroy(S.detail); S.detail = NULL; }
    if (S.join)   { gtk_widget_destroy(S.join);   S.join = NULL;   }
    gtk_stack_set_visible_child(GTK_STACK(S.stack), S.listpage);
    unselect_all_lists();

    anchor(S.pointer);
    gboolean was_warm = S.warm;
    if (!was_warm) {
        S.scanning = TRUE;
        refresh();
        S.scanning = FALSE;
    } else {
        fit();
    }
    gtk_widget_show(S.win);
    gdk_window_raise(gtk_widget_get_window(S.win));
    gdk_display_flush(gdk_display_get_default());
    S.shade = spawn_backdrop(S.below);
    gdk_window_raise(gtk_widget_get_window(S.win));
    grab_soon(S.win, G_CALLBACK(panel_close), NULL);
    if (was_warm) refresh_async();
    scan();
}

static void move_selection(int delta) {
    if (!S.rows || S.rows->len == 0) return;
    S.kb_nav = TRUE;
    GtkListBoxRow *cur = selected_row();
    int next = 0;
    if (cur) {
        int idx = 0;
        for (guint i = 0; i < S.rows->len; i++) {
            if (g_ptr_array_index(S.rows, i) == cur) { idx = (int)i; break; }
        }
        next = CLAMP(idx + delta, 0, (int)S.rows->len - 1);
    } else {
        next = delta >= 0 ? 0 : (int)S.rows->len - 1;
    }
    GtkListBoxRow *row = g_ptr_array_index(S.rows, next);
    select_only(row);
    /* Scroll into view if inside S.scroll. */
    if (gtk_widget_is_ancestor(GTK_WIDGET(row), S.scroll)) {
        GtkAdjustment *adj = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(S.scroll));
        GtkAllocation al;
        gtk_widget_get_allocation(GTK_WIDGET(row), &al);
        int wx = 0, wy = 0;
        gtk_widget_translate_coordinates(GTK_WIDGET(row), S.sections, 0, 0, &wx, &wy);
        double v = gtk_adjustment_get_value(adj), page = gtk_adjustment_get_page_size(adj);
        if (wy < v) gtk_adjustment_set_value(adj, wy);
        else if (wy + al.height > v + page) gtk_adjustment_set_value(adj, wy + al.height - page);
    }
}

static gboolean on_key(GtkWidget *_w, GdkEventKey *ev, gpointer _d) {
    guint k = ev->keyval;
    GtkWidget *focus = gtk_window_get_focus(GTK_WINDOW(S.win));
    gboolean typing = focus && GTK_IS_ENTRY(focus);
    gboolean on_list = showing(S.listpage);
    if (k == GDK_KEY_Escape) {
        if (on_list) panel_close();
        else show_list();
        return TRUE;
    }
    if (typing) return FALSE;
    if (!on_list && (k == GDK_KEY_Left || k == GDK_KEY_BackSpace)) { show_list(); return TRUE; }
    if (on_list && (k == GDK_KEY_Down || k == GDK_KEY_KP_Down)) { move_selection(1); return TRUE; }
    if (on_list && (k == GDK_KEY_Up || k == GDK_KEY_KP_Up)) { move_selection(-1); return TRUE; }
    if (on_list && (k == GDK_KEY_Return || k == GDK_KEY_KP_Enter)) {
        GtkListBoxRow *row = selected_row();
        if (!row && S.rows && S.rows->len > 0) row = g_ptr_array_index(S.rows, 0);
        if (row) on_activated(NULL, row, NULL);
        return TRUE;
    }
    if (on_list && k == GDK_KEY_Right) {
        GtkListBoxRow *row = selected_row();
        if (!row && S.rows && S.rows->len > 0) row = g_ptr_array_index(S.rows, 0);
        Net *n = row ? net_of(GTK_WIDGET(row)) : NULL;
        if (n) { show_detail(n->ssid); return TRUE; }
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
static gboolean on_term(gpointer _d) {
    if (S.daemon) {
        S.daemon = FALSE;
        const char *run = g_getenv("XDG_RUNTIME_DIR");
        char *fifo = g_build_filename(run && *run ? run : "/tmp", "wifi-panel.fifo", NULL);
        unlink(fifo);
        g_free(fifo);
    }
    panel_close();
    return G_SOURCE_REMOVE;
}
static void on_transition(GObject *_o, GParamSpec *_p, gpointer _d) {
    if (gtk_stack_get_transition_running(GTK_STACK(S.stack))) fit_follow(PAGE_ANIM_MS + 40);
    else fit();
}

/* -- build ------------------------------------------------------------------------ */
static char *css(const Palette *p) {
    /* The rice's fg at a few strengths; sentence case, the UI font only. */
    GString *c = g_string_new(NULL);
    g_string_append_printf(c, "@define-color fg %s;\n@define-color bg %s;\n", p->fg, p->bg);
    g_string_append_printf(c, "* { font-family: \"%s\"; }\n", p->font);
    g_string_append_printf(c,
        ".panel { background-color: @bg; border: 1px solid alpha(@fg, 0.10); border-radius: 14px; padding: %dpx; }\n",
        PAD);
    g_string_append(c,
        "window, stack { background: transparent; }\n"
        ".glyph { font-family: \"Material Design Icons Desktop\"; }\n"
        ".head { padding: 2px 4px 10px 4px; min-height: 28px; }\n"
        ".wifi-head { padding: 10px 12px; min-height: 26px; }\n"
        ".title { font-size: 11pt; font-weight: 600; color: @fg; }\n"
        ".page-title { font-size: 11.5pt; font-weight: 600; color: @fg; }\n"
        "switch { background-color: alpha(@fg, 0.14); border: none; border-radius: 999px; box-shadow: none;\n"
        "         min-width: 40px; min-height: 22px; background-image: none; outline: none; }\n"
        "switch:checked { background-color: @fg; }\n"
        "switch slider { background-color: @bg; border: none; border-radius: 999px; box-shadow: none;\n"
        "                min-width: 18px; min-height: 18px; margin: 2px; background-image: none; }\n"
        "switch:not(:checked) slider { background-color: alpha(@fg, 0.6); }\n"
        "switch image { color: transparent; }\n"
        ".section-head { padding: 12px 6px 5px 6px; }\n"
        ".section, .section-right { font-size: 9pt; color: alpha(@fg, 0.45); }\n"
        "scrolledwindow, viewport, list { background: transparent; border: none; }\n"
        "scrollbar { opacity: 0; }\n"
        "list.group { background-color: alpha(@fg, 0.05); border-radius: 10px; }\n"
        "list.group > row:not(:first-child) { border-top: 1px solid alpha(@fg, 0.06); }\n"
        "row.net { background: transparent; border-radius: 0px; padding: 9px 8px 9px 12px; outline: none;\n"
        "          transition: background-color 120ms ease-out; }\n"
        "list.group > row.net:first-child { border-top-left-radius: 10px; border-top-right-radius: 10px; }\n"
        "list.group > row.net:last-child { border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; }\n"
        "list.active-group > row.net { border-bottom-left-radius: 10px; border-bottom-right-radius: 10px; }\n"
        "row.net:hover { background-color: alpha(@fg, 0.05); }\n"
        "row.net:selected { background-color: alpha(@fg, 0.08); }\n"
        ".check { font-size: 15px; color: @fg; }\n"
        ".name { font-size: 10.2pt; color: alpha(@fg, 0.88); }\n"
        "row.active .name { color: @fg; font-weight: 600; }\n"
        "row.other .name { color: alpha(@fg, 0.72); }\n"
        ".sub { font-size: 9pt; color: alpha(@fg, 0.5); }\n"
        "row.busy .sub, row.error .sub { color: @fg; }\n"
        ".lock { font-size: 12px; color: alpha(@fg, 0.42); }\n"
        ".signal { color: @fg; }\n"
        "button { background-image: none; box-shadow: none; text-shadow: none; -gtk-icon-shadow: none;\n"
        "         border: none; outline: none; min-height: 0; min-width: 0; }\n"
        "button.info, button.eye, button.back { background: transparent; padding: 0px 4px; border-radius: 6px;\n"
        "         font-family: \"Material Design Icons Desktop\"; font-size: 16px; color: alpha(@fg, 0.45); }\n"
        "button.back { font-size: 20px; color: @fg; padding: 0px 2px; }\n"
        "button.info:hover, button.eye:hover { color: @fg; }\n"
        "button.back:hover { background-color: alpha(@fg, 0.08); }\n"
        ".group { background-color: alpha(@fg, 0.05); border-radius: 10px; }\n"
        ".grow { padding: 8px 12px; min-height: 24px; }\n"
        ".group > *:not(:first-child) { border-top: 1px solid alpha(@fg, 0.06); }\n"
        ".gkey { font-size: 10pt; color: @fg; }\n"
        ".gval { font-size: 10pt; color: alpha(@fg, 0.55); }\n"
        "button.action { background: transparent; padding: 10px 12px; border-radius: 0px;\n"
        "                font-size: 10pt; color: @fg; }\n"
        ".group > button.action:first-child { border-radius: 10px 10px 0px 0px; }\n"
        ".group > button.action:last-child { border-radius: 0px 0px 10px 10px; }\n"
        "button.action:hover { background-color: alpha(@fg, 0.05); }\n"
        "button.action.confirm { font-weight: 600; }\n"
        ".qr { padding: 4px 0px 12px 0px; }\n"
        ".hint { font-size: 9pt; color: alpha(@fg, 0.5); padding: 0px 2px; }\n"
        ".err { font-size: 9.5pt; color: @fg; padding: 0px 2px; }\n"
        ".field { background-color: alpha(@fg, 0.06); border-radius: 10px; padding: 3px 6px 3px 12px; }\n"
        "entry { background: none; border: none; box-shadow: none; outline: none; padding: 6px 0px;\n"
        "        min-height: 0px; font-size: 10.5pt; color: @fg; caret-color: @fg; }\n"
        "entry selection, label selection { background-color: alpha(@fg, 0.25); color: @fg; }\n"
        "button.primary { background-color: alpha(@fg, 0.16); border-radius: 10px; padding: 9px 14px;\n"
        "                 font-size: 10pt; font-weight: 600; color: @fg; }\n"
        "button.primary:hover { background-color: alpha(@fg, 0.22); }\n"
        "button.primary:disabled { background-color: alpha(@fg, 0.06); color: alpha(@fg, 0.35); }\n"
        ".off-hint { font-size: 9.5pt; color: alpha(@fg, 0.45); padding: 14px 6px 4px 6px; }\n");
    return g_string_free(c, FALSE);
}

static void build(void) {
    Palette p;
    palette_load(&p);
    char *c = css(&p);
    css_load(c);
    g_free(c);

    S.rows = g_ptr_array_new();

    S.win = overlay_window("WifiPanel", 0, 0, WIDTH, 100);
    gtk_widget_set_size_request(S.win, WIDTH, -1);
    gtk_window_set_default_size(GTK_WINDOW(S.win), -1, -1);
    gtk_window_set_resizable(GTK_WINDOW(S.win), TRUE);
    gtk_widget_add_events(S.win, GDK_KEY_PRESS_MASK | GDK_BUTTON_PRESS_MASK);
    g_signal_connect(S.win, "key-press-event", G_CALLBACK(on_key), NULL);
    g_signal_connect(S.win, "button-press-event", G_CALLBACK(on_outside), NULL);
    g_signal_connect(S.win, "size-allocate", G_CALLBACK(on_alloc), NULL);

    /* Top iOS group card: "Wi-Fi" + switch, and (when connected) the active network row. */
    S.top_group = group();
    GtkWidget *head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    add_class(head, "wifi-head");
    gtk_box_pack_start(GTK_BOX(head), label("Wi-Fi", "title", 0), TRUE, TRUE, 0);
    S.sw = gtk_switch_new();
    gtk_widget_set_valign(S.sw, GTK_ALIGN_CENTER);
    gtk_widget_set_can_focus(S.sw, FALSE);
    g_signal_connect(S.sw, "state-set", G_CALLBACK(on_switch), NULL);
    gtk_box_pack_end(GTK_BOX(head), S.sw, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.top_group), head, FALSE, FALSE, 0);

    S.off_label = label("Wi-Fi is off", "off-hint", 0);
    gtk_widget_set_no_show_all(S.off_label, TRUE);

    S.sections = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    S.scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(S.scroll), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_propagate_natural_height(GTK_SCROLLED_WINDOW(S.scroll), TRUE);
    gtk_scrolled_window_set_max_content_height(GTK_SCROLLED_WINDOW(S.scroll), LIST_MAX_H);
    gtk_container_add(GTK_CONTAINER(S.scroll), S.sections);

    S.listpage = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start(GTK_BOX(S.listpage), S.top_group, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.listpage), S.off_label, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.listpage), S.scroll, FALSE, FALSE, 0);

    S.stack = gtk_stack_new();
    gtk_stack_set_transition_type(GTK_STACK(S.stack), GTK_STACK_TRANSITION_TYPE_SLIDE_LEFT_RIGHT);
    gtk_stack_set_transition_duration(GTK_STACK(S.stack), PAGE_ANIM_MS);
    gtk_stack_set_vhomogeneous(GTK_STACK(S.stack), FALSE);
    gtk_stack_set_interpolate_size(GTK_STACK(S.stack), TRUE);
    gtk_stack_add_named(GTK_STACK(S.stack), S.listpage, "list");
    g_signal_connect(S.stack, "notify::transition-running", G_CALLBACK(on_transition), NULL);

    S.box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    add_class(S.box, "panel");
    gtk_box_pack_start(GTK_BOX(S.box), S.stack, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(S.win), S.box);
    gtk_widget_show_all(S.box);
    gtk_widget_realize(S.win);
}

/* -- daemon & single-instance ----------------------------------------------------- */
static char fifo_buf[1024];
static size_t fifo_len;

static gboolean on_fifo(gint fd, GIOCondition _c, gpointer _d) {
    ssize_t n = read(fd, fifo_buf + fifo_len, sizeof fifo_buf - 1 - fifo_len);
    if (n <= 0) return G_SOURCE_CONTINUE;
    fifo_len += n;
    fifo_buf[fifo_len] = 0;
    char *start = fifo_buf, *nl;
    while ((nl = strchr(start, '\n'))) {
        *nl = 0;
        if (g_str_has_prefix(start, "toggle")) {
            if (S.is_open) panel_close();
            else panel_open(strstr(start, "--pointer") != NULL);
        } else if (g_str_has_prefix(start, "open")) {
            if (!S.is_open) panel_open(strstr(start, "--pointer") != NULL);
        } else if (g_str_has_prefix(start, "close")) {
            panel_close();
        } else if (g_str_has_prefix(start, "quit")) {
            on_term(NULL);
        }
        start = nl + 1;
    }
    fifo_len = strlen(start);
    memmove(fifo_buf, start, fifo_len + 1);
    return G_SOURCE_CONTINUE;
}

static gboolean client_try_fifo(const char *cmd) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    char *fifo = g_build_filename(run && *run ? run : "/tmp", "wifi-panel.fifo", NULL);
    int fd = open(fifo, O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    g_free(fifo);
    if (fd < 0) return FALSE;
    char buf[128];
    int len = g_snprintf(buf, sizeof buf, "%s\n", cmd);
    ssize_t wr = write(fd, buf, (size_t)len);
    close(fd);
    return wr == len;
}

static int serve(void) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    if (!run || !*run) run = "/tmp";
    char *pidfile = g_build_filename(run, "wifi-panel.pid", NULL);
    char *fifo = g_build_filename(run, "wifi-panel.fifo", NULL);
    int lock = open(pidfile, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    g_free(pidfile);
    if (lock < 0 || flock(lock, LOCK_EX | LOCK_NB) != 0) { g_free(fifo); return 0; }
    if (ftruncate(lock, 0) == 0) {
        char t[32];
        int n = g_snprintf(t, sizeof t, "%d\n", (int)getpid());
        if (pwrite(lock, t, n, 0) < 0) { /* lock holds */ }
    }
    unlink(fifo);
    if (mkfifo(fifo, 0600) != 0 && errno != EEXIST) { g_free(fifo); return 1; }
    int fd = open(fifo, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) { unlink(fifo); g_free(fifo); return 1; }
    g_free(fifo);
    g_unix_fd_add(fd, G_IO_IN, on_fifo, NULL);
    anchor(FALSE);                               /* cache monitor rect & bar padding */
    refresh_async();                             /* warm the initial list in the background */
    heap_trim();
    gtk_main();
    return 0;
}

/* One panel: a second run closes the open one (second click on the icon). */
static int single(void) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    char *path = g_build_filename(run && *run ? run : "/tmp", "wifi-panel.pid", NULL);
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
        if (pwrite(fd, t, n, 0) < 0) { /* the lock still holds */ }
    }
    return 1;                                    /* fd stays open: the lock */
}

int main(int argc, char **argv) {
    gboolean daemon = FALSE, pointer = FALSE;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--daemon")) daemon = TRUE;
        else if (!strcmp(argv[i], "--pointer")) pointer = TRUE;
        else if (!strcmp(argv[i], "close")) {
            if (client_try_fifo("close")) return 0;
            return 0;
        }
    }
    if (!daemon) {
        /* Fast path (~0.3 ms): ask the resident daemon to toggle. */
        if (client_try_fifo(pointer ? "toggle --pointer" : "toggle")) return 0;
        if (!single()) return 0;
    }

    g_setenv("GDK_GL", "disable", FALSE);
    g_setenv("NO_AT_BRIDGE", "1", FALSE);
    g_setenv("GIO_USE_VFS", "local", FALSE);

    S.daemon = daemon;
    S.pointer = pointer;
    if (!daemon) {
        prespawn_refresh();                      /* overlap 3 nmcli calls with gtk_init + build */
        close_bar_popups("wifi");
    }
    gtk_env_lean();
    gtk_init(&argc, &argv);
    g_unix_signal_add(SIGTERM, on_term, NULL);
    find_ifname();
    build();
    if (daemon) return serve();
    panel_open(pointer);
    gtk_main();
    return 0;
}
