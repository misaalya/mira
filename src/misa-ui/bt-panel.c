/* BluetoothPanel - the Bluetooth panel of the bar (bin/BluetoothPanel [--pointer]).
 *
 * The Bluetooth screen of iOS (Jakob's law), built like WifiPanel:
 *
 *   list    "Bluetooth" + switch; "My devices" (paired): type icon, name,
 *           "Connected" / "Not connected", an info button. A click connects
 *           (or opens the details when it is connected already). "Other
 *           devices" (found by the scan that runs while the panel is open):
 *           a click pairs, trusts and connects. A second line only while
 *           something happens ("Connecting…", "Couldn't pair").
 *   detail  (info button, or →) Connect / Disconnect, Forget This Device
 *           (asks once more), then Name, Type, Battery, Address.
 *
 * Esc / ← / Backspace go back; Esc on the list (or a click outside) closes.
 * Everything goes through bluetoothctl, asynchronously, except the quick
 * reads. Scanning stops when the panel closes.
 *
 * Placement, backdrop, grab and the single-instance toggle as WifiPanel.
 * picom: class BluetoothPanel (the "Bar panels" rule).
 */
#include "common.h"

#include <glib-unix.h>

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#define WIDTH 380
#define PAD 14
#define GAP 6
#define LIST_MAX_H 460
#define SCAN_SECONDS "15"
#define ELLIPSIS "\xe2\x80\xa6"

/* Material Design Icons code points. */
enum {
    IC_BLUETOOTH = 0xF00AF, IC_HEADPHONES = 0xF02CB, IC_KEYBOARD = 0xF030C, IC_MOUSE = 0xF037D,
    IC_PHONE = 0xF011C, IC_LAPTOP = 0xF0322, IC_GAMEPAD = 0xF0297, IC_SPEAKER = 0xF04C3,
    IC_INFO = 0xF02FD, IC_BACK = 0xF0141,
};

typedef struct {
    char *mac, *name, *icon;
    gboolean paired, connected, busy;
    int battery;                     /* -1: unknown */
    char *status;
    GtkWidget *row, *sub;
} Dev;

static struct {
    GtkWidget *win, *box, *stack, *listpage, *detail;
    GtkWidget *sw, *list, *scroll, *empty;
    GtkWidget *d_status;
    GList *devs;
    char *sig;                       /* what the list shows, to skip rebuilds */
    gboolean avail, powered, scanning, pointer, closing, syncing, d_confirm;
    char *page_mac;
    GSubprocess *scan;
    guint scan_tick;
    Rect mon;
    char edge[8];
    int off, px, shade;
} S;

static void refresh(void);
static void show_list(void);
static void scan(void);

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

static GtkWidget *glyph(gunichar g, const char *cls) {
    char *t = u8(g);
    GtkWidget *l = gtk_label_new(t);
    g_free(t);
    add_class(l, "glyph");
    if (cls) add_class(l, cls);
    return l;
}

static GtkWidget *glyph_button(gunichar g, const char *cls) {
    char *t = u8(g);
    GtkWidget *b = gtk_button_new_with_label(t);
    g_free(t);
    add_class(b, cls);
    gtk_widget_set_can_focus(b, FALSE);
    return b;
}

static GtkWidget *label(const char *t, const char *cls, float xalign) {
    GtkWidget *l = gtk_label_new(t);
    gtk_label_set_xalign(GTK_LABEL(l), xalign);
    gtk_label_set_ellipsize(GTK_LABEL(l), PANGO_ELLIPSIZE_END);
    if (cls) add_class(l, cls);
    return l;
}

static gboolean showing(GtkWidget *page) {
    return page && gtk_stack_get_visible_child(GTK_STACK(S.stack)) == page;
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

static GSubprocess *run_async(const char *const *argv, Done cb, const char *key) {
    GSubprocess *p = g_subprocess_newv(argv, G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDERR_PIPE, NULL);
    if (!p) { if (cb) cb(FALSE, "", "", (char *)key); return NULL; }
    Job *j = g_new0(Job, 1);
    j->cb = cb;
    j->key = g_strdup(key);
    g_subprocess_communicate_utf8_async(p, NULL, NULL, job_done, j);
    return p;
}

static char *ctl(const char *a, const char *b) {
    const char *argv[] = {"bluetoothctl", a, b, NULL};
    return run_out(argv);
}

static Dev *dev_by_mac(const char *mac) {
    for (GList *l = S.devs; l && mac; l = l->next)
        if (!strcmp(((Dev *)l->data)->mac, mac)) return l->data;
    return NULL;
}

static Dev *dev_of(GtkWidget *row) {
    return row ? g_object_get_data(G_OBJECT(row), "dev") : NULL;
}

static gunichar type_icon(const char *icon) {
    if (!icon || !*icon) return IC_BLUETOOTH;
    if (strstr(icon, "headset") || strstr(icon, "headphones")) return IC_HEADPHONES;
    if (strstr(icon, "audio")) return IC_SPEAKER;
    if (strstr(icon, "keyboard")) return IC_KEYBOARD;
    if (strstr(icon, "mouse") || strstr(icon, "tablet")) return IC_MOUSE;
    if (strstr(icon, "gaming")) return IC_GAMEPAD;
    if (strstr(icon, "phone")) return IC_PHONE;
    if (strstr(icon, "computer")) return IC_LAPTOP;
    return IC_BLUETOOTH;
}

static const char *type_name(const char *icon) {
    if (!icon || !*icon) return NULL;
    if (strstr(icon, "headset")) return "Headset";
    if (strstr(icon, "headphones")) return "Headphones";
    if (strstr(icon, "audio")) return "Speaker";
    if (strstr(icon, "keyboard")) return "Keyboard";
    if (strstr(icon, "mouse")) return "Mouse";
    if (strstr(icon, "tablet")) return "Tablet";
    if (strstr(icon, "gaming")) return "Game controller";
    if (strstr(icon, "phone")) return "Phone";
    if (strstr(icon, "computer")) return "Computer";
    return NULL;
}

/* BlueZ names a device without a name after its address ("AA-BB-..."). */
static gboolean nameless(const char *name, const char *mac) {
    if (!*name) return TRUE;
    if (strlen(name) != 17) return FALSE;
    for (int i = 0; i < 17; i++)
        if ((i % 3 == 2) ? (name[i] != '-' && name[i] != ':') : !g_ascii_isxdigit(name[i])) return FALSE;
    return TRUE;
}

/* -- placement (as WifiPanel) ------------------------------------------------- */
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

/* -- page scaffolding ---------------------------------------------------------- */
static void go_back(GtkWidget *_w, gpointer _d) { show_list(); }

static GtkWidget *page_head(const char *title) {
    GtkWidget *back = glyph_button(IC_BACK, "back");
    gtk_widget_set_tooltip_text(back, "Back");
    g_signal_connect(back, "clicked", G_CALLBACK(go_back), NULL);
    GtkWidget *h = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    add_class(h, "head");
    gtk_box_pack_start(GTK_BOX(h), back, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(h), label(title, "page-title", 0), TRUE, TRUE, 0);
    return h;
}

static GtkWidget *group(void) {
    GtkWidget *g = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    add_class(g, "group");
    return g;
}

static void info_row(GtkWidget *grp, const char *key, const char *val) {
    if (!val || !*val) return;
    GtkWidget *r = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    add_class(r, "grow");
    gtk_box_pack_start(GTK_BOX(r), label(key, "gkey", 0), FALSE, FALSE, 0);
    GtkWidget *v = label(val, "gval", 1);
    gtk_label_set_selectable(GTK_LABEL(v), TRUE);
    gtk_widget_set_can_focus(v, FALSE);
    gtk_box_pack_end(GTK_BOX(r), v, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(grp), r, FALSE, FALSE, 0);
}

static GtkWidget *action_row(GtkWidget *grp, const char *text, GCallback cb) {
    GtkWidget *b = gtk_button_new_with_label(text);
    add_class(b, "action");
    gtk_widget_set_halign(gtk_bin_get_child(GTK_BIN(b)), GTK_ALIGN_START);
    g_signal_connect(b, "clicked", cb, NULL);
    gtk_box_pack_start(GTK_BOX(grp), b, FALSE, FALSE, 0);
    return b;
}

/* -- power ----------------------------------------------------------------------- */
static void update_switch(void) {
    S.syncing = TRUE;
    gtk_switch_set_active(GTK_SWITCH(S.sw), S.powered);
    gtk_switch_set_state(GTK_SWITCH(S.sw), S.powered);
    gtk_widget_set_sensitive(S.sw, S.avail);
    S.syncing = FALSE;
}

/* The switch keeps the position the user chose until the adapter really
 * is there (refresh() reads `pending`), so it never jumps back and forth. */
static gboolean pending, want;
static int settle_tries;

static gboolean adapter_on(void) {
    char *show = ctl("show", NULL);
    gboolean on = show && strstr(show, "Powered: yes") != NULL;
    g_free(show);
    return on;
}

static gboolean settle(gpointer _d) {
    if (S.closing) return G_SOURCE_REMOVE;
    if (adapter_on() != want && ++settle_tries < 25) return G_SOURCE_CONTINUE;
    pending = FALSE;
    g_free(S.sig);
    S.sig = NULL;
    refresh();
    if (S.powered) scan();
    return G_SOURCE_REMOVE;
}

static void power_done(int ok, const char *out, const char *err, char *key) {
    settle_tries = ok ? 0 : 99;                             /* failed: show the real state */
    g_timeout_add(300, settle, NULL);
}

static gboolean on_switch(GtkSwitch *sw, gboolean on, gpointer _d) {
    if (S.syncing) return FALSE;
    gtk_switch_set_state(sw, on);
    if (on == (pending ? want : S.powered)) return TRUE;
    pending = TRUE;
    want = on;
    g_free(S.sig);
    S.sig = NULL;
    refresh();                                              /* "Bluetooth is off" at once */
    if (on) {
        /* A soft rfkill block makes `power on` fail: lift it first. */
        const char *argv[] = {"sh", "-c", "rfkill unblock bluetooth 2>/dev/null; bluetoothctl power on", NULL};
        run_async(argv, power_done, NULL);
    } else {
        if (S.scan) g_subprocess_send_signal(S.scan, SIGTERM);
        const char *argv[] = {"bluetoothctl", "power", "off", NULL};
        run_async(argv, power_done, NULL);
    }
    return TRUE;
}

/* -- device actions ------------------------------------------------------------------ */
static void set_status(Dev *d, const char *status, gboolean busy) {
    if (!d) return;
    g_free(d->status);
    d->status = status ? g_strdup(status) : NULL;
    d->busy = busy;
    if (d->row && d->sub) {
        gtk_label_set_text(GTK_LABEL(d->sub), d->status ? d->status : "");
        gtk_widget_set_visible(d->sub, d->status != NULL);
        set_class(d->row, "busy", busy);
        set_class(d->row, "error", d->status && !busy);
    }
    g_free(S.sig);                   /* the next refresh redraws */
    S.sig = NULL;
    g_idle_add(fit_idle, NULL);
}

static void page_status(const char *t) {
    if (!showing(S.detail)) return;
    gtk_label_set_text(GTK_LABEL(S.d_status), t ? t : "");
    gtk_widget_set_visible(S.d_status, t != NULL);
    g_idle_add(fit_idle, NULL);
}

static void action_done(int ok, const char *out, const char *err, char *key) {
    Dev *d = dev_by_mac(key);
    const char *what = d && d->status && g_str_has_prefix(d->status, "Pairing") ? "Couldn't pair"
                     : d && d->status && g_str_has_prefix(d->status, "Disconnecting") ? "Couldn't disconnect"
                     : "Couldn't connect";
    if (ok) {
        set_status(d, NULL, FALSE);
        show_list();
        return;
    }
    set_status(d, what, FALSE);
    page_status(what);
}

static void do_connect(Dev *d) {
    if (!d || d->busy) return;
    set_status(d, "Connecting" ELLIPSIS, TRUE);
    page_status("Connecting" ELLIPSIS);
    const char *argv[] = {"bluetoothctl", "--timeout", "20", "connect", d->mac, NULL};
    run_async(argv, action_done, d->mac);
}

static void do_disconnect(Dev *d) {
    if (!d || d->busy) return;
    set_status(d, "Disconnecting" ELLIPSIS, TRUE);
    page_status("Disconnecting" ELLIPSIS);
    const char *argv[] = {"bluetoothctl", "disconnect", d->mac, NULL};
    run_async(argv, action_done, d->mac);
}

/* New device: pair, trust (so it reconnects by itself), connect. */
static void do_pair(Dev *d) {
    if (!d || d->busy) return;
    set_status(d, "Pairing" ELLIPSIS, TRUE);
    const char *argv[] = {"sh", "-c",
                          "bluetoothctl --timeout 30 pair \"$1\" && bluetoothctl trust \"$1\" && "
                          "bluetoothctl --timeout 20 connect \"$1\"",
                          "sh", d->mac, NULL};
    run_async(argv, action_done, d->mac);
}

/* -- detail page --------------------------------------------------------------------- */
static void on_conn(GtkWidget *_b, gpointer _d) {
    Dev *d = dev_by_mac(S.page_mac);
    if (d && d->connected) do_disconnect(d);
    else do_connect(d);
}

static void forget_done(int ok, const char *out, const char *err, char *key) {
    if (ok) show_list();
    else page_status("Couldn't forget the device");
}

static void on_forget(GtkWidget *b, gpointer _d) {
    if (!S.d_confirm) {
        S.d_confirm = TRUE;
        gtk_button_set_label(GTK_BUTTON(b), "Forget? Click again to confirm");
        gtk_widget_set_halign(gtk_bin_get_child(GTK_BIN(b)), GTK_ALIGN_START);
        add_class(b, "confirm");
        return;
    }
    page_status("Forgetting" ELLIPSIS);
    const char *argv[] = {"bluetoothctl", "remove", S.page_mac, NULL};
    run_async(argv, forget_done, S.page_mac);
}

static void show_detail(const char *mac) {
    Dev *d = dev_by_mac(mac);
    if (!d) return;
    char *keep = g_strdup(mac);
    g_free(S.page_mac);
    S.page_mac = keep;
    S.d_confirm = FALSE;
    GtkWidget *p = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start(GTK_BOX(p), page_head(d->name), FALSE, FALSE, 0);
    GtkWidget *body = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);

    S.d_status = label("", "err", 0);
    gtk_widget_set_no_show_all(S.d_status, TRUE);
    gtk_box_pack_start(GTK_BOX(body), S.d_status, FALSE, FALSE, 0);

    GtkWidget *g1 = group();
    action_row(g1, d->connected ? "Disconnect" : "Connect", G_CALLBACK(on_conn));
    action_row(g1, "Forget This Device", G_CALLBACK(on_forget));
    gtk_box_pack_start(GTK_BOX(body), g1, FALSE, FALSE, 0);

    GtkWidget *g2 = group();
    info_row(g2, "Name", d->name);
    info_row(g2, "Type", type_name(d->icon));
    if (d->battery >= 0) {
        char *bat = g_strdup_printf("%d%%", d->battery);
        info_row(g2, "Battery", bat);
        g_free(bat);
    }
    info_row(g2, "Status", d->connected ? "Connected" : "Not connected");
    info_row(g2, "Address", d->mac);
    gtk_box_pack_start(GTK_BOX(body), g2, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(p), body, FALSE, FALSE, 0);

    GtkWidget *old = S.detail;
    S.detail = p;
    gtk_widget_show_all(p);
    gtk_stack_add_named(GTK_STACK(S.stack), p, NULL);
    gtk_stack_set_visible_child(GTK_STACK(S.stack), p);
    if (old) gtk_widget_destroy(old);
    g_idle_add(fit_idle, NULL);
}

/* -- list page ---------------------------------------------------------------------- */
static void on_activated(GtkListBox *_l, GtkListBoxRow *row, gpointer _d) {
    Dev *d = dev_of(GTK_WIDGET(row));
    if (!d || d->busy) return;
    if (!d->paired) do_pair(d);
    else if (d->connected) show_detail(d->mac);
    else do_connect(d);
}

static void on_info(GtkWidget *_b, gpointer data) {
    show_detail(((Dev *)data)->mac);
}

/* A row of the iOS Bluetooth screen: the device type (small, no badge),
 * the name, "Connected" / "Not connected" and an info button for paired
 * devices; a second line only while something happens. */
static GtkWidget *make_row(Dev *d) {
    d->row = gtk_list_box_row_new();
    g_object_set_data(G_OBJECT(d->row), "dev", d);
    add_class(d->row, "dev");
    if (d->connected) add_class(d->row, "active");

    d->sub = label("", "sub", 0);
    gtk_widget_set_no_show_all(d->sub, TRUE);
    GtkWidget *texts = gtk_box_new(GTK_ORIENTATION_VERTICAL, 1);
    gtk_box_pack_start(GTK_BOX(texts), label(d->name, "name", 0), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(texts), d->sub, FALSE, FALSE, 0);
    gtk_widget_set_valign(texts, GTK_ALIGN_CENTER);

    GtkWidget *r = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    GtkWidget *ic = glyph(type_icon(d->icon), "type");
    gtk_widget_set_size_request(ic, 20, -1);
    gtk_box_pack_start(GTK_BOX(r), ic, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(r), texts, TRUE, TRUE, 0);
    if (d->paired) {
        gtk_box_pack_start(GTK_BOX(r), label(d->connected ? "Connected" : "Not connected", "state", 1),
                           FALSE, FALSE, 0);
        GtkWidget *info = glyph_button(IC_INFO, "info");
        gtk_widget_set_tooltip_text(info, "Details");
        g_signal_connect(info, "clicked", G_CALLBACK(on_info), d);
        gtk_box_pack_start(GTK_BOX(r), info, FALSE, FALSE, 0);
    }
    gtk_container_add(GTK_CONTAINER(d->row), r);
    if (d->status) {
        gtk_label_set_text(GTK_LABEL(d->sub), d->status);
        gtk_widget_show(d->sub);
        set_class(d->row, "busy", d->busy);
        set_class(d->row, "error", !d->busy);
    }
    return d->row;
}

/* "My devices" / "Other devices" (· Searching… while the scan runs). */
static void header(GtkListBoxRow *row, GtkListBoxRow *before, gpointer _d) {
    Dev *d = dev_of(GTK_WIDGET(row)), *b = before ? dev_of(GTK_WIDGET(before)) : NULL;
    if (!d || (b && b->paired == d->paired)) { gtk_list_box_row_set_header(row, NULL); return; }
    const char *t = d->paired ? "My devices"
                  : S.scanning ? "Other devices  \xc2\xb7  Searching" ELLIPSIS : "Other devices";
    GtkWidget *h = label(t, "section", 0);
    gtk_widget_show(h);
    gtk_list_box_row_set_header(row, h);
}

static void free_dev(Dev *d) {
    g_free(d->mac);
    g_free(d->name);
    g_free(d->icon);
    g_free(d->status);
    g_free(d);
}

static gint by_kind(gconstpointer a, gconstpointer b) {
    const Dev *x = a, *y = b;
    if (x->paired != y->paired) return y->paired - x->paired;
    if (x->connected != y->connected) return y->connected - x->connected;
    return g_utf8_collate(x->name, y->name);
}

/* "Device AA:BB:.. Name" lines -> mac set / list. */
static GHashTable *macs(const char *filter) {
    GHashTable *h = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
    char *out = ctl("devices", filter);
    char **lines = g_strsplit(out ? out : "", "\n", -1);
    for (int i = 0; lines[i]; i++) {
        if (!g_str_has_prefix(lines[i], "Device ") || strlen(lines[i]) < 24) continue;
        char *mac = g_strndup(lines[i] + 7, 17);
        const char *name = lines[i][24] == ' ' ? lines[i] + 25 : "";
        g_hash_table_insert(h, mac, g_strdup(name));
    }
    g_strfreev(lines);
    g_free(out);
    return h;
}

static void read_info(Dev *d) {
    char *out = ctl("info", d->mac);
    char **lines = g_strsplit(out ? out : "", "\n", -1);
    for (int i = 0; lines[i]; i++) {
        char *l = g_strstrip(lines[i]);
        if (g_str_has_prefix(l, "Icon: ")) { g_free(d->icon); d->icon = g_strdup(l + 6); }
        else if (g_str_has_prefix(l, "Alias: ") && l[7]) { g_free(d->name); d->name = g_strdup(l + 7); }
        else if (g_str_has_prefix(l, "Connected: ")) d->connected = !strcmp(l + 11, "yes");
        else if (g_str_has_prefix(l, "Battery Percentage: ")) {
            const char *par = strchr(l, '(');
            if (par) d->battery = atoi(par + 1);
        }
    }
    g_strfreev(lines);
    g_free(out);
}

static void refresh(void) {
    if (S.closing || !showing(S.listpage)) return;
    char *show = ctl("show", NULL);
    S.avail = show && strstr(show, "Controller ") != NULL;
    S.powered = S.avail && (pending ? want : strstr(show, "Powered: yes") != NULL);
    g_free(show);
    update_switch();

    /* Keep busy states across the rebuild. */
    GHashTable *keep = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);
    GHashTable *keep_busy = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    for (GList *l = S.devs; l; l = l->next) {
        Dev *d = l->data;
        if (d->status) {
            g_hash_table_insert(keep, g_strdup(d->mac), g_strdup(d->status));
            if (d->busy) g_hash_table_add(keep_busy, g_strdup(d->mac));
        }
    }

    GList *devs = NULL;
    if (S.powered) {
        GHashTable *all = macs(NULL), *paired = macs("Paired");
        GHashTableIter it;
        gpointer k, v;
        g_hash_table_iter_init(&it, all);
        while (g_hash_table_iter_next(&it, &k, &v)) {
            gboolean pd = g_hash_table_contains(paired, k);
            if (!pd && nameless(v, k)) continue;
            Dev *d = g_new0(Dev, 1);
            d->mac = g_strdup(k);
            d->name = g_strdup(v);
            d->paired = pd;
            d->battery = -1;
            if (pd) read_info(d);
            const char *st = g_hash_table_lookup(keep, k);
            if (st) { d->status = g_strdup(st); d->busy = g_hash_table_contains(keep_busy, k); }
            devs = g_list_prepend(devs, d);
        }
        g_hash_table_destroy(all);
        g_hash_table_destroy(paired);
    }
    g_hash_table_destroy(keep);
    g_hash_table_destroy(keep_busy);
    devs = g_list_sort(devs, by_kind);

    /* Rebuild only when something changed (the scan refreshes often). */
    GString *sig = g_string_new(S.scanning ? "s" : "-");
    for (GList *l = devs; l; l = l->next) {
        Dev *d = l->data;
        g_string_append_printf(sig, "|%s %s %s %d%d%d %d %s", d->mac, d->name, d->icon ? d->icon : "",
                               d->paired, d->connected, d->busy, d->battery, d->status ? d->status : "");
    }
    if (S.sig && !strcmp(S.sig, sig->str)) {
        g_string_free(sig, TRUE);
        g_list_free_full(devs, (GDestroyNotify)free_dev);
        return;
    }
    g_free(S.sig);
    S.sig = g_string_free(sig, FALSE);

    char *sel_mac = NULL;
    GtkListBoxRow *selrow = gtk_list_box_get_selected_row(GTK_LIST_BOX(S.list));
    if (selrow && dev_of(GTK_WIDGET(selrow))) sel_mac = g_strdup(dev_of(GTK_WIDGET(selrow))->mac);
    GList *kids = gtk_container_get_children(GTK_CONTAINER(S.list));
    for (GList *l = kids; l; l = l->next) gtk_widget_destroy(l->data);
    g_list_free(kids);
    g_list_free_full(S.devs, (GDestroyNotify)free_dev);
    S.devs = devs;

    GtkListBoxRow *pick = NULL;
    for (GList *l = S.devs; l; l = l->next) {
        Dev *d = l->data;
        gtk_container_add(GTK_CONTAINER(S.list), make_row(d));
        gtk_widget_show_all(d->row);
        if (d->sub && !d->status) gtk_widget_hide(d->sub);
        if (sel_mac && !strcmp(sel_mac, d->mac)) pick = GTK_LIST_BOX_ROW(d->row);
    }
    g_free(sel_mac);
    if (!pick && S.devs) pick = GTK_LIST_BOX_ROW(((Dev *)S.devs->data)->row);
    if (pick) {
        gtk_list_box_select_row(GTK_LIST_BOX(S.list), pick);
        gtk_widget_grab_focus(GTK_WIDGET(pick));
    }
    gtk_label_set_text(GTK_LABEL(S.empty), !S.avail ? "No Bluetooth adapter"
                                         : !S.powered ? "Bluetooth is off"
                                         : S.scanning ? "Searching" ELLIPSIS : "No devices found");
    gtk_list_box_invalidate_headers(GTK_LIST_BOX(S.list));
    fit();
}

static void show_list(void) {
    gtk_stack_set_visible_child(GTK_STACK(S.stack), S.listpage);
    g_free(S.sig);
    S.sig = NULL;
    refresh();
}

/* -- scanning: while the panel is open, as a phone does --------------------------------- */
static gboolean scan_tick(gpointer _d) {
    refresh();
    return G_SOURCE_CONTINUE;
}

static gboolean rescan(gpointer _d) { scan(); return G_SOURCE_REMOVE; }

static void scan_done(int ok, const char *out, const char *err, char *key) {
    S.scan = NULL;
    S.scanning = FALSE;
    if (S.scan_tick) { g_source_remove(S.scan_tick); S.scan_tick = 0; }
    refresh();
    if (S.powered) g_timeout_add_seconds(4, rescan, NULL);
}

static void scan(void) {
    if (S.scan || !S.powered || S.closing) return;
    const char *argv[] = {"bluetoothctl", "--timeout", SCAN_SECONDS, "scan", "on", NULL};
    S.scan = run_async(argv, scan_done, NULL);
    if (!S.scan) return;
    S.scanning = TRUE;
    S.scan_tick = g_timeout_add(1500, scan_tick, NULL);
    refresh();
}

/* -- open / close ----------------------------------------------------------------- */
static void panel_close(void) {
    if (S.closing) return;
    S.closing = TRUE;
    if (S.scan) g_subprocess_send_signal(S.scan, SIGTERM);   /* stops discovery */
    ungrab();
    gtk_widget_hide(S.win);
    drop_backdrop(S.shade);
    gtk_main_quit();
}

static gboolean on_key(GtkWidget *_w, GdkEventKey *ev, gpointer _d) {
    guint k = ev->keyval;
    gboolean on_list = showing(S.listpage);
    if (k == GDK_KEY_Escape) {
        if (on_list) panel_close();
        else show_list();
        return TRUE;
    }
    if (!on_list && (k == GDK_KEY_Left || k == GDK_KEY_BackSpace)) { show_list(); return TRUE; }
    if (on_list && k == GDK_KEY_Right) {
        GtkListBoxRow *row = gtk_list_box_get_selected_row(GTK_LIST_BOX(S.list));
        Dev *d = row ? dev_of(GTK_WIDGET(row)) : NULL;
        if (d && d->paired) { show_detail(d->mac); return TRUE; }
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
static void on_transition(GObject *_o, GParamSpec *_p, gpointer _d) { fit(); }

/* -- build ------------------------------------------------------------------------ */
static char *css(const Palette *p) {
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
        ".title { font-size: 13pt; font-weight: 600; color: @fg; }\n"
        ".page-title { font-size: 11.5pt; font-weight: 600; color: @fg; }\n"
        "switch { background-color: alpha(@fg, 0.14); border: none; border-radius: 999px; box-shadow: none;\n"
        "         min-width: 40px; min-height: 22px; background-image: none; outline: none; }\n"
        "switch:checked { background-color: @fg; }\n"
        "switch slider { background-color: @bg; border: none; border-radius: 999px; box-shadow: none;\n"
        "                min-width: 18px; min-height: 18px; margin: 2px; background-image: none; }\n"
        "switch:not(:checked) slider { background-color: alpha(@fg, 0.6); }\n"
        "switch:disabled { opacity: 0.4; }\n"
        "switch image { color: transparent; }\n"
        ".section { font-size: 9pt; color: alpha(@fg, 0.45); padding: 10px 6px 4px 6px; }\n"
        "scrolledwindow, viewport, list { background: transparent; border: none; }\n"
        "scrollbar { opacity: 0; }\n"
        "row.dev { background: transparent; border-radius: 10px; padding: 9px 4px 9px 6px; outline: none;\n"
        "          transition: background-color 120ms ease-out; }\n"
        "row.dev:hover { background-color: alpha(@fg, 0.05); }\n"
        "row.dev:selected { background-color: alpha(@fg, 0.08); }\n"
        ".type { font-size: 16px; color: alpha(@fg, 0.55); }\n"
        "row.active .type { color: @fg; }\n"
        ".name { font-size: 10.5pt; color: alpha(@fg, 0.88); }\n"
        "row.active .name { color: @fg; font-weight: 600; }\n"
        ".sub { font-size: 9pt; color: alpha(@fg, 0.5); }\n"
        "row.busy .sub, row.error .sub { color: @fg; }\n"
        ".state { font-size: 9.5pt; color: alpha(@fg, 0.45); }\n"
        "row.active .state { color: alpha(@fg, 0.75); }\n"
        "button { background-image: none; box-shadow: none; text-shadow: none; -gtk-icon-shadow: none;\n"
        "         border: none; outline: none; min-height: 0; min-width: 0; }\n"
        "button.info, button.back { background: transparent; padding: 0px 4px; border-radius: 6px;\n"
        "         font-family: \"Material Design Icons Desktop\"; font-size: 16px; color: alpha(@fg, 0.45); }\n"
        "button.back { font-size: 20px; color: @fg; padding: 0px 2px; }\n"
        "button.info:hover { color: @fg; }\n"
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
        "label selection { background-color: alpha(@fg, 0.25); color: @fg; }\n"
        ".err { font-size: 9.5pt; color: @fg; padding: 0px 2px; }\n"
        ".empty { font-size: 10pt; color: alpha(@fg, 0.5); padding: 18px; }\n");
    return g_string_free(c, FALSE);
}

static void build(void) {
    Palette p;
    palette_load(&p);
    char *c = css(&p);
    css_load(c);
    g_free(c);

    S.win = overlay_window("BluetoothPanel", 0, 0, WIDTH, 100);
    gtk_widget_set_size_request(S.win, WIDTH, -1);
    gtk_window_set_default_size(GTK_WINDOW(S.win), -1, -1);
    gtk_window_set_resizable(GTK_WINDOW(S.win), TRUE);
    gtk_widget_add_events(S.win, GDK_KEY_PRESS_MASK | GDK_BUTTON_PRESS_MASK);
    g_signal_connect(S.win, "key-press-event", G_CALLBACK(on_key), NULL);
    g_signal_connect(S.win, "button-press-event", G_CALLBACK(on_outside), NULL);
    g_signal_connect(S.win, "size-allocate", G_CALLBACK(on_alloc), NULL);

    GtkWidget *head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    add_class(head, "head");
    gtk_box_pack_start(GTK_BOX(head), label("Bluetooth", "title", 0), TRUE, TRUE, 0);
    S.sw = gtk_switch_new();
    gtk_widget_set_valign(S.sw, GTK_ALIGN_CENTER);
    gtk_widget_set_can_focus(S.sw, FALSE);
    g_signal_connect(S.sw, "state-set", G_CALLBACK(on_switch), NULL);
    gtk_box_pack_end(GTK_BOX(head), S.sw, FALSE, FALSE, 0);

    S.list = gtk_list_box_new();
    gtk_list_box_set_selection_mode(GTK_LIST_BOX(S.list), GTK_SELECTION_BROWSE);
    gtk_list_box_set_activate_on_single_click(GTK_LIST_BOX(S.list), TRUE);
    gtk_list_box_set_header_func(GTK_LIST_BOX(S.list), header, NULL, NULL);
    g_signal_connect(S.list, "row-activated", G_CALLBACK(on_activated), NULL);
    S.empty = gtk_label_new("");
    add_class(S.empty, "empty");
    gtk_list_box_set_placeholder(GTK_LIST_BOX(S.list), S.empty);
    gtk_widget_show(S.empty);
    S.scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(S.scroll), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_propagate_natural_height(GTK_SCROLLED_WINDOW(S.scroll), TRUE);
    gtk_scrolled_window_set_max_content_height(GTK_SCROLLED_WINDOW(S.scroll), LIST_MAX_H);
    gtk_container_add(GTK_CONTAINER(S.scroll), S.list);

    S.listpage = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start(GTK_BOX(S.listpage), head, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.listpage), S.scroll, FALSE, FALSE, 0);

    S.stack = gtk_stack_new();
    gtk_stack_set_transition_type(GTK_STACK(S.stack), GTK_STACK_TRANSITION_TYPE_SLIDE_LEFT_RIGHT);
    gtk_stack_set_transition_duration(GTK_STACK(S.stack), 180);
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

static int single(void) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    char *path = g_build_filename(run && *run ? run : "/tmp", "bt-panel.pid", NULL);
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
    return 1;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--pointer")) S.pointer = TRUE;
    if (!single()) return 0;
    close_bar_popups("bt");
    gtk_init(&argc, &argv);
    g_unix_signal_add(SIGTERM, on_term, NULL);
    build();
    anchor(S.pointer);
    refresh();
    S.shade = backdrop(S.mon, 0, NULL);
    gtk_widget_show(S.win);
    gdk_window_raise(gtk_widget_get_window(S.win));
    grab_soon(S.win, G_CALLBACK(panel_close), NULL);
    scan();
    gtk_main();
    return 0;
}
