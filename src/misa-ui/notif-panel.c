/* NotifPanel - ultra-clean Apple-style notification daemon & Notification Center
 * for the BSPWM rice (bin/NotifPanel).
 *
 * Design principles (Mira monochrome + Apple/iOS clarity):
 *   - Radical reduction: no box-in-box nesting, no loud borders, no redundant
 *     badges or stacked indicator pills.
 *   - Banners: slim 348 px black cards with a quiet 1 px hairline border,
 *     16 px radius, clean 34 px icon, two-line typography (SemiBold title +
 *     muted body), subtle "now" fading to a bare × on hover, and a 4 px pill
 *     progress bar with right-aligned percentage for Brightness / Volume OSD.
 *   - Notification Center: same visual family as WifiPanel & BluetoothPanel —
 *     "Notifications" + plain "Clear" text action, borderless "Do Not Disturb"
 *     row with the iOS switch, a single hairline divider, and flat rows that
 *     highlight softly on hover.
 *
 * Modes:
 *   NotifPanel --daemon             owns org.freedesktop.Notifications on D-Bus
 *   NotifPanel [--pointer]          toggles the Notification Center (also `toggle`)
 *   NotifPanel open [--pointer]     opens the Notification Center
 *   NotifPanel close                closes the Notification Center
 *   NotifPanel close-all            dismisses all visible banners
 *   NotifPanel clear                clears notification history
 *   NotifPanel dnd [on|off|toggle]  controls Do Not Disturb
 *   NotifPanel reload               reloads theme from shared.rasi
 *   NotifPanel status               prints "<count> <dnd>" (<1 ms, no GTK)
 *   NotifPanel quit                 stops the daemon cleanly
 *
 * Performance:
 *   - CLI subcommands finish BEFORE gtk_init() (<1 ms, zero GTK overhead).
 *   - GDK_GL=disable, NO_AT_BRIDGE=1, GIO_USE_VFS=local, direct filesystem
 *     icon lookup (no GtkIconTheme cache mmap), mallopt(M_ARENA_MAX, 1).
 *   - 0 polling timers while idle; polybar updates via IPC hooks.
 */
#include "common.h"

#include <gio/gio.h>
#include <glib-unix.h>

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* -- layout constants -------------------------------------------------------- */
#define TOAST_W        348
#define TOAST_GAP      8
#define TOAST_MARGIN_X 14
#define TOAST_MARGIN_Y 10
#define MAX_TOASTS     3
#define HISTORY_MAX    50
#define ICON_CACHE_MAX 32
#define ICON_SZ        34
#define NC_ICON_SZ     32
#define THUMB_SZ       40
#define NC_WIDTH       380
#define NC_PAD         14
#define NC_GAP         6
#define NC_LIST_MAX_H  460
#define GLIDE_MS       220

#define DEFAULT_TIMEOUT_LOW    3500
#define DEFAULT_TIMEOUT_NORMAL 5000
#define HOVER_RESUME_MIN_MS    1600

/* Material Design Icons Desktop code points. */
enum {
    IC_BELL     = 0xF009C,
    IC_CLOSE    = 0xF0156,
    IC_ALERT    = 0xF05D6,
    IC_CAMERA   = 0xF0E51,
};

/* -- D-Bus introspection XML (Desktop Notifications Spec 1.2) ---------------- */
static const char INTROSPECTION_XML[] =
    "<node>"
    "  <interface name='org.freedesktop.Notifications'>"
    "    <method name='GetCapabilities'>"
    "      <arg name='capabilities' type='as' direction='out'/>"
    "    </method>"
    "    <method name='Notify'>"
    "      <arg name='app_name' type='s' direction='in'/>"
    "      <arg name='replaces_id' type='u' direction='in'/>"
    "      <arg name='app_icon' type='s' direction='in'/>"
    "      <arg name='summary' type='s' direction='in'/>"
    "      <arg name='body' type='s' direction='in'/>"
    "      <arg name='actions' type='as' direction='in'/>"
    "      <arg name='hints' type='a{sv}' direction='in'/>"
    "      <arg name='expire_timeout' type='i' direction='in'/>"
    "      <arg name='id' type='u' direction='out'/>"
    "    </method>"
    "    <method name='CloseNotification'>"
    "      <arg name='id' type='u' direction='in'/>"
    "    </method>"
    "    <method name='GetServerInformation'>"
    "      <arg name='name' type='s' direction='out'/>"
    "      <arg name='vendor' type='s' direction='out'/>"
    "      <arg name='version' type='s' direction='out'/>"
    "      <arg name='spec_version' type='s' direction='out'/>"
    "    </method>"
    "    <signal name='NotificationClosed'>"
    "      <arg name='id' type='u'/>"
    "      <arg name='reason' type='u'/>"
    "    </signal>"
    "    <signal name='ActionInvoked'>"
    "      <arg name='id' type='u'/>"
    "      <arg name='action_key' type='s'/>"
    "    </signal>"
    "  </interface>"
    "</node>";

/* -- data types -------------------------------------------------------------- */
typedef struct {
    char *key;
    char *label;
} Action;

typedef struct {
    int refcount;
    guint32 id;
    char *app_name;
    char *summary;
    char *body;
    char *icon_name;      /* app_icon, resolved desktop icon, or file path */
    char *image_path;     /* separate attachment file path */
    GdkPixbuf *raw_icon;  /* small pre-scaled pixbuf only for D-Bus image-data */
    GdkPixbuf *raw_thumb; /* small pre-scaled attachment only for D-Bus image-data */
    char *stack_tag;
    guint8 urgency;       /* 0=low, 1=normal, 2=critical */
    int progress;         /* -1 if none, else 0..100 */
    gboolean transient;
    gboolean resident;
    gint64 created_at;
    int timeout_ms;
    Action *actions;
    int n_actions;
    gboolean has_default;
} Notif;

typedef struct {
    Notif *n;
    GtkWidget *win;
    GtkWidget *card;
    GtkWidget *icon_slot;
    GtkWidget *thumb_slot;
    GtkWidget *crit_dot;
    GtkWidget *title_lbl;
    GtkWidget *body_lbl;
    GtkWidget *time_stack;
    GtkWidget *time_lbl;
    GtkWidget *prog_bar;
    GtkWidget *actions_box;
    guint expire_src;
    gint64 remaining_ms;
    gint64 timer_start_us;
    gboolean hovered;
    /* Geometry & Y glide */
    int x, y, h;
    int anim_from_y, anim_to_y;
    gint64 anim_t0;
    gboolean gliding;
    guint tick_id;
} Toast;

static struct {
    /* Theme */
    Palette pal;
    char urgent[16];
    char icon_theme_name[128];
    GtkCssProvider *css_prov;
    GHashTable *icon_cache;

    /* D-Bus */
    GDBusConnection *bus;
    guint bus_owner_id;
    guint bus_reg_id;
    GDBusNodeInfo *introspection;
    guint32 next_id;

    /* Notifications state */
    GList *toasts;         /* Toast*, newest first */
    GList *history;        /* Notif*, newest first */
    int history_len;
    gboolean dnd;

    /* Notification Center */
    gboolean nc_open;
    gboolean nc_pointer;
    gboolean nc_syncing;
    GtkWidget *nc_win;
    GtkWidget *nc_box;
    GtkWidget *nc_clear_btn;
    GtkWidget *nc_dnd_sw;
    GtkWidget *nc_list;
    GtkWidget *nc_empty;
    GtkWidget *nc_scroll;
    guint nc_time_src;
    Rect nc_mon;
    char nc_edge[8];
    int nc_off, nc_px, nc_shade;

    /* Runtime paths & timers */
    char state_path[4096];
    guint bar_poke_src;
    guint trim_src;
    int lock_fd;
    int fifo_fd;
    char fifo_buf[4096];
    size_t fifo_len;
} S;

/* -- forward declarations ---------------------------------------------------- */
static void nc_open(gboolean pointer);
static void nc_close(void);
static void nc_rebuild_list(void);
static void nc_fit(void);
static void toast_close(Toast *t, guint32 reason, gboolean animate_rest);
static void toasts_layout(gboolean animate);
static void schedule_bar_sync(void);
static void schedule_trim(void);

/* -- small utilities --------------------------------------------------------- */
static const char *runtime_dir(void) {
    const char *r = g_getenv("XDG_RUNTIME_DIR");
    return (r && *r) ? r : "/tmp";
}

static void add_class(GtkWidget *w, const char *c) {
    gtk_style_context_add_class(gtk_widget_get_style_context(w), c);
}

static char *u8_glyph(gunichar cp) {
    char *s = g_malloc0(8);
    g_unichar_to_utf8(cp, s);
    return s;
}

static GtkWidget *make_glyph(gunichar cp, const char *cls) {
    char *t = u8_glyph(cp);
    GtkWidget *l = gtk_label_new(t);
    g_free(t);
    add_class(l, "glyph");
    if (cls) add_class(l, cls);
    return l;
}

static void clear_container(GtkWidget *box) {
    GList *kids = gtk_container_get_children(GTK_CONTAINER(box));
    for (GList *l = kids; l; l = l->next) gtk_widget_destroy(GTK_WIDGET(l->data));
    g_list_free(kids);
}

static gboolean is_generic_app(const char *app) {
    if (!app || !*app) return TRUE;
    return !g_ascii_strcasecmp(app, "notify-send") ||
           !g_ascii_strcasecmp(app, "dunstify") ||
           !g_ascii_strcasecmp(app, "dunst");
}

/* Strip lightweight HTML markup and decode common XML entities so body text
 * always renders in crisp, uniform Inter. */
static char *clean_text(const char *src, gboolean single_line) {
    if (!src || !*src) return g_strdup("");
    GString *out = g_string_sized_new(strlen(src) + 1);
    const char *p = src;
    while (*p) {
        if (*p == '<') {
            const char *end = strchr(p + 1, '>');
            if (end && (end - p) < 256) {
                if ((end - p == 3 && g_ascii_strncasecmp(p + 1, "br", 2) == 0) ||
                    (end - p == 4 && g_ascii_strncasecmp(p + 1, "br/", 3) == 0) ||
                    (end - p == 5 && g_ascii_strncasecmp(p + 1, "br /", 4) == 0)) {
                    g_string_append_c(out, single_line ? ' ' : '\n');
                }
                p = end + 1;
                continue;
            }
        }
        if (*p == '&') {
            if (g_str_has_prefix(p, "&amp;"))  { g_string_append_c(out, '&');  p += 5; continue; }
            if (g_str_has_prefix(p, "&lt;"))   { g_string_append_c(out, '<');  p += 4; continue; }
            if (g_str_has_prefix(p, "&gt;"))   { g_string_append_c(out, '>');  p += 4; continue; }
            if (g_str_has_prefix(p, "&quot;")) { g_string_append_c(out, '"');  p += 6; continue; }
            if (g_str_has_prefix(p, "&apos;")) { g_string_append_c(out, '\''); p += 6; continue; }
            if (g_str_has_prefix(p, "&#39;"))  { g_string_append_c(out, '\''); p += 5; continue; }
            if (g_str_has_prefix(p, "&nbsp;")) { g_string_append_c(out, ' ');  p += 6; continue; }
        }
        if (*p == '\r') { p++; continue; }
        if (*p == '\n') {
            if (single_line) {
                if (out->len && out->str[out->len - 1] != ' ') g_string_append_c(out, ' ');
            } else {
                if (!(out->len >= 1 && out->str[out->len - 1] == '\n'))
                    g_string_append_c(out, '\n');
            }
            p++;
            continue;
        }
        g_string_append_c(out, *p++);
    }
    char *s = g_string_free(out, FALSE);
    g_strstrip(s);
    if (!g_utf8_validate(s, -1, NULL)) {
        char *valid = g_utf8_make_valid(s, -1);
        g_free(s);
        s = valid;
    }
    return s;
}

/* Compact Apple-style relative time: "now", "2m", "1h", "1d". */
static void fmt_relative(gint64 created_at, char out[32]) {
    gint64 dt = (gint64)time(NULL) - created_at;
    if (dt < 45)         g_strlcpy(out, "now", 32);
    else if (dt < 3600)  g_snprintf(out, 32, "%ldm", (long)((dt + 15) / 60));
    else if (dt < 86400) g_snprintf(out, 32, "%ldh", (long)(dt / 3600));
    else                 g_snprintf(out, 32, "%ldd", (long)(dt / 86400));
}

/* For OSD progress titles like "Brightness 72%" or "Volume 65%", strip the
 * trailing percentage so the title sits cleanly on the left and "72%" aligns
 * neatly on the right. */
static void split_osd_title(const char *title, int progress, char out_title[128], char out_pct[16]) {
    g_snprintf(out_pct, 16, "%d%%", CLAMP(progress, 0, 100));
    g_strlcpy(out_title, title, 128);
    size_t len = strlen(out_title);
    if (len >= 2 && out_title[len - 1] == '%') {
        size_t i = len - 1;
        while (i > 0 && isdigit((unsigned char)out_title[i - 1])) i--;
        if (i < len - 1) {
            while (i > 0 && (out_title[i - 1] == ' ' || out_title[i - 1] == ':')) i--;
            if (i > 0) out_title[i] = 0;
        }
    }
}

/* -- Notif refcounting ------------------------------------------------------- */
static Notif *notif_ref(Notif *n) {
    if (n) n->refcount++;
    return n;
}

static void notif_unref(Notif *n) {
    if (!n || --n->refcount > 0) return;
    g_free(n->app_name);
    g_free(n->summary);
    g_free(n->body);
    g_free(n->icon_name);
    g_free(n->image_path);
    g_free(n->stack_tag);
    if (n->raw_icon) g_object_unref(n->raw_icon);
    if (n->raw_thumb) g_object_unref(n->raw_thumb);
    for (int i = 0; i < n->n_actions; i++) {
        g_free(n->actions[i].key);
        g_free(n->actions[i].label);
    }
    g_free(n->actions);
    g_free(n);
}

/* -- palette & CSS ----------------------------------------------------------- */
static gboolean rasi_str(const char *text, const char *key, char *out, size_t n) {
    char pat[128];
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

static void load_theme(void) {
    palette_load(&S.pal);
    g_strlcpy(S.urgent, "#e5675f", sizeof S.urgent);
    g_strlcpy(S.icon_theme_name, "Papirus-Dark", sizeof S.icon_theme_name);

    char *path = mira_config_path("config/rofi/shared.rasi");
    char *text = NULL;
    if (g_file_get_contents(path, &text, NULL, NULL) && text) {
        char v[160];
        if (rasi_str(text, "urgent", v, sizeof v) && v[0] == '#' && strlen(v) >= 7)
            g_strlcpy(S.urgent, v, 8);
        if (rasi_str(text, "icon-theme", v, sizeof v) && *v)
            g_strlcpy(S.icon_theme_name, v, sizeof S.icon_theme_name);
        g_free(text);
    }
    g_free(path);

    if (S.icon_cache) g_hash_table_remove_all(S.icon_cache);
    else S.icon_cache = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, g_free);

    GString *c = g_string_new(NULL);
    g_string_append_printf(c,
        "@define-color bg %s;\n"
        "@define-color fg %s;\n"
        "@define-color urgent %s;\n"
        "* { font-family: \"%s\"; }\n",
        S.pal.bg, S.pal.fg, S.urgent, S.pal.font);
    g_string_append_printf(c,
        "window, stack, scrolledwindow, viewport { background: transparent; border: none; }\n"
        "scrollbar { opacity: 0; }\n"
        ".glyph { font-family: \"Material Design Icons Desktop\"; }\n"
        "button { background-image: none; box-shadow: none; text-shadow: none; -gtk-icon-shadow: none;\n"
        "         border: none; outline: none; min-height: 0; min-width: 0; }\n"

        /* -- Toast banner: quiet, sharp, minimal (Apple x Mira) -- */
        ".toast {\n"
        "    background-color: @bg;\n"
        "    border: 1px solid alpha(@fg, 0.09);\n"
        "    border-radius: 16px;\n"
        "    padding: 12px 14px;\n"
        "}\n"
        ".t-title { font-size: 9.8pt; font-weight: 600; color: @fg; }\n"
        ".t-body  { font-size: 9.2pt; font-weight: 400; color: alpha(@fg, 0.62); }\n"
        ".t-meta  { font-size: 8.2pt; font-weight: 400; color: alpha(@fg, 0.34);\n"
        "           font-feature-settings: \"tnum\"; }\n"
        ".t-meta.pct { font-size: 9pt; font-weight: 500; color: alpha(@fg, 0.52); }\n"
        ".crit-dot { background-color: @urgent; border-radius: 999px; min-width: 5px; min-height: 5px; }\n"

        /* Subtle squircle only for symbolic / fallback glyphs. */
        ".icon-badge {\n"
        "    background-color: alpha(@fg, 0.06);\n"
        "    border-radius: 9px;\n"
        "    min-width: %dpx; min-height: %dpx;\n"
        "}\n"
        ".icon-badge.small { border-radius: 8px; min-width: %dpx; min-height: %dpx; }\n"
        ".icon-badge .glyph { font-size: 17px; color: alpha(@fg, 0.78); }\n"
        ".icon-badge.small .glyph { font-size: 15px; color: alpha(@fg, 0.75); }\n"
        ".icon-badge.critical { background-color: alpha(@urgent, 0.13); }\n"
        ".icon-badge.critical .glyph { color: @urgent; }\n"

        /* Bare × close glyph on hover (no loud grey circle until hovered). */
        "button.t-close {\n"
        "    font-family: \"Material Design Icons Desktop\";\n"
        "    font-size: 12px; color: alpha(@fg, 0.42);\n"
        "    background-color: transparent;\n"
        "    border-radius: 999px;\n"
        "    min-width: 20px; min-height: 20px; padding: 0px;\n"
        "    transition: background-color 120ms ease-out, color 120ms ease-out;\n"
        "}\n"
        "button.t-close:hover { background-color: alpha(@fg, 0.10); color: @fg; }\n"

        /* Slim 4 px progress pill (Volume / Brightness OSD). */
        "progressbar { min-height: 4px; margin-top: 7px; }\n"
        "progressbar trough {\n"
        "    min-height: 4px; border-radius: 999px; border: none; box-shadow: none;\n"
        "    background-image: none; background-color: alpha(@fg, 0.12);\n"
        "}\n"
        "progressbar progress {\n"
        "    min-height: 4px; border-radius: 999px; border: none; box-shadow: none;\n"
        "    background-image: none; background-color: @fg;\n"
        "}\n"

        /* Quiet hairline-separated actions row inside a banner. */
        ".t-actions { margin-top: 9px; padding-top: 8px; border-top: 1px solid alpha(@fg, 0.06); }\n"
        "button.t-act {\n"
        "    background-color: alpha(@fg, 0.07); border-radius: 7px;\n"
        "    padding: 4px 11px; font-size: 8.8pt; font-weight: 500; color: alpha(@fg, 0.88);\n"
        "    transition: background-color 120ms ease-out, color 120ms ease-out;\n"
        "}\n"
        "button.t-act:hover { background-color: alpha(@fg, 0.15); color: @fg; }\n"

        /* -- Notification Center (flat & quiet, same family as WifiPanel) -- */
        ".panel {\n"
        "    background-color: @bg;\n"
        "    border: 1px solid alpha(@fg, 0.10);\n"
        "    border-radius: 14px;\n"
        "    padding: %dpx;\n"
        "}\n"
        ".head { padding: 2px 6px 8px 6px; min-height: 28px; }\n"
        ".title { font-size: 13pt; font-weight: 600; color: @fg; }\n"
        "button.clear-btn {\n"
        "    background: transparent; border-radius: 6px;\n"
        "    padding: 2px 6px; font-size: 9.5pt; font-weight: 500; color: alpha(@fg, 0.45);\n"
        "    transition: color 120ms ease-out;\n"
        "}\n"
        "button.clear-btn:hover { color: @fg; }\n"

        /* Borderless Do Not Disturb row + single hairline divider below. */
        ".dnd-row {\n"
        "    padding: 6px 6px 12px 6px;\n"
        "    border-bottom: 1px solid alpha(@fg, 0.07);\n"
        "}\n"
        ".dnd-label { font-size: 10pt; font-weight: 400; color: alpha(@fg, 0.78); }\n"
        "switch { background-color: alpha(@fg, 0.14); border: none; border-radius: 999px; box-shadow: none;\n"
        "         min-width: 40px; min-height: 22px; background-image: none; outline: none; }\n"
        "switch:checked { background-color: @fg; }\n"
        "switch slider { background-color: @bg; border: none; border-radius: 999px; box-shadow: none;\n"
        "                min-width: 18px; min-height: 18px; margin: 2px; background-image: none; }\n"
        "switch:not(:checked) slider { background-color: alpha(@fg, 0.6); }\n"
        "switch image { color: transparent; }\n"

        /* Flat notification rows (no nested boxes, subtle hover surface). */
        ".nc-list { padding-top: 6px; }\n"
        ".nc-row {\n"
        "    background: transparent;\n"
        "    border-radius: 10px;\n"
        "    padding: 9px 6px;\n"
        "    transition: background-color 120ms ease-out;\n"
        "}\n"
        ".nc-row:hover { background-color: alpha(@fg, 0.05); }\n"
        ".nc-title { font-size: 9.8pt; font-weight: 600; color: @fg; }\n"
        ".nc-body  { font-size: 9pt; font-weight: 400; color: alpha(@fg, 0.60); }\n"
        ".nc-meta  { font-size: 8.2pt; color: alpha(@fg, 0.35); font-feature-settings: \"tnum\"; }\n"
        ".empty    { font-size: 9.8pt; color: alpha(@fg, 0.40); padding: 28px 12px 18px 12px; }\n",
        ICON_SZ, ICON_SZ, NC_ICON_SZ, NC_ICON_SZ, NC_PAD);

    GdkScreen *scr = gdk_screen_get_default();
    if (S.css_prov) {
        gtk_style_context_remove_provider_for_screen(scr, GTK_STYLE_PROVIDER(S.css_prov));
        g_object_unref(S.css_prov);
    }
    S.css_prov = gtk_css_provider_new();
    gtk_css_provider_load_from_data(S.css_prov, c->str, -1, NULL);
    gtk_style_context_add_provider_for_screen(scr, GTK_STYLE_PROVIDER(S.css_prov),
                                              GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_string_free(c, TRUE);
}

/* -- icon & picture helpers -------------------------------------------------- */
static gboolean has_opaque_corners(GdkPixbuf *pb) {
    if (!gdk_pixbuf_get_has_alpha(pb)) return TRUE;
    int w = gdk_pixbuf_get_width(pb), h = gdk_pixbuf_get_height(pb);
    if (w < 4 || h < 4) return FALSE;
    int rs = gdk_pixbuf_get_rowstride(pb);
    const guchar *px = gdk_pixbuf_get_pixels(pb);
    int opaque = 0;
    int pts[4][2] = {{1, 1}, {w - 2, 1}, {1, h - 2}, {w - 2, h - 2}};
    for (int i = 0; i < 4; i++) {
        if (px[pts[i][1] * rs + pts[i][0] * 4 + 3] > 200) opaque++;
    }
    return opaque >= 3;
}

static char *resolve_file_path(const char *spec) {
    if (!spec || !*spec) return NULL;
    char *p = NULL;
    if (g_str_has_prefix(spec, "file://"))
        p = g_uri_unescape_string(spec + 7, NULL);
    else if (spec[0] == '/')
        p = g_strdup(spec);
    else if (g_str_has_prefix(spec, "~/"))
        p = g_build_filename(g_get_home_dir(), spec + 2, NULL);
    if (p && g_file_test(p, G_FILE_TEST_IS_REGULAR)) return p;
    g_free(p);
    return NULL;
}

static char *desktop_icon(const char *entry) {
    if (!entry || !*entry) return NULL;
    char *fname = g_str_has_suffix(entry, ".desktop") ? g_strdup(entry)
                                                      : g_strconcat(entry, ".desktop", NULL);
    const char *dirs[] = { g_get_user_data_dir(), "/usr/local/share", "/usr/share", NULL };
    char *found = NULL;
    for (int i = 0; dirs[i] && !found; i++) {
        char *path = g_build_filename(dirs[i], "applications", fname, NULL);
        GKeyFile *kf = g_key_file_new();
        if (g_key_file_load_from_file(kf, path, G_KEY_FILE_NONE, NULL)) {
            char *ic = g_key_file_get_string(kf, "Desktop Entry", "Icon", NULL);
            if (ic && *ic) found = ic;
            else g_free(ic);
        }
        g_key_file_free(kf);
        g_free(path);
    }
    g_free(fname);
    return found;
}

static GdkPixbuf *decode_image_data(GVariant *v, int sz, int rad) {
    if (!v || !g_variant_is_of_type(v, G_VARIANT_TYPE("(iiibiiay)"))) return NULL;
    gint32 w = 0, h = 0, rs = 0, bps = 0, ch = 0;
    gboolean has_alpha = FALSE;
    GVariant *bytes = NULL;
    g_variant_get(v, "(iiibii@ay)", &w, &h, &rs, &has_alpha, &bps, &ch, &bytes);
    if (w <= 0 || h <= 0 || w > 4096 || h > 4096 || bps != 8 || (ch != 3 && ch != 4) || rs < w * ch) {
        if (bytes) g_variant_unref(bytes);
        return NULL;
    }
    gsize len = 0;
    const guchar *src = g_variant_get_fixed_array(bytes, &len, 1);
    gsize need = (gsize)(h - 1) * (gsize)rs + (gsize)w * (gsize)ch;
    GdkPixbuf *out = NULL;
    if (src && len >= need) {
        GdkPixbuf *raw = gdk_pixbuf_new_from_data(src, GDK_COLORSPACE_RGB, has_alpha, 8, w, h, rs, NULL, NULL);
        if (raw) {
            if (has_opaque_corners(raw)) {
                out = cover_pixbuf(raw, sz, sz, rad, FALSE);
            } else {
                double s = MIN((double)sz / w, (double)sz / h);
                int nw = MAX(1, (int)nearbyint(w * s)), nh = MAX(1, (int)nearbyint(h * s));
                out = gdk_pixbuf_scale_simple(raw, nw, nh, GDK_INTERP_BILINEAR);
            }
            g_object_unref(raw);
        }
    }
    g_variant_unref(bytes);
    return out;
}

static gboolean is_symbolic_spec(const char *spec) {
    if (!spec || !*spec) return FALSE;
    return g_str_has_suffix(spec, "-symbolic") || g_str_has_suffix(spec, "-symbolic.svg");
}

static char *find_theme_icon(const char *name) {
    if (!name || !*name) return NULL;
    if (S.icon_cache) {
        const char *hit = g_hash_table_lookup(S.icon_cache, name);
        if (hit) return *hit ? g_strdup(hit) : NULL;
    }

    char *home_icons = g_build_filename(g_get_home_dir(), ".local/share/icons", NULL);
    const char *roots[] = { home_icons, "/usr/share/icons", NULL };
    const char *themes[6];
    int nt = 0;
    themes[nt++] = S.icon_theme_name;
    if (g_str_has_prefix(S.icon_theme_name, "Papirus") && strcmp(S.icon_theme_name, "Papirus") != 0)
        themes[nt++] = "Papirus";
    if (strcmp(S.icon_theme_name, "hicolor") != 0) themes[nt++] = "hicolor";
    if (strcmp(S.icon_theme_name, "Adwaita") != 0) themes[nt++] = "Adwaita";

    static const char *sizes[] = { "48x48", "64x64", "32x32", "scalable", "symbolic", "24x24", NULL };
    static const char *cats[]  = { "apps", "status", "devices", "actions", "categories", "places", "mimetypes", NULL };
    static const char *exts[]  = { ".png", ".svg", NULL };

    char *low = g_ascii_strdown(name, -1);
    const char *names[3] = { name, strcmp(low, name) ? low : NULL, NULL };
    char *found = NULL;

    for (int ni = 0; names[ni] && !found; ni++) {
        for (int ti = 0; ti < nt && !found; ti++) {
            for (int ri = 0; roots[ri] && !found; ri++) {
                char tdir[512];
                g_snprintf(tdir, sizeof tdir, "%s/%s", roots[ri], themes[ti]);
                if (access(tdir, X_OK) != 0) continue;
                for (int si = 0; sizes[si] && !found; si++) {
                    for (int ci = 0; cats[ci] && !found; ci++) {
                        for (int ei = 0; exts[ei] && !found; ei++) {
                            char p[1024];
                            g_snprintf(p, sizeof p, "%s/%s/%s/%s%s", tdir, sizes[si], cats[ci], names[ni], exts[ei]);
                            if (access(p, R_OK) == 0) { found = g_strdup(p); break; }
                        }
                    }
                }
            }
        }
        for (int ei = 0; exts[ei] && !found; ei++) {
            char p[512];
            g_snprintf(p, sizeof p, "/usr/share/pixmaps/%s%s", names[ni], exts[ei]);
            if (access(p, R_OK) == 0) found = g_strdup(p);
        }
    }

    g_free(low);
    g_free(home_icons);

    if (S.icon_cache) {
        if (g_hash_table_size(S.icon_cache) >= ICON_CACHE_MAX) g_hash_table_remove_all(S.icon_cache);
        g_hash_table_insert(S.icon_cache, g_strdup(name), g_strdup(found ? found : ""));
    }
    return found;
}

static GdkPixbuf *load_icon_pixbuf(const char *spec, int sz, int rad, gboolean *wants_badge) {
    *wants_badge = FALSE;
    if (!spec || !*spec) return NULL;
    char *fpath = resolve_file_path(spec);
    gboolean from_theme = FALSE;
    if (!fpath) {
        fpath = find_theme_icon(spec);
        from_theme = fpath != NULL;
    }
    if (!fpath) return NULL;

    gboolean symbolic = is_symbolic_spec(spec) || strstr(fpath, "/symbolic/") != NULL;
    int target = symbolic ? MAX(16, sz - 14) : sz;
    GdkPixbuf *src = g_str_has_suffix(fpath, ".svg")
                         ? gdk_pixbuf_new_from_file_at_scale(fpath, target, target, TRUE, NULL)
                         : load_cover_size(fpath, target, target);
    if (!src) { g_free(fpath); return NULL; }

    GdkPixbuf *out = NULL;
    if (symbolic) {
        *wants_badge = TRUE;
        out = src;
    } else if (has_opaque_corners(src)) {
        out = cover_pixbuf(src, sz, sz, rad, FALSE);
        g_object_unref(src);
    } else {
        int sw = gdk_pixbuf_get_width(src), sh = gdk_pixbuf_get_height(src);
        if (!from_theme && g_str_has_suffix(fpath, ".svg") && sw <= 128) {
            *wants_badge = TRUE;
            int bsz = MAX(16, sz - 14);
            out = gdk_pixbuf_scale_simple(src, bsz, bsz, GDK_INTERP_BILINEAR);
        } else if (sw == sz && sh == sz) {
            out = g_object_ref(src);
        } else {
            double s = MIN((double)sz / sw, (double)sz / sh);
            int nw = MAX(1, (int)nearbyint(sw * s)), nh = MAX(1, (int)nearbyint(sh * s));
            out = gdk_pixbuf_scale_simple(src, nw, nh, GDK_INTERP_BILINEAR);
        }
        g_object_unref(src);
    }
    g_free(fpath);
    return out;
}

/* Returns whether a dedicated icon was rendered (if FALSE, a fallback badge
 * was used — for critical alerts the badge itself carries the red accent, so
 * we never duplicate a second red dot next to the title). */
static GtkWidget *build_icon_widget(const Notif *n, int sz, gboolean small_badge, gboolean *used_crit_badge) {
    if (used_crit_badge) *used_crit_badge = FALSE;
    int rad = small_badge ? 8 : 9;
    if (n->raw_icon) {
        GdkPixbuf *pb = (gdk_pixbuf_get_width(n->raw_icon) == sz && gdk_pixbuf_get_height(n->raw_icon) == sz)
                            ? g_object_ref(n->raw_icon)
                            : gdk_pixbuf_scale_simple(n->raw_icon, sz, sz, GDK_INTERP_BILINEAR);
        GtkWidget *img = gtk_image_new_from_pixbuf(pb);
        g_object_unref(pb);
        gtk_widget_set_size_request(img, sz, sz);
        return img;
    }

    gboolean badge = FALSE;
    GdkPixbuf *pb = load_icon_pixbuf(n->icon_name, sz, rad, &badge);
    if (pb) {
        GtkWidget *img = gtk_image_new_from_pixbuf(pb);
        g_object_unref(pb);
        if (badge) {
            GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
            add_class(box, "icon-badge");
            if (small_badge) add_class(box, "small");
            if (n->urgency == 2) {
                add_class(box, "critical");
                if (used_crit_badge) *used_crit_badge = TRUE;
            }
            gtk_widget_set_halign(img, GTK_ALIGN_CENTER);
            gtk_widget_set_valign(img, GTK_ALIGN_CENTER);
            gtk_box_pack_start(GTK_BOX(box), img, TRUE, TRUE, 0);
            return box;
        }
        gtk_widget_set_size_request(img, sz, sz);
        return img;
    }

    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    add_class(box, "icon-badge");
    if (small_badge) add_class(box, "small");
    if (n->urgency == 2) {
        add_class(box, "critical");
        if (used_crit_badge) *used_crit_badge = TRUE;
    }
    GtkWidget *gl = make_glyph(n->urgency == 2 ? IC_ALERT : IC_BELL, NULL);
    gtk_widget_set_halign(gl, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(gl, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(box), gl, TRUE, TRUE, 0);
    return box;
}

static GtkWidget *build_thumb_widget(const Notif *n, int sz) {
    GdkPixbuf *pb = NULL;
    if (n->raw_thumb) {
        pb = (gdk_pixbuf_get_width(n->raw_thumb) == sz)
                 ? g_object_ref(n->raw_thumb)
                 : gdk_pixbuf_scale_simple(n->raw_thumb, sz, sz, GDK_INTERP_BILINEAR);
    } else if (n->image_path) {
        pb = cover_file(n->image_path, sz, sz, 8, FALSE);
    }
    if (!pb) return NULL;
    GtkWidget *img = gtk_image_new_from_pixbuf(pb);
    g_object_unref(pb);
    gtk_widget_set_size_request(img, sz, sz);
    return img;
}

/* -- bar state & IPC sync ---------------------------------------------------- */
static void write_state_file(void) {
    char tmp[4200];
    g_snprintf(tmp, sizeof tmp, "%s.tmp", S.state_path);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return;
    char buf[64];
    int len = g_snprintf(buf, sizeof buf, "%d %d\n", S.history_len, S.dnd ? 1 : 0);
    if (write(fd, buf, (size_t)len) < 0) { /* ignore */ }
    close(fd);
    rename(tmp, S.state_path);
}

static gboolean poke_bar_cb(gpointer _d) {
    S.bar_poke_src = 0;
    const char *argv[] = {"polybar-msg", "action", "#notif.hook.0", NULL};
    g_spawn_async(NULL, (char **)argv, NULL,
                  G_SPAWN_SEARCH_PATH | G_SPAWN_STDOUT_TO_DEV_NULL | G_SPAWN_STDERR_TO_DEV_NULL,
                  NULL, NULL, NULL, NULL);
    return G_SOURCE_REMOVE;
}

static void schedule_bar_sync(void) {
    write_state_file();
    if (!S.bar_poke_src) S.bar_poke_src = g_timeout_add(40, poke_bar_cb, NULL);
}

static gboolean trim_cb(gpointer _d) {
    S.trim_src = 0;
    if (!S.toasts && !S.nc_open) heap_trim();
    return G_SOURCE_REMOVE;
}

static void schedule_trim(void) {
    if (S.trim_src) g_source_remove(S.trim_src);
    S.trim_src = g_timeout_add(450, trim_cb, NULL);
}

/* -- D-Bus signal emitters --------------------------------------------------- */
static void emit_closed(guint32 id, guint32 reason) {
    if (!S.bus || !id) return;
    g_dbus_connection_emit_signal(S.bus, NULL, "/org/freedesktop/Notifications",
                                  "org.freedesktop.Notifications", "NotificationClosed",
                                  g_variant_new("(uu)", id, reason), NULL);
}

static void emit_action(guint32 id, const char *key) {
    if (!S.bus || !id || !key) return;
    g_dbus_connection_emit_signal(S.bus, NULL, "/org/freedesktop/Notifications",
                                  "org.freedesktop.Notifications", "ActionInvoked",
                                  g_variant_new("(us)", id, key), NULL);
}

/* -- Toast banners (top-right, below the bar) -------------------------------- */
static Toast *toast_by_id(guint32 id) {
    for (GList *l = S.toasts; l; l = l->next)
        if (((Toast *)l->data)->n->id == id) return l->data;
    return NULL;
}

static Toast *toast_by_tag(const char *tag) {
    if (!tag || !*tag) return NULL;
    for (GList *l = S.toasts; l; l = l->next) {
        Toast *t = l->data;
        if (t->n->stack_tag && !strcmp(t->n->stack_tag, tag)) return t;
    }
    return NULL;
}

static gboolean toast_expire_cb(gpointer data) {
    Toast *t = data;
    t->expire_src = 0;
    toast_close(t, 1, TRUE);
    return G_SOURCE_REMOVE;
}

static void toast_start_timer(Toast *t, gint64 ms) {
    if (t->expire_src) {
        g_source_remove(t->expire_src);
        t->expire_src = 0;
    }
    t->remaining_ms = ms;
    if (ms <= 0 || t->hovered) return;
    t->timer_start_us = g_get_monotonic_time();
    t->expire_src = g_timeout_add((guint)ms, toast_expire_cb, t);
}

static void toast_pause_timer(Toast *t) {
    if (!t->expire_src) return;
    gint64 elapsed = (g_get_monotonic_time() - t->timer_start_us) / 1000;
    t->remaining_ms = MAX(HOVER_RESUME_MIN_MS, t->remaining_ms - elapsed);
    g_source_remove(t->expire_src);
    t->expire_src = 0;
}

static gboolean toast_glide_tick(GtkWidget *w, GdkFrameClock *clock, gpointer data) {
    Toast *t = data;
    if (!t->gliding) { t->tick_id = 0; return G_SOURCE_REMOVE; }
    double frac = MIN(1.0, (double)(gdk_frame_clock_get_frame_time(clock) - t->anim_t0) / ((double)GLIDE_MS * 1000.0));
    double ease = 1.0 - pow(1.0 - frac, 4.0);
    t->y = (int)nearbyint(t->anim_from_y + (t->anim_to_y - t->anim_from_y) * ease);
    gtk_window_move(GTK_WINDOW(t->win), t->x, t->y);
    if (frac >= 1.0) {
        t->gliding = FALSE;
        t->tick_id = 0;
        return G_SOURCE_REMOVE;
    }
    return G_SOURCE_CONTINUE;
}

static int toast_measure(Toast *t) {
    int nh = 58;
    gtk_widget_get_preferred_height_for_width(t->card, TOAST_W, NULL, &nh);
    t->h = MAX(52, nh);
    gtk_window_resize(GTK_WINDOW(t->win), TOAST_W, t->h);
    return t->h;
}

static void toasts_layout(gboolean animate) {
    if (!S.toasts) return;
    Rect area = below_bar(focused_monitor());
    int x = area.x + area.w - TOAST_W - TOAST_MARGIN_X;
    int y = area.y + TOAST_MARGIN_Y;

    for (GList *l = S.toasts; l; l = l->next) {
        Toast *t = l->data;
        toast_measure(t);
        t->x = x;
        GdkFrameClock *clk = gtk_widget_get_frame_clock(t->win);
        if (animate && gtk_widget_get_visible(t->win) && clk && abs(t->y - y) >= 2) {
            t->anim_from_y = t->y;
            t->anim_to_y = y;
            t->anim_t0 = gdk_frame_clock_get_frame_time(clk);
            t->gliding = TRUE;
            if (!t->tick_id) t->tick_id = gtk_widget_add_tick_callback(t->win, toast_glide_tick, t, NULL);
        } else {
            if (t->tick_id) {
                gtk_widget_remove_tick_callback(t->win, t->tick_id);
                t->tick_id = 0;
            }
            t->gliding = FALSE;
            t->y = y;
            gtk_window_move(GTK_WINDOW(t->win), t->x, t->y);
        }
        y += t->h + TOAST_GAP;
    }
}

static void toast_close(Toast *t, guint32 reason, gboolean animate_rest) {
    if (!t) return;
    S.toasts = g_list_remove(S.toasts, t);
    if (t->expire_src) g_source_remove(t->expire_src);
    if (t->tick_id && t->win) gtk_widget_remove_tick_callback(t->win, t->tick_id);
    guint32 id = t->n->id;
    if (t->win) gtk_widget_destroy(t->win);
    notif_unref(t->n);
    g_free(t);
    if (reason) emit_closed(id, reason);
    toasts_layout(animate_rest);
    if (!S.toasts) schedule_trim();
}

static void toasts_close_all(guint32 reason) {
    while (S.toasts) toast_close(S.toasts->data, reason, FALSE);
    schedule_trim();
}

static gboolean on_toast_enter(GtkWidget *_w, GdkEventCrossing *ev, gpointer data) {
    if (ev->detail == GDK_NOTIFY_INFERIOR) return FALSE;
    Toast *t = data;
    if (t->hovered) return FALSE;
    t->hovered = TRUE;
    toast_pause_timer(t);
    gtk_stack_set_visible_child_name(GTK_STACK(t->time_stack), "close");
    return FALSE;
}

static gboolean on_toast_leave(GtkWidget *_w, GdkEventCrossing *ev, gpointer data) {
    if (ev->detail == GDK_NOTIFY_INFERIOR) return FALSE;
    Toast *t = data;
    if (!t->hovered) return FALSE;
    t->hovered = FALSE;
    gtk_stack_set_visible_child_name(GTK_STACK(t->time_stack), "time");
    if (t->remaining_ms > 0) toast_start_timer(t, t->remaining_ms);
    return FALSE;
}

static void on_toast_close_btn(GtkWidget *_b, gpointer data) {
    toast_close((Toast *)data, 2, TRUE);
}

static void on_toast_action_btn(GtkWidget *b, gpointer data) {
    Toast *t = data;
    const char *key = g_object_get_data(G_OBJECT(b), "action-key");
    if (key) emit_action(t->n->id, key);
    if (!t->n->resident) toast_close(t, 2, TRUE);
}

static gboolean on_toast_press(GtkWidget *_w, GdkEventButton *ev, gpointer data) {
    Toast *t = data;
    if (ev->button == 1) {
        if (t->n->has_default) emit_action(t->n->id, "default");
        if (!t->n->resident) toast_close(t, 2, TRUE);
        return TRUE;
    }
    if (ev->button == 3) {
        toast_close(t, 2, TRUE);
        return TRUE;
    }
    return FALSE;
}

static void toast_populate(Toast *t) {
    Notif *n = t->n;

    /* Primary icon */
    clear_container(t->icon_slot);
    gboolean used_crit_badge = FALSE;
    GtkWidget *ic = build_icon_widget(n, ICON_SZ, FALSE, &used_crit_badge);
    gtk_widget_set_valign(ic, (n->n_actions > 0) ? GTK_ALIGN_START : GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(t->icon_slot), ic, FALSE, FALSE, 0);
    gtk_widget_show_all(t->icon_slot);

    /* Optional right-side attachment thumbnail (explicit show: parent has no_show_all) */
    clear_container(t->thumb_slot);
    GtkWidget *th = build_thumb_widget(n, THUMB_SZ);
    if (th) {
        gtk_widget_set_valign(th, GTK_ALIGN_CENTER);
        gtk_box_pack_start(GTK_BOX(t->thumb_slot), th, FALSE, FALSE, 0);
        gtk_widget_show(th);
        gtk_widget_show(t->thumb_slot);
    } else {
        gtk_widget_hide(t->thumb_slot);
    }

    /* One quiet signal for critical: only show the 5 px dot if the icon itself
     * is not already the red alert badge (never duplicate red elements). */
    gtk_widget_set_visible(t->crit_dot, n->urgency == 2 && !used_crit_badge);

    const char *raw_title = (n->summary && *n->summary) ? n->summary
                          : (!is_generic_app(n->app_name) ? n->app_name : "Notification");

    GtkStyleContext *msc = gtk_widget_get_style_context(t->time_lbl);
    if (n->progress >= 0) {
        char clean_t[128], pct[16];
        split_osd_title(raw_title, n->progress, clean_t, pct);
        gtk_label_set_text(GTK_LABEL(t->title_lbl), clean_t);
        gtk_label_set_text(GTK_LABEL(t->time_lbl), pct);
        gtk_style_context_add_class(msc, "pct");
    } else {
        gtk_label_set_text(GTK_LABEL(t->title_lbl), raw_title);
        gtk_label_set_text(GTK_LABEL(t->time_lbl), "now");
        gtk_style_context_remove_class(msc, "pct");
    }

    /* Body (max 2 clean wrapped lines, hidden if empty) */
    if (n->body && *n->body) {
        gtk_label_set_text(GTK_LABEL(t->body_lbl), n->body);
        gtk_widget_show(t->body_lbl);
    } else {
        gtk_label_set_text(GTK_LABEL(t->body_lbl), "");
        gtk_widget_hide(t->body_lbl);
    }

    /* Slim 4 px progress pill */
    if (n->progress >= 0) {
        gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(t->prog_bar),
                                      CLAMP(n->progress, 0, 100) / 100.0);
        gtk_widget_show(t->prog_bar);
    } else {
        gtk_widget_hide(t->prog_bar);
    }

    /* Action buttons (explicit show per button: parent has no_show_all) */
    clear_container(t->actions_box);
    int shown = 0;
    for (int i = 0; i < n->n_actions && shown < 3; i++) {
        if (!strcmp(n->actions[i].key, "default")) continue;
        const char *lbl = (n->actions[i].label && *n->actions[i].label) ? n->actions[i].label : n->actions[i].key;
        GtkWidget *b = gtk_button_new_with_label(lbl);
        add_class(b, "t-act");
        gtk_widget_set_can_focus(b, FALSE);
        g_object_set_data_full(G_OBJECT(b), "action-key", g_strdup(n->actions[i].key), g_free);
        g_signal_connect(b, "clicked", G_CALLBACK(on_toast_action_btn), t);
        gtk_box_pack_start(GTK_BOX(t->actions_box), b, FALSE, FALSE, 0);
        gtk_widget_show(b);
        shown++;
    }
    if (shown > 0) gtk_widget_show(t->actions_box);
    else gtk_widget_hide(t->actions_box);
}

static Toast *toast_create(Notif *n) {
    Toast *t = g_new0(Toast, 1);
    t->n = notif_ref(n);

    Rect area = below_bar(focused_monitor());
    t->x = area.x + area.w - TOAST_W - TOAST_MARGIN_X;
    t->y = area.y + TOAST_MARGIN_Y;

    t->win = overlay_window("NotifToast", t->x, t->y, TOAST_W, 58);
    gtk_window_set_accept_focus(GTK_WINDOW(t->win), FALSE);
    gtk_window_set_focus_on_map(GTK_WINDOW(t->win), FALSE);
    gtk_widget_set_size_request(t->win, TOAST_W, -1);
    gtk_window_set_default_size(GTK_WINDOW(t->win), TOAST_W, -1);
    gtk_window_set_resizable(GTK_WINDOW(t->win), TRUE);
    gtk_widget_add_events(t->win, GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK | GDK_BUTTON_PRESS_MASK);
    g_signal_connect(t->win, "enter-notify-event", G_CALLBACK(on_toast_enter), t);
    g_signal_connect(t->win, "leave-notify-event", G_CALLBACK(on_toast_leave), t);
    g_signal_connect(t->win, "button-press-event", G_CALLBACK(on_toast_press), t);

    t->card = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    add_class(t->card, "toast");

    /* Upper content row: [icon] [title + body + progress] [optional thumb] */
    GtkWidget *main_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);

    t->icon_slot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_box_pack_start(GTK_BOX(main_row), t->icon_slot, FALSE, FALSE, 0);

    GtkWidget *mid = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    gtk_widget_set_valign(mid, GTK_ALIGN_CENTER);

    GtkWidget *top = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    t->crit_dot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    add_class(t->crit_dot, "crit-dot");
    gtk_widget_set_valign(t->crit_dot, GTK_ALIGN_CENTER);
    gtk_widget_set_no_show_all(t->crit_dot, TRUE);

    t->title_lbl = gtk_label_new("");
    add_class(t->title_lbl, "t-title");
    gtk_label_set_xalign(GTK_LABEL(t->title_lbl), 0.0);
    gtk_label_set_ellipsize(GTK_LABEL(t->title_lbl), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(t->title_lbl), 1);

    t->time_stack = gtk_stack_new();
    gtk_stack_set_transition_type(GTK_STACK(t->time_stack), GTK_STACK_TRANSITION_TYPE_CROSSFADE);
    gtk_stack_set_transition_duration(GTK_STACK(t->time_stack), 110);
    gtk_widget_set_valign(t->time_stack, GTK_ALIGN_CENTER);

    t->time_lbl = gtk_label_new("now");
    add_class(t->time_lbl, "t-meta");
    gtk_label_set_xalign(GTK_LABEL(t->time_lbl), 1.0);

    char *x_str = u8_glyph(IC_CLOSE);
    GtkWidget *close_btn = gtk_button_new_with_label(x_str);
    g_free(x_str);
    add_class(close_btn, "t-close");
    gtk_widget_set_can_focus(close_btn, FALSE);
    gtk_widget_set_halign(close_btn, GTK_ALIGN_END);
    gtk_widget_set_valign(close_btn, GTK_ALIGN_CENTER);
    g_signal_connect(close_btn, "clicked", G_CALLBACK(on_toast_close_btn), t);

    gtk_stack_add_named(GTK_STACK(t->time_stack), t->time_lbl, "time");
    gtk_stack_add_named(GTK_STACK(t->time_stack), close_btn, "close");
    gtk_stack_set_visible_child_name(GTK_STACK(t->time_stack), "time");

    gtk_box_pack_start(GTK_BOX(top), t->crit_dot, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(top), t->title_lbl, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(top), t->time_stack, FALSE, FALSE, 0);

    t->body_lbl = gtk_label_new("");
    add_class(t->body_lbl, "t-body");
    gtk_label_set_xalign(GTK_LABEL(t->body_lbl), 0.0);
    gtk_label_set_line_wrap(GTK_LABEL(t->body_lbl), TRUE);
    gtk_label_set_line_wrap_mode(GTK_LABEL(t->body_lbl), PANGO_WRAP_WORD_CHAR);
    gtk_label_set_lines(GTK_LABEL(t->body_lbl), 2);
    gtk_label_set_ellipsize(GTK_LABEL(t->body_lbl), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(t->body_lbl), 1);
    gtk_widget_set_no_show_all(t->body_lbl, TRUE);

    t->prog_bar = gtk_progress_bar_new();
    gtk_widget_set_no_show_all(t->prog_bar, TRUE);

    gtk_box_pack_start(GTK_BOX(mid), top, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(mid), t->body_lbl, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(mid), t->prog_bar, FALSE, FALSE, 0);

    t->thumb_slot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_widget_set_no_show_all(t->thumb_slot, TRUE);

    gtk_box_pack_start(GTK_BOX(main_row), mid, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(main_row), t->thumb_slot, FALSE, FALSE, 0);

    /* Full-width hairline-separated actions row below content */
    t->actions_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    add_class(t->actions_box, "t-actions");
    gtk_widget_set_no_show_all(t->actions_box, TRUE);

    gtk_box_pack_start(GTK_BOX(t->card), main_row, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(t->card), t->actions_box, FALSE, FALSE, 0);

    gtk_container_add(GTK_CONTAINER(t->win), t->card);
    gtk_widget_show_all(t->card);
    toast_populate(t);
    return t;
}

static void show_toast(Notif *n) {
    Toast *existing = toast_by_id(n->id);
    if (!existing && n->stack_tag) existing = toast_by_tag(n->stack_tag);
    if (existing) {
        notif_unref(existing->n);
        existing->n = notif_ref(n);
        toast_populate(existing);
        toast_start_timer(existing, n->timeout_ms);
        toasts_layout(TRUE);
        return;
    }

    while (g_list_length(S.toasts) >= MAX_TOASTS) {
        GList *last = g_list_last(S.toasts);
        toast_close((Toast *)last->data, 1, FALSE);
    }

    Toast *t = toast_create(n);
    S.toasts = g_list_prepend(S.toasts, t);
    toasts_layout(TRUE);
    gtk_widget_show(t->win);
    gdk_window_raise(gtk_widget_get_window(t->win));
    toast_start_timer(t, n->timeout_ms);
}

/* -- History management ------------------------------------------------------ */
static void history_add(Notif *n) {
    if (n->transient || n->progress >= 0) return;

    for (GList *l = S.history; l; l = l->next) {
        Notif *old = l->data;
        if (old->id == n->id || (n->stack_tag && old->stack_tag && !strcmp(old->stack_tag, n->stack_tag))) {
            S.history = g_list_delete_link(S.history, l);
            S.history_len--;
            notif_unref(old);
            break;
        }
    }

    S.history = g_list_prepend(S.history, notif_ref(n));
    S.history_len++;
    while (S.history_len > HISTORY_MAX) {
        GList *last = g_list_last(S.history);
        Notif *drop = last->data;
        S.history = g_list_delete_link(S.history, last);
        S.history_len--;
        notif_unref(drop);
    }
    schedule_bar_sync();
    if (S.nc_open) nc_rebuild_list();
}

static void history_remove(Notif *target) {
    GList *l = g_list_find(S.history, target);
    if (!l) return;
    S.history = g_list_delete_link(S.history, l);
    S.history_len--;
    notif_unref(target);
    schedule_bar_sync();
    if (S.nc_open) nc_rebuild_list();
}

static void history_clear(void) {
    if (!S.history) return;
    g_list_free_full(S.history, (GDestroyNotify)notif_unref);
    S.history = NULL;
    S.history_len = 0;
    schedule_bar_sync();
    if (S.nc_open) nc_rebuild_list();
    schedule_trim();
}

/* -- Do Not Disturb ---------------------------------------------------------- */
static void set_dnd(gboolean on) {
    if (S.dnd == on) return;
    S.dnd = on;
    if (on) toasts_close_all(2);
    if (S.nc_dnd_sw) {
        S.nc_syncing = TRUE;
        gtk_switch_set_active(GTK_SWITCH(S.nc_dnd_sw), on);
        gtk_switch_set_state(GTK_SWITCH(S.nc_dnd_sw), on);
        S.nc_syncing = FALSE;
    }
    schedule_bar_sync();
}

/* -- Notification Center (flat, quiet, same family as WifiPanel) ------------- */
static void nc_anchor(gboolean pointer) {
    S.nc_mon = (Rect){0, 0, 1920, 1080};
    g_strlcpy(S.nc_edge, "top", sizeof S.nc_edge);
    S.nc_off = 0;
    const char *argv[] = {"BarAnchor", pointer ? "--pointer" : NULL, NULL};
    char *out = run_out(argv);
    gboolean ok = FALSE;
    if (out) {
        char **f = g_strsplit_set(g_strstrip(out), " \t\n", -1);
        if (g_strv_length(f) >= 3 && monitor_rect(f[0], &S.nc_mon)) {
            g_strlcpy(S.nc_edge, f[1], sizeof S.nc_edge);
            S.nc_off = atoi(f[2]);
            ok = TRUE;
        }
        g_strfreev(f);
        g_free(out);
    }
    if (!ok) S.nc_mon = focused_monitor();
    int py = 0;
    gdk_device_get_position(gdk_seat_get_pointer(gdk_display_get_default_seat(gdk_display_get_default())),
                            NULL, &S.nc_px, &py);
}

static void nc_place(int h) {
    if (!S.nc_win) return;
    if (h <= 0) h = gtk_widget_get_allocated_height(S.nc_win);
    if (h <= 0) h = 140;
    int x = S.nc_pointer ? S.nc_px - NC_WIDTH / 2 : S.nc_mon.x + S.nc_mon.w - NC_WIDTH - 8;
    x = MAX(S.nc_mon.x + 8, MIN(x, S.nc_mon.x + S.nc_mon.w - NC_WIDTH - 8));
    int y = strcmp(S.nc_edge, "top") == 0 ? S.nc_mon.y + S.nc_off + NC_GAP
                                          : S.nc_mon.y + S.nc_mon.h - S.nc_off - NC_GAP - h;
    int cx = 0, cy = 0;
    gtk_window_get_position(GTK_WINDOW(S.nc_win), &cx, &cy);
    if (cx != x || cy != y) gtk_window_move(GTK_WINDOW(S.nc_win), x, y);
}

static void nc_fit(void) {
    if (!S.nc_win || !S.nc_box) return;
    if (S.history) {
        int list_h = 0;
        gtk_widget_get_preferred_height_for_width(S.nc_list, NC_WIDTH - 2 * NC_PAD, NULL, &list_h);
        int clamped = MAX(48, MIN(NC_LIST_MAX_H, list_h));
        gtk_scrolled_window_set_min_content_height(GTK_SCROLLED_WINDOW(S.nc_scroll), clamped);
        gtk_scrolled_window_set_max_content_height(GTK_SCROLLED_WINDOW(S.nc_scroll), clamped);
    }
    int nh = 120;
    gtk_widget_get_preferred_height_for_width(S.nc_box, NC_WIDTH, NULL, &nh);
    gtk_window_resize(GTK_WINDOW(S.nc_win), NC_WIDTH, MAX(1, nh));
    nc_place(nh);
}

static gboolean nc_fit_idle(gpointer _d) {
    if (S.nc_open) nc_fit();
    return G_SOURCE_REMOVE;
}

typedef struct {
    Notif *n;
    GtkWidget *stack;
    GtkWidget *time_lbl;
} NcRow;

static void nc_row_free(gpointer data) {
    NcRow *c = data;
    notif_unref(c->n);
    g_free(c);
}

static gboolean on_nc_row_enter(GtkWidget *_w, GdkEventCrossing *ev, gpointer data) {
    if (ev->detail == GDK_NOTIFY_INFERIOR) return FALSE;
    NcRow *c = data;
    gtk_stack_set_visible_child_name(GTK_STACK(c->stack), "close");
    return FALSE;
}

static gboolean on_nc_row_leave(GtkWidget *_w, GdkEventCrossing *ev, gpointer data) {
    if (ev->detail == GDK_NOTIFY_INFERIOR) return FALSE;
    NcRow *c = data;
    gtk_stack_set_visible_child_name(GTK_STACK(c->stack), "time");
    return FALSE;
}

static void on_nc_row_rm(GtkWidget *_b, gpointer data) {
    NcRow *c = data;
    history_remove(c->n);
}

static gboolean on_nc_row_click(GtkWidget *_w, GdkEventButton *ev, gpointer data) {
    NcRow *c = data;
    if (ev->button == 1 && c->n->has_default) {
        emit_action(c->n->id, "default");
        history_remove(c->n);
        nc_close();
        return TRUE;
    }
    if (ev->button == 3) {
        history_remove(c->n);
        return TRUE;
    }
    return FALSE;
}

static GtkWidget *nc_make_row(Notif *n) {
    NcRow *c = g_new0(NcRow, 1);
    c->n = notif_ref(n);

    GtkWidget *ev = gtk_event_box_new();
    gtk_event_box_set_visible_window(GTK_EVENT_BOX(ev), FALSE);
    gtk_widget_add_events(ev, GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK | GDK_BUTTON_PRESS_MASK);
    g_object_set_data_full(G_OBJECT(ev), "nc-row", c, nc_row_free);
    g_signal_connect(ev, "enter-notify-event", G_CALLBACK(on_nc_row_enter), c);
    g_signal_connect(ev, "leave-notify-event", G_CALLBACK(on_nc_row_leave), c);
    g_signal_connect(ev, "button-press-event", G_CALLBACK(on_nc_row_click), c);

    GtkWidget *row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 11);
    add_class(row, "nc-row");

    gboolean used_crit_badge = FALSE;
    GtkWidget *ic = build_icon_widget(n, NC_ICON_SZ, TRUE, &used_crit_badge);
    gtk_widget_set_valign(ic, GTK_ALIGN_START);
    gtk_box_pack_start(GTK_BOX(row), ic, FALSE, FALSE, 0);

    GtkWidget *mid = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    gtk_widget_set_valign(mid, GTK_ALIGN_CENTER);

    GtkWidget *top = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    if (n->urgency == 2 && !used_crit_badge) {
        GtkWidget *dot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
        add_class(dot, "crit-dot");
        gtk_widget_set_valign(dot, GTK_ALIGN_CENTER);
        gtk_box_pack_start(GTK_BOX(top), dot, FALSE, FALSE, 0);
    }

    const char *title = (n->summary && *n->summary) ? n->summary
                      : (!is_generic_app(n->app_name) ? n->app_name : "Notification");
    GtkWidget *tl = gtk_label_new(title);
    add_class(tl, "nc-title");
    gtk_label_set_xalign(GTK_LABEL(tl), 0.0);
    gtk_label_set_ellipsize(GTK_LABEL(tl), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(tl), 1);

    c->stack = gtk_stack_new();
    gtk_stack_set_transition_type(GTK_STACK(c->stack), GTK_STACK_TRANSITION_TYPE_CROSSFADE);
    gtk_stack_set_transition_duration(GTK_STACK(c->stack), 110);
    gtk_widget_set_valign(c->stack, GTK_ALIGN_CENTER);

    char rel[32];
    fmt_relative(n->created_at, rel);
    c->time_lbl = gtk_label_new(rel);
    add_class(c->time_lbl, "nc-meta");
    gtk_label_set_xalign(GTK_LABEL(c->time_lbl), 1.0);

    char *x_str = u8_glyph(IC_CLOSE);
    GtkWidget *rm = gtk_button_new_with_label(x_str);
    g_free(x_str);
    add_class(rm, "t-close");
    gtk_widget_set_can_focus(rm, FALSE);
    gtk_widget_set_halign(rm, GTK_ALIGN_END);
    g_signal_connect(rm, "clicked", G_CALLBACK(on_nc_row_rm), c);

    gtk_stack_add_named(GTK_STACK(c->stack), c->time_lbl, "time");
    gtk_stack_add_named(GTK_STACK(c->stack), rm, "close");
    gtk_stack_set_visible_child_name(GTK_STACK(c->stack), "time");

    gtk_box_pack_start(GTK_BOX(top), tl, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(top), c->stack, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(mid), top, FALSE, FALSE, 0);

    if (n->body && *n->body) {
        GtkWidget *bl = gtk_label_new(n->body);
        add_class(bl, "nc-body");
        gtk_label_set_xalign(GTK_LABEL(bl), 0.0);
        gtk_label_set_line_wrap(GTK_LABEL(bl), TRUE);
        gtk_label_set_line_wrap_mode(GTK_LABEL(bl), PANGO_WRAP_WORD_CHAR);
        gtk_label_set_lines(GTK_LABEL(bl), 2);
        gtk_label_set_ellipsize(GTK_LABEL(bl), PANGO_ELLIPSIZE_END);
        gtk_label_set_max_width_chars(GTK_LABEL(bl), 1);
        gtk_box_pack_start(GTK_BOX(mid), bl, FALSE, FALSE, 0);
    }

    gtk_box_pack_start(GTK_BOX(row), mid, TRUE, TRUE, 0);

    GtkWidget *th = build_thumb_widget(n, 36);
    if (th) {
        gtk_widget_set_valign(th, GTK_ALIGN_CENTER);
        gtk_box_pack_end(GTK_BOX(row), th, FALSE, FALSE, 0);
    }

    gtk_container_add(GTK_CONTAINER(ev), row);
    return ev;
}

static void nc_rebuild_list(void) {
    if (!S.nc_list) return;
    clear_container(S.nc_list);
    gboolean has_items = S.history != NULL;
    gtk_widget_set_visible(S.nc_clear_btn, has_items);
    gtk_widget_set_visible(S.nc_scroll, has_items);
    gtk_widget_set_visible(S.nc_empty, !has_items);

    for (GList *l = S.history; l; l = l->next) {
        GtkWidget *row = nc_make_row((Notif *)l->data);
        gtk_box_pack_start(GTK_BOX(S.nc_list), row, FALSE, FALSE, 0);
        gtk_widget_show_all(row);
    }
    g_idle_add(nc_fit_idle, NULL);
}

static gboolean nc_refresh_times(gpointer _d) {
    if (!S.nc_open || !S.nc_list) { S.nc_time_src = 0; return G_SOURCE_REMOVE; }
    GList *kids = gtk_container_get_children(GTK_CONTAINER(S.nc_list));
    for (GList *l = kids; l; l = l->next) {
        NcRow *c = g_object_get_data(G_OBJECT(l->data), "nc-row");
        if (!c || !c->time_lbl) continue;
        char rel[32];
        fmt_relative(c->n->created_at, rel);
        gtk_label_set_text(GTK_LABEL(c->time_lbl), rel);
    }
    g_list_free(kids);
    return G_SOURCE_CONTINUE;
}

static gboolean on_nc_dnd_switch(GtkSwitch *sw, gboolean on, gpointer _d) {
    if (S.nc_syncing) return FALSE;
    gtk_switch_set_state(sw, on);
    set_dnd(on);
    return TRUE;
}

static void on_nc_clear(GtkWidget *_b, gpointer _d) {
    history_clear();
}

static gboolean on_nc_key(GtkWidget *_w, GdkEventKey *ev, gpointer _d) {
    if (ev->keyval == GDK_KEY_Escape) {
        nc_close();
        return TRUE;
    }
    return FALSE;
}

static gboolean on_nc_outside(GtkWidget *_w, GdkEventButton *ev, gpointer _d) {
    int x = 0, y = 0, w = 0, h = 0;
    gtk_window_get_position(GTK_WINDOW(S.nc_win), &x, &y);
    gtk_window_get_size(GTK_WINDOW(S.nc_win), &w, &h);
    if (!(ev->x_root >= x && ev->x_root < x + w && ev->y_root >= y && ev->y_root < y + h)) {
        nc_close();
        return TRUE;
    }
    return FALSE;
}

static void on_nc_alloc(GtkWidget *_w, GdkRectangle *_a, gpointer _d) {
    if (S.nc_open) nc_place(0);
}

static void nc_build(void) {
    S.nc_win = overlay_window("NotifCenter", 0, 0, NC_WIDTH, 140);
    gtk_widget_set_size_request(S.nc_win, NC_WIDTH, -1);
    gtk_window_set_default_size(GTK_WINDOW(S.nc_win), -1, -1);
    gtk_window_set_resizable(GTK_WINDOW(S.nc_win), TRUE);
    gtk_widget_add_events(S.nc_win, GDK_KEY_PRESS_MASK | GDK_BUTTON_PRESS_MASK);
    g_signal_connect(S.nc_win, "key-press-event", G_CALLBACK(on_nc_key), NULL);
    g_signal_connect(S.nc_win, "button-press-event", G_CALLBACK(on_nc_outside), NULL);
    g_signal_connect(S.nc_win, "size-allocate", G_CALLBACK(on_nc_alloc), NULL);

    /* Header: "Notifications" + bare "Clear" text button (iOS nav bar style) */
    GtkWidget *head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    add_class(head, "head");
    GtkWidget *title = gtk_label_new("Notifications");
    add_class(title, "title");
    gtk_label_set_xalign(GTK_LABEL(title), 0.0);
    S.nc_clear_btn = gtk_button_new_with_label("Clear");
    add_class(S.nc_clear_btn, "clear-btn");
    gtk_widget_set_valign(S.nc_clear_btn, GTK_ALIGN_CENTER);
    gtk_widget_set_can_focus(S.nc_clear_btn, FALSE);
    gtk_widget_set_no_show_all(S.nc_clear_btn, TRUE);
    g_signal_connect(S.nc_clear_btn, "clicked", G_CALLBACK(on_nc_clear), NULL);
    gtk_box_pack_start(GTK_BOX(head), title, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(head), S.nc_clear_btn, FALSE, FALSE, 0);

    /* Borderless Do Not Disturb row + hairline divider below */
    GtkWidget *dnd_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    add_class(dnd_row, "dnd-row");
    GtkWidget *dnd_lbl = gtk_label_new("Do Not Disturb");
    add_class(dnd_lbl, "dnd-label");
    gtk_label_set_xalign(GTK_LABEL(dnd_lbl), 0.0);
    S.nc_dnd_sw = gtk_switch_new();
    gtk_widget_set_valign(S.nc_dnd_sw, GTK_ALIGN_CENTER);
    gtk_widget_set_can_focus(S.nc_dnd_sw, FALSE);
    g_signal_connect(S.nc_dnd_sw, "state-set", G_CALLBACK(on_nc_dnd_switch), NULL);
    gtk_box_pack_start(GTK_BOX(dnd_row), dnd_lbl, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(dnd_row), S.nc_dnd_sw, FALSE, FALSE, 0);

    /* Scrollable flat notification rows */
    S.nc_list = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    add_class(S.nc_list, "nc-list");
    S.nc_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(S.nc_scroll), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_propagate_natural_height(GTK_SCROLLED_WINDOW(S.nc_scroll), TRUE);
    gtk_scrolled_window_set_max_content_height(GTK_SCROLLED_WINDOW(S.nc_scroll), NC_LIST_MAX_H);
    gtk_container_add(GTK_CONTAINER(S.nc_scroll), S.nc_list);
    gtk_widget_show_all(S.nc_scroll);
    gtk_widget_set_no_show_all(S.nc_scroll, TRUE);
    gtk_widget_hide(S.nc_scroll);

    /* Quiet empty placeholder */
    S.nc_empty = gtk_label_new("No notifications");
    add_class(S.nc_empty, "empty");
    gtk_widget_set_no_show_all(S.nc_empty, TRUE);

    S.nc_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    add_class(S.nc_box, "panel");
    gtk_box_pack_start(GTK_BOX(S.nc_box), head, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.nc_box), dnd_row, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.nc_box), S.nc_scroll, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.nc_box), S.nc_empty, FALSE, FALSE, 0);

    gtk_container_add(GTK_CONTAINER(S.nc_win), S.nc_box);
    gtk_widget_show_all(S.nc_box);
    gtk_widget_realize(S.nc_win);
}

static void nc_close(void) {
    if (!S.nc_open) return;
    S.nc_open = FALSE;
    if (S.nc_time_src) {
        g_source_remove(S.nc_time_src);
        S.nc_time_src = 0;
    }
    ungrab();
    gtk_widget_hide(S.nc_win);
    drop_backdrop(S.nc_shade);
    S.nc_shade = 0;
    clear_container(S.nc_list);
    schedule_trim();
}

static void nc_open(gboolean pointer) {
    if (S.nc_open) {
        nc_close();
        return;
    }
    close_bar_popups("notif");
    if (!S.nc_win) nc_build();
    toasts_close_all(0);
    S.nc_open = TRUE;
    S.nc_pointer = pointer;
    nc_anchor(pointer);

    S.nc_syncing = TRUE;
    gtk_switch_set_active(GTK_SWITCH(S.nc_dnd_sw), S.dnd);
    gtk_switch_set_state(GTK_SWITCH(S.nc_dnd_sw), S.dnd);
    S.nc_syncing = FALSE;

    nc_rebuild_list();
    nc_fit();
    S.nc_shade = backdrop(S.nc_mon, 0, NULL);
    gtk_widget_show(S.nc_win);
    gdk_window_raise(gtk_widget_get_window(S.nc_win));
    grab_soon(S.nc_win, G_CALLBACK(nc_close), NULL);
    S.nc_time_src = g_timeout_add_seconds(30, nc_refresh_times, NULL);
}

/* -- D-Bus org.freedesktop.Notifications handler ----------------------------- */
static void handle_notify(GVariant *params, GDBusMethodInvocation *inv) {
    const char *app_name = NULL, *app_icon = NULL, *summary = NULL, *body = NULL;
    guint32 replaces_id = 0;
    gint32 expire_timeout = -1;
    GVariantIter *actions_it = NULL, *hints_it = NULL;

    g_variant_get(params, "(&su&s&s&sasa{sv}i)",
                  &app_name, &replaces_id, &app_icon, &summary, &body,
                  &actions_it, &hints_it, &expire_timeout);

    Notif *n = g_new0(Notif, 1);
    n->refcount = 1;
    n->urgency = 1;
    n->progress = -1;
    n->created_at = (gint64)time(NULL);
    n->app_name = g_strdup(app_name ? app_name : "");
    n->summary = clean_text(summary, TRUE);
    n->body = clean_text(body, FALSE);

    GPtrArray *act_arr = g_ptr_array_new();
    const char *akey = NULL, *albl = NULL;
    while (g_variant_iter_next(actions_it, "&s", &akey)) {
        if (!g_variant_iter_next(actions_it, "&s", &albl)) break;
        if (!strcmp(akey, "default")) {
            n->has_default = TRUE;
        } else {
            Action *a = g_new0(Action, 1);
            a->key = g_strdup(akey);
            a->label = g_strdup(albl);
            g_ptr_array_add(act_arr, a);
        }
    }
    g_variant_iter_free(actions_it);
    if (act_arr->len > 0) {
        n->n_actions = (int)act_arr->len;
        n->actions = g_new0(Action, n->n_actions);
        for (int i = 0; i < n->n_actions; i++) {
            Action *a = g_ptr_array_index(act_arr, i);
            n->actions[i] = *a;
            g_free(a);
        }
    }
    g_ptr_array_free(act_arr, TRUE);

    char *desktop_entry = NULL;
    char *hint_img_path = NULL;
    GVariant *hint_img_data = NULL;
    const char *hkey = NULL;
    GVariant *hval = NULL;
    while (g_variant_iter_next(hints_it, "{&sv}", &hkey, &hval)) {
        if (!strcmp(hkey, "urgency")) {
            if (g_variant_is_of_type(hval, G_VARIANT_TYPE_BYTE))
                n->urgency = g_variant_get_byte(hval);
            else if (g_variant_is_of_type(hval, G_VARIANT_TYPE_INT32))
                n->urgency = (guint8)CLAMP(g_variant_get_int32(hval), 0, 2);
            else if (g_variant_is_of_type(hval, G_VARIANT_TYPE_UINT32))
                n->urgency = (guint8)MIN(g_variant_get_uint32(hval), 2u);
        } else if (!strcmp(hkey, "x-dunst-stack-tag") ||
                   !strcmp(hkey, "synchronous") ||
                   !strcmp(hkey, "private-synchronous")) {
            if (g_variant_is_of_type(hval, G_VARIANT_TYPE_STRING)) {
                g_free(n->stack_tag);
                n->stack_tag = g_strdup(g_variant_get_string(hval, NULL));
            }
        } else if (!strcmp(hkey, "value")) {
            if (g_variant_is_of_type(hval, G_VARIANT_TYPE_INT32))
                n->progress = CLAMP(g_variant_get_int32(hval), 0, 100);
            else if (g_variant_is_of_type(hval, G_VARIANT_TYPE_UINT32))
                n->progress = (int)MIN(g_variant_get_uint32(hval), 100u);
            else if (g_variant_is_of_type(hval, G_VARIANT_TYPE_DOUBLE))
                n->progress = CLAMP((int)nearbyint(g_variant_get_double(hval)), 0, 100);
        } else if (!strcmp(hkey, "transient")) {
            if (g_variant_is_of_type(hval, G_VARIANT_TYPE_BOOLEAN))
                n->transient = g_variant_get_boolean(hval);
            else if (g_variant_is_of_type(hval, G_VARIANT_TYPE_INT32))
                n->transient = g_variant_get_int32(hval) != 0;
        } else if (!strcmp(hkey, "resident")) {
            if (g_variant_is_of_type(hval, G_VARIANT_TYPE_BOOLEAN))
                n->resident = g_variant_get_boolean(hval);
        } else if (!strcmp(hkey, "desktop-entry")) {
            if (g_variant_is_of_type(hval, G_VARIANT_TYPE_STRING)) {
                g_free(desktop_entry);
                desktop_entry = g_strdup(g_variant_get_string(hval, NULL));
            }
        } else if (!strcmp(hkey, "image-path") || !strcmp(hkey, "image_path")) {
            if (g_variant_is_of_type(hval, G_VARIANT_TYPE_STRING)) {
                g_free(hint_img_path);
                hint_img_path = g_strdup(g_variant_get_string(hval, NULL));
            }
        } else if (!strcmp(hkey, "image-data") || !strcmp(hkey, "image_data") || !strcmp(hkey, "icon_data")) {
            if (!hint_img_data || strcmp(hkey, "icon_data") != 0) {
                if (hint_img_data) g_variant_unref(hint_img_data);
                hint_img_data = g_variant_ref(hval);
            }
        }
        g_variant_unref(hval);
    }
    g_variant_iter_free(hints_it);

    char *primary_icon = NULL;
    if (app_icon && *app_icon) {
        primary_icon = g_strdup(app_icon);
    } else if (desktop_entry && *desktop_entry) {
        primary_icon = desktop_icon(desktop_entry);
        if (!primary_icon) primary_icon = g_strdup(desktop_entry);
    } else if (!is_generic_app(n->app_name)) {
        char *found = find_theme_icon(n->app_name);
        if (found) primary_icon = found;
    }

    if (hint_img_data) {
        if (primary_icon) {
            n->icon_name = primary_icon;
            n->raw_thumb = decode_image_data(hint_img_data, THUMB_SZ, 8);
        } else {
            n->raw_icon = decode_image_data(hint_img_data, ICON_SZ, 9);
        }
        g_variant_unref(hint_img_data);
    } else if (hint_img_path && *hint_img_path) {
        char *resolved = resolve_file_path(hint_img_path);
        if (primary_icon && resolved && g_strcmp0(primary_icon, resolved) != 0) {
            n->icon_name = primary_icon;
            n->image_path = resolved;
        } else {
            g_free(primary_icon);
            n->icon_name = resolved ? resolved : g_strdup(hint_img_path);
        }
    } else {
        n->icon_name = primary_icon;
    }
    g_free(desktop_entry);
    g_free(hint_img_path);

    if (replaces_id > 0) {
        n->id = replaces_id;
        if (replaces_id >= S.next_id) S.next_id = replaces_id + 1;
    } else {
        if (S.next_id == 0) S.next_id = 1;
        n->id = S.next_id++;
    }

    if (n->urgency == 2) {
        n->timeout_ms = (expire_timeout > 0) ? expire_timeout : 0;
    } else if (expire_timeout == 0) {
        n->timeout_ms = 0;
    } else if (expire_timeout > 0) {
        n->timeout_ms = MAX(900, expire_timeout);
    } else {
        n->timeout_ms = (n->urgency == 0) ? DEFAULT_TIMEOUT_LOW : DEFAULT_TIMEOUT_NORMAL;
    }

    history_add(n);
    gboolean suppress_toast = (S.dnd && n->urgency < 2) || S.nc_open;
    if (!suppress_toast) show_toast(n);

    g_dbus_method_invocation_return_value(inv, g_variant_new("(u)", n->id));
    notif_unref(n);
}

static void on_dbus_call(GDBusConnection *_conn, const char *_sender, const char *_path,
                         const char *_iface, const char *method, GVariant *params,
                         GDBusMethodInvocation *inv, gpointer _data) {
    if (!strcmp(method, "GetCapabilities")) {
        const char *caps[] = {"body", "body-markup", "actions", "icon-static", "persistence", "action-icons", NULL};
        g_dbus_method_invocation_return_value(inv, g_variant_new("(^as)", caps));
    } else if (!strcmp(method, "GetServerInformation")) {
        g_dbus_method_invocation_return_value(inv, g_variant_new("(ssss)", "NotifPanel", "misa", "1.0", "1.2"));
    } else if (!strcmp(method, "Notify")) {
        handle_notify(params, inv);
    } else if (!strcmp(method, "CloseNotification")) {
        guint32 id = 0;
        g_variant_get(params, "(u)", &id);
        Toast *t = toast_by_id(id);
        if (t) toast_close(t, 3, TRUE);
        else emit_closed(id, 3);
        g_dbus_method_invocation_return_value(inv, NULL);
    } else {
        g_dbus_method_invocation_return_error(inv, G_DBUS_ERROR, G_DBUS_ERROR_UNKNOWN_METHOD,
                                              "Unknown method %s", method);
    }
}

static const GDBusInterfaceVTable DBUS_VTABLE = { on_dbus_call, NULL, NULL, {0} };

static void on_bus_acquired(GDBusConnection *conn, const char *_name, gpointer _d) {
    S.bus = conn;
    S.bus_reg_id = g_dbus_connection_register_object(conn, "/org/freedesktop/Notifications",
                                                     S.introspection->interfaces[0],
                                                     &DBUS_VTABLE, NULL, NULL, NULL);
}

static void on_name_lost(GDBusConnection *_conn, const char *_name, gpointer _d) {
    if (!S.bus) return;
    gtk_main_quit();
}

/* -- Control FIFO ------------------------------------------------------------ */
static void handle_cmd(const char *line) {
    if (g_str_has_prefix(line, "toggle")) {
        nc_open(strstr(line, "--pointer") != NULL);
    } else if (g_str_has_prefix(line, "open")) {
        if (!S.nc_open) nc_open(strstr(line, "--pointer") != NULL);
    } else if (!strcmp(line, "close")) {
        nc_close();
    } else if (!strcmp(line, "close-all")) {
        toasts_close_all(2);
    } else if (!strcmp(line, "clear")) {
        history_clear();
    } else if (!strcmp(line, "dnd on")) {
        set_dnd(TRUE);
    } else if (!strcmp(line, "dnd off")) {
        set_dnd(FALSE);
    } else if (g_str_has_prefix(line, "dnd")) {
        set_dnd(!S.dnd);
    } else if (!strcmp(line, "reload")) {
        load_theme();
        for (GList *l = S.toasts; l; l = l->next) toast_populate((Toast *)l->data);
        toasts_layout(FALSE);
        if (S.nc_open) nc_rebuild_list();
        schedule_bar_sync();
    } else if (!strcmp(line, "quit")) {
        nc_close();
        toasts_close_all(0);
        gtk_main_quit();
    }
}

static gboolean on_fifo(gint fd, GIOCondition _c, gpointer _d) {
    ssize_t n = read(fd, S.fifo_buf + S.fifo_len, sizeof S.fifo_buf - 1 - S.fifo_len);
    if (n <= 0) return G_SOURCE_CONTINUE;
    S.fifo_len += (size_t)n;
    S.fifo_buf[S.fifo_len] = 0;
    char *start = S.fifo_buf, *nl;
    while ((nl = strchr(start, '\n'))) {
        *nl = 0;
        if (*start) handle_cmd(start);
        start = nl + 1;
    }
    S.fifo_len = strlen(start);
    memmove(S.fifo_buf, start, S.fifo_len + 1);
    return G_SOURCE_CONTINUE;
}

static gboolean on_signal_quit(gpointer _d) {
    nc_close();
    toasts_close_all(0);
    gtk_main_quit();
    return G_SOURCE_REMOVE;
}

/* -- Fast non-GTK client helpers (<1 ms) ------------------------------------- */
static int client_send(const char *cmd) {
    char fifo[4096];
    g_snprintf(fifo, sizeof fifo, "%s/notif-panel.fifo", runtime_dir());
    int fd = open(fifo, O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return 1;
    char buf[256];
    int len = g_snprintf(buf, sizeof buf, "%s\n", cmd);
    ssize_t wr = write(fd, buf, (size_t)len);
    close(fd);
    return wr == len ? 0 : 1;
}

static int client_status(void) {
    char path[4096];
    g_snprintf(path, sizeof path, "%s/notif-panel.state", runtime_dir());
    FILE *f = fopen(path, "r");
    if (!f) return 1;
    int count = 0, dnd = 0;
    int ok = fscanf(f, "%d %d", &count, &dnd) == 2;
    fclose(f);
    if (!ok) return 1;
    printf("%d %d\n", count, dnd);
    return 0;
}

/* -- Daemon startup ---------------------------------------------------------- */
static int run_daemon(int argc, char **argv) {
    mallopt(M_ARENA_MAX, 1);

    const char *run = runtime_dir();
    char pidfile[4096], fifo[4096];
    g_snprintf(pidfile, sizeof pidfile, "%s/notif-panel.pid", run);
    g_snprintf(fifo, sizeof fifo, "%s/notif-panel.fifo", run);
    g_snprintf(S.state_path, sizeof S.state_path, "%s/notif-panel.state", run);

    S.lock_fd = open(pidfile, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (S.lock_fd < 0 || flock(S.lock_fd, LOCK_EX | LOCK_NB) != 0) return 0;
    if (ftruncate(S.lock_fd, 0) == 0) {
        char t[32];
        int n = g_snprintf(t, sizeof t, "%d\n", (int)getpid());
        if (write(S.lock_fd, t, (size_t)n) < 0) { /* lock holds */ }
    }

    unlink(fifo);
    if (mkfifo(fifo, 0600) != 0 && errno != EEXIST) return 1;
    S.fifo_fd = open(fifo, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (S.fifo_fd < 0) return 1;

    g_setenv("GDK_GL", "disable", FALSE);
    g_setenv("NO_AT_BRIDGE", "1", FALSE);
    g_setenv("GIO_USE_VFS", "local", FALSE);

    gtk_init(&argc, &argv);
    S.next_id = 1;
    load_theme();

    S.introspection = g_dbus_node_info_new_for_xml(INTROSPECTION_XML, NULL);
    S.bus_owner_id = g_bus_own_name(
        G_BUS_TYPE_SESSION,
        "org.freedesktop.Notifications",
        G_BUS_NAME_OWNER_FLAGS_ALLOW_REPLACEMENT | G_BUS_NAME_OWNER_FLAGS_REPLACE,
        on_bus_acquired, NULL, on_name_lost, NULL, NULL);

    g_unix_fd_add(S.fifo_fd, G_IO_IN, on_fifo, NULL);
    g_unix_signal_add(SIGTERM, on_signal_quit, NULL);
    g_unix_signal_add(SIGINT, on_signal_quit, NULL);

    schedule_bar_sync();
    heap_trim();
    gtk_main();

    if (S.bar_poke_src) g_source_remove(S.bar_poke_src);
    if (S.trim_src) g_source_remove(S.trim_src);
    if (S.bus_reg_id && S.bus) g_dbus_connection_unregister_object(S.bus, S.bus_reg_id);
    if (S.bus_owner_id) g_bus_unown_name(S.bus_owner_id);
    if (S.introspection) g_dbus_node_info_unref(S.introspection);
    g_list_free_full(S.history, (GDestroyNotify)notif_unref);
    if (S.icon_cache) g_hash_table_destroy(S.icon_cache);
    if (S.css_prov) g_object_unref(S.css_prov);
    unlink(fifo);
    unlink(S.state_path);
    close(S.fifo_fd);
    close(S.lock_fd);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--daemon"))
        return run_daemon(argc, argv);

    if (argc == 1) return client_send("toggle");
    if (!strcmp(argv[1], "--pointer")) return client_send("toggle --pointer");
    if (!strcmp(argv[1], "status")) return client_status();
    if (!strcmp(argv[1], "toggle"))
        return client_send((argc >= 3 && !strcmp(argv[2], "--pointer")) ? "toggle --pointer" : "toggle");
    if (!strcmp(argv[1], "open"))
        return client_send((argc >= 3 && !strcmp(argv[2], "--pointer")) ? "open --pointer" : "open");
    if (!strcmp(argv[1], "close"))     return client_send("close");
    if (!strcmp(argv[1], "close-all")) return client_send("close-all");
    if (!strcmp(argv[1], "clear"))     return client_send("clear");
    if (!strcmp(argv[1], "reload"))    return client_send("reload");
    if (!strcmp(argv[1], "quit"))      return client_send("quit");
    if (!strcmp(argv[1], "dnd")) {
        if (argc >= 3 && !strcmp(argv[2], "on"))  return client_send("dnd on");
        if (argc >= 3 && !strcmp(argv[2], "off")) return client_send("dnd off");
        return client_send("dnd toggle");
    }

    fprintf(stderr, "Usage: NotifPanel [--daemon | [toggle|open] [--pointer] | close | close-all | clear | dnd [on|off|toggle] | reload | status | quit]\n");
    return 2;
}
