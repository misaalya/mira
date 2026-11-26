/* WindowSwitch - Alt-Tab (bound to alt + Tab in sxhkdrc).
 *
 * A card in the middle of the focused monitor with one tile per window, most
 * recently used first: a live thumbnail (the app's icon for windows that are
 * not on screen, e.g. the hidden columns of the Scroll strip), the app icon
 * and the title. The tile after the current window is selected.
 *
 * Hold Alt and press Tab / Shift+Tab (or the arrows) to move, release Alt to
 * switch; Enter or a click switches too, Esc cancels. A quick Alt+Tab tap
 * (Alt already released when this starts) switches straight to the previous
 * window without showing anything.
 *
 * Motion: the card pops in (picom, class WindowSwitch), the tiles come in
 * one after another, the selected tile lifts and lights up (CSS transitions)
 * and the row glides when there are more tiles than fit. Behind it the
 * monitor is dimmed and blurred (bin/misa-backdrop).
 *
 * An override-redirect window with a keyboard grab, like rofi: bspwm never
 * manages it. Switching goes through _NET_ACTIVE_WINDOW (wmctrl -a), which
 * bspwm turns into node_activate (Scroll brings hidden columns in).
 * Tunables: config/windowswitch.conf (TILE_HEIGHT, MAX_WIDTH_PERCENT).
 * Testing: WINDOWSWITCH_DEMO=<file of "title|/path/thumb.png" lines>.
 */
#include "common.h"
#include "json.h"

#include <gdk/gdkx.h>
#include <gio/gdesktopappinfo.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static int TILE_H = 150, MAX_W_PCT = 90;

typedef struct {
    unsigned long xid;
    char *title, *cls, *inst;
    gboolean on_screen;
} Win;

static struct {
    Win *wins;
    int n, cur;
    gboolean done, demo;
    int last_x, last_y;
    GtkWidget *win, *scroll;
    GtkWidget **tiles;
    int shade;
    Glide glide;
} S = {.cur = -1, .last_x = -1, .last_y = -1};

static void read_conf(void) {
    char *path = mira_config_path("config/windowswitch.conf");
    char *text = NULL;
    if (g_file_get_contents(path, &text, NULL, NULL)) {
        char **lines = g_strsplit(text, "\n", -1);
        for (int i = 0; lines[i]; i++) {
            char *hash = strchr(lines[i], '#');
            if (hash) *hash = 0;
            char *eq = strchr(lines[i], '=');
            if (!eq) continue;
            *eq = 0;
            char *k = g_strstrip(lines[i]), *v = g_strstrip(eq + 1);
            if (!*v || strspn(v, "0123456789") != strlen(v)) continue;
            if (!strcmp(k, "TILE_HEIGHT")) TILE_H = atoi(v);
            else if (!strcmp(k, "MAX_WIDTH_PERCENT")) MAX_W_PCT = atoi(v);
        }
        g_strfreev(lines);
        g_free(text);
    }
    g_free(path);
}

/* -- windows, most recently used first --------------------------------------- */
typedef struct { unsigned long id; char *cls, *inst; gboolean vis; } Client;

static void walk(J *node, gboolean visible, GArray *out) {
    if (!node || node->type != J_OBJ) return;
    J *c = jget(node, "client"), *first = jget(node, "firstChild");
    if (c && c->type == J_OBJ && (!first || first->type == J_NULL)) {
        Client cl = {(unsigned long)jnum(jget(node, "id")), g_strdup(jstr(jget(c, "className"))),
                     g_strdup(jstr(jget(c, "instanceName"))), visible && !jtrue(jget(node, "hidden"))};
        g_array_append_val(out, cl);
    }
    walk(first, visible, out);
    walk(jget(node, "secondChild"), visible, out);
}

static int windows(void) {
    const char *a1[] = {"bspc", "wm", "-d", NULL};
    char *out = run_out(a1);
    J *st = json_parse(out);
    g_free(out);
    if (!st) return 0;
    GArray *clients = g_array_new(FALSE, FALSE, sizeof(Client));
    J *mons = jget(st, "monitors");
    for (J *m = mons ? mons->child : NULL; m; m = m->next) {
        double fd = jnum(jget(m, "focusedDesktopId"));
        J *ds = jget(m, "desktops");
        for (J *d = ds ? ds->child : NULL; d; d = d->next)
            walk(jget(d, "root"), jnum(jget(d, "id")) == fd, clients);
    }
    /* titles */
    GHashTable *titles = g_hash_table_new_full(g_int64_hash, g_int64_equal, g_free, g_free);
    const char *a2[] = {"wmctrl", "-l", NULL};
    char *wl = run_out(a2);
    if (wl) {
        char **lines = g_strsplit(wl, "\n", -1);
        for (int i = 0; lines[i]; i++) {
            char **parts = g_strsplit_set(lines[i], " \t", 0);
            /* split like Python's str.split(None, 3) */
            int k = 0;
            char *fields[4] = {0};
            const char *s = lines[i];
            for (k = 0; k < 4 && *s; k++) {
                while (*s == ' ' || *s == '\t') s++;
                if (!*s) break;
                if (k == 3) { fields[3] = (char *)s; k++; break; }
                const char *e = s;
                while (*e && *e != ' ' && *e != '\t') e++;
                fields[k] = g_strndup(s, e - s);
                s = e;
            }
            if (fields[0] && fields[2]) {
                gint64 *id = g_new(gint64, 1);
                *id = (gint64)strtoull(fields[0], NULL, 16);
                g_hash_table_replace(titles, id, g_strdup(fields[3] ? fields[3] : ""));
            }
            for (int j = 0; j < 3; j++) g_free(fields[j]);
            g_strfreev(parts);
        }
        g_strfreev(lines);
        g_free(wl);
    }
    /* order: focus history, newest first, then the rest */
    GArray *order = g_array_new(FALSE, FALSE, sizeof(int));
    J *hist = jget(st, "focusHistory");
    int hn = 0;
    for (J *h = hist ? hist->child : NULL; h; h = h->next) hn++;
    J **hv = g_new(J *, hn + 1);
    int i = 0;
    for (J *h = hist ? hist->child : NULL; h; h = h->next) hv[i++] = h;
    for (i = hn - 1; i >= 0; i--) {
        unsigned long nid = (unsigned long)jnum(jget(hv[i], "nodeId"));
        for (guint c = 0; c < clients->len; c++) {
            if (g_array_index(clients, Client, c).id != nid) continue;
            gboolean seen = FALSE;
            for (guint o = 0; o < order->len; o++) seen |= g_array_index(order, int, o) == (int)c;
            if (!seen) { int ci = c; g_array_append_val(order, ci); }
        }
    }
    g_free(hv);
    for (guint c = 0; c < clients->len; c++) {
        gboolean seen = FALSE;
        for (guint o = 0; o < order->len; o++) seen |= g_array_index(order, int, o) == (int)c;
        if (!seen) { int ci = c; g_array_append_val(order, ci); }
    }
    S.n = order->len;
    S.wins = g_new0(Win, S.n + 1);
    for (int k = 0; k < S.n; k++) {
        Client *cl = &g_array_index(clients, Client, g_array_index(order, int, k));
        gint64 key = cl->id;
        const char *t = g_hash_table_lookup(titles, &key);
        S.wins[k] = (Win){cl->id, g_strdup(t ? t : ""), cl->cls, cl->inst, cl->vis};
    }
    g_array_free(order, TRUE);
    g_array_free(clients, TRUE);
    g_hash_table_destroy(titles);
    json_free(st);
    return S.n;
}

/* -- pictures ------------------------------------------------------------------- */
static GdkPixbuf *app_icon(const char *cls, const char *inst, int size) {
    GtkIconTheme *theme = gtk_icon_theme_get_default();
    GPtrArray *names = g_ptr_array_new_with_free_func(g_free);
    gchar ***found = g_desktop_app_info_search(cls);
    GdkPixbuf *direct = NULL;
    if (found && found[0] && found[0][0]) {
        GDesktopAppInfo *info = g_desktop_app_info_new(found[0][0]);
        GIcon *icon = info ? g_app_info_get_icon(G_APP_INFO(info)) : NULL;
        if (icon && G_IS_THEMED_ICON(icon)) {
            const char *const *ns = g_themed_icon_get_names(G_THEMED_ICON(icon));
            for (int i = 0; ns && ns[i]; i++) g_ptr_array_add(names, g_strdup(ns[i]));
        } else if (icon && G_IS_FILE_ICON(icon)) {
            char *p = g_file_get_path(g_file_icon_get_file(G_FILE_ICON(icon)));
            if (p) direct = gdk_pixbuf_new_from_file_at_size(p, size, size, NULL);
            g_free(p);
        }
        if (info) g_object_unref(info);
    }
    if (found) {
        for (int i = 0; found[i]; i++) g_strfreev(found[i]);
        g_free(found);
    }
    if (direct) { g_ptr_array_free(names, TRUE); return direct; }
    char *lc = g_ascii_strdown(cls, -1), *li = g_ascii_strdown(inst, -1);
    char *dot = strrchr(lc, '.');
    g_ptr_array_add(names, g_strdup(lc));
    g_ptr_array_add(names, g_strdup(li));
    g_ptr_array_add(names, g_strdup(dot ? dot + 1 : lc));
    g_ptr_array_add(names, g_strdup("application-x-executable"));
    g_free(lc);
    g_free(li);
    GdkPixbuf *pb = NULL;
    for (guint i = 0; i < names->len && !pb; i++) {
        const char *n = names->pdata[i];
        if (*n && gtk_icon_theme_has_icon(theme, n))
            pb = gtk_icon_theme_load_icon(theme, n, size, GTK_ICON_LOOKUP_FORCE_SIZE, NULL);
    }
    g_ptr_array_free(names, TRUE);
    return pb;
}

/* The window's content, or NULL (not on screen). */
static GdkPixbuf *thumbnail(unsigned long xid) {
    GdkDisplay *d = gdk_display_get_default();
    gdk_x11_display_error_trap_push(d);
    GdkWindow *gw = gdk_x11_window_foreign_new_for_display(d, xid);
    GdkPixbuf *pb = NULL;
    if (gw && gdk_window_is_viewable(gw)) {
        int w = gdk_window_get_width(gw), h = gdk_window_get_height(gw);
        if (w >= 2 && h >= 2) pb = gdk_pixbuf_get_from_window(gw, 0, 0, w, h);
    }
    if (gw) g_object_unref(gw);
    gdk_x11_display_error_trap_pop_ignored(d);
    return pb;
}

/* -- selection / actions -------------------------------------------------------------- */
static void centre(gboolean animate) {
    GtkAllocation a;
    gtk_widget_get_allocation(gtk_widget_get_parent(S.tiles[S.cur]), &a);
    glide_to(&S.glide, a.x + a.width / 2.0 - gtk_adjustment_get_page_size(S.glide.adj) / 2, animate);
}

static void select_tile(int i, gboolean animate) {
    i = ((i % S.n) + S.n) % S.n;
    if (i == S.cur) return;
    if (S.cur >= 0) gtk_style_context_remove_class(gtk_widget_get_style_context(S.tiles[S.cur]), "sel");
    S.cur = i;
    gtk_style_context_add_class(gtk_widget_get_style_context(S.tiles[i]), "sel");
    centre(animate);
}

static void quit(void) {
    ungrab();
    gtk_widget_hide(S.win);
    drop_backdrop(S.shade);
    gtk_main_quit();
}

static void switch_to(unsigned long xid) {
    char id[32];
    g_snprintf(id, sizeof id, "0x%08lx", xid);
    const char *argv[] = {"wmctrl", "-i", "-a", id, NULL};
    g_spawn_async(NULL, (char **)argv, NULL, G_SPAWN_SEARCH_PATH | G_SPAWN_STDERR_TO_DEV_NULL,
                  NULL, NULL, NULL, NULL);
}

static gboolean activate(gpointer _) {
    if (S.done) return G_SOURCE_REMOVE;
    S.done = TRUE;
    unsigned long xid = S.wins[S.cur].xid;
    quit();
    if (!S.demo) switch_to(xid);
    return G_SOURCE_REMOVE;
}

static gboolean alt_held(void) {
    GdkKeymap *km = gdk_keymap_get_for_display(gdk_display_get_default());
    return (gdk_keymap_get_modifier_state(km) & GDK_MOD1_MASK) != 0;
}

static gboolean on_hover(GtkWidget *_w, GdkEventMotion *ev, gpointer data) {
    int x = (int)lround(ev->x_root), y = (int)lround(ev->y_root);
    if (x != S.last_x || y != S.last_y) {
        S.last_x = x;
        S.last_y = y;
        select_tile(GPOINTER_TO_INT(data), TRUE);
    }
    return FALSE;
}

static gboolean on_click(GtkWidget *_w, GdkEventButton *ev, gpointer data) {
    if (ev->button == 1) {
        select_tile(GPOINTER_TO_INT(data), FALSE);
        activate(NULL);
    }
    return TRUE;
}

static gboolean on_key(GtkWidget *_w, GdkEventKey *ev, gpointer _) {
    guint k = ev->keyval;
    if (k == GDK_KEY_Escape) { S.done = TRUE; quit(); }
    else if (k == GDK_KEY_Return || k == GDK_KEY_KP_Enter) activate(NULL);
    else if (k == GDK_KEY_Tab || k == GDK_KEY_Right || k == GDK_KEY_Down) select_tile(S.cur + 1, TRUE);
    else if (k == GDK_KEY_ISO_Left_Tab || k == GDK_KEY_Left || k == GDK_KEY_Up) select_tile(S.cur - 1, TRUE);
    return TRUE;
}

static gboolean on_key_release(GtkWidget *_w, GdkEventKey *ev, gpointer _) {
    guint k = ev->keyval;
    if (k == GDK_KEY_Alt_L || k == GDK_KEY_Alt_R || k == GDK_KEY_Meta_L || k == GDK_KEY_Meta_R)
        activate(NULL);
    return TRUE;
}

static gboolean on_outside(GtkWidget *_w, GdkEventButton *_e, gpointer _) {
    quit();
    return TRUE;
}

static gboolean centre_later(gpointer _) {
    centre(FALSE);
    return G_SOURCE_REMOVE;
}

static void grab_failed(gpointer _) {
    S.done = TRUE;
    quit();
}

/* -- build ------------------------------------------------------------------------------ */
static char *css(const Palette *p, int n) {
    GString *st = g_string_new("");
    for (int i = 0; i < MIN(n, 12); i++)
        g_string_append_printf(st, "%s.tile-%d { animation-delay: %dms; }", i ? "\n" : "", i, 20 + i * 28);
    const char *fg = p->fg;
    char *s = g_strdup_printf(
        "* { font-family: \"%s\"; }\n"
        "window { background: transparent; }\n"
        "scrolledwindow, viewport { background: transparent; border: none; }\n"
        "scrollbar { opacity: 0; }\n"
        ".panel {\n"
        "    background-color: %s;\n"
        "    border: 1px solid alpha(%s, 0.12);\n"
        "    border-radius: 18px;\n"
        "    padding: 14px;\n"
        "}\n"
        ".enter { animation: enter 360ms cubic-bezier(0.2, 0.9, 0.25, 1) both; }\n"
        "@keyframes enter {\n"
        "    from { opacity: 0; padding-top: 26px; }\n"
        "    to   { opacity: 1; padding-top: 0px; }\n"
        "}\n"
        "%s\n"
        ".tile {\n"
        "    padding: 8px 8px 10px 8px;\n"
        "    margin-top: 6px; margin-bottom: 0px;\n"
        "    border-radius: 14px;\n"
        "    border: 2px solid transparent;\n"
        "    opacity: 0.5;\n"
        "    transition: all 200ms cubic-bezier(0.25, 1.35, 0.45, 1);\n"
        "}\n"
        ".tile.sel {\n"
        "    margin-top: 0px; margin-bottom: 6px;\n"
        "    background-color: alpha(%s, 0.10);\n"
        "    border-color: %s;\n"
        "    opacity: 1;\n"
        "}\n"
        ".thumb-empty { background-color: alpha(%s, 0.05); border-radius: 9px; }\n"
        ".title { font-size: 12px; font-weight: 600; color: %s; }\n",
        p->font, p->bg, fg, st->str, fg, fg, fg, fg);
    g_string_free(st, TRUE);
    return s;
}

static void add_class(GtkWidget *w, const char *c) {
    gtk_style_context_add_class(gtk_widget_get_style_context(w), c);
}

static void build(void) {
    Rect mon = focused_monitor();
    Palette p;
    palette_load(&p);

    S.win = gtk_window_new(GTK_WINDOW_POPUP);
    gtk_window_set_title(GTK_WINDOW(S.win), "WindowSwitch");
    G_GNUC_BEGIN_IGNORE_DEPRECATIONS
    gtk_window_set_wmclass(GTK_WINDOW(S.win), "windowswitch", "WindowSwitch");
    G_GNUC_END_IGNORE_DEPRECATIONS
    GdkVisual *vis = gdk_screen_get_rgba_visual(gtk_widget_get_screen(S.win));
    if (vis) gtk_widget_set_visual(S.win, vis);
    gtk_widget_set_app_paintable(S.win, TRUE);
    char *c = css(&p, S.n);
    css_load(c);
    g_free(c);

    GtkWidget *row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    S.tiles = g_new0(GtkWidget *, S.n);
    for (int i = 0; i < S.n; i++) {
        Win *w = &S.wins[i];
        GdkPixbuf *pb = NULL;
        const char *title = w->title;
        char *demo_title = NULL;
        if (S.demo) {
            char *bar = strchr(w->title, '|');
            if (bar) {
                pb = gdk_pixbuf_new_from_file(bar + 1, NULL);
                demo_title = g_strndup(w->title, bar - w->title);
                title = demo_title;
            }
        } else if (w->on_screen) {
            pb = thumbnail(w->xid);
        }
        int tw = TILE_H * 16 / 10;
        GtkWidget *pic;
        if (pb) {
            double r = (double)TILE_H * gdk_pixbuf_get_width(pb) / gdk_pixbuf_get_height(pb);
            tw = MAX((int)(TILE_H * 0.75), MIN((int)(TILE_H * 1.9), (int)nearbyint(r)));
            GdkPixbuf *cv = cover_pixbuf(pb, tw, TILE_H, 9, FALSE);
            pic = gtk_image_new_from_pixbuf(cv);
            g_object_unref(cv);
            g_object_unref(pb);
        } else {
            /* Not on screen: the app's icon, big, on an empty frame. */
            pic = gtk_overlay_new();
            GtkWidget *frame = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
            add_class(frame, "thumb-empty");
            gtk_widget_set_size_request(frame, tw, TILE_H);
            gtk_container_add(GTK_CONTAINER(pic), frame);
            GdkPixbuf *ic = app_icon(w->cls, w->inst, 64);
            if (ic) {
                GtkWidget *big = gtk_image_new_from_pixbuf(ic);
                gtk_widget_set_halign(big, GTK_ALIGN_CENTER);
                gtk_widget_set_valign(big, GTK_ALIGN_CENTER);
                gtk_overlay_add_overlay(GTK_OVERLAY(pic), big);
                g_object_unref(ic);
            }
        }
        GtkWidget *cap = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 7);
        GdkPixbuf *ic = app_icon(w->cls, w->inst, 18);
        if (ic) {
            gtk_box_pack_start(GTK_BOX(cap), gtk_image_new_from_pixbuf(ic), FALSE, FALSE, 0);
            g_object_unref(ic);
        }
        GtkWidget *lb = gtk_label_new(*title ? title : w->cls);
        gtk_label_set_xalign(GTK_LABEL(lb), 0);
        gtk_label_set_ellipsize(GTK_LABEL(lb), PANGO_ELLIPSIZE_END);
        gtk_label_set_max_width_chars(GTK_LABEL(lb), 1);
        add_class(lb, "title");
        gtk_box_pack_start(GTK_BOX(cap), lb, TRUE, TRUE, 0);
        gtk_widget_set_size_request(cap, tw, -1);
        g_free(demo_title);

        GtkWidget *tile = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
        add_class(tile, "tile");
        gtk_box_pack_start(GTK_BOX(tile), pic, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(tile), cap, FALSE, FALSE, 0);
        GtkWidget *ev = gtk_event_box_new();
        gtk_container_add(GTK_CONTAINER(ev), tile);
        add_class(ev, "enter");
        char cls[16];
        g_snprintf(cls, sizeof cls, "tile-%d", i);
        add_class(ev, cls);
        gtk_widget_add_events(ev, GDK_POINTER_MOTION_MASK | GDK_BUTTON_PRESS_MASK);
        g_signal_connect(ev, "motion-notify-event", G_CALLBACK(on_hover), GINT_TO_POINTER(i));
        g_signal_connect(ev, "button-press-event", G_CALLBACK(on_click), GINT_TO_POINTER(i));
        S.tiles[i] = tile;
        gtk_box_pack_start(GTK_BOX(row), ev, FALSE, FALSE, 0);
    }

    gtk_widget_show_all(row);                    /* hidden widgets measure 0 */
    int nat_w = 0, nat_h = 0;
    gtk_widget_get_preferred_width(row, NULL, &nat_w);
    gtk_widget_get_preferred_height(row, NULL, &nat_h);
    int max_w = mon.w * MAX_W_PCT / 100 - 30;
    S.scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(S.scroll), GTK_POLICY_EXTERNAL, GTK_POLICY_NEVER);
    gtk_container_add(GTK_CONTAINER(S.scroll), row);
    gtk_widget_set_size_request(S.scroll, MIN(nat_w, max_w), nat_h + 6);
    glide_init(&S.glide, S.scroll, gtk_scrolled_window_get_hadjustment(GTK_SCROLLED_WINDOW(S.scroll)), 260);

    GtkWidget *panel = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    add_class(panel, "panel");
    gtk_box_pack_start(GTK_BOX(panel), S.scroll, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(S.win), panel);
    gtk_widget_add_events(S.win, GDK_KEY_PRESS_MASK | GDK_KEY_RELEASE_MASK | GDK_BUTTON_PRESS_MASK);
    g_signal_connect(S.win, "key-press-event", G_CALLBACK(on_key), NULL);
    g_signal_connect(S.win, "key-release-event", G_CALLBACK(on_key_release), NULL);
    g_signal_connect(S.win, "button-press-event", G_CALLBACK(on_outside), NULL);

    gtk_widget_show_all(panel);
    GtkRequisition req;
    gtk_widget_get_preferred_size(panel, NULL, &req);
    gtk_window_move(GTK_WINDOW(S.win), mon.x + (mon.w - req.width) / 2, mon.y + (mon.h - req.height) / 2);
    /* The monitor behind dimmed and blurred (not for a quick tap, which
     * switches without ever showing the panel for long). */
    S.shade = (S.demo || alt_held()) ? backdrop(mon, 0, NULL) : 0;
    gtk_widget_show_all(S.win);
    select_tile(S.n > 1 ? 1 : 0, FALSE);
    g_timeout_add(40, centre_later, NULL);
    grab_soon(S.win, G_CALLBACK(grab_failed), NULL);
    /* Alt released while we were starting up: switch now. */
    if (!S.demo && !alt_held()) g_idle_add(activate, NULL);
}

int main(int argc, char **argv) {
    gtk_env_lean();
    gtk_init(&argc, &argv);
    read_conf();
    const char *demo = g_getenv("WINDOWSWITCH_DEMO");
    if (demo && *demo) {
        /* Testing: "title|/path/thumb.png" lines instead of real windows. */
        char *text = NULL;
        if (!g_file_get_contents(demo, &text, NULL, NULL)) return 0;
        char **lines = g_strsplit(text, "\n", -1);
        GArray *a = g_array_new(FALSE, TRUE, sizeof(Win));
        for (int i = 0; lines[i]; i++) {
            char *l = g_strstrip(lines[i]);
            if (!*l) continue;
            Win w = {(unsigned long)a->len, g_strdup(l), g_strdup(""), g_strdup(""), TRUE};
            g_array_append_val(a, w);
        }
        S.n = a->len;
        S.wins = (Win *)g_array_free(a, FALSE);
        g_strfreev(lines);
        g_free(text);
        S.demo = TRUE;
    } else {
        windows();
    }
    if (S.n == 0) return 0;
    if (!S.demo && S.n > 1 && !alt_held()) {
        /* Quick tap: straight to the previous window, nothing shown. */
        switch_to(S.wins[1].xid);
        return 0;
    }
    build();
    gtk_main();
    return 0;
}
