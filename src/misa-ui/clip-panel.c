/* ClipPanel - the clipboard history popup (bin/ClipMenu starts it).
 *
 * A panel against the bar, under the clipboard icon when opened from it
 * (--pointer), else at the bar's left end (shortcut), like the launcher:
 * a search line with the entry count, the history newest first, keycaps.
 *
 *   - text entries: one line, cut with "...";
 *   - images: a picture copied to the clipboard (misa-clip stores it as a
 *     PNG file and lists its path) or a copied image path / file:// URI,
 *     shown as the picture itself across the panel, no text;
 *   - type to filter (text, or an image's file name), up / down to move,
 *     Enter or click to put the entry back on the clipboard, Shift+Delete
 *     to delete it, Esc or a click outside to close.
 *
 * Behind the panel the monitor is dimmed and blurred (bin/misa-backdrop).
 *
 * Resident: `ClipPanel --daemon` (from bspwmrc) builds the window once and
 * waits, hidden, on a FIFO ($XDG_RUNTIME_DIR/clip-panel.fifo); bin/ClipMenu
 * writes "open" or "open --pointer" to it. Without the daemon, ClipMenu runs
 * `ClipPanel [--pointer]` once (C + GTK starts in a few ms anyway).
 *
 * Images are decoded at their shown width, one per idle turn after the
 * panel is up (sizes come from the file header first, so nothing jumps),
 * and the scaled picture is cached in ~/.cache/misa-clip/thumbs (path +
 * mtime + size), so the next opening only reads small PNGs. Rows and
 * pictures are dropped on close.
 *
 * An override-redirect window with a keyboard / pointer grab (like rofi);
 * picom: class ClipPanel (the "Bar panels" rule).
 */
#include "common.h"

#include <glib-unix.h>

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define WIDTH 560
#define PAD 14
#define ROW_PAD 10
#define IMG_W (WIDTH - 2 * PAD - 2 * ROW_PAD)
#define IMG_MAX_H 240
#define LIST_MAX_H 470
#define GAP 6

typedef struct {
    GtkWidget *row, *holder;
    int idx;
    gboolean image;
    char *value, *key;
    int w, h;
} Row;

static struct {
    gboolean daemon, is_open, pointer;
    GtkWidget *win, *box, *entry, *count, *list, *scroll;
    GList *rows;          /* Row*, newest first */
    GList *pending;       /* Row* with a picture to load */
    guint load_id;
    Rect mon;
    char edge[8];
    int off, px;
    int shade;
    char thumbs[4096];
} S;

static const char *IMAGE_EXT[] = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif",
                                  ".svg", ".tif", ".tiff", NULL};

/* -- misa-clip ------------------------------------------------------------- */
static char *clip(const char *a, const char *b) {
    const char *argv[] = {"misa-clip", a, b, NULL};
    return run_out(argv);
}

static char *image_path(const char *text) {
    char *first = g_strdup(text);
    g_strstrip(first);
    char *nl = strchr(first, '\n');
    if (nl) *nl = 0;
    g_strstrip(first);
    char *p = NULL;
    if (g_str_has_prefix(first, "file://")) {
        p = g_uri_unescape_string(first + 7, NULL);
    } else if (g_str_has_prefix(first, "~/")) {
        p = g_build_filename(g_get_home_dir(), first + 2, NULL);
    } else {
        p = g_strdup(first);
    }
    g_free(first);
    if (!p || p[0] != '/') { g_free(p); return NULL; }
    char *low = g_ascii_strdown(p, -1);
    gboolean ext = FALSE;
    for (int i = 0; IMAGE_EXT[i]; i++) {
        if (g_str_has_suffix(low, IMAGE_EXT[i])) { ext = TRUE; break; }
    }
    g_free(low);
    if (!ext || !g_file_test(p, G_FILE_TEST_IS_REGULAR)) { g_free(p); return NULL; }
    return p;
}

/* -- placement -------------------------------------------------------------- */
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
        x = MAX(S.mon.x + 8, MIN(x, S.mon.x + S.mon.w - WIDTH - 8));
    } else {
        x = S.mon.x + 8;
    }
    int y = strcmp(S.edge, "top") == 0 ? S.mon.y + S.off + GAP : S.mon.y + S.mon.h - S.off - GAP - h;
    int cx, cy;
    gtk_window_get_position(GTK_WINDOW(S.win), &cx, &cy);
    if (cx != x || cy != y) gtk_window_move(GTK_WINDOW(S.win), x, y);
}

/* Window height = the contents' (the list scrolls past LIST_MAX_H). */
static void fit(void) {
    GtkRequisition nat;
    gtk_widget_get_preferred_size(S.box, NULL, &nat);
    gtk_window_resize(GTK_WINDOW(S.win), WIDTH, MAX(1, nat.height));
    place(nat.height);
}

/* -- rows -------------------------------------------------------------------- */
static Row *row_of(GtkWidget *w) {
    return w ? g_object_get_data(G_OBJECT(w), "row") : NULL;
}

static gboolean keep(GtkListBoxRow *lr, gpointer _) {
    Row *r = row_of(GTK_WIDGET(lr));
    const char *q = gtk_entry_get_text(GTK_ENTRY(S.entry));
    char *ql = g_ascii_strdown(q, -1);
    g_strstrip(ql);
    gboolean ok = TRUE;
    if (*ql) {
        char **words = g_strsplit_set(ql, " \t", -1);
        for (int i = 0; words[i]; i++) {
            if (*words[i] && !strstr(r->key, words[i])) { ok = FALSE; break; }
        }
        g_strfreev(words);
    }
    g_free(ql);
    return ok;
}

static GList *visible_rows(void) {
    GList *v = NULL;
    for (GList *l = S.rows; l; l = l->next) {
        Row *r = l->data;
        if (keep(GTK_LIST_BOX_ROW(r->row), NULL)) v = g_list_append(v, r);
    }
    return v;
}

static void update_count(void) {
    GList *v = visible_rows();
    char t[32];
    g_snprintf(t, sizeof t, "%u / %u", g_list_length(v), g_list_length(S.rows));
    gtk_label_set_text(GTK_LABEL(S.count), t);
    g_list_free(v);
}

static void select_first(void) {
    GList *v = visible_rows();
    if (v) {
        gtk_list_box_select_row(GTK_LIST_BOX(S.list), GTK_LIST_BOX_ROW(((Row *)v->data)->row));
        gtk_adjustment_set_value(gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(S.scroll)), 0);
    }
    g_list_free(v);
}

static void add_class(GtkWidget *w, const char *c) {
    gtk_style_context_add_class(gtk_widget_get_style_context(w), c);
}

static Row *make_row(int idx, gboolean image, const char *value) {
    Row *r = g_new0(Row, 1);
    r->idx = idx;
    r->image = image;
    r->value = g_strdup(value);
    r->row = gtk_list_box_row_new();
    g_object_set_data(G_OBJECT(r->row), "row", r);
    add_class(r->row, image ? "image" : "text");
    if (image) {
        char *base = g_path_get_basename(value);
        r->key = g_ascii_strdown(base, -1);
        g_free(base);
        /* Always the full panel width; the height follows the picture,
         * capped (a tall one is cropped to its middle). */
        r->w = IMG_W;
        r->h = IMG_MAX_H / 2;
        int fw = 0, fh = 0;
        if (gdk_pixbuf_get_file_info(value, &fw, &fh) && fw > 0) {
            r->h = MAX(48, MIN(IMG_MAX_H, (int)(fh * (double)IMG_W / fw + 0.5)));
        }
        r->holder = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
        gtk_widget_set_size_request(r->holder, r->w, r->h);
        gtk_widget_set_halign(r->holder, GTK_ALIGN_CENTER);
        add_class(r->holder, "placeholder");
        gtk_container_add(GTK_CONTAINER(r->row), r->holder);
        S.pending = g_list_append(S.pending, r);
    } else {
        r->key = g_ascii_strdown(value, -1);
        GtkWidget *lb = gtk_label_new(value);
        gtk_label_set_xalign(GTK_LABEL(lb), 0);
        gtk_label_set_ellipsize(GTK_LABEL(lb), PANGO_ELLIPSIZE_END);
        gtk_label_set_max_width_chars(GTK_LABEL(lb), 1);
        add_class(lb, "entry-text");
        gtk_container_add(GTK_CONTAINER(r->row), lb);
    }
    return r;
}

static void free_row(Row *r) {
    g_free(r->value);
    g_free(r->key);
    g_free(r);
}

static void load_entries(void) {
    char *out = clip("list", NULL);
    if (!out) return;
    char **lines = g_strsplit(out, "\n", -1);
    int n = g_strv_length(lines);
    if (n && lines[n - 1][0] == 0) n--;
    for (int i = 0; i < n; i++) {
        const char *line = lines[i];
        Row *r = NULL;
        if (g_str_has_prefix(line, "/") || g_str_has_prefix(line, "~/") ||
            g_str_has_prefix(line, "file://")) {
            char idx[16];
            g_snprintf(idx, sizeof idx, "%d", i);
            char *full = clip("get", idx);
            char *img = full ? image_path(full) : NULL;
            if (img) r = make_row(i, TRUE, img);
            g_free(img);
            g_free(full);
        }
        if (!r) r = make_row(i, FALSE, line);
        S.rows = g_list_append(S.rows, r);
        gtk_container_add(GTK_CONTAINER(S.list), r->row);
        gtk_widget_show_all(r->row);
    }
    g_strfreev(lines);
    g_free(out);
}

/* -- pictures ---------------------------------------------------------------- */
static GdkPixbuf *scaled(const char *path, int w, int h) {
    struct stat st;
    if (stat(path, &st) != 0) return NULL;
    long long ns = (long long)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
    char *k = g_strdup_printf("%s %lld %dx%d", path, ns, w, h);
    char *sum = g_compute_checksum_for_string(G_CHECKSUM_SHA1, k, -1);
    sum[24] = 0;
    char *cached = g_strdup_printf("%s/%s.png", S.thumbs, sum);
    g_free(k);
    g_free(sum);
    GdkPixbuf *pb = gdk_pixbuf_new_from_file(cached, NULL);
    if (!pb) {
        /* decoded at the shown width only (a 4K screenshot never sits in
         * memory at full size), then cropped to w x h */
        GdkPixbuf *src = gdk_pixbuf_new_from_file_at_scale(path, w, -1, TRUE, NULL);
        if (src) {
            pb = cover_pixbuf(src, w, h, 10, FALSE);
            g_object_unref(src);
            g_mkdir_with_parents(S.thumbs, 0700);
            gdk_pixbuf_save(pb, cached, "png", NULL, NULL);
        }
    }
    g_free(cached);
    return pb;
}

static gboolean load_next(gpointer _) {
    if (!S.pending) { S.load_id = 0; return G_SOURCE_REMOVE; }
    Row *r = S.pending->data;
    S.pending = g_list_delete_link(S.pending, S.pending);
    GdkPixbuf *pb = scaled(r->value, r->w, r->h);
    if (pb && r->holder && gtk_widget_get_parent(r->holder) == r->row) {
        gtk_container_remove(GTK_CONTAINER(r->row), r->holder);
        r->holder = NULL;
        GtkWidget *img = gtk_image_new_from_pixbuf(pb);
        gtk_widget_set_halign(img, GTK_ALIGN_CENTER);
        gtk_container_add(GTK_CONTAINER(r->row), img);
        gtk_widget_show(img);
    }
    if (pb) g_object_unref(pb);
    if (!S.pending) { S.load_id = 0; return G_SOURCE_REMOVE; }
    return G_SOURCE_CONTINUE;
}

/* Drop cached pictures not used for a month (their entry is long gone). */
static gboolean prune(gpointer _) {
    GDir *d = g_dir_open(S.thumbs, 0, NULL);
    if (!d) return G_SOURCE_REMOVE;
    time_t old = time(NULL) - 30 * 86400;
    const char *n;
    while ((n = g_dir_read_name(d))) {
        char *p = g_build_filename(S.thumbs, n, NULL);
        struct stat st;
        if (stat(p, &st) == 0 && st.st_atime < old && st.st_mtime < old) unlink(p);
        g_free(p);
    }
    g_dir_close(d);
    heap_trim();
    return G_SOURCE_REMOVE;
}

/* -- open / close ------------------------------------------------------------- */
static void panel_close(void) {
    if (!S.is_open) return;
    S.is_open = FALSE;
    ungrab();
    gtk_widget_hide(S.win);
    drop_backdrop(S.shade);
    S.shade = 0;
    if (!S.daemon) {
        gtk_main_quit();
        return;
    }
    /* Hidden again: drop rows and pictures, hand the heap back. */
    if (S.load_id) { g_source_remove(S.load_id); S.load_id = 0; }
    g_list_free(S.pending);
    S.pending = NULL;
    for (GList *l = S.rows; l; l = l->next) {
        Row *r = l->data;
        gtk_widget_destroy(r->row);
        free_row(r);
    }
    g_list_free(S.rows);
    S.rows = NULL;
    g_idle_add(prune, NULL);
}

static void panel_open(gboolean pointer) {
    if (S.is_open) { panel_close(); return; }
    close_bar_popups("clip");
    S.is_open = TRUE;
    S.pointer = pointer;
    anchor(pointer);
    gtk_entry_set_text(GTK_ENTRY(S.entry), "");
    load_entries();
    update_count();
    select_first();
    fit();
    S.shade = backdrop(S.mon, 0, NULL);        /* under the panel: map it first */
    gtk_widget_show(S.win);
    gdk_window_raise(gtk_widget_get_window(S.win));
    grab_soon(S.win, G_CALLBACK(panel_close), NULL);
    gtk_widget_grab_focus(S.entry);
    /* Pictures after the panel is up, one per idle turn. */
    if (S.pending && !S.load_id) S.load_id = g_idle_add(load_next, NULL);
}

/* -- actions ------------------------------------------------------------------ */
static void pick(GtkListBoxRow *lr) {
    Row *r = row_of(GTK_WIDGET(lr));
    if (!r) return;
    char idx[16];
    g_snprintf(idx, sizeof idx, "%d", r->idx);
    g_free(clip("select", idx));
    panel_close();
}

static void move_sel(int step) {
    GList *v = visible_rows();
    if (!v) return;
    Row *cur = row_of(GTK_WIDGET(gtk_list_box_get_selected_row(GTK_LIST_BOX(S.list))));
    int n = g_list_length(v), i = cur ? g_list_index(v, cur) : -1;
    i = MAX(0, MIN(n - 1, i + step));
    Row *r = g_list_nth_data(v, i);
    gtk_list_box_select_row(GTK_LIST_BOX(S.list), GTK_LIST_BOX_ROW(r->row));
    /* Keep it in view. */
    GtkAllocation a;
    gtk_widget_get_allocation(r->row, &a);
    GtkAdjustment *adj = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(S.scroll));
    double val = gtk_adjustment_get_value(adj), page = gtk_adjustment_get_page_size(adj);
    if (a.y < val) gtk_adjustment_set_value(adj, a.y);
    else if (a.y + a.height > val + page) gtk_adjustment_set_value(adj, a.y + a.height - page);
    g_list_free(v);
}

static void delete_sel(void) {
    GtkListBoxRow *lr = gtk_list_box_get_selected_row(GTK_LIST_BOX(S.list));
    Row *r = row_of(GTK_WIDGET(lr));
    if (!r) return;
    GList *v = visible_rows();
    int pos = MAX(0, g_list_index(v, r));
    g_list_free(v);
    char idx[16];
    g_snprintf(idx, sizeof idx, "%d", r->idx);
    g_free(clip("delete", idx));
    for (GList *l = S.rows; l; l = l->next) {      /* history indices shift up */
        Row *o = l->data;
        if (o->idx > r->idx) o->idx--;
    }
    S.rows = g_list_remove(S.rows, r);
    S.pending = g_list_remove(S.pending, r);
    gtk_widget_destroy(r->row);
    free_row(r);
    v = visible_rows();
    if (v) {
        Row *s = g_list_nth_data(v, MIN(pos, (int)g_list_length(v) - 1));
        gtk_list_box_select_row(GTK_LIST_BOX(S.list), GTK_LIST_BOX_ROW(s->row));
    }
    g_list_free(v);
    update_count();
    fit();
}

static void on_filter(GtkEditable *_e, gpointer _d) {
    gtk_list_box_invalidate_filter(GTK_LIST_BOX(S.list));
    update_count();
    select_first();
    fit();
}

static gboolean on_key(GtkWidget *_w, GdkEventKey *ev, gpointer _d) {
    guint k = ev->keyval;
    gboolean shift = ev->state & GDK_SHIFT_MASK;
    if (k == GDK_KEY_Escape) panel_close();
    else if ((k == GDK_KEY_Down || k == GDK_KEY_Tab) && !(k == GDK_KEY_Tab && shift)) move_sel(1);
    else if (k == GDK_KEY_Up || k == GDK_KEY_ISO_Left_Tab || (k == GDK_KEY_Tab && shift)) move_sel(-1);
    else if (k == GDK_KEY_Page_Down) move_sel(5);
    else if (k == GDK_KEY_Page_Up) move_sel(-5);
    else if (k == GDK_KEY_Return || k == GDK_KEY_KP_Enter)
        pick(gtk_list_box_get_selected_row(GTK_LIST_BOX(S.list)));
    else if (k == GDK_KEY_Delete && shift) delete_sel();
    else return FALSE;                       /* typing goes to the search entry */
    return TRUE;
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

static void on_activated(GtkListBox *_l, GtkListBoxRow *row, gpointer _d) {
    pick(row);
}

static void on_alloc(GtkWidget *_w, GdkRectangle *_a, gpointer _d) {
    place(0);
}

/* -- build -------------------------------------------------------------------- */
static char *css(const Palette *p) {
    const char *fg = p->fg;
    return g_strdup_printf(
        "* { font-family: \"%s\"; }\n"
        "window { background: transparent; }\n"
        ".panel {\n"
        "    background-color: %s;\n"
        "    border: 1px solid alpha(%s, 0.12);\n"
        "    border-radius: 14px;\n"
        "    padding: %dpx;\n"
        "}\n"
        ".search { border-bottom: 1px solid alpha(%s, 0.12); padding: 4px 6px 14px 6px; }\n"
        ".glyph { font-family: \"Material Design Icons Desktop\"; font-size: 18px; color: alpha(%s, 0.55); }\n"
        "entry {\n"
        "    background: none; border: none; box-shadow: none; outline: none;\n"
        "    padding: 0px; min-height: 0px;\n"
        "    font-size: 15px; font-weight: 500; color: %s; caret-color: %s;\n"
        "}\n"
        "entry selection { background-color: alpha(%s, 0.25); }\n"
        ".count { font-family: \"JetBrainsMono Nerd Font\"; font-size: 9px; color: alpha(%s, 0.55); }\n"
        "scrolledwindow, viewport, list { background: transparent; border: none; }\n"
        "scrollbar { opacity: 0; }\n"
        "row {\n"
        "    background: transparent; border-radius: 10px; outline: none;\n"
        "    transition: background-color 120ms ease-out;\n"
        "}\n"
        "row.text { padding: 10px 12px; }\n"
        "row.image { padding: %dpx; }\n"
        "row:hover { background-color: alpha(%s, 0.06); }\n"
        "row:selected { background-color: alpha(%s, 0.18); }\n"
        ".entry-text { font-size: 11pt; font-weight: 500; color: %s; }\n"
        ".placeholder { background-color: alpha(%s, 0.05); border-radius: 10px; }\n"
        ".empty { font-size: 11pt; color: alpha(%s, 0.55); padding: 18px; }\n"
        ".footer { border-top: 1px solid alpha(%s, 0.12); padding: 10px 4px 0px 4px; }\n"
        ".key {\n"
        "    font-family: \"%s\"; font-size: 9px; color: %s;\n"
        "    border: 1px solid alpha(%s, 0.55); border-radius: 5px; padding: 2px 6px;\n"
        "}\n"
        ".keylabel { font-family: \"%s\"; font-size: 9px; color: alpha(%s, 0.55); margin-right: 12px; }\n",
        p->font, p->bg, fg, PAD, fg, fg, fg, fg, fg, fg, ROW_PAD, fg, fg, fg, fg, fg, fg,
        p->display, fg, fg, p->display, fg);
}

static void build(void) {
    Palette p;
    palette_load(&p);
    char *c = css(&p);
    css_load(c);
    g_free(c);

    S.win = overlay_window("ClipPanel", 0, 0, WIDTH, 100);
    gtk_widget_set_size_request(S.win, WIDTH, -1);
    gtk_window_set_default_size(GTK_WINDOW(S.win), -1, -1);   /* size from the contents */
    gtk_window_set_resizable(GTK_WINDOW(S.win), TRUE);
    gtk_widget_add_events(S.win, GDK_KEY_PRESS_MASK | GDK_BUTTON_PRESS_MASK);
    g_signal_connect(S.win, "key-press-event", G_CALLBACK(on_key), NULL);
    g_signal_connect(S.win, "button-press-event", G_CALLBACK(on_outside), NULL);
    g_signal_connect(S.win, "size-allocate", G_CALLBACK(on_alloc), NULL);

    /* Search line. */
    GtkWidget *glyph = gtk_label_new("\xf3\xb0\x85\x8c");        /* U+F014C clipboard */
    add_class(glyph, "glyph");
    S.entry = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(S.entry), "Search clipboard");
    g_signal_connect(S.entry, "changed", G_CALLBACK(on_filter), NULL);
    S.count = gtk_label_new("");
    add_class(S.count, "count");
    GtkWidget *search = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    add_class(search, "search");
    gtk_box_pack_start(GTK_BOX(search), glyph, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(search), S.entry, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(search), S.count, FALSE, FALSE, 0);

    /* The history. */
    S.list = gtk_list_box_new();
    gtk_list_box_set_selection_mode(GTK_LIST_BOX(S.list), GTK_SELECTION_BROWSE);
    gtk_list_box_set_activate_on_single_click(GTK_LIST_BOX(S.list), TRUE);
    g_signal_connect(S.list, "row-activated", G_CALLBACK(on_activated), NULL);
    gtk_list_box_set_filter_func(GTK_LIST_BOX(S.list), keep, NULL, NULL);
    GtkWidget *empty = gtk_label_new("Nothing copied yet");
    add_class(empty, "empty");
    gtk_list_box_set_placeholder(GTK_LIST_BOX(S.list), empty);
    gtk_widget_show(empty);
    S.scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(S.scroll), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_propagate_natural_height(GTK_SCROLLED_WINDOW(S.scroll), TRUE);
    gtk_scrolled_window_set_max_content_height(GTK_SCROLLED_WINDOW(S.scroll), LIST_MAX_H);
    gtk_container_add(GTK_CONTAINER(S.scroll), S.list);

    /* Keycaps. */
    GtkWidget *footer = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    add_class(footer, "footer");
    const char *keys[][2] = {{"ENTER", "COPY"}, {"SHIFT DEL", "DELETE"}, {"ESC", "CLOSE"}};
    for (int i = 0; i < 3; i++) {
        GtkWidget *k = gtk_label_new(keys[i][0]);
        add_class(k, "key");
        GtkWidget *lb = gtk_label_new(keys[i][1]);
        add_class(lb, "keylabel");
        gtk_box_pack_start(GTK_BOX(footer), k, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(footer), lb, FALSE, FALSE, 0);
    }

    S.box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    add_class(S.box, "panel");
    gtk_box_pack_start(GTK_BOX(S.box), search, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.box), S.scroll, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.box), footer, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(S.win), S.box);
    gtk_widget_show_all(S.box);
    gtk_widget_realize(S.win);
}

/* -- daemon -------------------------------------------------------------------- */
static char fifo_buf[4096];
static size_t fifo_len;

static gboolean on_fifo(gint fd, GIOCondition _c, gpointer _d) {
    ssize_t n = read(fd, fifo_buf + fifo_len, sizeof fifo_buf - 1 - fifo_len);
    if (n <= 0) return G_SOURCE_CONTINUE;
    fifo_len += n;
    fifo_buf[fifo_len] = 0;
    char *start = fifo_buf, *nl;
    while ((nl = strchr(start, '\n'))) {
        *nl = 0;
        if (g_str_has_prefix(start, "open")) panel_open(strstr(start, "--pointer") != NULL);
        else if (g_str_has_prefix(start, "close")) panel_close();
        else if (g_str_has_prefix(start, "quit")) gtk_main_quit();
        start = nl + 1;
    }
    fifo_len = strlen(start);
    memmove(fifo_buf, start, fifo_len + 1);
    return G_SOURCE_CONTINUE;
}

static int serve(void) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    if (!run) run = "/tmp";
    char *pidfile = g_build_filename(run, "clip-panel.pid", NULL);
    char *fifo = g_build_filename(run, "clip-panel.fifo", NULL);
    int lock = open(pidfile, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (lock < 0 || flock(lock, LOCK_EX | LOCK_NB) != 0) return 0;   /* one is running */
    if (ftruncate(lock, 0) == 0) {
        char t[32];
        int n = g_snprintf(t, sizeof t, "%d\n", (int)getpid());
        if (write(lock, t, n) < 0) { /* the lock still holds */ }
    }
    unlink(fifo);
    if (mkfifo(fifo, 0600) != 0 && errno != EEXIST) return 1;
    /* O_RDWR: the FIFO never reports EOF between writers. */
    int fd = open(fifo, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return 1;
    g_unix_fd_add(fd, G_IO_IN, on_fifo, NULL);
    heap_trim();
    gtk_main();
    return 0;
}

int main(int argc, char **argv) {
    gboolean daemon = FALSE, pointer = FALSE;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--daemon")) daemon = TRUE;
        else if (!strcmp(argv[i], "--pointer")) pointer = TRUE;
    }
    gtk_init(&argc, &argv);
    const char *cache = g_getenv("XDG_CACHE_HOME");
    char *base = cache && *cache ? g_strdup(cache) : g_build_filename(g_get_home_dir(), ".cache", NULL);
    g_snprintf(S.thumbs, sizeof S.thumbs, "%s/misa-clip/thumbs", base);
    g_free(base);
    S.daemon = daemon;
    build();
    if (daemon) return serve();
    panel_open(pointer);
    gtk_main();
    prune(NULL);           /* after the panel is gone: never delays opening */
    return 0;
}
