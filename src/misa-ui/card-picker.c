/* CardPicker - the full-screen pickers of the rice, over the dimmed, blurred
 * desktop (picom: class CardPicker).
 *
 *     printf 'TITLE\tsubtitle\t/path/image\n' ... | CardPicker [options]
 *
 *     --layout gallery   wallpapers: the selected one fills the screen as it
 *                        will look (the bar stays above it, sharp), picked
 *                        from a filmstrip of thumbnails in a dock below
 *     --layout showcase  big names on the left, the big preview of the
 *                        selected one on the right (rices)
 *     --layout strips    the pictures themselves in a vertical list, all at
 *                        one scale (bars: each picture is a bar strip)
 *     --card WxH         preview aspect (showcase) / canvas width (strips)
 *     --select I         start on item I       --active I   mark "IN USE"
 *     --hint-move TEXT   the keycap of the MOVE hint
 *
 * A 4th column (showcase) is a picture under the preview: the preview is
 * then laid on it like a screenshot on a desk (rices: their wallpaper).
 * gallery: the 3rd column is the thumbnail, the 4th the wallpaper itself; a
 * line without a thumbnail is the "add" tile.
 *
 * Prints the index (0-based) of the chosen item; exit 1 when closed.
 *
 * Why GTK and not rofi: rofi cannot animate anything inside its window.
 * gallery:  the wallpaper cross-fades (the thumbnail at once, the full
 *           picture decoded in a thread), the filmstrip glides.
 * showcase: the selected name slides in and brightens, the list glides to
 *           keep it centred, the preview slides to the next one.
 * strips:   the selected strip slides in and lights up, the list glides.
 *
 * Keys: arrows / Tab / Home / End move, Enter or click picks, Esc, right
 * click or a click on the backdrop closes, the wheel moves. Hovering selects
 * (gallery: a click selects, a click on the selected one or Enter applies).
 * The window leaves the bar uncovered (below_bar), so it never blurs.
 */
#include "common.h"
#include "live-gl.h"

#include <glib/gstdio.h>
#include <malloc.h>
#include <math.h>
#include <sys/stat.h>
#include <utime.h>
#include <stdio.h>
#include <string.h>

#define LIFT 16       /* px rows rise while coming in */
#define TW 176        /* gallery thumbnail */
#define TH 99

enum { GALLERY, SHOWCASE, STRIPS };

typedef struct { char *title, *sub, *image, *under; } Item;

static struct {
    int layout, card_w, card_h, select, active;
    char *hint_move;
} A = {GALLERY, 16, 9, -1, -1, NULL};

static struct {
    Item *items;
    int n, cur, result;
    GtkWidget *win, *stack;
    GtkWidget **marks;
    gboolean by_click;               /* selected by a click: the list stays put */
    Glide glide;
} S = {.cur = -1, .result = -1};

static char *css(const Palette *p, int n) {
    GString *st = g_string_new("");
    for (int i = 0; i < MIN(n, 14); i++)
        g_string_append_printf(st, "%s.item-%d { animation-delay: %dms; }", i ? "\n" : "", i, 30 + i * 35);
    const char *fg = p->fg, *bg = p->bg, *font = p->font, *disp = p->display;
    GString *s = g_string_new("");
    g_string_append_printf(s,
        "* { font-family: \"%s\"; }\n"
        "window { background-color: alpha(black, 0.74); }\n"
        "scrolledwindow, viewport, flowbox { background: transparent; border: none; padding: 0; margin: 0; opacity: 1; }\n"
        "scrollbar { opacity: 0; }\n"
        "flowboxchild { padding: 0; outline: none; background: transparent; }\n"
        "\n"
        "/* coming in, one after another */\n"
        ".enter { animation: enter 460ms cubic-bezier(0.2, 0.9, 0.25, 1) both; }\n"
        "@keyframes enter {\n"
        "    from { opacity: 0; padding-top: %dpx; }\n"
        "    to   { opacity: 1; padding-top: 0px; }\n"
        "}\n"
        "%s\n", font, LIFT + 60, st->str);
    g_string_append_printf(s,
        "\n/* ---- gallery ---- */\n"
        "window.gallery { background-color: black; }\n"
        ".dock {\n"
        "    background-color: %s;\n"
        "    border: 1px solid alpha(%s, 0.12);\n"
        "    border-radius: 18px;\n"
        "    padding: 18px 20px 12px 20px;\n"
        "    box-shadow: 0 18px 48px alpha(black, 0.55);\n"
        "    animation: rise 420ms cubic-bezier(0.2, 0.9, 0.25, 1) both;\n"
        "}\n"
        "@keyframes rise {\n"
        "    from { opacity: 0; margin-bottom: 0px; }\n"
        "    to   { opacity: 1; margin-bottom: 28px; }\n"
        "}\n"
        ".g-title { font-size: 16px; font-weight: 600; color: %s; }\n"
        ".g-meta { font-size: 12px; font-weight: 600; color: alpha(%s, 0.55); }\n"
        ".g-meta-row .badge { margin: 0px 0px 0px 8px; padding: 2px 7px; box-shadow: none; }\n"
        ".tframe {\n"
        "    padding: 3px; margin-top: 6px; margin-bottom: 0px;\n"
        "    border: 2px solid transparent; border-radius: 14px;\n"
        "    opacity: 0.55;\n"
        "    transition: all 220ms cubic-bezier(0.25, 1.35, 0.45, 1);\n"
        "}\n"
        ".th:hover .tframe { opacity: 0.88; }\n"
        ".tile.sel .tframe { border-color: %s; opacity: 1; margin-top: 0px; margin-bottom: 6px; }\n"
        ".dot { min-width: 4px; min-height: 4px; border-radius: 2px; margin-top: 2px; }\n"
        ".tile.active .dot { background-color: %s; }\n"
        ".addtile {\n"
        "    border: 1px dashed alpha(%s, 0.3); border-radius: 10px;\n"
        "    background-color: alpha(%s, 0.04);\n"
        "}\n"
        ".thumb-slot {\n"
        "    border-radius: 10px;\n"
        "    background-color: alpha(%s, 0.05);\n"
        "}\n"
        ".plus { font-size: 30px; font-weight: 300; color: alpha(%s, 0.85); }\n"
        ".apply, .apply:hover, .apply:active {\n"
        "    background-image: none; border: none; box-shadow: none; text-shadow: none;\n"
        "    -gtk-icon-shadow: none; outline: none;\n"
        "    border-radius: 10px; padding: 7px 16px; margin-left: 8px;\n"
        "}\n"
        ".apply { background-color: %s; transition: background-color 160ms ease-out; }\n"
        ".apply:hover { background-color: mix(%s, %s, 0.15); }\n"
        ".apply:active { background-color: mix(%s, %s, 0.3); }\n"
        ".apply label { color: %s; font-size: 13px; font-weight: 700; }\n",
        bg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, fg, bg, fg, bg, bg);
    g_string_append_printf(s,
        ".badge {\n"
        "    margin: 10px; padding: 3px 9px; border-radius: 7px;\n"
        "    background-color: %s; color: %s;\n"
        "    font-family: \"%s\"; font-size: 10px;\n"
        "    box-shadow: 0 2px 8px alpha(black, 0.5);\n"
        "}\n",
        fg, bg, disp);
    g_string_append_printf(s,
        "\n/* ---- showcase ---- */\n"
        ".row {\n"
        "    padding: 6px 0px 6px 0px;\n"
        "    opacity: 0.28;\n"
        "    transition: all 260ms cubic-bezier(0.25, 1.3, 0.45, 1);\n"
        "}\n"
        ".row.sel { padding-left: 22px; opacity: 1; }\n"
        ".num { font-family: \"%s\"; font-size: 13px; color: alpha(%s, 0.6); min-width: 30px; }\n"
        ".name { font-family: \"%s\"; font-size: 34px; color: %s; }\n"
        ".row .badge { margin: 0px 0px 0px 12px; }\n"
        ".preview { border-radius: 16px; }\n"
        "\n/* ---- strips ---- */\n"
        ".strip {\n"
        "    padding: 0px;\n"
        "    opacity: 0.32;\n"
        "    transition: all 260ms cubic-bezier(0.25, 1.3, 0.45, 1);\n"
        "}\n"
        ".strip.sel { padding-left: 22px; opacity: 1; }\n"
        ".strip-name { font-family: \"%s\"; font-size: 18px; color: %s; }\n"
        ".strip-sub { font-size: 12px; font-weight: 600; color: alpha(%s, 0.55); }\n"
        ".strip .badge { margin: 0px 0px 0px 6px; }\n"
        ".plate {\n"
        "    padding: 16px;\n"
        "    border-radius: 14px;\n"
        "    border: 2px solid alpha(%s, 0.06);\n"
        "    background-color: mix(%s, %s, 0.05);\n"
        "    transition: border-color 220ms ease-out;\n"
        "}\n"
        ".strip.sel .plate { border-color: %s; }\n",
        disp, fg, disp, fg, disp, fg, fg, fg, bg, fg, fg);
    g_string_append_printf(s,
        "\n/* ---- key hints ---- */\n"
        ".key {\n"
        "    font-family: \"%s\"; font-size: 11px; color: %s;\n"
        "    border: 2px solid %s; border-radius: 5px; padding: 2px 7px;\n"
        "}\n"
        ".key-label {\n"
        "    font-family: \"%s\"; font-size: 11px;\n"
        "    color: alpha(%s, 0.55); margin-right: 16px;\n"
        "}\n",
        disp, fg, fg, disp, fg);
    g_string_free(st, TRUE);
    return g_string_free(s, FALSE);
}

static void add_class(GtkWidget *w, const char *c) {
    gtk_style_context_add_class(gtk_widget_get_style_context(w), c);
}

static GtkWidget *label(const char *text, const char *cls, gboolean ellipsize) {
    GtkWidget *lb = gtk_label_new(text);
    gtk_label_set_xalign(GTK_LABEL(lb), 0);
    if (ellipsize) {
        gtk_label_set_ellipsize(GTK_LABEL(lb), PANGO_ELLIPSIZE_END);
        gtk_label_set_max_width_chars(GTK_LABEL(lb), 1);   /* width comes from the container */
    }
    char **cs = g_strsplit(cls, " ", -1);
    for (int i = 0; cs[i]; i++)
        if (*cs[i]) add_class(lb, cs[i]);
    g_strfreev(cs);
    return lb;
}

static GtkWidget *badge(void) {
    GtkWidget *b = gtk_label_new("IN USE");
    gtk_widget_set_halign(b, GTK_ALIGN_START);
    gtk_widget_set_valign(b, GTK_ALIGN_CENTER);
    add_class(b, "badge");
    return b;
}

static GtkWidget *scrolled(GtkWidget *child, GtkPolicyType h, GtkPolicyType v) {
    GtkWidget *sc = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(sc), h, v);
    gtk_container_add(GTK_CONTAINER(sc), child);
    return sc;
}

static void preview_selected(int i, gboolean animate);
static void showcase_cleanup(void);
static void gallery_selected(int i, gboolean animate);
static gboolean gallery_applying(void);
static void pick(int i);

/* -- selection ------------------------------------------------------------------- */
static void centre(gboolean animate) {
    GtkWidget *holder = gtk_widget_get_parent(S.marks[S.cur]);   /* the hover EventBox */
    double page = gtk_adjustment_get_page_size(S.glide.adj);
    GtkAllocation a;
    if (A.layout == GALLERY) {
        /* Tiles are in the viewport's own window: already absolute. Kept
         * centred, but the strip never scrolls past its ends (no half-empty
         * dock at the first or last one). */
        gtk_widget_get_allocation(holder, &a);
        double end = gtk_adjustment_get_upper(S.glide.adj) - page;
        glide_to(&S.glide, MAX(0, MIN(a.x + a.width / 2.0 - page / 2, end)), animate);
    } else {
        /* Rows are in the viewport's own window: already absolute. */
        gtk_widget_get_allocation(holder, &a);
        glide_to(&S.glide, a.y + a.height / 2.0 - page / 2, animate);
    }
}

static void select_item(int i, gboolean animate) {
    if (gallery_applying()) return;
    i = MAX(0, MIN(i, S.n - 1));
    if (i == S.cur) return;
    int prev = S.cur;
    S.cur = i;
    if (prev >= 0) gtk_style_context_remove_class(gtk_widget_get_style_context(S.marks[prev]), "sel");
    add_class(S.marks[i], "sel");
    if (A.layout == SHOWCASE) {
        preview_selected(i, animate && prev >= 0);
        if (S.stack) {
            char name[16];
            g_snprintf(name, sizeof name, "%d", i);
            GtkStackTransitionType kind = !animate || prev < 0 ? GTK_STACK_TRANSITION_TYPE_NONE
                                          : i > prev           ? GTK_STACK_TRANSITION_TYPE_SLIDE_UP
                                                               : GTK_STACK_TRANSITION_TYPE_SLIDE_DOWN;
            gtk_stack_set_visible_child_full(GTK_STACK(S.stack), name, kind);
        }
    } else if (A.layout == GALLERY) {
        gallery_selected(i, animate && prev >= 0);
    }
    /* The pointer picks what is under it; the list moving away under it
     * would make the cursor scroll the list on its own. */
    if (!S.by_click) centre(animate);
}

static void gallery_cleanup(void);
static void gallery_confirm(int i);

static void quit(void) {
    gallery_cleanup();
    showcase_cleanup();
    ungrab();
    gtk_main_quit();
}

static void pick(int i) {
    S.result = i;
    if (A.layout == GALLERY && i > 0 && i < S.n && *S.items[i].under) {
        gallery_confirm(i);
        return;
    }
    quit();
}

/* -- input --------------------------------------------------------------------------- */
/* A click selects; a click on the selected item (or Enter) applies it. */
static gboolean on_item_click(GtkWidget *_w, GdkEventButton *ev, gpointer data) {
    if (ev->button != 1 || ev->type != GDK_BUTTON_PRESS) return ev->button == 1;
    int i = GPOINTER_TO_INT(data);
    if (i == S.cur) { pick(i); return TRUE; }
    /* The list does not glide: a second click on the same spot must still
     * hit the item (and apply it), not the empty space next to it. */
    S.by_click = TRUE;
    select_item(i, TRUE);
    S.by_click = FALSE;
    return TRUE;
}

static gboolean on_preview_click(GtkWidget *_w, GdkEventButton *ev, gpointer _) {
    if (ev->button == 1) {
        pick(S.cur);
        return TRUE;
    }
    return FALSE;
}

static gboolean on_button(GtkWidget *_w, GdkEventButton *_e, gpointer _) {
    /* Reached only when no item took the click: the backdrop. Close. */
    if (!gallery_applying()) quit();
    return TRUE;
}

static gboolean on_key(GtkWidget *_w, GdkEventKey *ev, gpointer _) {
    if (gallery_applying()) return TRUE;
    guint k = ev->keyval;
    int move = 0;
    switch (k) {
        case GDK_KEY_Tab: move = 1; break;
        case GDK_KEY_ISO_Left_Tab: move = -1; break;
        case GDK_KEY_Right: move = 1; break;
        case GDK_KEY_Left: move = -1; break;
        case GDK_KEY_Down: move = 1; break;
        case GDK_KEY_Up: move = -1; break;
    }
    if (k == GDK_KEY_Escape) quit();
    else if (k == GDK_KEY_Return || k == GDK_KEY_KP_Enter || k == GDK_KEY_space) pick(S.cur);
    else if (k == GDK_KEY_Home) select_item(0, TRUE);
    else if (k == GDK_KEY_End) select_item(S.n - 1, TRUE);
    else if (move) select_item(S.cur + move, TRUE);
    return TRUE;
}

static gboolean on_wheel(GtkWidget *_w, GdkEventScroll *ev, gpointer _) {
    int d = 0;
    if (ev->direction == GDK_SCROLL_UP || ev->direction == GDK_SCROLL_LEFT) d = -1;
    else if (ev->direction == GDK_SCROLL_DOWN || ev->direction == GDK_SCROLL_RIGHT) d = 1;
    else if (ev->direction == GDK_SCROLL_SMOOTH) {
        double dx = 0, dy = 0;
        gdk_event_get_scroll_deltas((GdkEvent *)ev, &dx, &dy);
        double v = dy != 0 ? dy : dx;
        d = v > 0 ? 1 : v < 0 ? -1 : 0;
    }
    if (d) select_item(S.cur + d, TRUE);
    return TRUE;
}

static GtkWidget *hoverable(GtkWidget *widget, int i) {
    GtkWidget *ev = gtk_event_box_new();
    gtk_container_add(GTK_CONTAINER(ev), widget);
    /* The pointer does not select (user's wish): only a click, the keys or
     * the wheel move the selection. */
    gtk_widget_add_events(ev, GDK_BUTTON_PRESS_MASK);
    g_signal_connect(ev, "button-press-event", G_CALLBACK(on_item_click), GINT_TO_POINTER(i));
    add_class(ev, "enter");
    char c[24];
    g_snprintf(c, sizeof c, "item-%d", i);
    add_class(ev, c);
    return ev;
}

/* -- gallery layout -------------------------------------------------------------------- */
/* The wallpaper fills the window exactly as misa-wall / feh --bg-fill will put
 * it on the monitor (cover, centred), minus the bar's part: what you see is
 * what you get, with the real bar above it.
 *
 * Transition: Mira's original Luma-Halftone Develop, matching misa-wall down
 * to the pixel. Rather than a directional sweep, the entire screen develops
 * organically at once through a crisp 4x4 ordered Bayer halftone lattice at
 * 4x4 pixel pitch guided by auto-leveled target luminance and soft harmonic
 * darkroom pockets, with a precomputed uint16 threshold map for near-zero CPU
 * cost per frame. */
#define DEV_W_BAYER    6881
#define DEV_W_LUMA     6226
#define DEV_W_ORG      3276
#define DEV_HALF_ORG   (DEV_W_ORG / 2)
#define DEV_MAX_THRESH (DEV_W_BAYER + DEV_W_LUMA + DEV_W_ORG)
#define DEV_FEATHER     960
#define DEV_RECIP      ((256 * 65536 + DEV_FEATHER / 2) / DEV_FEATHER)
#define DEV_DUR_BROWSE_MS 460
#define DEV_DUR_APPLY_MS  280

static const int BAYER4_RAW[4][4] = {
    { 0,  8,  2, 10},
    {12,  4, 14,  6},
    { 3, 11,  1,  9},
    {15,  7, 13,  5},
};

static struct {
    int mw, mh, top;                      /* monitor size, bar height above us */
    int ww, wh;                           /* window */
    GtkWidget *over, *canvas, *dock_ev;
    LiveGl *gl;
    gboolean use_gl;
    cairo_surface_t *surf;
    guchar *cur_buf, *src_buf, *tgt_buf;
    guint16 *thresh_buf;
    int *org_sa, *org_cc, *org_cb, *org_sd;
    int bayer4[4][4];
    gint64 trans_start_us;
    int trans_dur_ms, prev_head;
    gboolean has_cur, trans_done, full_ready, applying, alive;
    guint tick_id;
    int shown;
    guint gen;                            /* bumped per selection: late decodes are dropped */
    guint settle;                         /* the full decode waits until the selection rests */
    gboolean loading;                     /* a quick picture is being made in a thread */
    int pending;                          /* the selection that came in meanwhile, or -1 */
    GtkWidget *title, *meta, *badge, *apply;
    GtkWidget **thumb_img;                /* filmstrip images filled asynchronously */
    char **thumb_cache;                   /* shared on-disk 176x99 rounded PNG path per item */
    char **frag_path;                     /* sidecar <stem>.frag per item, or NULL for custom */
    GArray *thumb_order;                  /* cold-cache / off-screen tile indices queued after GL init */
    GCancellable *thumb_cancel;
} G = {.shown = -2, .pending = -1, .prev_head = -1};

static void develop_init_tables(void) {
    for (int by = 0; by < 4; by++)
        for (int bx = 0; bx < 4; bx++)
            G.bayer4[by][bx] = (BAYER4_RAW[by][bx] * DEV_W_BAYER) / 15;

    G.thresh_buf = g_new0(guint16, (size_t)G.ww * (size_t)G.wh);
    G.org_sa = g_new(int, G.ww);
    G.org_cc = g_new(int, G.ww);
    G.org_cb = g_new(int, G.wh);
    G.org_sd = g_new(int, G.wh);

    double inv_w = G.ww > 1 ? 1.0 / (G.ww - 1) : 0.0;
    double inv_h = G.wh > 1 ? 1.0 / (G.wh - 1) : 0.0;
    double tau = 2.0 * G_PI;

    for (int x = 0; x < G.ww; x++) {
        double u = x * inv_w;
        G.org_sa[x] = (int)llround(sin(tau * 1.35 * u + 0.4) * 256.0);
        G.org_cc[x] = (int)llround(cos(tau * 2.15 * u - 0.9) * 256.0);
    }
    for (int y = 0; y < G.wh; y++) {
        double v = y * inv_h;
        G.org_cb[y] = (int)llround(0.55 * cos(tau * 1.15 * v - 0.3) * DEV_HALF_ORG);
        G.org_sd[y] = (int)llround(0.45 * sin(tau * 1.75 * v + 0.7) * DEV_HALF_ORG);
    }
}

static void develop_build_thresh(void) {
    int l_min = 255, l_max = 0;
    for (int y = 0; y < G.wh; y += 16) {
        const guchar *row = G.tgt_buf + (size_t)y * (size_t)G.ww * 4;
        for (int x = 0; x < G.ww; x += 16) {
            size_t off = (size_t)x * 4;
            int luma = (row[off + 2] * 77 + row[off + 1] * 150 + row[off] * 29) >> 8;
            if (luma < l_min) l_min = luma;
            if (luma > l_max) l_max = luma;
        }
    }
    int span = l_max - l_min;
    if (span < 48) span = 48;
    int luma_scale = (DEV_W_LUMA * 256 + span / 2) / span;

    for (int y = 0; y < G.wh; y++) {
        const int *b_row = G.bayer4[(y >> 2) & 3];
        int k_cb = G.org_cb[y], k_sd = G.org_sd[y];
        const guchar *t_row = G.tgt_buf + (size_t)y * (size_t)G.ww * 4;
        guint16 *th_row = G.thresh_buf + (size_t)y * (size_t)G.ww;
        for (int x = 0; x < G.ww; x++) {
            size_t off = (size_t)x * 4;
            int luma = (t_row[off + 2] * 77 + t_row[off + 1] * 150 + t_row[off] * 29) >> 8;
            int inv_l = l_max - luma;
            if (inv_l < 0) inv_l = 0;
            else if (inv_l > span) inv_l = span;
            int luma_term = (inv_l * luma_scale) >> 8;
            int org_term = DEV_HALF_ORG + ((G.org_sa[x] * k_cb + G.org_cc[x] * k_sd) >> 8);
            int th = b_row[(x >> 2) & 3] + luma_term + org_term;
            if (th < 0) th = 0;
            else if (th > DEV_MAX_THRESH) th = DEV_MAX_THRESH;
            th_row[x] = (guint16)th;
        }
    }
}

static void develop_begin(void) {
    memcpy(G.src_buf, G.cur_buf, (size_t)G.ww * (size_t)G.wh * 4);
    develop_build_thresh();
    G.prev_head = -1;
    G.trans_dur_ms = G.applying ? DEV_DUR_APPLY_MS : DEV_DUR_BROWSE_MS;
    G.trans_start_us = g_get_monotonic_time();
}

/* src scaled to cover the monitor (centred, like feh --bg-fill), only the
 * window's part of it: one scaling pass straight into the window's size. */
static void file_sig(GString *k, const char *path);
static char *cache_dir(const char *sub);

static GdkPixbuf *screen_part(GdkPixbuf *src) {
    int sw = gdk_pixbuf_get_width(src), sh = gdk_pixbuf_get_height(src);
    double k = MAX((double)G.mw / sw, (double)G.mh / sh);
    double ox = (sw * k - G.mw) / 2, oy = (sh * k - G.mh) / 2 + MIN(G.top, G.mh - G.wh);
    GdkPixbuf *out = gdk_pixbuf_new(GDK_COLORSPACE_RGB, FALSE, 8, G.ww, G.wh);
    GdkInterpType interp = k == 1.0 ? GDK_INTERP_NEAREST : GDK_INTERP_BILINEAR;
    if (gdk_pixbuf_get_has_alpha(src))
        gdk_pixbuf_composite_color(src, out, 0, 0, G.ww, G.wh, -ox, -oy, k, k,
                                   interp, 255, 0, 0, 16, 0x000000, 0x000000);
    else
        gdk_pixbuf_scale(src, out, 0, 0, G.ww, G.wh, -ox, -oy, k, k, interp);
    return out;
}

static void pixbuf_to_bgrx(GdkPixbuf *pb, guchar *dst) {
    int w = MIN(G.ww, gdk_pixbuf_get_width(pb));
    int h = MIN(G.wh, gdk_pixbuf_get_height(pb));
    int rs = gdk_pixbuf_get_rowstride(pb);
    int ch = gdk_pixbuf_get_n_channels(pb);
    const guchar *px = gdk_pixbuf_get_pixels(pb);
    for (int y = 0; y < h; y++) {
        const guchar *s = px + (size_t)y * rs;
        guchar *d = dst + (size_t)y * G.ww * 4;
        for (int x = 0; x < w; x++) {
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
            d[3] = 0;
            s += ch;
            d += 4;
        }
    }
}

static void gallery_apply_root(int i) {
    if (i <= 0 || i >= S.n || !*S.items[i].under) return;
    char *bin = g_build_filename(self_dir(), "misa-wall", NULL);
    if (g_file_test(bin, G_FILE_TEST_IS_EXECUTABLE)) {
        const char *argv[] = {bin, "--no-transition", S.items[i].under, NULL};
        g_spawn_sync(NULL, (char **)argv, NULL,
                     G_SPAWN_STDOUT_TO_DEV_NULL | G_SPAWN_STDERR_TO_DEV_NULL,
                     NULL, NULL, NULL, NULL, NULL, NULL);
    }
    g_free(bin);
}

static gboolean gallery_finish_idle(gpointer _d) {
    gallery_apply_root(S.result);
    if (G.use_gl) live_gl_pause_desktop_daemon(FALSE);
    quit();
    return G_SOURCE_REMOVE;
}

static void on_gl_trans_finished(void *_u) {
    G.trans_done = TRUE;
    if (G.applying && G.full_ready) g_idle_add(gallery_finish_idle, NULL);
}

/* One row of Mira's Luma-Halftone Develop: skips pixels already settled on the
 * previous frame and pixels not yet reached, touching only the ~6% actively
 * blooming halftone dots plus newly settled copies. */
static gboolean develop_row(int y, int head, int prev_lim) {
    size_t px_off = (size_t)y * (size_t)G.ww;
    const guint16 *th_row = G.thresh_buf + px_off;
    guchar *cur = G.cur_buf + px_off * 4;
    const guchar *src = G.src_buf + px_off * 4;
    const guchar *tgt = G.tgt_buf + px_off * 4;
    gboolean touched = FALSE;

    for (int x = 0; x < G.ww; x++) {
        int th = th_row[x];
        if (th >= head || th <= prev_lim) continue;
        touched = TRUE;
        size_t off = (size_t)x * 4;
        int diff = head - th;
        if (diff >= DEV_FEATHER) {
            cur[off] = tgt[off];
            cur[off + 1] = tgt[off + 1];
            cur[off + 2] = tgt[off + 2];
            cur[off + 3] = 0;
        } else {
            int a = (diff * DEV_RECIP) >> 16;
            int sb = src[off], sg = src[off + 1], sr = src[off + 2];
            int tb = tgt[off], tg = tgt[off + 1], tr = tgt[off + 2];
            cur[off] = (guchar)(sb + (((tb - sb) * a) >> 8));
            cur[off + 1] = (guchar)(sg + (((tg - sg) * a) >> 8));
            cur[off + 2] = (guchar)(sr + (((tr - sr) * a) >> 8));
            cur[off + 3] = 0;
        }
    }
    return touched;
}

static gboolean gallery_tick(GtkWidget *w, GdkFrameClock *_clk, gpointer _d) {
    double elapsed_ms = (g_get_monotonic_time() - G.trans_start_us) / 1000.0;
    double dur = G.trans_dur_ms > 0 ? (double)G.trans_dur_ms : 1.0;
    double t = (elapsed_ms + 16.0) / dur;
    gboolean finish = t >= 1.0;
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
    double smooth = t * t * (3.0 - 2.0 * t);
    double inv = 1.0 - t;
    double cubic_out = 1.0 - inv * inv * inv;
    double eased = 0.80 * smooth + 0.20 * cubic_out;
    int head = finish ? (DEV_MAX_THRESH + DEV_FEATHER)
                      : (int)llround(eased * (DEV_MAX_THRESH + DEV_FEATHER));
    int prev_lim = G.prev_head < 0 ? -1 : (G.prev_head - DEV_FEATHER);

    int y0 = -1, y1 = -1;
    for (int y = 0; y < G.wh; y++) {
        if (develop_row(y, head, prev_lim)) {
            if (y0 < 0) y0 = y;
            y1 = y;
        }
    }
    G.prev_head = head;

    if (y0 >= 0) {
        cairo_surface_mark_dirty_rectangle(G.surf, 0, y0, G.ww, y1 - y0 + 1);
        gtk_widget_queue_draw_area(w, 0, y0, G.ww, y1 - y0 + 1);
    }
    if (finish) {
        G.trans_done = TRUE;
        G.tick_id = 0;
        if (G.applying && G.full_ready) g_idle_add(gallery_finish_idle, NULL);
        return G_SOURCE_REMOVE;
    }
    return G_SOURCE_CONTINUE;
}

static void start_gallery_tick(void) {
    G.trans_done = FALSE;
    if (!G.tick_id && G.canvas)
        G.tick_id = gtk_widget_add_tick_callback(G.canvas, gallery_tick, NULL, NULL);
}

static gboolean on_gallery_draw(GtkWidget *_w, cairo_t *cr, gpointer _d) {
    if (G.surf) {
        cairo_set_source_surface(cr, G.surf, 0, 0);
        cairo_paint(cr);
    }
    return TRUE;
}

/* The screen-sized picture is cached (JPEG, ~10 ms to read back): a big
 * photo takes up to ~0.75 s to decode and scale. Keyed by the file (path,
 * mtime, size) and the window's place on the monitor. */
static char *wall_cache_path(const char *path) {
    static char *dir;
    if (!dir) dir = cache_dir("walls");
    GString *k = g_string_new("");
    file_sig(k, path);
    g_string_append_printf(k, "%dx%d %d %dx%d", G.mw, G.mh, G.top, G.ww, G.wh);
    char *sum = g_compute_checksum_for_string(G_CHECKSUM_SHA1, k->str, -1);
    sum[24] = 0;
    char *name = g_strconcat(sum, ".jpg", NULL);
    char *out = g_build_filename(dir, name, NULL);
    g_free(name);
    g_free(sum);
    g_string_free(k, TRUE);
    return out;
}

static GdkPixbuf *screen_picture(const char *path, gboolean want) {
    char *cached = wall_cache_path(path);
    GdkPixbuf *out = NULL;
    if (g_file_test(cached, G_FILE_TEST_EXISTS)) {
        utime(cached, NULL);                      /* still in use */
        if (want) out = gdk_pixbuf_new_from_file(cached, NULL);
        if (!want || out) {
            g_free(cached);
            return out;
        }
    }
    GdkPixbuf *src = load_cover_size(path, G.mw, G.mh);
    if (src) {
        out = screen_part(src);
        g_object_unref(src);
        char *dir = g_path_get_dirname(cached), *tmp = g_strconcat(cached, ".tmp", NULL);
        g_mkdir_with_parents(dir, 0700);
        if (gdk_pixbuf_save(out, tmp, "jpeg", NULL, "quality", "92", NULL)) g_rename(tmp, cached);
        else g_unlink(tmp);
        g_free(tmp);
        g_free(dir);
    }
    g_free(cached);
    if (out && !want) {
        g_object_unref(out);
        out = NULL;
    }
    return out;
}

/* A picture as the window's BGRX bytes, made in the thread: the main loop
 * only swaps a pointer. */
static guchar *to_bgrx(GdkPixbuf *pb) {
    guchar *buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
    pixbuf_to_bgrx(pb, buf);
    return buf;
}

/* Tightly packed RGB888 payload produced on a worker thread for GPU upload. */
typedef struct {
    guchar *rgb;
    int w, h;
    float luma_max, luma_span;
    char *frag;
} GpuPic;

static void gpu_pic_free(gpointer p) {
    GpuPic *g = p;
    if (!g) return;
    g_free(g->rgb);
    g_free(g->frag);
    g_free(g);
}

static GpuPic *pixbuf_to_gpu_pic(GdkPixbuf *pb, const char *frag) {
    if (!pb) return NULL;
    int w = gdk_pixbuf_get_width(pb);
    int h = gdk_pixbuf_get_height(pb);
    int rs = gdk_pixbuf_get_rowstride(pb);
    int ch = gdk_pixbuf_get_n_channels(pb);
    const guchar *px = gdk_pixbuf_get_pixels(pb);
    if (w <= 0 || h <= 0 || (ch != 3 && ch != 4) || !px) return NULL;

    GpuPic *gp = g_new0(GpuPic, 1);
    gp->w = w;
    gp->h = h;
    gp->frag = frag ? g_strdup(frag) : NULL;
    live_gl_scan_luma_rgb(px, w, h, rs, ch, &gp->luma_max, &gp->luma_span);

    size_t row_rgb = (size_t)w * 3;
    gp->rgb = g_malloc(row_rgb * (size_t)h);
    if (ch == 3) {
        for (int y = 0; y < h; y++)
            memcpy(gp->rgb + (size_t)y * row_rgb, px + (size_t)y * (size_t)rs, row_rgb);
    } else {
        for (int y = 0; y < h; y++) {
            const guchar *s = px + (size_t)y * (size_t)rs;
            guchar *d = gp->rgb + (size_t)y * row_rgb;
            for (int x = 0; x < w; x++) {
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
                s += 4; d += 3;
            }
        }
    }
    return gp;
}

/* Full-monitor cover JPEG cache for large custom static photos on the GPU path. */
static char *gpu_wall_cache_path(const char *path) {
    static char *dir;
    if (!dir) dir = cache_dir("walls");
    GString *k = g_string_new("gpu ");
    file_sig(k, path);
    g_string_append_printf(k, "%dx%d", G.mw, G.mh);
    char *sum = g_compute_checksum_for_string(G_CHECKSUM_SHA1, k->str, -1);
    sum[24] = 0;
    char *name = g_strconcat(sum, ".jpg", NULL);
    char *out = g_build_filename(dir, name, NULL);
    g_free(name);
    g_free(sum);
    g_string_free(k, TRUE);
    return out;
}

static GdkPixbuf *gpu_full_pixbuf(const char *path, const char *frag, gboolean want) {
    if (frag && *frag) {
        /* Live shader wallpaper: decode straight at 2x cell grid (~12 ms, ~2.7 MB,
         * matches misa-wall VRAM footprint — no disk cache write needed). */
        if (!want) return NULL;
        int cw = 1280, ch = 720;
        live_gl_tex_cap(G.mw, G.mh, &cw, &ch);
        int iw = 0, ih = 0;
        if (gdk_pixbuf_get_file_info(path, &iw, &ih) && iw > 0 && ih > 0 && (iw > cw || ih > ch))
            return gdk_pixbuf_new_from_file_at_scale(path, cw, ch, TRUE, NULL);
        return gdk_pixbuf_new_from_file(path, NULL);
    }
    /* Custom static photo: keep full monitor resolution, cached on disk for instant repeats. */
    char *cached = gpu_wall_cache_path(path);
    GdkPixbuf *out = NULL;
    if (g_file_test(cached, G_FILE_TEST_EXISTS)) {
        utime(cached, NULL);
        if (want) out = gdk_pixbuf_new_from_file(cached, NULL);
        if (!want || out) {
            g_free(cached);
            return out;
        }
    }
    GdkPixbuf *src = load_cover_size(path, G.mw, G.mh);
    if (src) {
        out = cover_pixbuf(src, G.mw, G.mh, 0, FALSE);
        g_object_unref(src);
        if (out) {
            char *dir = g_path_get_dirname(cached), *tmp = g_strconcat(cached, ".tmp", NULL);
            g_mkdir_with_parents(dir, 0700);
            if (gdk_pixbuf_save(out, tmp, "jpeg", NULL, "quality", "92", NULL)) g_rename(tmp, cached);
            else g_unlink(tmp);
            g_free(tmp);
            g_free(dir);
        }
    }
    g_free(cached);
    if (out && !want) {
        g_object_unref(out);
        out = NULL;
    }
    return out;
}

typedef struct { char *path, *frag; } FullReq;
static void full_req_free(gpointer p) {
    FullReq *r = p;
    g_free(r->path);
    g_free(r->frag);
    g_free(r);
}

static void decode_full(GTask *task, gpointer _src, gpointer data, GCancellable *_c) {
    FullReq *r = data;
    if (G.use_gl) {
        GdkPixbuf *pb = gpu_full_pixbuf(r->path, r->frag, TRUE);
        GpuPic *gp = pixbuf_to_gpu_pic(pb, r->frag);
        if (pb) g_object_unref(pb);
        g_task_return_pointer(task, gp, gpu_pic_free);
        return;
    }
    GdkPixbuf *pb = screen_picture(r->path, TRUE);
    guchar *buf = pb ? to_bgrx(pb) : NULL;
    if (pb) g_object_unref(pb);
    g_task_return_pointer(task, buf, g_free);
}

static void take_target(guchar *buf) {
    g_free(G.tgt_buf);
    G.tgt_buf = buf;
}

/* Neighbours made ready on disk, one after another, while nothing moves. */
static void warm(GTask *task, gpointer _src, gpointer data, GCancellable *_c) {
    for (char **p = data; *p; p++) {
        if (G.use_gl) {
            char *fr = live_gl_find_sidecar(*p);
            if (!fr) gpu_full_pixbuf(*p, NULL, FALSE);
            g_free(fr);
        } else {
            screen_picture(*p, FALSE);
        }
    }
    g_task_return_boolean(task, TRUE);
}

static void full_done(GObject *_o, GAsyncResult *res, gpointer gen) {
    gpointer payload = g_task_propagate_pointer(G_TASK(res), NULL);
    gboolean current = GPOINTER_TO_UINT(gen) == G.gen;
    if (G.use_gl) {
        GpuPic *gp = payload;
        if (gp && current) {
            G.full_ready = TRUE;
            live_gl_refine_target(G.gl, gp->rgb, gp->w, gp->h, gp->frag, gp->luma_max, gp->luma_span);
            if (G.applying && live_gl_trans_done(G.gl)) g_idle_add(gallery_finish_idle, NULL);
        }
        gpu_pic_free(gp);
    } else {
        guchar *pb = payload;
        if (pb && current) {
            take_target(pb);
            pb = NULL;
            G.full_ready = TRUE;
            if (G.trans_done) {
                memcpy(G.cur_buf, G.tgt_buf, (size_t)G.ww * (size_t)G.wh * 4);
                cairo_surface_mark_dirty(G.surf);
                gtk_widget_queue_draw(G.canvas);
                if (G.applying) g_idle_add(gallery_finish_idle, NULL);
            } else {
                develop_build_thresh();
                G.prev_head = -1;
                start_gallery_tick();
            }
        }
        g_free(pb);
    }
    heap_trim();
    if (!current) return;
    GPtrArray *next = g_ptr_array_new();
    for (int d = 1; d >= -1; d -= 2) {
        int j = S.cur + d;
        if (j >= 0 && j < S.n && *S.items[j].under) g_ptr_array_add(next, g_strdup(S.items[j].under));
    }
    g_ptr_array_add(next, NULL);
    GTask *t = g_task_new(NULL, NULL, NULL, NULL);
    g_task_set_task_data(t, g_ptr_array_free(next, FALSE), (GDestroyNotify)g_strfreev);
    g_task_run_in_thread(t, warm);
    g_object_unref(t);
}

#define SETTLE_MS 180

static gboolean decode_later(gpointer data) {
    G.settle = 0;
    FullReq *orig = data;
    FullReq *req = g_new0(FullReq, 1);
    req->path = g_strdup(orig->path);
    req->frag = orig->frag ? g_strdup(orig->frag) : NULL;
    GTask *t = g_task_new(NULL, NULL, full_done, GUINT_TO_POINTER(G.gen));
    g_task_set_task_data(t, req, full_req_free);
    g_task_run_in_thread(t, decode_full);
    g_object_unref(t);
    return G_SOURCE_REMOVE;
}

/* The quick picture of item i: on GPU live wallpapers (.frag) decode straight at
 * 2x cell grid (~12 ms, full=TRUE at once!); on custom photos uses cached JPEG or
 * cached thumbnail scaled up while full-res decodes in the background. */
typedef struct {
    char *cached, *thumb, *under, *frag;
    guchar *buf;
    GpuPic *gpu;
    gboolean full, allow_full, use_gl;
} Quick;

static void quick_free(gpointer p) {
    Quick *q = p;
    g_free(q->cached);
    g_free(q->thumb);
    g_free(q->under);
    g_free(q->frag);
    g_free(q->buf);
    gpu_pic_free(q->gpu);
    g_free(q);
}

static void make_quick(Quick *q) {
    if (q->use_gl) {
        GdkPixbuf *pb = NULL;
        if (q->frag && *q->frag) {
            pb = gpu_full_pixbuf(q->under, q->frag, TRUE);
            q->full = pb != NULL;
        } else {
            if (g_file_test(q->cached, G_FILE_TEST_EXISTS)) {
                utime(q->cached, NULL);
                pb = gdk_pixbuf_new_from_file(q->cached, NULL);
                q->full = pb != NULL;
            }
            if (!pb && *q->thumb && g_file_test(q->thumb, G_FILE_TEST_EXISTS)) {
                pb = gdk_pixbuf_new_from_file(q->thumb, NULL);
            }
            if (!pb && q->allow_full && *q->under) {
                pb = gpu_full_pixbuf(q->under, NULL, TRUE);
                q->full = pb != NULL;
            }
        }
        if (pb) {
            q->gpu = pixbuf_to_gpu_pic(pb, q->frag);
            g_object_unref(pb);
        }
        return;
    }

    GdkPixbuf *pb = NULL;
    if (g_file_test(q->cached, G_FILE_TEST_EXISTS)) {
        utime(q->cached, NULL);
        pb = gdk_pixbuf_new_from_file(q->cached, NULL);
        q->full = pb != NULL;
    }
    if (!pb && *q->thumb && g_file_test(q->thumb, G_FILE_TEST_EXISTS)) {
        GdkPixbuf *small = gdk_pixbuf_new_from_file(q->thumb, NULL);
        pb = small ? screen_part(small) : NULL;
        if (small) g_object_unref(small);
    }
    if (!pb && q->allow_full && *q->under) {
        pb = screen_picture(q->under, TRUE);
        q->full = pb != NULL;
    }
    if (pb) {
        q->buf = to_bgrx(pb);
        g_object_unref(pb);
    } else {
        q->buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
    }
}

static Quick *quick_new(int i, gboolean allow_full) {
    Quick *q = g_new0(Quick, 1);
    q->use_gl = G.use_gl;
    q->frag = (G.frag_path && G.frag_path[i]) ? g_strdup(G.frag_path[i]) : NULL;
    q->cached = G.use_gl ? gpu_wall_cache_path(S.items[i].under) : wall_cache_path(S.items[i].under);
    const char *th = *S.items[i].image ? S.items[i].image
                     : (G.thumb_cache && G.thumb_cache[i] ? G.thumb_cache[i] : "");
    q->thumb = g_strdup(th);
    q->under = g_strdup(S.items[i].under);
    q->allow_full = allow_full;
    return q;
}

static void quick_thread(GTask *task, gpointer _src, gpointer data, GCancellable *_c) {
    make_quick(data);
    g_task_return_boolean(task, TRUE);
}

/* The full picture once the selection rests (at once when applying). */
static void after_quick(int i, gboolean full, gboolean animate) {
    G.full_ready = full;
    if (G.settle) { g_source_remove(G.settle); G.settle = 0; }
    if (!full) {
        FullReq *req = g_new0(FullReq, 1);
        req->path = g_strdup(S.items[i].under);
        req->frag = (G.frag_path && G.frag_path[i]) ? g_strdup(G.frag_path[i]) : NULL;
        G.settle = g_timeout_add_full(G_PRIORITY_DEFAULT, animate && !G.applying ? SETTLE_MS : 0,
                                      decode_later, req, full_req_free);
    }
}

static void request_quick(int i);

static void quick_done(GObject *_o, GAsyncResult *res, gpointer data) {
    GTask *task = G_TASK(res);
    Quick *q = g_task_get_task_data(task);
    guint gen = GPOINTER_TO_UINT(data) >> 16;
    int i = (int)(GPOINTER_TO_UINT(data) & 0xFFFF);
    G.loading = FALSE;
    if ((gen & 0xFFFF) == (G.gen & 0xFFFF) && G.pending < 0) {
        if (G.use_gl) {
            if (q->gpu) {
                G.trans_done = FALSE;
                int dur = G.applying ? DEV_DUR_APPLY_MS : DEV_DUR_BROWSE_MS;
                live_gl_set_wallpaper(G.gl, q->gpu->rgb, q->gpu->w, q->gpu->h,
                                      q->gpu->frag, q->gpu->luma_max, q->gpu->luma_span,
                                      TRUE, dur);
                if (G.applying) live_gl_set_finish_cb(G.gl, on_gl_trans_finished, NULL);
            }
            after_quick(i, q->full, TRUE);
        } else {
            take_target(q->buf);
            q->buf = NULL;
            develop_begin();
            start_gallery_tick();
            after_quick(i, q->full, TRUE);
        }
    }
    if (G.pending >= 0) {
        int j = G.pending;
        G.pending = -1;
        request_quick(j);
    }
}

/* One quick picture in flight at a time; keys held down skip the ones
 * passed over (only the latest selection is made next). */
static void request_quick(int i) {
    if (G.loading) {
        G.pending = i;
        return;
    }
    G.loading = TRUE;
    guint tag = ((G.gen & 0xFFFF) << 16) | ((guint)i & 0xFFFF);
    GTask *t = g_task_new(NULL, NULL, quick_done, GUINT_TO_POINTER(tag));
    g_task_set_task_data(t, quick_new(i, TRUE), quick_free);
    g_task_run_in_thread(t, quick_thread);
    g_object_unref(t);
}

/* Item i on the screen; the "add" tile keeps the wallpaper in use. */
static void show_wallpaper(int i, gboolean animate) {
    if (!*S.items[i].under) i = A.active >= 0 && A.active < S.n ? A.active : -1;
    if (i < 0 || i == G.shown) return;
    G.shown = i;
    G.gen++;
    G.full_ready = FALSE;
    if (G.settle) { g_source_remove(G.settle); G.settle = 0; }

    /* The first picture is made before anything shows (no black frame) when
     * already cached on disk; uncached cold starts never block window map. */
    if (animate && G.has_cur) {
        request_quick(i);
        return;
    }
    Quick *q = quick_new(i, G.use_gl && G.frag_path && G.frag_path[i] != NULL);
    make_quick(q);
    if (G.use_gl) {
        if (q->gpu) {
            live_gl_set_wallpaper(G.gl, q->gpu->rgb, q->gpu->w, q->gpu->h,
                                  q->gpu->frag, q->gpu->luma_max, q->gpu->luma_span,
                                  FALSE, 0);
            G.has_cur = TRUE;
        }
        G.trans_done = TRUE;
        after_quick(i, q->full, animate);
        quick_free(q);
        return;
    }
    take_target(q->buf);
    q->buf = NULL;
    memcpy(G.cur_buf, G.tgt_buf, (size_t)G.ww * (size_t)G.wh * 4);
    memcpy(G.src_buf, G.tgt_buf, (size_t)G.ww * (size_t)G.wh * 4);
    G.has_cur = TRUE;
    G.trans_done = TRUE;
    cairo_surface_mark_dirty(G.surf);
    gtk_widget_queue_draw(G.canvas);
    after_quick(i, q->full, animate);
    quick_free(q);
}

static gboolean gallery_applying(void) {
    return G.applying;
}

static void gallery_cleanup(void) {
    G.alive = FALSE;
    if (G.thumb_cancel) g_cancellable_cancel(G.thumb_cancel);
    if (G.thumb_order) {
        g_array_free(G.thumb_order, TRUE);
        G.thumb_order = NULL;
    }
    if (G.gl) live_gl_stop(G.gl);
    if (G.use_gl) live_gl_pause_desktop_daemon(FALSE);
    if (G.tick_id && G.canvas) {
        gtk_widget_remove_tick_callback(G.canvas, G.tick_id);
        G.tick_id = 0;
    }
    if (G.settle) {
        g_source_remove(G.settle);
        G.settle = 0;
    }
}

static void gallery_confirm(int i) {
    if (G.applying) return;
    gboolean done = G.use_gl ? live_gl_trans_done(G.gl) : G.trans_done;
    if (done && G.full_ready) {
        gallery_apply_root(i);
        if (G.use_gl) live_gl_pause_desktop_daemon(FALSE);
        quit();
        return;
    }
    G.applying = TRUE;
    ungrab();
    if (G.dock_ev) gtk_widget_hide(G.dock_ev);
    if (G.settle) {
        g_source_remove(G.settle);
        G.settle = 0;
        FullReq req = {S.items[i].under, G.frag_path ? G.frag_path[i] : NULL};
        decode_later(&req);
    }
    if (G.use_gl) {
        if (!live_gl_trans_done(G.gl)) live_gl_set_finish_cb(G.gl, on_gl_trans_finished, NULL);
        else if (G.full_ready) g_idle_add(gallery_finish_idle, NULL);
    } else if (!G.trans_done) {
        start_gallery_tick();
    }
}

static void caption(int i) {
    Item *it = &S.items[i];
    gtk_label_set_text(GTK_LABEL(G.title), it->title);
    gboolean add = !*it->under;
    gboolean is_live = G.frag_path && G.frag_path[i] != NULL;
    int w = 0, h = 0;
    char *meta;
    if (add) meta = g_strdup(it->sub);
    else if (gdk_pixbuf_get_file_info(it->under, &w, &h))
        meta = g_strdup_printf("%s%s%d \xc3\x97 %d%s", it->sub, *it->sub ? "  \xc2\xb7  " : "", w, h,
                               is_live ? "  \xc2\xb7  LIVE" : "");
    else meta = g_strdup_printf("%s%s", it->sub, is_live ? "  \xc2\xb7  LIVE" : "");
    gtk_label_set_text(GTK_LABEL(G.meta), meta);
    g_free(meta);
    gtk_widget_set_visible(G.badge, i == A.active);
    gtk_label_set_text(GTK_LABEL(G.apply), add ? "ADD" : "APPLY");
}

static void gallery_selected(int i, gboolean animate) {
    caption(i);
    show_wallpaper(i, animate);
}

static gboolean on_tile_click(GtkWidget *_w, GdkEventButton *ev, gpointer data) {
    if (ev->button != 1) return FALSE;
    int i = GPOINTER_TO_INT(data);
    if (i == S.cur) pick(i);
    else select_item(i, TRUE);
    return TRUE;
}

static gboolean swallow(GtkWidget *_w, GdkEventButton *ev, gpointer _) {
    return ev->button == 1;          /* the dock's own empty space: not "outside" */
}

static GdkPixbuf *bake_dither_pixbuf(const char *cache_dir_path, const char *wall,
                                     const char *frag, int pw, int ph);

/* Shared on-disk filmstrip thumbnail cache (exact TW x TH with rounded corners
 * baked in, keyed by source file + optional .frag shader mtime/size). */
static char *thumb_cache_path(const char *src, const char *frag) {
    static char *dir;
    if (!dir) dir = cache_dir("thumbs");
    GString *k = g_string_new(frag && *frag ? "live-v1|" : "");
    file_sig(k, src);
    if (frag && *frag) file_sig(k, frag);
    g_string_append_printf(k, "%dx%d r10", TW, TH);
    char *sum = g_compute_checksum_for_string(G_CHECKSUM_SHA1, k->str, -1);
    sum[24] = 0;
    char *name = g_strconcat(sum, ".png", NULL);
    char *out = g_build_filename(dir, name, NULL);
    g_free(name);
    g_free(sum);
    g_string_free(k, TRUE);
    return out;
}

typedef struct { int idx; char *src, *frag, *cache; } ThumbJob;
typedef struct { ThumbJob *jobs; int n; char *dir; gboolean use_gl; GCancellable *cancel; } ThumbBatch;
typedef struct { int idx; GdkPixbuf *pb; } ThumbResult;
typedef struct { int idx, w, h; guchar *rgb; char *frag, *cache, *dir; } LiveThumbReq;

static void thumb_batch_free(gpointer p) {
    ThumbBatch *b = p;
    for (int i = 0; i < b->n; i++) {
        g_free(b->jobs[i].src);
        g_free(b->jobs[i].frag);
        g_free(b->jobs[i].cache);
    }
    g_free(b->jobs);
    g_free(b->dir);
    if (b->cancel) g_object_unref(b->cancel);
    g_free(b);
}

static void thumb_result_free(gpointer p) {
    ThumbResult *r = p;
    if (r->pb) g_object_unref(r->pb);
    g_free(r);
}

static void live_thumb_req_free(gpointer p) {
    LiveThumbReq *r = p;
    g_free(r->rgb);
    g_free(r->frag);
    g_free(r->cache);
    g_free(r->dir);
    g_free(r);
}

static void rgb_pixbuf_free(guchar *px, gpointer _d) {
    (void)_d;
    g_free(px);
}

static void save_thumb_cache(const char *dir, const char *cache, GdkPixbuf *pb) {
    if (!dir || !cache || !pb) return;
    g_mkdir_with_parents(dir, 0700);
    char *tmp = g_strdup_printf("%s.%lu.tmp", cache, (unsigned long)g_thread_self());
    if (gdk_pixbuf_save(pb, tmp, "png", NULL, "compression", "1", NULL))
        g_rename(tmp, cache);
    else
        g_unlink(tmp);
    g_free(tmp);
}

static gboolean apply_thumb_idle(gpointer p) {
    ThumbResult *r = p;
    if (G.alive && r->pb && G.thumb_img && r->idx >= 0 && r->idx < S.n && G.thumb_img[r->idx]) {
        gtk_style_context_remove_class(gtk_widget_get_style_context(G.thumb_img[r->idx]), "thumb-slot");
        gtk_image_set_from_pixbuf(GTK_IMAGE(G.thumb_img[r->idx]), r->pb);
    }
    return G_SOURCE_REMOVE;
}

static gboolean apply_live_thumb_idle(gpointer p) {
    LiveThumbReq *r = p;
    if (!G.alive || !G.thumb_img || r->idx < 0 || r->idx >= S.n || !G.thumb_img[r->idx])
        return G_SOURCE_REMOVE;

    GdkPixbuf *pb = NULL;
    if (G.use_gl && G.gl && live_gl_ready(G.gl)) {
        guchar *rendered = live_gl_render_thumb(G.gl, r->rgb, r->w, r->h, r->frag, TW, TH, 4.2f);
        if (rendered) {
            GdkPixbuf *raw = gdk_pixbuf_new_from_data(rendered, GDK_COLORSPACE_RGB, FALSE, 8,
                                                      TW, TH, TW * 3, rgb_pixbuf_free, NULL);
            if (raw) {
                pb = round_corners(raw, 10, FALSE);
                g_object_unref(raw);
            } else {
                g_free(rendered);
            }
        }
    }
    if (!pb && r->rgb && r->w > 0 && r->h > 0) {
        GdkPixbuf *src = gdk_pixbuf_new_from_data(r->rgb, GDK_COLORSPACE_RGB, FALSE, 8,
                                                  r->w, r->h, r->w * 3, NULL, NULL);
        if (src) {
            pb = cover_pixbuf(src, TW, TH, 10, FALSE);
            g_object_unref(src);
        }
    }
    if (pb) {
        gtk_style_context_remove_class(gtk_widget_get_style_context(G.thumb_img[r->idx]), "thumb-slot");
        gtk_image_set_from_pixbuf(GTK_IMAGE(G.thumb_img[r->idx]), pb);
        save_thumb_cache(r->dir, r->cache, pb);
        g_object_unref(pb);
    }
    return G_SOURCE_REMOVE;
}

static void thumb_worker(GTask *task, gpointer _src, gpointer data, GCancellable *_c) {
    ThumbBatch *b = data;
    for (int k = 0; k < b->n; k++) {
        if (b->cancel && g_cancellable_is_cancelled(b->cancel)) break;
        ThumbJob *j = &b->jobs[k];
        GdkPixbuf *pb = NULL;
        if (g_file_test(j->cache, G_FILE_TEST_EXISTS)) {
            utime(j->cache, NULL);
            pb = gdk_pixbuf_new_from_file(j->cache, NULL);
        }
        if (!pb && j->frag && *j->frag) {
            if (b->use_gl) {
                GdkPixbuf *src_pb = load_cover_size(j->src, TW * 2, TH * 2);
                GpuPic *gp = src_pb ? pixbuf_to_gpu_pic(src_pb, NULL) : NULL;
                if (src_pb) g_object_unref(src_pb);
                if (gp && !(b->cancel && g_cancellable_is_cancelled(b->cancel))) {
                    LiveThumbReq *req = g_new0(LiveThumbReq, 1);
                    req->idx = j->idx;
                    req->w = gp->w;
                    req->h = gp->h;
                    req->rgb = gp->rgb;
                    gp->rgb = NULL;
                    req->frag = g_strdup(j->frag);
                    req->cache = g_strdup(j->cache);
                    req->dir = g_strdup(b->dir);
                    gpu_pic_free(gp);
                    g_idle_add_full(G_PRIORITY_HIGH_IDLE, apply_live_thumb_idle,
                                    req, live_thumb_req_free);
                    continue;
                }
                gpu_pic_free(gp);
            } else {
                GdkPixbuf *baked = bake_dither_pixbuf(b->dir, j->src, j->frag, TW, TH);
                if (baked) {
                    pb = round_corners(baked, 10, FALSE);
                    g_object_unref(baked);
                    if (pb) save_thumb_cache(b->dir, j->cache, pb);
                }
            }
        }
        if (!pb) {
            pb = cover_file(j->src, TW, TH, 10, FALSE);
            if (pb) save_thumb_cache(b->dir, j->cache, pb);
        }
        if (pb && !(b->cancel && g_cancellable_is_cancelled(b->cancel))) {
            ThumbResult *r = g_new0(ThumbResult, 1);
            r->idx = j->idx;
            r->pb = pb;
            g_idle_add_full(G_PRIORITY_HIGH_IDLE, apply_thumb_idle, r, thumb_result_free);
        } else if (pb) {
            g_object_unref(pb);
        }
    }
    g_task_return_boolean(task, TRUE);
}

static void gallery_start_thumbs(void) {
    if (!G.thumb_order || G.thumb_order->len == 0) {
        if (G.thumb_order) { g_array_free(G.thumb_order, TRUE); G.thumb_order = NULL; }
        return;
    }
    GArray *order = G.thumb_order;
    G.thumb_order = NULL;
    int nw = (int)CLAMP((int)g_get_num_processors(), 2, 4);
    if ((guint)nw > order->len) nw = (int)order->len;
    char *tdir = cache_dir("thumbs");
    for (int w = 0; w < nw; w++) {
        int count = (int)((order->len - (guint)w + (guint)nw - 1) / (guint)nw);
        ThumbBatch *b = g_new0(ThumbBatch, 1);
        b->jobs = g_new0(ThumbJob, count);
        b->n = count;
        b->dir = g_strdup(tdir);
        b->use_gl = G.use_gl;
        b->cancel = g_object_ref(G.thumb_cancel);
        for (int k = 0; k < count; k++) {
            int idx = g_array_index(order, int, w + k * nw);
            const char *src = *S.items[idx].image ? S.items[idx].image : S.items[idx].under;
            b->jobs[k].idx = idx;
            b->jobs[k].src = g_strdup(src);
            b->jobs[k].frag = (G.frag_path && G.frag_path[idx]) ? g_strdup(G.frag_path[idx]) : NULL;
            b->jobs[k].cache = g_strdup(G.thumb_cache[idx]);
        }
        GTask *t = g_task_new(NULL, NULL, NULL, NULL);
        g_task_set_task_data(t, b, thumb_batch_free);
        g_task_run_in_thread(t, thumb_worker);
        g_object_unref(t);
    }
    g_free(tdir);
    g_array_free(order, TRUE);
}

static void gallery_init_cairo_fallback(void) {
    if (G.canvas) return;
    G.cur_buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
    G.src_buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
    G.tgt_buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
    develop_init_tables();
    G.surf = cairo_image_surface_create_for_data(G.cur_buf, CAIRO_FORMAT_RGB24, G.ww, G.wh, G.ww * 4);
    G.canvas = gtk_drawing_area_new();
    gtk_widget_set_size_request(G.canvas, G.ww, G.wh);
    g_signal_connect(G.canvas, "draw", G_CALLBACK(on_gallery_draw), NULL);
    if (G.over) {
        GtkWidget *old = gtk_bin_get_child(GTK_BIN(G.over));
        if (old) gtk_container_remove(GTK_CONTAINER(G.over), old);
        G.gl = NULL;
        gtk_container_add(GTK_CONTAINER(G.over), G.canvas);
        gtk_widget_show(G.canvas);
    }
}

static void gallery_after_show(void) {
    if (G.gl) {
        gtk_widget_realize(live_gl_widget(G.gl));
        if (live_gl_ready(G.gl)) {
            G.use_gl = TRUE;
            live_gl_pause_desktop_daemon(TRUE);
            gallery_start_thumbs();
            return;
        }
    }
    G.use_gl = FALSE;
    gallery_init_cairo_fallback();
    gallery_start_thumbs();
}

static gboolean on_gallery_motion(GtkWidget *_w, GdkEventMotion *ev, gpointer _) {
    if (G.use_gl && G.gl && G.mw > 0 && G.mh > 0) {
        float nx = (float)(ev->x / G.mw) * 2.0f - 1.0f;
        float ny = (float)((ev->y + G.top) / G.mh) * 2.0f - 1.0f;
        if (nx < -1.0f) nx = -1.0f; else if (nx > 1.0f) nx = 1.0f;
        if (ny < -1.0f) ny = -1.0f; else if (ny > 1.0f) ny = 1.0f;
        live_gl_set_mouse(G.gl, nx, ny);
    }
    return FALSE;
}

static GtkWidget *build_gallery(Rect mon, Rect win) {
    G.mw = mon.w;
    G.mh = mon.h;
    G.top = win.y - mon.y;
    G.ww = win.w;
    G.wh = win.h;
    G.alive = TRUE;
    G.thumb_img = g_new0(GtkWidget *, S.n);
    G.thumb_cache = g_new0(char *, S.n);
    G.frag_path = g_new0(char *, S.n);
    G.thumb_cancel = g_cancellable_new();
    add_class(S.win, "gallery");

    for (int i = 0; i < S.n; i++)
        if (*S.items[i].under) G.frag_path[i] = live_gl_find_sidecar(S.items[i].under);

    /* GPU live-shader & Luma-Halftone Develop surface (zero 28 MB CPU buffers
     * allocated unless OpenGL fails to realize in gallery_after_show()). */
    G.gl = live_gl_new(G.mw, G.mh, G.top, G.ww, G.wh);
    GtkWidget *base_view = live_gl_widget(G.gl);
    g_object_set_data_full(G_OBJECT(base_view), "live-gl", G.gl, (GDestroyNotify)live_gl_destroy);

    /* caption: name, type and size, IN USE; hints and the button right */
    G.title = label("", "g-title", TRUE);
    G.meta = label("", "g-meta", FALSE);
    G.badge = badge();
    gtk_widget_set_no_show_all(G.badge, TRUE);
    GtkWidget *meta_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
    add_class(meta_row, "g-meta-row");
    gtk_box_pack_start(GTK_BOX(meta_row), G.meta, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(meta_row), G.badge, FALSE, FALSE, 0);
    GtkWidget *info = gtk_box_new(GTK_ORIENTATION_VERTICAL, 3);
    gtk_widget_set_valign(info, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(info), G.title, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(info), meta_row, FALSE, FALSE, 0);

    GtkWidget *hints = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_set_valign(hints, GTK_ALIGN_CENTER);
    /* Like the other pickers: key hints, no button (a white one glared). */
    G.apply = label("APPLY", "key-label", FALSE);
    gtk_box_pack_start(GTK_BOX(hints), label("ENTER", "key", FALSE), FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(hints), G.apply, FALSE, FALSE, 0);
    const char *keys[2][2] = {{"\xe2\x86\x90 \xe2\x86\x92", "MOVE"}, {"ESC", "CLOSE"}};
    for (int k = 0; k < 2; k++) {
        gtk_box_pack_start(GTK_BOX(hints), label(keys[k][0], "key", FALSE), FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(hints), label(keys[k][1], "key-label", FALSE), FALSE, FALSE, 0);
    }

    GtkWidget *head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 16);
    gtk_box_pack_start(GTK_BOX(head), info, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(head), hints, FALSE, FALSE, 0);

    /* filmstrip: load already-cached visible tiles synchronously (~2 ms),
     * and decode everything else (off-screen or cold-cache) on worker threads
     * radiating outward from the initial selection so the window opens at once. */
    int init_sel = A.select >= 0 && A.select < S.n ? A.select : 0;
    int dock_w = MIN(1240, win.w - 2 * 48);
    int view_w = dock_w - 2 * 20 - 2;                 /* minus padding and border */
    GtkWidget *row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    gboolean *loaded = g_new0(gboolean, S.n);
    for (int i = 0; i < S.n; i++) {
        Item *it = &S.items[i];
        const char *src = *it->image ? it->image : it->under;
        GtkWidget *pic;
        if (*src) {
            G.thumb_cache[i] = thumb_cache_path(src, G.frag_path[i]);
            GdkPixbuf *pb = NULL;
            if (abs(i - init_sel) <= 5 && g_file_test(G.thumb_cache[i], G_FILE_TEST_EXISTS)) {
                utime(G.thumb_cache[i], NULL);
                pb = gdk_pixbuf_new_from_file(G.thumb_cache[i], NULL);
            }
            if (pb) {
                pic = gtk_image_new_from_pixbuf(pb);
                g_object_unref(pb);
                loaded[i] = TRUE;
            } else {
                pic = gtk_image_new();
                add_class(pic, "thumb-slot");
            }
            G.thumb_img[i] = pic;
        } else {
            pic = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
            add_class(pic, "addtile");
            GtkWidget *plus = label("+", "plus", FALSE);
            gtk_widget_set_halign(plus, GTK_ALIGN_CENTER);
            gtk_box_set_center_widget(GTK_BOX(pic), plus);
            loaded[i] = TRUE;
        }
        gtk_widget_set_size_request(pic, TW, TH);
        GtkWidget *frame = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        add_class(frame, "tframe");
        gtk_box_pack_start(GTK_BOX(frame), pic, FALSE, FALSE, 0);
        GtkWidget *dot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
        add_class(dot, "dot");
        gtk_widget_set_halign(dot, GTK_ALIGN_CENTER);
        GtkWidget *tile = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
        add_class(tile, "tile");
        if (i == A.active) add_class(tile, "active");
        gtk_box_pack_start(GTK_BOX(tile), frame, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(tile), dot, FALSE, FALSE, 0);
        S.marks[i] = tile;
        GtkWidget *ev = gtk_event_box_new();
        add_class(ev, "th");
        gtk_container_add(GTK_CONTAINER(ev), tile);
        gtk_widget_add_events(ev, GDK_BUTTON_PRESS_MASK | GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK);
        g_signal_connect(ev, "button-press-event", G_CALLBACK(on_tile_click), GINT_TO_POINTER(i));
        gtk_box_pack_start(GTK_BOX(row), ev, FALSE, FALSE, 0);
    }

    /* Defer async thumbnail loading/GPU shader baking until gallery_after_show()
     * finishes realizing LiveGl so live wallpapers render with dither + atmosphere. */
    GArray *order = g_array_new(FALSE, FALSE, sizeof(int));
    for (int d = 0; d < S.n; d++) {
        int cand[2] = {init_sel + d, init_sel - d};
        int cnt = d == 0 ? 1 : 2;
        for (int k = 0; k < cnt; k++) {
            int idx = cand[k];
            if (idx >= 0 && idx < S.n && !loaded[idx] && G.thumb_cache[idx])
                g_array_append_val(order, idx);
        }
    }
    g_free(loaded);
    G.thumb_order = order;
    GtkWidget *scroll = scrolled(row, GTK_POLICY_EXTERNAL, GTK_POLICY_NEVER);
    gtk_widget_set_size_request(scroll, view_w, -1);
    glide_init(&S.glide, scroll, gtk_scrolled_window_get_hadjustment(GTK_SCROLLED_WINDOW(scroll)), 300);

    GtkWidget *dock = gtk_box_new(GTK_ORIENTATION_VERTICAL, 14);
    add_class(dock, "dock");
    gtk_box_pack_start(GTK_BOX(dock), head, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(dock), scroll, FALSE, FALSE, 0);
    GtkWidget *dev = gtk_event_box_new();
    G.dock_ev = dev;
    gtk_widget_set_halign(dev, GTK_ALIGN_CENTER);
    gtk_widget_set_valign(dev, GTK_ALIGN_END);
    gtk_widget_set_size_request(dev, dock_w, -1);
    gtk_container_add(GTK_CONTAINER(dev), dock);
    g_signal_connect(dev, "button-press-event", G_CALLBACK(swallow), NULL);

    GtkWidget *over = gtk_overlay_new();
    G.over = over;
    gtk_container_add(GTK_CONTAINER(over), base_view);
    gtk_overlay_add_overlay(GTK_OVERLAY(over), dev);
    return over;
}

/* -- showcase previews ------------------------------------------------------------------ */
static struct {
    int w, h;
    GtkWidget **img;
    gboolean *loaded;
    gboolean *pending;
    guint idle;
    char *cache;
    GCancellable *cancel;
    gboolean alive;
} PV;

#define PREVIEW_KEEP 3       /* previews kept around the selected one */
#define PREVIEW_DAYS 30      /* cached previews unused this long are removed */

static void showcase_cleanup(void) {
    PV.alive = FALSE;
    if (PV.cancel) g_cancellable_cancel(PV.cancel);
    if (PV.idle) { g_source_remove(PV.idle); PV.idle = 0; }
}

static void file_sig(GString *k, const char *path) {
    struct stat st;
    if (*path && stat(path, &st) == 0)
        g_string_append_printf(k, "%s %lld %lld|", path,
                               (long long)st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec,
                               (long long)st.st_size);
    else
        g_string_append_printf(k, "%s - -|", path);
}

/* The finished picture (dithered + atmosphere base + bar overlay, rounded) is
 * cached on disk, keyed by wallpaper, sidecar .frag shader, bar image, and size. */
static char *cache_dir(const char *sub) {
    const char *xc = g_getenv("XDG_CACHE_HOME");
    return xc && *xc ? g_build_filename(xc, "misa-ui", sub, NULL)
                     : g_build_filename(g_get_home_dir(), ".cache", "misa-ui", sub, NULL);
}

static char *preview_cache_path(const Item *it, const char *frag) {
    if (!PV.cache) PV.cache = cache_dir("previews");
    GString *k = g_string_new("dither-v2|");
    file_sig(k, it->under);
    file_sig(k, frag ? frag : "");
    file_sig(k, it->image);
    g_string_append_printf(k, "%dx%d", PV.w, PV.h);
    char *sum = g_compute_checksum_for_string(G_CHECKSUM_SHA1, k->str, -1);
    sum[24] = 0;
    char *name = g_strconcat(sum, ".png", NULL);
    char *path = g_build_filename(PV.cache, name, NULL);
    g_free(name);
    g_free(sum);
    g_string_free(k, TRUE);
    return path;
}

/* Offscreen-bake ONE crisp 3px Bayer dither + atmosphere snapshot via
 * `misa-wall --bake-frame` on cache miss (zero OpenGL inside Theme Picker!). */
static GdkPixbuf *bake_dither_pixbuf(const char *cache_dir_path, const char *wall,
                                     const char *frag, int pw, int ph) {
    char *bin = g_build_filename(self_dir(), "misa-wall", NULL);
    if (!g_file_test(bin, G_FILE_TEST_IS_EXECUTABLE)) {
        g_free(bin);
        return NULL;
    }
    g_mkdir_with_parents(cache_dir_path, 0700);
    char *ppm = g_strdup_printf("%s/.bake-%d-%lu.ppm", cache_dir_path,
                                (int)getpid(), (unsigned long)g_thread_self());
    const char *argv[] = {bin, "--bake-frame", wall, frag, ppm, "4.2", NULL};
    int status = 1;
    GdkPixbuf *out = NULL;
    if (g_spawn_sync(NULL, (char **)argv, NULL,
                     G_SPAWN_STDOUT_TO_DEV_NULL | G_SPAWN_STDERR_TO_DEV_NULL,
                     NULL, NULL, NULL, NULL, &status, NULL)
        && status == 0 && g_file_test(ppm, G_FILE_TEST_IS_REGULAR)) {
        GdkPixbuf *raw = gdk_pixbuf_new_from_file(ppm, NULL);
        if (raw) {
            GdkInterpType interp = pw <= 240 ? GDK_INTERP_BILINEAR : GDK_INTERP_NEAREST;
            out = gdk_pixbuf_scale_simple(raw, pw, ph, interp);
            g_object_unref(raw);
        }
    }
    g_unlink(ppm);
    g_free(ppm);
    g_free(bin);
    return out;
}

static GdkPixbuf *compose_preview_bg(const char *cache_dir_path, const char *under,
                                     const char *image, const char *frag, int pw, int ph) {
    GdkPixbuf *pb = NULL;
    if (*under && frag && *frag)
        pb = bake_dither_pixbuf(cache_dir_path, under, frag, pw, ph);
    if (!pb && *under)
        pb = cover_file(under, pw, ph, 0, FALSE);
    if (pb && *image) {
        /* The picture is a transparent canvas of the screen (a bar where it
         * sits): stretched over the base, decoded at the preview size. */
        GdkPixbuf *s = gdk_pixbuf_new_from_file_at_scale(image, pw, ph, FALSE, NULL);
        if (s) {
            gdk_pixbuf_composite(s, pb, 0, 0, pw, ph, 0, 0, 1, 1, GDK_INTERP_NEAREST, 255);
            g_object_unref(s);
        }
    } else if (!pb && *image) {
        pb = cover_file(image, pw, ph, 0, FALSE);
    }
    if (!pb) return NULL;
    GdkPixbuf *r = round_corners(pb, 16, FALSE);
    g_object_unref(pb);
    return r;
}

typedef struct {
    int idx, w, h;
    char *under, *image, *frag, *cached, *cache_dir;
    GCancellable *cancel;
} BakeJob;

static void bake_job_free(gpointer p) {
    BakeJob *j = p;
    g_free(j->under);
    g_free(j->image);
    g_free(j->frag);
    g_free(j->cached);
    g_free(j->cache_dir);
    if (j->cancel) g_object_unref(j->cancel);
    g_free(j);
}

static void bake_preview_thread(GTask *task, gpointer _src, gpointer data, GCancellable *_c) {
    BakeJob *j = data;
    if (j->cancel && g_cancellable_is_cancelled(j->cancel)) {
        g_task_return_pointer(task, NULL, NULL);
        return;
    }
    GdkPixbuf *pb = compose_preview_bg(j->cache_dir, j->under, j->image, j->frag, j->w, j->h);
    if (pb) {
        g_mkdir_with_parents(j->cache_dir, 0700);
        char *tmp = g_strdup_printf("%s.%lu.tmp", j->cached, (unsigned long)g_thread_self());
        if (gdk_pixbuf_save(pb, tmp, "png", NULL, "compression", "1", NULL)) g_rename(tmp, j->cached);
        else g_unlink(tmp);
        g_free(tmp);
    }
    g_task_return_pointer(task, pb, g_object_unref);
}

static gboolean prefetch(gpointer _);

static void bake_preview_done(GObject *_o, GAsyncResult *res, gpointer data) {
    int i = GPOINTER_TO_INT(data);
    GdkPixbuf *pb = g_task_propagate_pointer(G_TASK(res), NULL);
    if (PV.alive && i >= 0 && i < S.n) {
        PV.pending[i] = FALSE;
        if (pb) {
            PV.loaded[i] = TRUE;
            gtk_image_set_from_pixbuf(GTK_IMAGE(PV.img[i]), pb);
        }
        if (!PV.idle) PV.idle = g_idle_add_full(G_PRIORITY_LOW, prefetch, NULL, NULL);
    }
    if (pb) g_object_unref(pb);
    heap_trim();
}

static void ensure_preview(int i) {
    if (i < 0 || i >= S.n || PV.loaded[i] || PV.pending[i]) return;
    char *frag = *S.items[i].under ? live_gl_find_sidecar(S.items[i].under) : NULL;
    char *cached = preview_cache_path(&S.items[i], frag);
    GdkPixbuf *pb = gdk_pixbuf_new_from_file(cached, NULL);
    if (pb && (gdk_pixbuf_get_width(pb) != PV.w || gdk_pixbuf_get_height(pb) != PV.h)) {
        g_object_unref(pb);
        pb = NULL;
    }
    if (pb) {
        /* Instant cache hit (~4 ms, zero OpenGL, zero main-loop stall!) */
        utime(cached, NULL);
        PV.loaded[i] = TRUE;
        gtk_image_set_from_pixbuf(GTK_IMAGE(PV.img[i]), pb);
        g_object_unref(pb);
        g_free(cached);
        g_free(frag);
        return;
    }
    /* Cache miss: bake offscreen in a background worker thread so Alt+Space
     * opens instantaneously even on cold start or right after a shader edit. */
    PV.pending[i] = TRUE;
    BakeJob *j = g_new0(BakeJob, 1);
    j->idx = i;
    j->w = PV.w;
    j->h = PV.h;
    j->under = g_strdup(S.items[i].under);
    j->image = g_strdup(S.items[i].image);
    j->frag = frag;
    j->cached = cached;
    j->cache_dir = g_strdup(PV.cache);
    j->cancel = PV.cancel ? g_object_ref(PV.cancel) : NULL;
    GTask *t = g_task_new(NULL, NULL, bake_preview_done, GINT_TO_POINTER(i));
    g_task_set_task_data(t, j, bake_job_free);
    g_task_run_in_thread(t, bake_preview_thread);
    g_object_unref(t);
}

/* One neighbour per idle call, nearest first; then drop the far ones. */
static gboolean prefetch(gpointer _) {
    for (int d = 1; d <= 2; d++)
        for (int sgn = 1; sgn >= -1; sgn -= 2) {
            int j = S.cur + sgn * d;
            if (j >= 0 && j < S.n && !PV.loaded[j] && !PV.pending[j]) {
                ensure_preview(j);
                if (PV.pending[j]) {
                    PV.idle = 0;
                    return G_SOURCE_REMOVE;
                }
                return G_SOURCE_CONTINUE;
            }
        }
    gboolean dropped = FALSE;
    for (int j = 0; j < S.n; j++)
        if (PV.loaded[j] && abs(j - S.cur) > PREVIEW_KEEP) {
            gtk_image_clear(GTK_IMAGE(PV.img[j]));
            PV.loaded[j] = FALSE;
            dropped = TRUE;
        }
    if (dropped) heap_trim();
    PV.idle = 0;
    return G_SOURCE_REMOVE;
}

static void preview_selected(int i, gboolean _animate) {
    (void)_animate;
    ensure_preview(i);
    if (!PV.idle) PV.idle = g_idle_add_full(G_PRIORITY_LOW, prefetch, NULL, NULL);
}

/* Cached pictures nobody used for PREVIEW_DAYS. */
static void prune_dir(const char *path) {
    GDir *dir = g_dir_open(path, 0, NULL);
    if (!dir) return;
    time_t old = time(NULL) - PREVIEW_DAYS * 86400;
    const char *n;
    while ((n = g_dir_read_name(dir))) {
        char *p = g_build_filename(path, n, NULL);
        struct stat st;
        if (stat(p, &st) == 0 && st.st_mtime < old) g_unlink(p);
        g_free(p);
    }
    g_dir_close(dir);
}

static void prune_previews(void) {
    char *d = cache_dir(A.layout == GALLERY ? "walls" : "previews");
    prune_dir(d);
    g_free(d);
    if (A.layout == GALLERY) {
        char *td = cache_dir("thumbs");
        prune_dir(td);
        g_free(td);
    }
}

/* -- showcase layout ------------------------------------------------------------------- */
static GtkWidget *build_showcase(int mw, int mh) {
    int aw = A.card_w, ah = A.card_h;
    int list_w = 380, gap = 64;
    int pw = mw - 2 * 96 - list_w - gap;
    int ph = (int)((double)pw * ah / aw);
    if (ph > mh * 0.66) {
        ph = (int)(mh * 0.66);
        pw = (int)((double)ph * aw / ah);
    }
    /* Names, one per row; padded so the first and last can sit centred. */
    GtkWidget *rows = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    gtk_widget_set_margin_top(rows, ph / 2 - 30);
    gtk_widget_set_margin_bottom(rows, ph / 2 - 30);
    for (int i = 0; i < S.n; i++) {
        GtkWidget *r = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
        add_class(r, "row");
        char num[16];
        g_snprintf(num, sizeof num, "%02d", i + 1);
        gtk_box_pack_start(GTK_BOX(r), label(num, "num", FALSE), FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(r), label(S.items[i].title, "name", FALSE), FALSE, FALSE, 0);
        if (i == A.active) gtk_box_pack_start(GTK_BOX(r), badge(), FALSE, FALSE, 0);
        S.marks[i] = r;
        gtk_box_pack_start(GTK_BOX(rows), hoverable(r, i), FALSE, FALSE, 0);
    }
    GtkWidget *scroll = scrolled(rows, GTK_POLICY_NEVER, GTK_POLICY_EXTERNAL);
    gtk_widget_set_size_request(scroll, list_w, ph);
    glide_init(&S.glide, scroll, gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(scroll)), 300);

    /* Zero-OpenGL static snapshot stack: pre-baked 3px Bayer dither + atmosphere
     * + topbar PNGs cached on disk (~35 ms open, 0% GPU, 0% CPU, -100 MiB RAM). */
    PV.w = pw;
    PV.h = ph;
    PV.alive = TRUE;
    PV.cancel = g_cancellable_new();
    PV.img = g_new0(GtkWidget *, S.n);
    PV.loaded = g_new0(gboolean, S.n);
    PV.pending = g_new0(gboolean, S.n);
    GtkWidget *stack = gtk_stack_new();
    gtk_stack_set_transition_duration(GTK_STACK(stack), 340);
    gtk_widget_set_size_request(stack, pw, ph);
    for (int i = 0; i < S.n; i++) {
        GtkWidget *pic = gtk_image_new();
        add_class(pic, "preview");
        PV.img[i] = pic;
        char name[16];
        g_snprintf(name, sizeof name, "%d", i);
        gtk_stack_add_named(GTK_STACK(stack), pic, name);
    }
    S.stack = stack;

    GtkWidget *pev = gtk_event_box_new();
    gtk_container_add(GTK_CONTAINER(pev), stack);
    g_signal_connect(pev, "button-press-event", G_CALLBACK(on_preview_click), NULL);

    GtkWidget *box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, gap);
    gtk_widget_set_halign(box, GTK_ALIGN_CENTER);
    gtk_box_pack_start(GTK_BOX(box), scroll, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(box), pev, FALSE, FALSE, 0);
    return box;
}

/* -- strips layout --------------------------------------------------------------------- */
static GtkWidget *build_strips(int mw, int mh) {
    /* One scale for all: the pictures were cut from canvases --card W wide;
     * that width maps to the plate width. */
    int canvas_w = A.card_w;
    int plate_w = MIN(1400, mw - 2 * 160);
    double k = (double)plate_w / canvas_w;

    GtkWidget *rows = gtk_box_new(GTK_ORIENTATION_VERTICAL, 18);
    for (int i = 0; i < S.n; i++) {
        Item *it = &S.items[i];
        /* decoded straight at the size shown */
        GdkPixbuf *pb = NULL;
        int iw = 0, ih = 0;
        if (*it->image && gdk_pixbuf_get_file_info(it->image, &iw, &ih) && iw > 0 && ih > 0) {
            double w = iw * k, h = ih * k;
            double f = MIN(MIN(1.0, 220 / h), plate_w / w);   /* tall/odd ones capped */
            pb = gdk_pixbuf_new_from_file_at_scale(it->image, MAX(1, (int)(w * f)), MAX(1, (int)(h * f)),
                                                   FALSE, NULL);
        }
        GtkWidget *head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
        char num[16];
        g_snprintf(num, sizeof num, "%02d", i + 1);
        gtk_box_pack_start(GTK_BOX(head), label(num, "num", FALSE), FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(head), label(it->title, "strip-name", FALSE), FALSE, FALSE, 0);
        if (*it->sub) gtk_box_pack_start(GTK_BOX(head), label(it->sub, "strip-sub", FALSE), FALSE, FALSE, 0);
        if (i == A.active) gtk_box_pack_start(GTK_BOX(head), badge(), FALSE, FALSE, 0);
        GtkWidget *plate = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
        add_class(plate, "plate");
        GtkWidget *pic = pb ? gtk_image_new_from_pixbuf(pb) : gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
        if (pb) g_object_unref(pb);
        gtk_widget_set_halign(pic, GTK_ALIGN_CENTER);
        gtk_widget_set_size_request(plate, plate_w, -1);
        gtk_box_set_center_widget(GTK_BOX(plate), pic);
        GtkWidget *r = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
        add_class(r, "strip");
        gtk_box_pack_start(GTK_BOX(r), head, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(r), plate, FALSE, FALSE, 0);
        S.marks[i] = r;
        gtk_box_pack_start(GTK_BOX(rows), hoverable(r, i), FALSE, FALSE, 0);
    }
    int page_h = (int)(mh * 0.74);
    gtk_widget_set_margin_top(rows, page_h / 2 - 40);
    gtk_widget_set_margin_bottom(rows, page_h / 2 - 40);

    GtkWidget *scroll = scrolled(rows, GTK_POLICY_NEVER, GTK_POLICY_EXTERNAL);
    gtk_widget_set_size_request(scroll, plate_w + 2 * 60, page_h);
    gtk_widget_set_halign(scroll, GTK_ALIGN_CENTER);
    glide_init(&S.glide, scroll, gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(scroll)), 300);
    return scroll;
}

/* -- window ----------------------------------------------------------------------------- */
static gboolean centre_later(gpointer _) {
    centre(FALSE);
    return G_SOURCE_REMOVE;
}

static void on_grab_fail(gpointer _d) {
    (void)_d;
    fprintf(stderr, "CardPicker: could not grab the keyboard\n");
    quit();
}

static gboolean build(void) {
    Rect mon = focused_monitor();
    Rect r = below_bar(mon);                  /* the bar stays above, sharp */
    S.win = overlay_window("CardPicker", r.x, r.y, r.w, r.h);
    if (A.layout == GALLERY)                  /* opaque: nothing to blur behind it */
        gtk_widget_set_visual(S.win, gdk_screen_get_system_visual(gtk_widget_get_screen(S.win)));
    Palette p;
    palette_load(&p);
    char *c = css(&p, S.n);
    css_load(c);
    g_free(c);

    S.marks = g_new0(GtkWidget *, S.n);
    GtkWidget *box;
    if (A.layout == GALLERY) {
        box = build_gallery(mon, r);
    } else {
    GtkWidget *body = A.layout == SHOWCASE ? build_showcase(r.w, r.h) : build_strips(r.w, r.h);

    GtkWidget *hints = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_set_halign(hints, GTK_ALIGN_END);
    gtk_widget_set_valign(hints, GTK_ALIGN_END);
    gtk_widget_set_margin_end(hints, 56);
    gtk_widget_set_margin_bottom(hints, 40);
    const char *move = A.hint_move && *A.hint_move ? A.hint_move
                                                   : "\xe2\x86\x91 \xe2\x86\x93";
    const char *keys[3][2] = {{"ENTER", "APPLY"}, {move, "MOVE"}, {"ESC", "BACK"}};
    for (int i = 0; i < 3; i++) {
        gtk_box_pack_start(GTK_BOX(hints), label(keys[i][0], "key", FALSE), FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(hints), label(keys[i][1], "key-label", FALSE), FALSE, FALSE, 0);
    }

    box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start(GTK_BOX(box), gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0), TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(box), body, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(box), hints, TRUE, TRUE, 0);
    }

    GtkWidget *ev = gtk_event_box_new();   /* input for the whole backdrop */
    gtk_container_add(GTK_CONTAINER(ev), box);
    g_signal_connect(ev, "button-press-event", G_CALLBACK(on_button), NULL);
    g_signal_connect(ev, "scroll-event", G_CALLBACK(on_wheel), NULL);
    if (A.layout == GALLERY) {
        g_signal_connect(ev, "motion-notify-event", G_CALLBACK(on_gallery_motion), NULL);
        gtk_widget_add_events(ev, GDK_SCROLL_MASK | GDK_SMOOTH_SCROLL_MASK | GDK_POINTER_MOTION_MASK);
    } else {
        gtk_widget_add_events(ev, GDK_SCROLL_MASK | GDK_SMOOTH_SCROLL_MASK);
    }
    gtk_container_add(GTK_CONTAINER(S.win), ev);

    g_signal_connect(S.win, "key-press-event", G_CALLBACK(on_key), NULL);
    gtk_widget_show_all(S.win);
    if (A.layout == GALLERY) gallery_after_show();
    select_item(A.select >= 0 && A.select < S.n ? A.select : 0, FALSE);
    gdk_display_flush(gdk_display_get_default());
    grab_soon(S.win, G_CALLBACK(on_grab_fail), NULL);
    /* Centre it again once everything has its final size. */
    g_timeout_add(60, centre_later, NULL);
    return TRUE;
}

/* -- main ---------------------------------------------------------------------------------- */
static char *opt_layout, *opt_card;

static gboolean read_items(void) {
    GString *in = g_string_new("");
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, stdin)) > 0) g_string_append_len(in, buf, n);
    char **lines = g_strsplit(in->str, "\n", -1);
    g_string_free(in, TRUE);
    GArray *a = g_array_new(FALSE, TRUE, sizeof(Item));
    for (int i = 0; lines[i]; i++) {
        char *l = lines[i];
        size_t len = strlen(l);
        if (len && l[len - 1] == '\r') l[len - 1] = 0;
        char *t = g_strdup(l);
        gboolean blank = !*g_strstrip(t);
        g_free(t);
        if (blank) continue;
        char **f = g_strsplit(l, "\t", 5);
        const char *v[4] = {"", "", "", ""};
        for (int k = 0; k < 4 && f[k]; k++) v[k] = f[k];
        Item it = {g_strdup(v[0]), g_strdup(v[1]), g_strdup(v[2]), g_strdup(v[3])};
        g_array_append_val(a, it);
        g_strfreev(f);
    }
    g_strfreev(lines);
    S.n = a->len;
    S.items = (Item *)g_array_free(a, FALSE);
    return S.n > 0;
}

int main(int argc, char **argv) {
    GOptionEntry entries[] = {
        {"layout", 0, 0, G_OPTION_ARG_STRING, &opt_layout, "gallery | showcase | strips", "L"},
        {"card", 0, 0, G_OPTION_ARG_STRING, &opt_card, "picture size WxH", "WxH"},
        {"select", 0, 0, G_OPTION_ARG_INT, &A.select, "start on item I", "I"},
        {"active", 0, 0, G_OPTION_ARG_INT, &A.active, "mark item I \"IN USE\"", "I"},
        {"hint-move", 0, 0, G_OPTION_ARG_STRING, &A.hint_move, "keycap of the MOVE hint", "TEXT"},
        {NULL}};
    GError *err = NULL;
    if (!gtk_init_with_args(&argc, &argv, "- full-screen picker", entries, NULL, &err)) {
        fprintf(stderr, "CardPicker: %s\n", err ? err->message : "cannot open the display");
        return 2;
    }
    if (opt_layout) {
        if (!strcmp(opt_layout, "gallery")) A.layout = GALLERY;
        else if (!strcmp(opt_layout, "showcase")) A.layout = SHOWCASE;
        else if (!strcmp(opt_layout, "strips")) A.layout = STRIPS;
        else {
            fprintf(stderr, "CardPicker: --layout gallery|showcase|strips\n");
            return 2;
        }
    }
    if (opt_card) {
        char *x = strchr(opt_card, 'x');
        if (!x) x = strchr(opt_card, 'X');
        char *e1 = NULL, *e2 = NULL;
        long w = x ? strtol(opt_card, &e1, 10) : 0, h = x ? strtol(x + 1, &e2, 10) : 0;
        if (!x || e1 != x || !e2 || *e2 || w <= 0 || h <= 0) {
            fprintf(stderr, "CardPicker: --card WxH\n");
            return 2;
        }
        A.card_w = w;
        A.card_h = h;
    }
    /* One malloc arena: the decode thread's own arena kept ~50 MB of freed
     * pictures that malloc_trim could not hand back. */
    mallopt(M_ARENA_MAX, 1);
    if (!read_items()) return 1;
    close_bar_popups(NULL);
    gboolean ok = build();
    if (ok) gtk_main();
    gtk_widget_destroy(S.win);
    prune_previews();
    if (S.result < 0) return 1;
    printf("%d\n", S.result);
    fflush(stdout);
    return 0;
}
