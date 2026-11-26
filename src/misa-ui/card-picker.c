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
 * The window covers the full monitor so blur and wallpaper transitions extend
 * seamlessly behind floating/island/minimal topbars, while Polybar / dock
 * windows on that monitor are raised directly above CardPicker so they stay
 * sharp. SHOWCASE and STRIPS pad their content below the bar strut.
 */
#include "common.h"

#include <fcntl.h>
#include <gdk/gdkx.h>
#include <glib-unix.h>
#include <glib/gstdio.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <malloc.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utime.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

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

typedef struct {
    Window bar;
    Window prev_above;
    Window prev_below;
} RaisedBar;

/* -- zero-OpenGL live media detection + MWA3 pread streaming player -------------------- */
static char *find_sidecar_frag(const char *image_path) {
    if (!image_path || !*image_path) return NULL;
    const char *env = g_getenv("MISA_WALL_LIVE");
    if (env && strcmp(env, "0") == 0) return NULL;
    char *canon = realpath(image_path, NULL);
    const char *base = canon ? canon : image_path;
    const char *slash = strrchr(base, '/');
    const char *dot = strrchr(base, '.');
    char *out = NULL;
    if (dot && (!slash || dot > slash)) {
        char *stem = g_strndup(base, (gsize)(dot - base));
        char *cand = g_strconcat(stem, ".frag", NULL);
        g_free(stem);
        if (g_file_test(cand, G_FILE_TEST_IS_REGULAR)) out = cand;
        else g_free(cand);
    }
    free(canon);
    return out;
}

static gboolean is_video_path(const char *path) {
    if (!path || !*path) return FALSE;
    const char *dot = strrchr(path, '.');
    if (!dot) return FALSE;
    return g_ascii_strcasecmp(dot, ".mp4") == 0 ||
           g_ascii_strcasecmp(dot, ".webm") == 0 ||
           g_ascii_strcasecmp(dot, ".mkv") == 0 ||
           g_ascii_strcasecmp(dot, ".mov") == 0 ||
           g_ascii_strcasecmp(dot, ".m4v") == 0;
}

static gboolean gif_is_animated_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return FALSE;
    unsigned char hdr[13];
    if (fread(hdr, 1, 13, fp) != 13 ||
        (memcmp(hdr, "GIF87a", 6) != 0 && memcmp(hdr, "GIF89a", 6) != 0)) {
        fclose(fp);
        return FALSE;
    }
    if (hdr[10] & 0x80) {
        long gct = 3L << ((hdr[10] & 7) + 1);
        if (fseek(fp, gct, SEEK_CUR) != 0) { fclose(fp); return FALSE; }
    }
    int frames = 0;
    for (;;) {
        int c = fgetc(fp);
        if (c == EOF || c == 0x3B) break;
        if (c == 0x00) continue;
        if (c == 0x21) {
            if (fgetc(fp) == EOF) break;
            for (;;) {
                int sz = fgetc(fp);
                if (sz <= 0) break;
                if (fseek(fp, sz, SEEK_CUR) != 0) { fclose(fp); return frames >= 2; }
            }
        } else if (c == 0x2C) {
            if (++frames >= 2) { fclose(fp); return TRUE; }
            unsigned char desc[9];
            if (fread(desc, 1, 9, fp) != 9) break;
            if (desc[8] & 0x80) {
                long lct = 3L << ((desc[8] & 7) + 1);
                if (fseek(fp, lct, SEEK_CUR) != 0) break;
            }
            if (fgetc(fp) == EOF) break;
            for (;;) {
                int sz = fgetc(fp);
                if (sz <= 0) break;
                if (fseek(fp, sz, SEEK_CUR) != 0) { fclose(fp); return FALSE; }
            }
        } else {
            break;
        }
    }
    fclose(fp);
    return FALSE;
}

static gboolean webp_is_animated_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return FALSE;
    unsigned char *buf = g_malloc(65536);
    size_t n = fread(buf, 1, 65536, fp);
    fclose(fp);
    gboolean anim = FALSE;
    if (n >= 20 && memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WEBP", 4) == 0) {
        size_t p = 12;
        while (p + 8 <= n) {
            uint32_t clen = (uint32_t)buf[p + 4] | ((uint32_t)buf[p + 5] << 8) |
                            ((uint32_t)buf[p + 6] << 16) | ((uint32_t)buf[p + 7] << 24);
            if (memcmp(buf + p, "VP8X", 4) == 0 && p + 9 <= n) {
                if (buf[p + 8] & 0x02) { anim = TRUE; break; }
            } else if (memcmp(buf + p, "ANIM", 4) == 0) {
                anim = TRUE;
                break;
            }
            size_t step = 8 + (size_t)clen + (size_t)(clen & 1u);
            if (step < 8 || step > n - p) break;
            p += step;
        }
    }
    g_free(buf);
    return anim;
}

static gboolean png_is_animated_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return FALSE;
    unsigned char *buf = g_malloc(65536);
    size_t n = fread(buf, 1, 65536, fp);
    fclose(fp);
    gboolean anim = FALSE;
    static const unsigned char sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (n >= 24 && memcmp(buf, sig, 8) == 0) {
        size_t p = 8;
        while (p + 12 <= n) {
            uint32_t clen = ((uint32_t)buf[p] << 24) | ((uint32_t)buf[p + 1] << 16) |
                            ((uint32_t)buf[p + 2] << 8) | (uint32_t)buf[p + 3];
            if (memcmp(buf + p + 4, "acTL", 4) == 0 && clen >= 4 && p + 12 <= n) {
                uint32_t nf = ((uint32_t)buf[p + 8] << 24) | ((uint32_t)buf[p + 9] << 16) |
                              ((uint32_t)buf[p + 10] << 8) | (uint32_t)buf[p + 11];
                anim = nf > 1;
                break;
            }
            if (memcmp(buf + p + 4, "IDAT", 4) == 0 || memcmp(buf + p + 4, "IEND", 4) == 0) break;
            size_t step = 12 + (size_t)clen;
            if (step < 12 || step > n - p) break;
            p += step;
        }
    }
    g_free(buf);
    return anim;
}

static gboolean detect_live_media(const char *path, const char *frag) {
    const char *env = g_getenv("MISA_WALL_LIVE");
    if (env && strcmp(env, "0") == 0) return FALSE;
    if (frag && *frag) return TRUE;
    if (!path || !*path) return FALSE;
    if (is_video_path(path)) return TRUE;
    FILE *fp = fopen(path, "rb");
    if (!fp) return FALSE;
    unsigned char m[12] = {0};
    size_t n = fread(m, 1, sizeof m, fp);
    fclose(fp);
    if (n >= 6 && (memcmp(m, "GIF87a", 6) == 0 || memcmp(m, "GIF89a", 6) == 0))
        return gif_is_animated_file(path);
    if (n >= 12 && memcmp(m, "RIFF", 4) == 0 && memcmp(m + 8, "WEBP", 4) == 0)
        return webp_is_animated_file(path);
    if (n >= 8 && m[0] == 0x89 && m[1] == 'P' && m[2] == 'N' && m[3] == 'G')
        return png_is_animated_file(path);
    return FALSE;
}

static void misa_wall_pause_daemon(gboolean pause) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    if (!run || !*run) run = "/tmp";
    const char *disp_raw = g_getenv("DISPLAY");
    if (!disp_raw || !*disp_raw) disp_raw = ":0";
    GString *disp = g_string_new("");
    for (const char *p = disp_raw; *p; p++)
        g_string_append_c(disp, g_ascii_isalnum(*p) ? *p : '_');
    if (disp->len == 0) g_string_append_c(disp, '0');
    char *path = g_strdup_printf("%s/misa-wall-live-%s.pause", run, disp->str);
    g_string_free(disp, TRUE);
    if (pause) {
        char *txt = g_strdup_printf("%d\n", (int)getpid());
        g_file_set_contents(path, txt, -1, NULL);
        g_free(txt);
    } else {
        g_unlink(path);
    }
    g_free(path);
}

#define MWA_FLAG_LINEAR 1u

typedef struct {
    uint64_t off;
    uint32_t comp_len;
    uint32_t raw_len;
    uint16_t dirty_x0, dirty_y0, dirty_x1, dirty_y1;
    uint16_t delay_ms;
} MwaFrame;

typedef struct {
    int fd;
    uint32_t cw, ch, n_frames, fps, flags, frame_idx;
    MwaFrame *frames;
    guchar *cell_bgrx;
    cairo_surface_t *cell_surf;
    guchar *comp_buf, *delta_buf;
    size_t comp_cap, delta_cap;
} MwaPlayer;

static inline uint16_t rd_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static inline uint32_t rd_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static inline uint64_t rd_u64_le(const uint8_t *p) {
    return (uint64_t)rd_u32_le(p) | ((uint64_t)rd_u32_le(p + 4) << 32);
}

static gboolean pread_exact(int fd, void *buf, size_t len, uint64_t off) {
    uint8_t *p = buf;
    size_t done = 0;
    while (done < len) {
        ssize_t r = pread(fd, p + done, len - done, (off_t)(off + done));
        if (r <= 0) return FALSE;
        done += (size_t)r;
    }
    return TRUE;
}

static ssize_t lz4_decompress_block(const uint8_t *src, size_t slen, uint8_t *dst, size_t dlen) {
    size_t sp = 0, dp = 0;
    while (sp < slen) {
        uint8_t tok = src[sp++];
        size_t lit = tok >> 4;
        if (lit == 15) {
            uint8_t b;
            do {
                if (sp >= slen) return -1;
                b = src[sp++];
                lit += b;
            } while (b == 255);
        }
        if (lit > slen - sp || lit > dlen - dp) return -1;
        memcpy(dst + dp, src + sp, lit);
        sp += lit;
        dp += lit;
        if (sp >= slen) break;
        if (slen - sp < 2) return -1;
        size_t off = (size_t)src[sp] | ((size_t)src[sp + 1] << 8);
        sp += 2;
        if (off == 0 || off > dp) return -1;
        size_t mlen = (size_t)(tok & 0x0Fu) + 4;
        if ((tok & 0x0Fu) == 15) {
            uint8_t b;
            do {
                if (sp >= slen) return -1;
                b = src[sp++];
                mlen += b;
            } while (b == 255);
        }
        if (mlen > dlen - dp) return -1;
        size_t mp = dp - off;
        if (off >= mlen) {
            memcpy(dst + dp, dst + mp, mlen);
            dp += mlen;
        } else if (off == 1) {
            memset(dst + dp, dst[mp], mlen);
            dp += mlen;
        } else {
            for (size_t k = 0; k < mlen; k++) dst[dp++] = dst[mp++];
        }
    }
    return (ssize_t)dp;
}

static void mwa_player_free(MwaPlayer *m) {
    if (!m) return;
    if (m->cell_surf) cairo_surface_destroy(m->cell_surf);
    if (m->fd >= 0) close(m->fd);
    g_free(m->frames);
    g_free(m->cell_bgrx);
    g_free(m->comp_buf);
    g_free(m->delta_buf);
    g_free(m);
}

static guint mwa_player_delay_ms(const MwaPlayer *m) {
    if (!m || m->n_frames == 0) return 80;
    uint16_t d = m->frames[m->frame_idx % m->n_frames].delay_ms;
    if (d > 0) return (guint)CLAMP((int)d, 16, 10000);
    return (guint)CLAMP((int)(1000u / MAX(m->fps, 1u)), 16, 10000);
}

static MwaPlayer *mwa_player_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 64) { close(fd); return NULL; }
    uint64_t fsize = (uint64_t)st.st_size;

    uint8_t hdr[64];
    if (!pread_exact(fd, hdr, 64, 0) || memcmp(hdr, "MWA3\0\0\0\0", 8) != 0 || rd_u32_le(hdr + 8) != 3) {
        close(fd);
        return NULL;
    }
    uint32_t flags = rd_u32_le(hdr + 12);
    uint32_t cw = rd_u32_le(hdr + 16);
    uint32_t ch = rd_u32_le(hdr + 20);
    uint32_t nf = rd_u32_le(hdr + 24);
    uint32_t fps = rd_u32_le(hdr + 28);
    uint64_t base_off = rd_u64_le(hdr + 40);
    uint32_t base_clen = rd_u32_le(hdr + 48);
    uint64_t tab_off = rd_u64_le(hdr + 52);

    if (cw == 0 || ch == 0 || cw > 1920 || ch > 1080 || nf == 0 || nf > 4096 ||
        tab_off + (uint64_t)nf * 44 > fsize || base_off + (uint64_t)base_clen > fsize) {
        close(fd);
        return NULL;
    }

    uint8_t *raw_tab = g_malloc((size_t)nf * 44);
    if (!pread_exact(fd, raw_tab, (size_t)nf * 44, tab_off)) {
        g_free(raw_tab);
        close(fd);
        return NULL;
    }
    MwaFrame *frames = g_new0(MwaFrame, nf);
    size_t max_comp = base_clen, max_raw = 16;
    for (uint32_t i = 0; i < nf; i++) {
        const uint8_t *e = raw_tab + (size_t)i * 44;
        frames[i].off = rd_u64_le(e);
        frames[i].comp_len = rd_u32_le(e + 8);
        frames[i].raw_len = rd_u32_le(e + 12);
        frames[i].dirty_x0 = rd_u16_le(e + 16);
        frames[i].dirty_y0 = rd_u16_le(e + 18);
        frames[i].dirty_x1 = rd_u16_le(e + 20);
        frames[i].dirty_y1 = rd_u16_le(e + 22);
        frames[i].delay_ms = rd_u16_le(e + 42);
        if (frames[i].comp_len > 0 && frames[i].off + frames[i].comp_len > fsize) {
            g_free(raw_tab);
            g_free(frames);
            close(fd);
            return NULL;
        }
        if (frames[i].comp_len > max_comp) max_comp = frames[i].comp_len;
        if (frames[i].raw_len > max_raw) max_raw = frames[i].raw_len;
    }
    g_free(raw_tab);

    size_t npx = (size_t)cw * (size_t)ch;
    uint8_t *comp = g_malloc(MAX(max_comp, 16));
    uint8_t *planar = g_malloc(npx * 3);
    if (!pread_exact(fd, comp, base_clen, base_off) ||
        lz4_decompress_block(comp, base_clen, planar, npx * 3) != (ssize_t)(npx * 3)) {
        g_free(comp);
        g_free(planar);
        g_free(frames);
        close(fd);
        return NULL;
    }

    guchar *bgrx = g_malloc(npx * 4);
    const uint8_t *bp = planar;
    const uint8_t *gp = planar + npx;
    const uint8_t *rp = planar + npx * 2;
    for (uint32_t y = 0; y < ch; y++) {
        size_t row = (size_t)y * cw;
        uint8_t pb = 0, pg = 0, pr = 0;
        for (uint32_t x = 0; x < cw; x++) {
            size_t p = row + x;
            pb = (uint8_t)(pb + bp[p]);
            pg = (uint8_t)(pg + gp[p]);
            pr = (uint8_t)(pr + rp[p]);
            guchar *d = bgrx + p * 4;
            d[0] = pb; d[1] = pg; d[2] = pr; d[3] = 0;
        }
    }
    g_free(planar);

    MwaPlayer *m = g_new0(MwaPlayer, 1);
    m->fd = fd;
    m->cw = cw;
    m->ch = ch;
    m->n_frames = nf;
    m->fps = CLAMP(fps, 1u, 60u);
    m->flags = flags;
    m->frame_idx = 0;
    m->frames = frames;
    m->cell_bgrx = bgrx;
    m->cell_surf = cairo_image_surface_create_for_data(bgrx, CAIRO_FORMAT_RGB24, (int)cw, (int)ch, (int)cw * 4);
    m->comp_buf = comp;
    m->comp_cap = MAX(max_comp, 16);
    m->delta_buf = g_malloc(MAX(max_raw, 16));
    m->delta_cap = MAX(max_raw, 16);
    return m;
}

static gboolean mwa_player_advance(MwaPlayer *m, int *out_x0, int *out_y0, int *out_x1, int *out_y1) {
    if (out_x0) *out_x0 = 0;
    if (out_y0) *out_y0 = 0;
    if (out_x1) *out_x1 = 0;
    if (out_y1) *out_y1 = 0;
    if (!m || m->n_frames <= 1) return FALSE;
    uint32_t next = (m->frame_idx + 1) % m->n_frames;
    const MwaFrame *f = &m->frames[next];
    if (f->comp_len > 0 && f->raw_len >= 8 &&
        f->comp_len <= m->comp_cap && f->raw_len <= m->delta_cap) {
        if (!pread_exact(m->fd, m->comp_buf, f->comp_len, f->off)) return FALSE;
        if (lz4_decompress_block(m->comp_buf, f->comp_len, m->delta_buf, f->raw_len) != (ssize_t)f->raw_len)
            return FALSE;
        uint32_t runs_len = rd_u32_le(m->delta_buf);
        uint32_t n_px = rd_u32_le(m->delta_buf + 4);
        if (8u + (uint64_t)runs_len + (uint64_t)n_px * 3u != f->raw_len) return FALSE;
        const uint8_t *runs = m->delta_buf + 8;
        const uint8_t *db = runs + runs_len;
        const uint8_t *dg = db + n_px;
        const uint8_t *dr = dg + n_px;
        size_t max_px = (size_t)m->cw * (size_t)m->ch;
        size_t rp = 0, dp = 0, cp = 0;
        while (rp < runs_len) {
            size_t skip = 0;
            while (rp < runs_len && runs[rp] == 255) { skip += 255; rp++; }
            if (rp >= runs_len) return FALSE;
            skip += runs[rp++];
            dp += skip;
            size_t cpy = 0;
            while (rp < runs_len && runs[rp] == 255) { cpy += 255; rp++; }
            if (rp >= runs_len) return FALSE;
            cpy += runs[rp++];
            if (dp + cpy > max_px || cp + cpy > n_px) return FALSE;
            for (size_t k = 0; k < cpy; k++) {
                guchar *p = m->cell_bgrx + (dp + k) * 4;
                size_t ci = cp + k;
                p[0] = (guchar)(p[0] + db[ci]);
                p[1] = (guchar)(p[1] + dg[ci]);
                p[2] = (guchar)(p[2] + dr[ci]);
            }
            cp += cpy;
            dp += cpy + 1;
        }
        int x0 = CLAMP((int)f->dirty_x0, 0, (int)m->cw);
        int y0 = CLAMP((int)f->dirty_y0, 0, (int)m->ch);
        int x1 = CLAMP((int)f->dirty_x1, x0, (int)m->cw);
        int y1 = CLAMP((int)f->dirty_y1, y0, (int)m->ch);
        if (x1 > x0 && y1 > y0) {
            cairo_surface_mark_dirty_rectangle(m->cell_surf, x0, y0, x1 - x0, y1 - y0);
            if (out_x0) *out_x0 = x0;
            if (out_y0) *out_y0 = y0;
            if (out_x1) *out_x1 = x1;
            if (out_y1) *out_y1 = y1;
        }
    }
    m->frame_idx = next;
    return TRUE;
}

static void mwa_player_paint_cover(const MwaPlayer *m, cairo_t *cr, int mw, int mh, int top, int ww, int wh) {
    if (!m || !m->cell_surf) return;
    double k = MAX((double)mw / (double)m->cw, (double)mh / (double)m->ch);
    double ox = ((double)m->cw * k - (double)mw) * 0.5;
    double oy = ((double)m->ch * k - (double)mh) * 0.5 + (double)MIN(top, mh - wh);
    cairo_save(cr);
    cairo_translate(cr, -ox, -oy);
    cairo_scale(cr, k, k);
    cairo_set_source_surface(cr, m->cell_surf, 0, 0);
    cairo_pattern_t *pat = cairo_get_source(cr);
    cairo_pattern_set_filter(pat, (m->flags & MWA_FLAG_LINEAR) ? CAIRO_FILTER_BILINEAR : CAIRO_FILTER_NEAREST);
    cairo_pattern_set_extend(pat, CAIRO_EXTEND_PAD);
    cairo_paint(cr);
    cairo_restore(cr);
}

static void mwa_player_snapshot_bgrx(const MwaPlayer *m, guchar *dst, int mw, int mh, int top, int ww, int wh) {
    cairo_surface_t *s = cairo_image_surface_create_for_data(dst, CAIRO_FORMAT_RGB24, ww, wh, ww * 4);
    cairo_t *cr = cairo_create(s);
    mwa_player_paint_cover(m, cr, mw, mh, top, ww, wh);
    cairo_destroy(cr);
    cairo_surface_flush(s);
    cairo_surface_destroy(s);
}

static char *ensure_mwa_cache_path(const char *media_path) {
    if (!media_path || !*media_path) return NULL;
    char *bin = g_build_filename(self_dir(), "misa-wall", NULL);
    if (!g_file_test(bin, G_FILE_TEST_IS_EXECUTABLE)) { g_free(bin); return NULL; }
    const char *argv[] = {bin, "--ensure-cache", media_path, NULL};
    char *out = run_out(argv);
    g_free(bin);
    if (!out) return NULL;
    g_strstrip(out);
    if (!*out || !g_file_test(out, G_FILE_TEST_IS_REGULAR)) {
        g_free(out);
        return NULL;
    }
    return out;
}

static struct {
    Rect mon;                             /* focused monitor geometry */
    int mw, mh, top;                      /* monitor size, bar height above us (0 in gallery) */
    int ww, wh;                           /* window */
    RaisedBar *raised_bars;
    int n_raised_bars;
    GtkWidget *over, *canvas, *dock_ev;
    cairo_surface_t *surf;
    guchar *cur_buf, *src_buf, *tgt_buf;
    guint16 *thresh_buf;
    int *org_sa, *org_cc, *org_cb, *org_sd;
    int bayer4[4][4];
    gint64 trans_start_us;
    int trans_dur_ms, prev_head;
    gboolean has_cur, trans_done, full_ready, applying, alive, use_avx2, dev_ready;
    guint tick_id;
    MwaPlayer *mwa;
    gboolean mwa_playing;
    guint mwa_timer;
    int shown;
    guint gen;                            /* bumped per selection: late decodes are dropped */
    guint settle;                         /* the full decode waits until the selection rests */
    gboolean loading;                     /* a quick picture is being made in a thread */
    int pending;                          /* the selection that came in meanwhile, or -1 */
    GtkWidget *title, *meta, *badge, *apply;
    GtkWidget **thumb_img;                /* filmstrip images filled asynchronously */
    char **thumb_cache;                   /* shared on-disk 176x99 rounded PNG path per item */
    char **frag_path;                     /* sidecar <stem>.frag per item, or NULL */
    gboolean *is_live;                    /* TRUE for .frag, animated GIF/WebP/APNG, video */
    GArray *thumb_order;                  /* cold-cache / off-screen tile indices */
    GCancellable *thumb_cancel;
} G = {.shown = -2, .pending = -1, .prev_head = -1};

/* Lazily allocate transition buffers (src_buf 8.3 MB + thresh_buf 4.1 MB = 12.4 MB)
 * ONLY when the user switches wallpapers — zero overhead on initial gallery open! */
static void develop_ensure_ready(void) {
    if (G.dev_ready) return;
    G.dev_ready = TRUE;
#if defined(__x86_64__)
    G.use_avx2 = __builtin_cpu_supports("avx2") ? TRUE : FALSE;
#endif
    for (int by = 0; by < 4; by++)
        for (int bx = 0; bx < 4; bx++)
            G.bayer4[by][bx] = (BAYER4_RAW[by][bx] * DEV_W_BAYER) / 15;

    if (!G.src_buf) G.src_buf = g_malloc((size_t)G.ww * (size_t)G.wh * 4);
    G.thresh_buf = g_new(guint16, (size_t)G.ww * (size_t)G.wh);
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
    develop_ensure_ready();
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
    develop_ensure_ready();
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

static void mwa_stop_timer(void) {
    if (G.mwa_timer) {
        g_source_remove(G.mwa_timer);
        G.mwa_timer = 0;
    }
    G.mwa_playing = FALSE;
}

static void mwa_drop(gboolean snapshot_to_cur) {
    if (snapshot_to_cur && G.mwa && G.mwa_playing && G.cur_buf) {
        mwa_player_snapshot_bgrx(G.mwa, G.cur_buf, G.mw, G.mh, G.top, G.ww, G.wh);
        if (G.surf) cairo_surface_mark_dirty(G.surf);
    }
    mwa_stop_timer();
    if (G.mwa) {
        mwa_player_free(G.mwa);
        G.mwa = NULL;
    }
}

static gboolean mwa_tick_cb(gpointer _d) {
    G.mwa_timer = 0;
    if (!G.alive || !G.mwa || !G.mwa_playing || !G.trans_done || G.applying)
        return G_SOURCE_REMOVE;
    int cx0 = 0, cy0 = 0, cx1 = 0, cy1 = 0;
    if (mwa_player_advance(G.mwa, &cx0, &cy0, &cx1, &cy1) && G.canvas && cx1 > cx0 && cy1 > cy0) {
        /* Map dirty cell bounding box through cover transform so Cairo only scales
         * & repaints the changed region (3-10x lower CPU than full-screen redraw!). */
        double k = MAX((double)G.mw / (double)G.mwa->cw, (double)G.mh / (double)G.mwa->ch);
        double ox = ((double)G.mwa->cw * k - (double)G.mw) * 0.5;
        double oy = ((double)G.mwa->ch * k - (double)G.mh) * 0.5 + (double)MIN(G.top, G.mh - G.wh);
        int pad = (G.mwa->flags & MWA_FLAG_LINEAR) ? (int)ceil(k) + 2 : 2;
        int sx0 = CLAMP((int)floor((double)cx0 * k - ox) - pad, 0, G.ww);
        int sy0 = CLAMP((int)floor((double)cy0 * k - oy) - pad, 0, G.wh);
        int sx1 = CLAMP((int)ceil((double)cx1 * k - ox) + pad, sx0, G.ww);
        int sy1 = CLAMP((int)ceil((double)cy1 * k - oy) + pad, sy0, G.wh);
        if (sx1 > sx0 && sy1 > sy0)
            gtk_widget_queue_draw_area(G.canvas, sx0, sy0, sx1 - sx0, sy1 - sy0);
    }
    G.mwa_timer = g_timeout_add(mwa_player_delay_ms(G.mwa), mwa_tick_cb, NULL);
    return G_SOURCE_REMOVE;
}

static void mwa_start_playback(void) {
    if (!G.alive || !G.mwa || !G.trans_done || G.applying || G.mwa->n_frames <= 1) return;
    if (!G.mwa_playing) {
        G.mwa_playing = TRUE;
        if (G.canvas) gtk_widget_queue_draw(G.canvas);
    }
    if (!G.mwa_timer)
        G.mwa_timer = g_timeout_add(mwa_player_delay_ms(G.mwa), mwa_tick_cb, NULL);
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
    misa_wall_pause_daemon(FALSE);
    quit();
    return G_SOURCE_REMOVE;
}

/* One row of Mira's Luma-Halftone Develop: skips pixels already settled on the
 * previous frame and pixels not yet reached, touching only the ~6% actively
 * blooming halftone dots plus newly settled copies. */
static gboolean develop_row_scalar(int y, int x_start, int head, int prev_lim) {
    size_t px_off = (size_t)y * (size_t)G.ww;
    const guint16 *th_row = G.thresh_buf + px_off;
    guchar *cur = G.cur_buf + px_off * 4;
    const guchar *src = G.src_buf + px_off * 4;
    const guchar *tgt = G.tgt_buf + px_off * 4;
    gboolean touched = FALSE;

    for (int x = x_start; x < G.ww; x++) {
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

#if defined(__x86_64__)
__attribute__((target("avx2")))
static gboolean develop_row_avx2(int y, int head, int prev_lim) {
    size_t px_off = (size_t)y * (size_t)G.ww;
    const guint16 *th_row = G.thresh_buf + px_off;
    guchar *cur = G.cur_buf + px_off * 4;
    const guchar *src = G.src_buf + px_off * 4;
    const guchar *tgt = G.tgt_buf + px_off * 4;

    const __m256i head_v = _mm256_set1_epi32(head);
    const __m256i prev_lim_v = _mm256_set1_epi32(prev_lim);
    const __m256i settle_lim_v = _mm256_set1_epi32(head - DEV_FEATHER);
    const __m256i zero = _mm256_setzero_si256();
    const __m256i feather_v = _mm256_set1_epi32(DEV_FEATHER);
    const __m256i recip_v = _mm256_set1_epi32(DEV_RECIP);
    const __m256i mask_ff = _mm256_set1_epi32(0xff);

    gboolean touched = FALSE;
    int x = 0;
    for (; x + 8 <= G.ww; x += 8) {
        __m128i th16 = _mm_loadu_si128((const __m128i *)(th_row + x));
        __m256i th = _mm256_cvtepu16_epi32(th16);

        __m256i reached = _mm256_cmpgt_epi32(head_v, th);
        __m256i unsettled = _mm256_cmpgt_epi32(th, prev_lim_v);
        __m256i active = _mm256_and_si256(reached, unsettled);
        if (_mm256_testz_si256(active, active)) continue;

        touched = TRUE;
        size_t off = (size_t)x * 4;
        __m256i not_done = _mm256_cmpgt_epi32(th, settle_lim_v);
        if (_mm256_testz_si256(not_done, not_done)) {
            __m256i t_px = _mm256_loadu_si256((const __m256i *)(tgt + off));
            _mm256_storeu_si256((__m256i *)(cur + off), t_px);
            continue;
        }

        __m256i c_px = _mm256_loadu_si256((const __m256i *)(cur + off));
        __m256i s_px = _mm256_loadu_si256((const __m256i *)(src + off));
        __m256i t_px = _mm256_loadu_si256((const __m256i *)(tgt + off));

        __m256i diff = _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(head_v, th), zero), feather_v);
        __m256i alpha = _mm256_srli_epi32(_mm256_mullo_epi32(diff, recip_v), 16);

        __m256i sb = _mm256_and_si256(s_px, mask_ff);
        __m256i sg = _mm256_and_si256(_mm256_srli_epi32(s_px, 8), mask_ff);
        __m256i sr = _mm256_and_si256(_mm256_srli_epi32(s_px, 16), mask_ff);

        __m256i tb = _mm256_and_si256(t_px, mask_ff);
        __m256i tg = _mm256_and_si256(_mm256_srli_epi32(t_px, 8), mask_ff);
        __m256i tr = _mm256_and_si256(_mm256_srli_epi32(t_px, 16), mask_ff);

        __m256i ob = _mm256_add_epi32(sb, _mm256_srai_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(tb, sb), alpha), 8));
        __m256i og = _mm256_add_epi32(sg, _mm256_srai_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(tg, sg), alpha), 8));
        __m256i or_ = _mm256_add_epi32(sr, _mm256_srai_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(tr, sr), alpha), 8));

        __m256i blended = _mm256_or_si256(_mm256_or_si256(ob, _mm256_slli_epi32(og, 8)), _mm256_slli_epi32(or_, 16));
        __m256i resolved = _mm256_blendv_epi8(t_px, blended, not_done);
        __m256i out = _mm256_blendv_epi8(c_px, resolved, active);
        _mm256_storeu_si256((__m256i *)(cur + off), out);
    }
    if (x < G.ww && develop_row_scalar(y, x, head, prev_lim)) touched = TRUE;
    return touched;
}
#endif

static inline gboolean develop_row(int y, int head, int prev_lim) {
#if defined(__x86_64__)
    if (G.use_avx2) return develop_row_avx2(y, head, prev_lim);
#endif
    return develop_row_scalar(y, 0, head, prev_lim);
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
        else if (G.mwa) mwa_start_playback();
        return G_SOURCE_REMOVE;
    }
    return G_SOURCE_CONTINUE;
}

static void start_gallery_tick(void) {
    G.trans_done = FALSE;
    mwa_stop_timer();
    if (!G.tick_id && G.canvas)
        G.tick_id = gtk_widget_add_tick_callback(G.canvas, gallery_tick, NULL, NULL);
}

static gboolean on_gallery_draw(GtkWidget *_w, cairo_t *cr, gpointer _d) {
    if (G.mwa && G.mwa_playing && G.trans_done) {
        mwa_player_paint_cover(G.mwa, cr, G.mw, G.mh, G.top, G.ww, G.wh);
    } else if (G.surf) {
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

static GdkPixbuf *bake_thumb_pixbuf(const char *cache_dir_path, const char *media, int pw, int ph);

/* A picture as the window's BGRX bytes, made in the thread: the main loop
 * only swaps a pointer. */
static guchar *to_bgrx(GdkPixbuf *pb) {
    guchar *buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
    pixbuf_to_bgrx(pb, buf);
    return buf;
}

typedef struct {
    guchar *buf;
    MwaPlayer *mwa;
} FullResult;

static void full_result_free(gpointer p) {
    FullResult *r = p;
    if (!r) return;
    g_free(r->buf);
    mwa_player_free(r->mwa);
    g_free(r);
}

typedef struct { char *path; gboolean is_live; } FullReq;
static void full_req_free(gpointer p) {
    FullReq *r = p;
    g_free(r->path);
    g_free(r);
}

static void decode_full(GTask *task, gpointer _src, gpointer data, GCancellable *_c) {
    FullReq *r = data;
    FullResult *res = g_new0(FullResult, 1);
    if (r->is_live) {
        char *mwa_path = ensure_mwa_cache_path(r->path);
        if (mwa_path) {
            MwaPlayer *mp = mwa_player_open(mwa_path);
            g_free(mwa_path);
            if (mp) {
                res->buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
                mwa_player_snapshot_bgrx(mp, res->buf, G.mw, G.mh, G.top, G.ww, G.wh);
                res->mwa = mp;
                g_task_return_pointer(task, res, full_result_free);
                return;
            }
        }
    }
    GdkPixbuf *pb = screen_picture(r->path, TRUE);
    if (!pb && is_video_path(r->path)) {
        char *tdir = cache_dir("walls");
        pb = bake_thumb_pixbuf(tdir, r->path, G.ww, G.wh);
        g_free(tdir);
    }
    res->buf = pb ? to_bgrx(pb) : NULL;
    if (pb) g_object_unref(pb);
    g_task_return_pointer(task, res, full_result_free);
}

static void take_target(guchar *buf) {
    g_free(G.tgt_buf);
    G.tgt_buf = buf;
}

typedef struct {
    char **paths;
    gboolean *is_live;
    int n;
    GCancellable *cancel;
} WarmReq;

static void warm_req_free(gpointer p) {
    WarmReq *w = p;
    if (!w) return;
    for (int i = 0; i < w->n; i++) g_free(w->paths[i]);
    g_free(w->paths);
    g_free(w->is_live);
    if (w->cancel) g_object_unref(w->cancel);
    g_free(w);
}

/* Neighbours made ready on disk, one after another, while nothing moves. */
static void warm(GTask *task, gpointer _src, gpointer data, GCancellable *_c) {
    WarmReq *w = data;
    for (int i = 0; i < w->n; i++) {
        if (w->cancel && g_cancellable_is_cancelled(w->cancel)) break;
        if (w->is_live[i]) {
            char *mp = ensure_mwa_cache_path(w->paths[i]);
            g_free(mp);
        } else {
            screen_picture(w->paths[i], FALSE);
        }
    }
    g_task_return_boolean(task, TRUE);
}

static void full_done(GObject *_o, GAsyncResult *res, gpointer gen) {
    FullResult *fr = g_task_propagate_pointer(G_TASK(res), NULL);
    gboolean current = GPOINTER_TO_UINT(gen) == G.gen;
    if (fr && current) {
        if (fr->buf) {
            take_target(fr->buf);
            fr->buf = NULL;
            G.has_cur = TRUE;
        }
        if (fr->mwa) {
            mwa_drop(FALSE);
            G.mwa = fr->mwa;
            fr->mwa = NULL;
        }
        G.full_ready = TRUE;
        if (G.trans_done) {
            if (G.tgt_buf) {
                memcpy(G.cur_buf, G.tgt_buf, (size_t)G.ww * (size_t)G.wh * 4);
                cairo_surface_mark_dirty(G.surf);
            }
            if (G.applying) g_idle_add(gallery_finish_idle, NULL);
            else if (G.mwa) mwa_start_playback();
            else if (G.canvas) gtk_widget_queue_draw(G.canvas);
        } else {
            develop_build_thresh();
            G.prev_head = -1;
            start_gallery_tick();
        }
    }
    full_result_free(fr);
    heap_trim();
    if (!current || !G.alive) return;
    WarmReq *w = g_new0(WarmReq, 1);
    w->paths = g_new0(char *, 2);
    w->is_live = g_new0(gboolean, 2);
    w->cancel = G.thumb_cancel ? g_object_ref(G.thumb_cancel) : NULL;
    for (int d = 1; d >= -1; d -= 2) {
        int j = S.cur + d;
        if (j >= 0 && j < S.n && *S.items[j].under) {
            w->paths[w->n] = g_strdup(S.items[j].under);
            w->is_live[w->n] = G.is_live && G.is_live[j];
            w->n++;
        }
    }
    if (w->n > 0) {
        GTask *t = g_task_new(NULL, NULL, NULL, NULL);
        g_task_set_task_data(t, w, warm_req_free);
        g_task_run_in_thread(t, warm);
        g_object_unref(t);
    } else {
        warm_req_free(w);
    }
}

#define SETTLE_MS 160

static gboolean decode_later(gpointer data) {
    G.settle = 0;
    FullReq *orig = data;
    FullReq *req = g_new0(FullReq, 1);
    req->path = g_strdup(orig->path);
    req->is_live = orig->is_live;
    GTask *t = g_task_new(NULL, NULL, full_done, GUINT_TO_POINTER(G.gen));
    g_task_set_task_data(t, req, full_req_free);
    g_task_run_in_thread(t, decode_full);
    g_object_unref(t);
    return G_SOURCE_REMOVE;
}

/* The quick picture of item i: for live items uses cached MWA3 (base frame 0 +
 * live player) in ~4 ms when cached; for static photos uses cached JPEG or
 * cached thumbnail scaled up while full-res decodes in the background. */
typedef struct {
    char *cached, *thumb, *under;
    guchar *buf;
    MwaPlayer *mwa;
    gboolean full, allow_full, is_live, have_buf;
} Quick;

static void quick_free(gpointer p) {
    Quick *q = p;
    g_free(q->cached);
    g_free(q->thumb);
    g_free(q->under);
    g_free(q->buf);
    mwa_player_free(q->mwa);
    g_free(q);
}

static void make_quick(Quick *q) {
    if (q->is_live && q->allow_full) {
        char *mwa_path = ensure_mwa_cache_path(q->under);
        if (mwa_path) {
            MwaPlayer *mp = mwa_player_open(mwa_path);
            g_free(mwa_path);
            if (mp) {
                q->buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
                mwa_player_snapshot_bgrx(mp, q->buf, G.mw, G.mh, G.top, G.ww, G.wh);
                q->mwa = mp;
                q->full = TRUE;
                q->have_buf = TRUE;
                return;
            }
        }
    }

    GdkPixbuf *pb = NULL;
    if (!q->is_live && g_file_test(q->cached, G_FILE_TEST_EXISTS)) {
        utime(q->cached, NULL);
        pb = gdk_pixbuf_new_from_file(q->cached, NULL);
        q->full = pb != NULL;
    }
    if (!pb && *q->thumb && g_file_test(q->thumb, G_FILE_TEST_EXISTS)) {
        GdkPixbuf *small = gdk_pixbuf_new_from_file(q->thumb, NULL);
        pb = small ? screen_part(small) : NULL;
        if (small) g_object_unref(small);
    }
    if (!pb && (q->allow_full || !is_video_path(q->under)) && *q->under) {
        if (!is_video_path(q->under)) {
            pb = screen_picture(q->under, TRUE);
            if (!q->is_live) q->full = pb != NULL;
        } else if (q->allow_full) {
            char *tdir = cache_dir("walls");
            pb = bake_thumb_pixbuf(tdir, q->under, G.ww, G.wh);
            g_free(tdir);
            q->full = pb != NULL;
        }
    }
    if (pb) {
        q->buf = to_bgrx(pb);
        q->have_buf = TRUE;
        g_object_unref(pb);
    } else {
        q->buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
    }
}

static Quick *quick_new(int i, gboolean allow_full) {
    Quick *q = g_new0(Quick, 1);
    q->is_live = G.is_live && G.is_live[i];
    q->cached = wall_cache_path(S.items[i].under);
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

/* The full picture once the selection rests (at once when applying or live initial). */
static void after_quick(int i, gboolean full, gboolean animate) {
    G.full_ready = full;
    if (G.settle) { g_source_remove(G.settle); G.settle = 0; }
    if (!full) {
        FullReq *req = g_new0(FullReq, 1);
        req->path = g_strdup(S.items[i].under);
        req->is_live = G.is_live && G.is_live[i];
        guint delay = (animate && !G.applying && !req->is_live) ? SETTLE_MS : 0;
        G.settle = g_timeout_add_full(G_PRIORITY_DEFAULT, delay, decode_later, req, full_req_free);
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
        mwa_drop(TRUE);
        if (q->mwa) {
            G.mwa = q->mwa;
            q->mwa = NULL;
        }
        take_target(q->buf);
        q->buf = NULL;
        develop_begin();
        start_gallery_tick();
        after_quick(i, q->full, TRUE);
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

    if (animate && G.has_cur) {
        request_quick(i);
        return;
    }
    mwa_drop(FALSE);
    /* Initial open: show cached thumb/image synchronously (no black frame) and
     * kick off async MWA3 ensure-cache for live wallpapers so open never stalls. */
    Quick *q = quick_new(i, FALSE);
    make_quick(q);
    take_target(q->buf);
    q->buf = NULL;
    memcpy(G.cur_buf, G.tgt_buf, (size_t)G.ww * (size_t)G.wh * 4);
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
    mwa_drop(FALSE);
    misa_wall_pause_daemon(FALSE);
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
    mwa_stop_timer();
    if (G.trans_done && G.full_ready) {
        gallery_apply_root(i);
        misa_wall_pause_daemon(FALSE);
        quit();
        return;
    }
    G.applying = TRUE;
    ungrab();
    if (G.dock_ev) gtk_widget_hide(G.dock_ev);
    if (G.settle) {
        g_source_remove(G.settle);
        G.settle = 0;
        FullReq req = {S.items[i].under, G.is_live && G.is_live[i]};
        decode_later(&req);
    }
    if (!G.trans_done) start_gallery_tick();
}

static void caption(int i) {
    Item *it = &S.items[i];
    gtk_label_set_text(GTK_LABEL(G.title), it->title);
    gboolean add = !*it->under;
    gboolean is_live = G.is_live && G.is_live[i];
    int w = 0, h = 0;
    char *meta;
    if (add) meta = g_strdup(it->sub);
    else if (!is_video_path(it->under) && gdk_pixbuf_get_file_info(it->under, &w, &h))
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
typedef struct { ThumbJob *jobs; int n; char *dir; GCancellable *cancel; } ThumbBatch;
typedef struct { int idx; GdkPixbuf *pb; } ThumbResult;

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
            GdkPixbuf *baked = bake_dither_pixbuf(b->dir, j->src, j->frag, TW, TH);
            if (baked) {
                pb = round_corners(baked, 10, FALSE);
                g_object_unref(baked);
                if (pb) save_thumb_cache(b->dir, j->cache, pb);
            }
        }
        if (!pb && is_video_path(j->src)) {
            GdkPixbuf *baked = bake_thumb_pixbuf(b->dir, j->src, TW, TH);
            if (baked) {
                pb = round_corners(baked, 10, FALSE);
                g_object_unref(baked);
                if (pb) save_thumb_cache(b->dir, j->cache, pb);
            }
        }
        if (!pb) {
            pb = cover_file(j->src, TW, TH, 10, FALSE);
            if (!pb) {
                GdkPixbuf *baked = bake_thumb_pixbuf(b->dir, j->src, TW, TH);
                if (baked) {
                    pb = round_corners(baked, 10, FALSE);
                    g_object_unref(baked);
                }
            }
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

/* Raise viewable Polybar / dock windows on the active monitor directly above
 * CardPicker's full-monitor gallery window so wallpaper transitions play
 * seamlessly behind floating/island/minimal topbars while the bar stays sharp. */
static gboolean is_monitor_bar_window(Display *dpy, Window w, Window self,
                                      Atom a_wm_type, Atom a_dock, Rect mon) {
    if (w == None || w == self) return FALSE;
    XWindowAttributes attr;
    if (!XGetWindowAttributes(dpy, w, &attr)) return FALSE;
    if (attr.map_state != IsViewable || attr.class != InputOutput) return FALSE;
    if (attr.x >= mon.x + mon.w || attr.x + attr.width <= mon.x ||
        attr.y >= mon.y + mon.h || attr.y + attr.height <= mon.y)
        return FALSE;

    gboolean matched = FALSE;
    XClassHint ch = {NULL, NULL};
    if (XGetClassHint(dpy, w, &ch)) {
        if ((ch.res_class && g_ascii_strcasecmp(ch.res_class, "Polybar") == 0) ||
            (ch.res_name && g_ascii_strcasecmp(ch.res_name, "polybar") == 0))
            matched = TRUE;
        if (ch.res_name) XFree(ch.res_name);
        if (ch.res_class) XFree(ch.res_class);
    }
    if (!matched && a_wm_type != None && a_dock != None) {
        Atom actual_type = None;
        int actual_fmt = 0;
        unsigned long nitems = 0, bytes_after = 0;
        unsigned char *prop = NULL;
        if (XGetWindowProperty(dpy, w, a_wm_type, 0, 16, False, XA_ATOM,
                               &actual_type, &actual_fmt, &nitems, &bytes_after, &prop) == Success &&
            prop) {
            if (actual_type == XA_ATOM && actual_fmt == 32) {
                Atom *atoms = (Atom *)prop;
                for (unsigned long k = 0; k < nitems; k++) {
                    if (atoms[k] == a_dock) { matched = TRUE; break; }
                }
            }
            XFree(prop);
        }
    }
    return matched;
}

static gboolean is_transient_overlay(Display *dpy, Window w) {
    gboolean overlay = FALSE;
    XClassHint ch = {NULL, NULL};
    if (XGetClassHint(dpy, w, &ch)) {
        if (ch.res_class &&
            (!strcmp(ch.res_class, "Backdrop") || !strcmp(ch.res_class, "CardPicker") ||
             !strcmp(ch.res_class, "SoundPanel") || !strcmp(ch.res_class, "MusicPanel") ||
             !strcmp(ch.res_class, "WifiPanel") || !strcmp(ch.res_class, "BluetoothPanel") ||
             !strcmp(ch.res_class, "ClipPanel") || !strcmp(ch.res_class, "NotifCenter") ||
             !strcmp(ch.res_class, "NotifToast") || !strcmp(ch.res_class, "WindowSwitch") ||
             !strcmp(ch.res_class, "MediaPill") || !strcmp(ch.res_class, "Rofi")))
            overlay = TRUE;
        if (ch.res_name) XFree(ch.res_name);
        if (ch.res_class) XFree(ch.res_class);
    }
    return overlay;
}

static gboolean has_other_active_overlay(Display *dpy, Window self, Rect mon) {
    Window root_ret = None, parent_ret = None, *kids = NULL;
    unsigned int nk = 0;
    if (!XQueryTree(dpy, DefaultRootWindow(dpy), &root_ret, &parent_ret, &kids, &nk) || !kids)
        return FALSE;
    gboolean found = FALSE;
    for (unsigned int i = 0; i < nk; i++) {
        Window w = kids[i];
        if (w == None || w == self) continue;
        XWindowAttributes attr;
        if (!XGetWindowAttributes(dpy, w, &attr)) continue;
        if (attr.map_state != IsViewable || attr.class != InputOutput) continue;
        if (attr.x >= mon.x + mon.w || attr.x + attr.width <= mon.x ||
            attr.y >= mon.y + mon.h || attr.y + attr.height <= mon.y)
            continue;
        XClassHint ch = {NULL, NULL};
        if (XGetClassHint(dpy, w, &ch)) {
            if (ch.res_class && (!strcmp(ch.res_class, "Backdrop") || !strcmp(ch.res_class, "CardPicker")))
                found = TRUE;
            if (ch.res_name) XFree(ch.res_name);
            if (ch.res_class) XFree(ch.res_class);
        }
        if (found) break;
    }
    XFree(kids);
    return found;
}

static void picker_raise_bars(void) {
    if (!S.win || G.n_raised_bars > 0) return;
    GdkWindow *gw = gtk_widget_get_window(S.win);
    if (!gw) return;
    GdkDisplay *gd = gdk_display_get_default();
    if (!gd || !GDK_IS_X11_DISPLAY(gd)) return;
    Display *dpy = GDK_DISPLAY_XDISPLAY(gd);
    Window card_xwin = gdk_x11_window_get_xid(gw);
    if (card_xwin == None) return;

    gdk_x11_display_error_trap_push(gd);
    Atom a_wm_type = XInternAtom(dpy, "_NET_WM_WINDOW_TYPE", True);
    Atom a_dock = XInternAtom(dpy, "_NET_WM_WINDOW_TYPE_DOCK", True);

    Window root_ret = None, parent_ret = None, *kids = NULL;
    unsigned int nk = 0;
    if (XQueryTree(dpy, DefaultRootWindow(dpy), &root_ret, &parent_ret, &kids, &nk) && kids && nk > 0) {
        gboolean *is_bar = g_new0(gboolean, nk);
        gboolean *is_stable = g_new0(gboolean, nk);
        int count = 0;
        for (unsigned int i = 0; i < nk; i++) {
            if (is_monitor_bar_window(dpy, kids[i], card_xwin, a_wm_type, a_dock, G.mon)) {
                is_bar[i] = TRUE;
                count++;
            } else if (kids[i] != card_xwin) {
                XWindowAttributes attr;
                if (XGetWindowAttributes(dpy, kids[i], &attr) &&
                    attr.map_state == IsViewable && attr.class == InputOutput &&
                    !is_transient_overlay(dpy, kids[i])) {
                    is_stable[i] = TRUE;
                }
            }
        }
        if (count > 0) {
            G.raised_bars = g_new0(RaisedBar, count);
            G.n_raised_bars = 0;
            Window anchor = card_xwin;
            for (unsigned int i = 0; i < nk; i++) {
                if (!is_bar[i]) continue;
                Window above = None, below = None;
                for (unsigned int k = i + 1; k < nk; k++) {
                    if (is_stable[k]) { above = kids[k]; break; }
                }
                for (unsigned int k = i; k > 0; k--) {
                    if (is_stable[k - 1]) { below = kids[k - 1]; break; }
                }
                G.raised_bars[G.n_raised_bars++] = (RaisedBar){kids[i], above, below};
                XWindowChanges wc;
                memset(&wc, 0, sizeof wc);
                wc.sibling = anchor;
                wc.stack_mode = Above;
                XConfigureWindow(dpy, kids[i], CWSibling | CWStackMode, &wc);
                anchor = kids[i];
            }
            XFlush(dpy);
        }
        g_free(is_stable);
        g_free(is_bar);
        XFree(kids);
    }
    gdk_x11_display_error_trap_pop_ignored(gd);
}

static gboolean window_in_list(Window w, const Window *kids, unsigned int nk) {
    if (w == None || !kids) return FALSE;
    for (unsigned int i = 0; i < nk; i++)
        if (kids[i] == w) return TRUE;
    return FALSE;
}

static void picker_restore_bars(void) {
    if (G.n_raised_bars <= 0 || !G.raised_bars) return;
    GdkDisplay *gd = gdk_display_get_default();
    if (gd && GDK_IS_X11_DISPLAY(gd)) {
        Display *dpy = GDK_DISPLAY_XDISPLAY(gd);
        gdk_x11_display_error_trap_push(gd);
        if (!has_other_active_overlay(dpy, None, G.mon)) {
            Window root_ret = None, parent_ret = None, *kids = NULL;
            unsigned int nk = 0;
            if (XQueryTree(dpy, DefaultRootWindow(dpy), &root_ret, &parent_ret, &kids, &nk) && kids) {
                for (int i = 0; i < G.n_raised_bars; i++) {
                    RaisedBar *rb = &G.raised_bars[i];
                    if (!window_in_list(rb->bar, kids, nk)) continue;
                    XWindowChanges wc;
                    memset(&wc, 0, sizeof wc);
                    if (window_in_list(rb->prev_below, kids, nk)) {
                        wc.sibling = rb->prev_below;
                        wc.stack_mode = Above;
                        XConfigureWindow(dpy, rb->bar, CWSibling | CWStackMode, &wc);
                    } else if (window_in_list(rb->prev_above, kids, nk)) {
                        wc.sibling = rb->prev_above;
                        wc.stack_mode = Below;
                        XConfigureWindow(dpy, rb->bar, CWSibling | CWStackMode, &wc);
                    }
                }
                XFree(kids);
                XFlush(dpy);
            }
        }
        gdk_x11_display_error_trap_pop_ignored(gd);
    }
    g_free(G.raised_bars);
    G.raised_bars = NULL;
    G.n_raised_bars = 0;
}

static void gallery_after_show(void) {
    misa_wall_pause_daemon(TRUE);
    gallery_start_thumbs();
}

static GtkWidget *build_gallery(Rect mon, Rect win) {
    G.mon = mon;
    G.mw = mon.w;
    G.mh = mon.h;
    G.top = win.y - mon.y;
    G.ww = win.w;
    G.wh = win.h;
    G.alive = TRUE;
    G.thumb_img = g_new0(GtkWidget *, S.n);
    G.thumb_cache = g_new0(char *, S.n);
    G.frag_path = g_new0(char *, S.n);
    G.is_live = g_new0(gboolean, S.n);
    G.thumb_cancel = g_cancellable_new();
    add_class(S.win, "gallery");

    for (int i = 0; i < S.n; i++) {
        if (*S.items[i].under) {
            G.frag_path[i] = find_sidecar_frag(S.items[i].under);
            G.is_live[i] = detect_live_media(S.items[i].under, G.frag_path[i]);
        }
    }

    /* Zero-OpenGL Cairo canvas: plays AVX2 60 FPS Luma-Halftone Develop transitions
     * and dirty-rect clipped MWA3 animations via pread with 0 MiB Mesa/GLX RAM!
     * src_buf (8.3 MB) + thresh_buf (4.1 MB) are allocated lazily on first switch. */
    G.cur_buf = g_malloc0((size_t)G.ww * (size_t)G.wh * 4);
    G.surf = cairo_image_surface_create_for_data(G.cur_buf, CAIRO_FORMAT_RGB24, G.ww, G.wh, G.ww * 4);
    G.canvas = gtk_drawing_area_new();
    gtk_widget_set_size_request(G.canvas, G.ww, G.wh);
    g_signal_connect(G.canvas, "draw", G_CALLBACK(on_gallery_draw), NULL);
    GtkWidget *base_view = G.canvas;

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

    /* Defer async thumbnail loading until gallery_after_show() so window maps immediately. */
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
 * `misa-wall --bake-frame` on cache miss (zero OpenGL inside CardPicker!). */
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

/* Extract a crisp cover-cropped first frame via `misa-wall --thumb` (videos / fallback). */
static GdkPixbuf *bake_thumb_pixbuf(const char *cache_dir_path, const char *media, int pw, int ph) {
    char *bin = g_build_filename(self_dir(), "misa-wall", NULL);
    if (!g_file_test(bin, G_FILE_TEST_IS_EXECUTABLE)) {
        g_free(bin);
        return NULL;
    }
    g_mkdir_with_parents(cache_dir_path, 0700);
    char *ppm = g_strdup_printf("%s/.thumb-%d-%lu.ppm", cache_dir_path,
                                (int)getpid(), (unsigned long)g_thread_self());
    char ws[16], hs[16];
    g_snprintf(ws, sizeof ws, "%d", pw);
    g_snprintf(hs, sizeof hs, "%d", ph);
    const char *argv[] = {bin, "--thumb", media, ppm, ws, hs, NULL};
    int status = 1;
    GdkPixbuf *out = NULL;
    if (g_spawn_sync(NULL, (char **)argv, NULL,
                     G_SPAWN_STDOUT_TO_DEV_NULL | G_SPAWN_STDERR_TO_DEV_NULL,
                     NULL, NULL, NULL, NULL, &status, NULL)
        && status == 0 && g_file_test(ppm, G_FILE_TEST_IS_REGULAR)) {
        out = gdk_pixbuf_new_from_file(ppm, NULL);
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
    if (!pb && *under && is_video_path(under))
        pb = bake_thumb_pixbuf(cache_dir_path, under, pw, ph);
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
    char *frag = *S.items[i].under ? find_sidecar_frag(S.items[i].under) : NULL;
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

static gboolean on_term_sig(gpointer _d) {
    (void)_d;
    quit();
    return G_SOURCE_REMOVE;
}

static gboolean build(void) {
    Rect mon = focused_monitor();
    Rect safe = below_bar(mon);
    G.mon = mon;
    /* All three layouts cover the full monitor so wallpaper transitions and
     * picom backdrop blur extend seamlessly behind floating/island/minimal
     * topbars, while Polybar / dock windows on this monitor are raised
     * directly above CardPicker so the bar stays sharp. SHOWCASE/STRIPS pad
     * their content to the `safe` area below the bar strut. */
    Rect r = mon;
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
    GtkWidget *body = A.layout == SHOWCASE ? build_showcase(safe.w, safe.h) : build_strips(safe.w, safe.h);

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
    gtk_widget_set_margin_top(box, safe.y - mon.y);
    gtk_widget_set_margin_bottom(box, (mon.y + mon.h) - (safe.y + safe.h));
    gtk_box_pack_start(GTK_BOX(box), gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0), TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(box), body, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(box), hints, TRUE, TRUE, 0);
    }

    GtkWidget *ev = gtk_event_box_new();   /* input for the whole backdrop */
    gtk_container_add(GTK_CONTAINER(ev), box);
    g_signal_connect(ev, "button-press-event", G_CALLBACK(on_button), NULL);
    g_signal_connect(ev, "scroll-event", G_CALLBACK(on_wheel), NULL);
    gtk_widget_add_events(ev, GDK_SCROLL_MASK | GDK_SMOOTH_SCROLL_MASK);
    gtk_container_add(GTK_CONTAINER(S.win), ev);

    g_signal_connect(S.win, "key-press-event", G_CALLBACK(on_key), NULL);
    gtk_widget_show_all(S.win);
    picker_raise_bars();
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
    gtk_env_lean();
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
    g_unix_signal_add(SIGTERM, on_term_sig, NULL);
    g_unix_signal_add(SIGINT, on_term_sig, NULL);
    gboolean ok = build();
    if (ok) gtk_main();
    gboolean wait_fade = (G.n_raised_bars > 0 && S.win
                          && gdk_screen_is_composited(gtk_widget_get_screen(S.win)));
    if (S.win) {
        gtk_widget_hide(S.win);
        gdk_display_flush(gdk_display_get_default());
        gtk_widget_destroy(S.win);
        S.win = NULL;
        gdk_display_flush(gdk_display_get_default());
    }
    if (wait_fade) g_usleep(160000);
    picker_restore_bars();
    prune_previews();
    if (S.result < 0) return 1;
    printf("%d\n", S.result);
    fflush(stdout);
    return 0;
}
