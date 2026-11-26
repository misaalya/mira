/* misa-ui common: shared bits of the rice's GTK overlays (C ports of the
 * former Python tools). See common.c. */
#ifndef MISA_UI_COMMON_H
#define MISA_UI_COMMON_H

#include <gtk/gtk.h>

typedef struct {
    char bg[8], fg[8];          /* "#rrggbb" */
    char font[128];             /* UI font family */
    char display[128];          /* keycap font ("Inter Black") */
} Palette;

typedef struct { int x, y, w, h; } Rect;

void gtk_env_lean(void);
void palette_load(Palette *p);
void css_load(const char *css);
GtkWidget *overlay_window(const char *wmclass, int x, int y, int w, int h);

/* Run argv, return stdout (g_free it) or NULL. */
char *run_out(const char *const *argv);

/* bspwm's focused monitor rectangle, else the pointer's monitor. */
Rect focused_monitor(void);
/* Rectangle of a bspwm monitor by name; FALSE if not found. */
gboolean monitor_rect(const char *name, Rect *r);

/* Resolve monitor rect, bar edge ("top"|"bottom"), and bar offset (px) directly
 * over BSPWM's Unix socket (zero fork, ~0.08 ms vs ~30 ms BarAnchor script). */
gboolean bar_anchor(gboolean pointer, Rect *mon, char edge[8], int *off);

/* r without the bar: bspwm's top/bottom padding of the monitor at r (the
 * bar's strut, when at least BAR_MIN px). Used for toast / panel / picker
 * content placement so UI elements never collide with the bar. */
#define BAR_MIN 8
Rect below_bar(Rect r);

/* Keyboard + pointer grab without blocking the main loop: retried every
 * 10 ms (the bar may still hold the click that opened us). on_fail runs if
 * it never succeeds. */
void grab_soon(GtkWidget *win, GCallback on_fail, gpointer data);
void ungrab(void);

/* Dim + blur behind a popup (bin/misa-backdrop) across the full monitor r;
 * misa-backdrop raises Polybar / dock windows above itself so the bar stays
 * sharp while transparent bar gaps blur seamlessly. Call before showing the
 * popup. Returns its pid (0 when unavailable). */
int backdrop(Rect r, int pid, const char *click);
void drop_backdrop(int pid);

/* Close every other top-bar popup (SoundPanel, MusicPanel, ClipPanel,
 * NotifPanel, WifiPanel, BluetoothPanel, rofi) so only one bar popup is ever
 * open. `except` is the caller's own name ("clip", "notif", "wifi", "bt",
 * "sound", "music") or NULL. */
void close_bar_popups(const char *except);

/* Pictures, the same pixels as misa_ui.py made: scaled to cover w x h
 * (centre crop) with r px antialiased corners (top ones only if top_only). */
GdkPixbuf *round_corners(GdkPixbuf *src, int r, gboolean top_only);
GdkPixbuf *cover_pixbuf(GdkPixbuf *src, int w, int h, int r, gboolean top_only);
GdkPixbuf *load_cover_size(const char *path, int w, int h);           /* NULL if unreadable */
GdkPixbuf *cover_file(const char *path, int w, int h, int r, gboolean top_only);  /* NULL if unreadable */
GdkPixbuf *circle_pixbuf(GdkPixbuf *src, int d);

/* Eased scrolling of an adjustment (ease-out quart), driven by widget's
 * frame clock. */
typedef struct {
    GtkWidget *widget;
    GtkAdjustment *adj;
    gint64 dur;                 /* us */
    double from, to;
    gint64 t0;
    gboolean on;
} Glide;
void glide_init(Glide *g, GtkWidget *widget, GtkAdjustment *adj, int ms);
void glide_to(Glide *g, double target, gboolean animate);

/* Grab keyboard + pointer for win (as rofi does), retrying for a moment
 * (blocking): the key that launched us may still hold sxhkd's grab. */
gboolean grab(GtkWidget *win);

/* Hand freed heap back (after dropping pictures). */
void heap_trim(void);

/* Directory of this executable (for bin/ siblings). */
const char *self_dir(void);

/* Resolve a path relative to Mira's config directory ($MIRA_CONFIG_DIR,
 * falling back to $XDG_CONFIG_HOME/bspwm). Caller must g_free(). */
char *mira_config_path(const char *rel);

#endif
