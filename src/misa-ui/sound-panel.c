/* SoundPanel - native GTK3 sound & per-app volume mixer popup for the top bar
 * (bin/SoundPanel [--pointer] [--side left|right]).
 *
 * Design (Mira monochrome + Apple/iOS clarity):
 *   - Master channels (Output & Microphone) remain the primary focus with bold
 *     26 px pill sliders.
 *   - Per-application audio is aggregated strictly 1 row per application (e.g.
 *     multiple YouTube tabs in Brave collapse into a single "Brave" slider that
 *     controls all of Brave's streams together).
 *   - Subordinate, single-line minimal app rows (8 px capsule slider with 22 px
 *     hit target, no clutter subtitles, hidden completely when no apps play).
 *   - Overflow handling: the main view caps at 3 inline apps so the popup never
 *     grows unboundedly; when >3 apps play, an "All N apps ›" action slides
 *     smoothly (GtkStack 180 ms) to a dedicated scrollable Apps page.
 *
 * Hyperoptimization:
 *   - Zero idle memory (one-shot), zero fork/exec audio IPC via dlopen of
 *     libpulse.so.0 + libpulse-mainloop-glib.so.0 directly on GTK's GMainLoop.
 *   - 100% event-driven via pa_context_subscribe() (0 polling timers, 0% idle).
 *   - Zero-fork bar placement via bar_anchor() over BSPWM's Unix socket.
 *   - Lazy secondary Apps page widgets (allocated only while viewing All Apps).
 *   - In-place widget diffing, dirty-checked updates, static UTF-8 glyphs,
 *     single cached cursor, 60 Hz leading-edge batch coalescing, M_ARENA_MAX=1.
 *
 * picom: class SoundPanel (the "Bar panels" rule).
 */
#include "common.h"

#include <glib-unix.h>

#include <ctype.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <malloc.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#define WIDTH             354
#define GAP               6
#define INLINE_APPS_MAX   3       /* max app rows shown inline on main page */
#define ALL_APPS_MAX_H    360     /* max scroll viewport height on All Apps page */
#define PAGE_ANIM_MS      180     /* GtkStack slide duration */
#define SEND_COALESCE_MS  16      /* ~60 Hz IPC ceiling during continuous drag */
#define ECHO_SUPPRESS_US  220000  /* per-control server echo suppression */
#define INITIAL_SHOW_MS   60      /* hard cap before mapping if PA is slow */

/* Material Design Icons Desktop UTF-8 literals (zero runtime heap alloc). */
#define GL_GEAR       "\U000F0493"
#define GL_VOL_MUTE   "\U000F075F"
#define GL_VOL_LOW    "\U000F057F"
#define GL_VOL_MED    "\U000F0580"
#define GL_VOL_HIGH   "\U000F057E"
#define GL_MIC        "\U000F036C"
#define GL_MIC_MUTE   "\U000F036D"
#define GL_SPEAKER    "\U000F04C3"
#define GL_HEADPHONES "\U000F02CB"
#define GL_WEB        "\U000F059F"
#define GL_MUSIC      "\U000F0387"
#define GL_MUSIC_BOX  "\U000F075A"
#define GL_VIDEO      "\U000F0567"
#define GL_GAMEPAD    "\U000F0297"
#define GL_CHAT       "\U000F0369"
#define GL_TERMINAL   "\U000F018D"
#define GL_BACK       "\U000F0141"
#define GL_CHEV_DN    "\U000F0140"
#define GL_CHEV_RT    "\U000F0142"
#define GL_CHECK      "\U000F012C"

/* -- PulseAudio C ABI (stable SONAME libpulse.so.0 / libpulse-mainloop-glib.so.0) -- */
#define PA_CHANNELS_MAX  32
#define PA_VOLUME_MUTED  0U
#define PA_VOLUME_NORM   0x10000U
#define PA_INVALID_INDEX ((uint32_t)-1)

enum {
    PA_CONTEXT_READY      = 4,
    PA_CONTEXT_FAILED     = 5,
    PA_CONTEXT_TERMINATED = 6,
};

enum {
    PA_SUBSCRIPTION_MASK_SINK       = 0x0001U,
    PA_SUBSCRIPTION_MASK_SOURCE     = 0x0002U,
    PA_SUBSCRIPTION_MASK_SINK_INPUT = 0x0004U,
    PA_SUBSCRIPTION_MASK_SERVER     = 0x0080U,
};

enum {
    PA_SUBSCRIPTION_EVENT_SINK          = 0x0000U,
    PA_SUBSCRIPTION_EVENT_SOURCE        = 0x0001U,
    PA_SUBSCRIPTION_EVENT_SINK_INPUT    = 0x0002U,
    PA_SUBSCRIPTION_EVENT_SERVER        = 0x0007U,
    PA_SUBSCRIPTION_EVENT_FACILITY_MASK = 0x000FU,
    PA_SUBSCRIPTION_EVENT_NEW           = 0x0000U,
    PA_SUBSCRIPTION_EVENT_CHANGE        = 0x0010U,
    PA_SUBSCRIPTION_EVENT_REMOVE        = 0x0020U,
    PA_SUBSCRIPTION_EVENT_TYPE_MASK     = 0x0030U,
};

typedef uint32_t pa_volume_t;
typedef uint64_t pa_usec_t;
typedef struct pa_glib_mainloop pa_glib_mainloop;
typedef struct pa_mainloop_api  pa_mainloop_api;
typedef struct pa_context       pa_context;
typedef struct pa_operation     pa_operation;
typedef struct pa_proplist      pa_proplist;

typedef struct {
    int format;
    uint32_t rate;
    uint8_t channels;
} pa_sample_spec;

typedef struct {
    uint8_t channels;
    int map[PA_CHANNELS_MAX];
} pa_channel_map;

typedef struct {
    uint8_t channels;
    pa_volume_t values[PA_CHANNELS_MAX];
} pa_cvolume;

typedef struct {
    const char *name;
    const char *description;
    uint32_t priority;
    int available;
} pa_sink_port_info;

typedef struct {
    const char *user_name;
    const char *host_name;
    const char *server_version;
    const char *server_name;
    pa_sample_spec sample_spec;
    const char *default_sink_name;
    const char *default_source_name;
    uint32_t cookie;
    pa_channel_map channel_map;
} pa_server_info;

typedef struct {
    const char *name;
    uint32_t index;
    const char *description;
    pa_sample_spec sample_spec;
    pa_channel_map channel_map;
    uint32_t owner_module;
    pa_cvolume volume;
    int mute;
    uint32_t monitor_source;
    const char *monitor_source_name;
    pa_usec_t latency;
    const char *driver;
    int flags;
    pa_proplist *proplist;
    pa_usec_t configured_latency;
    pa_volume_t base_volume;
    int state;
    uint32_t n_volume_steps;
    uint32_t card;
    uint32_t n_ports;
    pa_sink_port_info **ports;
    pa_sink_port_info *active_port;
} pa_sink_info;

typedef struct {
    const char *name;
    uint32_t index;
    const char *description;
    pa_sample_spec sample_spec;
    pa_channel_map channel_map;
    uint32_t owner_module;
    pa_cvolume volume;
    int mute;
    uint32_t monitor_of_sink;
    const char *monitor_of_sink_name;
    pa_usec_t latency;
    const char *driver;
    int flags;
    pa_proplist *proplist;
} pa_source_info;

typedef struct {
    uint32_t index;
    const char *name;
    uint32_t owner_module;
    uint32_t client;
    uint32_t sink;
    pa_sample_spec sample_spec;
    pa_channel_map channel_map;
    pa_cvolume volume;
    pa_usec_t buffer_usec;
    pa_usec_t sink_usec;
    const char *resample_method;
    const char *driver;
    int mute;
    pa_proplist *proplist;
    int corked;
    int has_volume;
    int volume_writable;
    void *format;
} pa_sink_input_info;

typedef void (*pa_context_notify_cb_t)(pa_context *c, void *userdata);
typedef void (*pa_context_success_cb_t)(pa_context *c, int success, void *userdata);
typedef void (*pa_context_subscribe_cb_t)(pa_context *c, int t, uint32_t idx, void *userdata);
typedef void (*pa_server_info_cb_t)(pa_context *c, const pa_server_info *i, void *userdata);
typedef void (*pa_sink_info_cb_t)(pa_context *c, const pa_sink_info *i, int eol, void *userdata);
typedef void (*pa_source_info_cb_t)(pa_context *c, const pa_source_info *i, int eol, void *userdata);
typedef void (*pa_sink_input_info_cb_t)(pa_context *c, const pa_sink_input_info *i, int eol, void *userdata);

static struct {
    void *h_pa;
    void *h_glib;
    pa_glib_mainloop *(*glib_mainloop_new)(GMainContext *c);
    pa_mainloop_api *(*glib_mainloop_get_api)(pa_glib_mainloop *g);
    void (*glib_mainloop_free)(pa_glib_mainloop *g);
    pa_context *(*context_new)(pa_mainloop_api *m, const char *name);
    void (*context_unref)(pa_context *c);
    void (*context_set_state_callback)(pa_context *c, pa_context_notify_cb_t cb, void *userdata);
    int (*context_connect)(pa_context *c, const char *server, int flags, const void *api);
    void (*context_disconnect)(pa_context *c);
    int (*context_get_state)(const pa_context *c);
    void (*context_set_subscribe_callback)(pa_context *c, pa_context_subscribe_cb_t cb, void *userdata);
    pa_operation *(*context_subscribe)(pa_context *c, int mask, pa_context_success_cb_t cb, void *userdata);
    pa_operation *(*context_get_server_info)(pa_context *c, pa_server_info_cb_t cb, void *userdata);
    pa_operation *(*context_get_sink_info_by_index)(pa_context *c, uint32_t idx, pa_sink_info_cb_t cb, void *userdata);
    pa_operation *(*context_get_sink_info_list)(pa_context *c, pa_sink_info_cb_t cb, void *userdata);
    pa_operation *(*context_get_source_info_by_name)(pa_context *c, const char *name, pa_source_info_cb_t cb, void *userdata);
    pa_operation *(*context_get_source_info_by_index)(pa_context *c, uint32_t idx, pa_source_info_cb_t cb, void *userdata);
    pa_operation *(*context_get_sink_input_info)(pa_context *c, uint32_t idx, pa_sink_input_info_cb_t cb, void *userdata);
    pa_operation *(*context_get_sink_input_info_list)(pa_context *c, pa_sink_input_info_cb_t cb, void *userdata);
    pa_operation *(*context_set_sink_volume_by_index)(pa_context *c, uint32_t idx, const pa_cvolume *vol, pa_context_success_cb_t cb, void *userdata);
    pa_operation *(*context_set_sink_mute_by_index)(pa_context *c, uint32_t idx, int mute, pa_context_success_cb_t cb, void *userdata);
    pa_operation *(*context_set_default_sink)(pa_context *c, const char *name, pa_context_success_cb_t cb, void *userdata);
    pa_operation *(*context_set_source_volume_by_index)(pa_context *c, uint32_t idx, const pa_cvolume *vol, pa_context_success_cb_t cb, void *userdata);
    pa_operation *(*context_set_source_mute_by_index)(pa_context *c, uint32_t idx, int mute, pa_context_success_cb_t cb, void *userdata);
    pa_operation *(*context_set_sink_input_volume)(pa_context *c, uint32_t idx, const pa_cvolume *vol, pa_context_success_cb_t cb, void *userdata);
    pa_operation *(*context_set_sink_input_mute)(pa_context *c, uint32_t idx, int mute, pa_context_success_cb_t cb, void *userdata);
    void (*operation_unref)(pa_operation *o);
    pa_volume_t (*cvolume_avg)(const pa_cvolume *a);
    pa_cvolume *(*cvolume_set)(pa_cvolume *a, unsigned channels, pa_volume_t v);
    pa_cvolume *(*cvolume_scale)(pa_cvolume *v, pa_volume_t max);
    const char *(*proplist_gets)(const pa_proplist *p, const char *key);
} PA;

/* -- Data model: Master controls, RawStreams, and aggregated AppGroups ----- */
typedef struct {
    gboolean is_mic;
    uint32_t pa_idx;
    pa_cvolume raw_vol;
    uint8_t channels;
    const char *base_name;

    GtkWidget *block;
    GtkWidget *icon_btn;
    GtkWidget *name_lbl;
    GtkWidget *num_lbl;
    GtkWidget *scale;

    int value;
    gboolean muted;
    gboolean dragging;

    const char *last_glyph;
    char last_title[64];
    int last_shown_pct;
    gboolean last_muted;
    gboolean last_zero;
    gboolean ui_inited;

    guint send_timer;
    int pending_val;
    gboolean has_pending;
    gboolean pending_unmute;
    gint64 suppress_until_us;
} MasterCtrl;

/* One underlying PulseAudio sink-input (e.g. a single browser tab). */
typedef struct {
    uint32_t pa_idx;
    char *app_key;
    char *display_name;
    const char *glyph;
    pa_cvolume raw_vol;
    uint8_t channels;
    int value;
    gboolean muted;
    gboolean corked;
    uint32_t seen_gen;
} RawStream;

typedef struct AppGroup AppGroup;

/* One UI row bound to an AppGroup (either on the inline main view or the
 * secondary All Apps page). */
typedef struct {
    AppGroup *grp;
    GtkWidget *row;
    GtkWidget *icon_btn;
    GtkWidget *name_lbl;
    GtkWidget *scale;
    GtkWidget *num_lbl;
    gboolean in_scroll;

    const char *last_glyph;
    char last_name[64];
    int last_shown_pct;
    gboolean last_muted;
    gboolean last_zero;
    gboolean ui_inited;
} AppRowUI;

/* One logical application (aggregates 1..N RawStreams sharing app_key). */
struct AppGroup {
    char *app_key;
    char *display_name;
    const char *glyph;
    int stream_count;
    int value;
    gboolean muted;
    gboolean all_corked;
    gboolean dragging;

    guint send_timer;
    int pending_val;
    gboolean has_pending;
    gboolean pending_unmute;
    gint64 suppress_until_us;

    AppRowUI *inline_ui;       /* NULL when hidden past INLINE_APPS_MAX */
    AppRowUI *page_ui;         /* NULL unless secondary All Apps page is open */
};

typedef struct {
    uint32_t index;
    char *name;
    char *desc;
    gboolean is_headphones;
} SinkDev;

static struct {
    GtkWidget *win, *box, *stack;
    GtkWidget *main_page, *apps_page;
    GtkWidget *apps_sec, *apps_group_box, *apps_inline_box;
    GtkWidget *apps_more_btn, *apps_more_lbl;
    GtkWidget *all_apps_count_lbl, *all_apps_scroll, *all_apps_box;
    GtkWidget *sinks_rev, *sinks_box;
    GtkWidget *foot_btn, *dev_icon_lbl, *dev_lbl, *dev_chev_lbl;
    GdkCursor *ptr_cursor;

    MasterCtrl out, mic;
    GPtrArray *raw_streams;    /* RawStream* */
    GPtrArray *groups;         /* AppGroup* (sorted, 1 per unique app) */
    GPtrArray *sinks;          /* SinkDev* */

    char *default_sink_name;
    char *default_source_name;
    uint32_t stream_gen;
    gboolean sinks_open;
    gboolean on_apps_page;

    /* Keyboard focus index (page-local) */
    int focus_idx;
    gboolean kb_active;

    /* PulseAudio GLib mainloop state */
    pa_glib_mainloop *pa_ml;
    pa_context *pa_ctx;
    gboolean pa_ok;
    gboolean pa_ready;
    gboolean init_sinks_done;
    gboolean init_source_done;
    gboolean init_streams_done;
    gboolean shown;

    /* Window placement & lifecycle */
    gboolean pointer, closing, syncing;
    char side[8];
    Rect mon;
    char edge[8];
    int off, px, shade;
    guint show_timer;
    guint fit_idle_id;
    guint fit_tick_id;
    gint64 fit_until_us;
} S;

static void panel_close(void);
static void schedule_fit(void);
static void fit_follow(int ms);
static void show_panel_now(void);
static void show_main_page(void);
static void show_apps_page(void);
static void update_focus_visuals(void);
static void refresh_all_state(void);

/* -- small helpers --------------------------------------------------------- */
static inline void add_class(GtkWidget *w, const char *c) {
    gtk_style_context_add_class(gtk_widget_get_style_context(w), c);
}

static inline void set_class(GtkWidget *w, const char *c, gboolean on) {
    GtkStyleContext *sc = gtk_widget_get_style_context(w);
    if (on) gtk_style_context_add_class(sc, c);
    else gtk_style_context_remove_class(sc, c);
}

static void on_clickable_realize(GtkWidget *w, gpointer _d) {
    GdkWindow *gw = gtk_widget_get_window(w);
    if (!gw) return;
    if (!S.ptr_cursor)
        S.ptr_cursor = gdk_cursor_new_from_name(gdk_window_get_display(gw), "pointer");
    if (S.ptr_cursor) gdk_window_set_cursor(gw, S.ptr_cursor);
}

static void make_clickable(GtkWidget *w) {
    g_signal_connect(w, "realize", G_CALLBACK(on_clickable_realize), NULL);
}

static void spawn_fire(const char *const *argv) {
    g_spawn_async(NULL, (char **)argv, NULL,
                  G_SPAWN_SEARCH_PATH | G_SPAWN_STDOUT_TO_DEV_NULL | G_SPAWN_STDERR_TO_DEV_NULL,
                  NULL, NULL, NULL, NULL);
}

static inline void op_unref(pa_operation *op) {
    if (op) PA.operation_unref(op);
}

static int cvolume_to_pct(const pa_cvolume *cv) {
    if (!cv || cv->channels == 0) return 0;
    pa_volume_t avg = PA.cvolume_avg(cv);
    return (int)llround((double)avg * 100.0 / (double)PA_VOLUME_NORM);
}

static void cvolume_apply_pct(pa_cvolume *cv, uint8_t fallback_ch, int pct) {
    pct = CLAMP(pct, 0, 100);
    pa_volume_t target = (pa_volume_t)llround((double)pct * (double)PA_VOLUME_NORM / 100.0);
    uint8_t ch = cv->channels;
    if (ch == 0 || ch > PA_CHANNELS_MAX)
        ch = (fallback_ch > 0 && fallback_ch <= PA_CHANNELS_MAX) ? fallback_ch : 2;
    cv->channels = ch;

    gboolean uniform = TRUE;
    pa_volume_t vmax = 0;
    for (uint8_t i = 0; i < ch; i++) {
        if (cv->values[i] != cv->values[0]) uniform = FALSE;
        if (cv->values[i] > vmax) vmax = cv->values[i];
    }
    if (uniform || vmax == 0) PA.cvolume_set(cv, ch, target);
    else PA.cvolume_scale(cv, target);
}

static gboolean str_contains_ci(const char *hay, const char *needle) {
    if (!hay || !*hay || !needle || !*needle) return FALSE;
    size_t nlen = strlen(needle);
    for (const char *p = hay; *p; p++) {
        if (g_ascii_strncasecmp(p, needle, nlen) == 0) return TRUE;
    }
    return FALSE;
}

static const char *pick_app_glyph(const char *app, const char *bin, const char *icon) {
    const char *keys[3] = {app, bin, icon};
    for (int i = 0; i < 3; i++) {
        const char *k = keys[i];
        if (!k || !*k) continue;
        if (str_contains_ci(k, "brave") || str_contains_ci(k, "chrom") ||
            str_contains_ci(k, "firefox") || str_contains_ci(k, "librewolf") ||
            str_contains_ci(k, "zen") || str_contains_ci(k, "vivaldi") ||
            str_contains_ci(k, "opera") || str_contains_ci(k, "edge") ||
            str_contains_ci(k, "epiphany") || str_contains_ci(k, "browser") ||
            str_contains_ci(k, "web"))
            return GL_WEB;
        if (str_contains_ci(k, "spotify") || str_contains_ci(k, "rhythmbox") ||
            str_contains_ci(k, "audacious") || str_contains_ci(k, "cmus") ||
            str_contains_ci(k, "ncmpcpp") || str_contains_ci(k, "mpd") ||
            str_contains_ci(k, "clementine") || str_contains_ci(k, "strawberry") ||
            str_contains_ci(k, "elisa") || str_contains_ci(k, "lollypop") ||
            str_contains_ci(k, "music") || str_contains_ci(k, "tidal"))
            return GL_MUSIC;
        if (str_contains_ci(k, "mpv") || str_contains_ci(k, "vlc") ||
            str_contains_ci(k, "celluloid") || str_contains_ci(k, "totem") ||
            str_contains_ci(k, "kodi") || str_contains_ci(k, "obs") ||
            str_contains_ci(k, "haruna") || str_contains_ci(k, "smplayer") ||
            str_contains_ci(k, "video"))
            return GL_VIDEO;
        if (str_contains_ci(k, "discord") || str_contains_ci(k, "vesktop") ||
            str_contains_ci(k, "telegram") || str_contains_ci(k, "slack") ||
            str_contains_ci(k, "signal") || str_contains_ci(k, "element") ||
            str_contains_ci(k, "zoom") || str_contains_ci(k, "teams") ||
            str_contains_ci(k, "skype") || str_contains_ci(k, "mumble") ||
            str_contains_ci(k, "whatsapp") || str_contains_ci(k, "chat"))
            return GL_CHAT;
        if (str_contains_ci(k, "steam") || str_contains_ci(k, "wine") ||
            str_contains_ci(k, "proton") || str_contains_ci(k, "lutris") ||
            str_contains_ci(k, "retroarch") || str_contains_ci(k, "minecraft") ||
            str_contains_ci(k, "heroic") || str_contains_ci(k, "game"))
            return GL_GAMEPAD;
        if (str_contains_ci(k, "kitty") || str_contains_ci(k, "alacritty") ||
            str_contains_ci(k, "foot") || str_contains_ci(k, "wezterm") ||
            str_contains_ci(k, "xterm") || str_contains_ci(k, "term") ||
            str_contains_ci(k, "pacat") || str_contains_ci(k, "paplay") ||
            str_contains_ci(k, "pw-"))
            return GL_TERMINAL;
    }
    return GL_MUSIC_BOX;
}

static gboolean should_ignore_stream(const pa_sink_input_info *info) {
    if (!info) return TRUE;
    if (!info->has_volume || !info->volume_writable) return TRUE;
    if (info->proplist) {
        const char *role = PA.proplist_gets(info->proplist, "media.role");
        if (role && strcmp(role, "event") == 0) return TRUE;
        const char *aid = PA.proplist_gets(info->proplist, "application.id");
        if (aid && (strcmp(aid, "org.PulseAudio.pavucontrol") == 0 ||
                    strcmp(aid, "org.freedesktop.pavucontrol") == 0))
            return TRUE;
    }
    return FALSE;
}

/* Collapse multi-tab / multi-stream apps (e.g. Brave tabs + WebRTC) under one
 * clean app key & display name so one application always gets one slider. */
static void resolve_app_identity(const pa_sink_input_info *info,
                                 char **out_key, char **out_name, const char **out_glyph) {
    const char *app  = info->proplist ? PA.proplist_gets(info->proplist, "application.name") : NULL;
    const char *node = info->proplist ? PA.proplist_gets(info->proplist, "node.name") : NULL;
    const char *bin  = info->proplist ? PA.proplist_gets(info->proplist, "application.process.binary") : NULL;
    const char *icon = info->proplist ? PA.proplist_gets(info->proplist, "application.icon_name") : NULL;

    /* Canonical browser identities: Chromium-based browsers can expose WebRTC
     * or secondary tab streams under "Chromium"/"WEBRTC VoiceEngine" even when
     * the process binary is brave/vivaldi/edge/opera. */
    static const struct { const char *bin_pat; const char *key; const char *name; } browsers[] = {
        { "brave",     "brave",     "Brave" },
        { "firefox",   "firefox",   "Firefox" },
        { "librewolf", "librewolf", "LibreWolf" },
        { "vivaldi",   "vivaldi",   "Vivaldi" },
        { "zen",       "zen",       "Zen" },
        { "chromium",  "chromium",  "Chromium" },
        { "chrome",    "chrome",    "Chrome" },
        { "opera",     "opera",     "Opera" },
        { "msedge",    "edge",      "Edge" },
        { NULL, NULL, NULL }
    };
    if (bin && *bin) {
        for (int i = 0; browsers[i].bin_pat; i++) {
            if (str_contains_ci(bin, browsers[i].bin_pat)) {
                *out_key   = g_strdup(browsers[i].key);
                *out_name  = g_strdup(browsers[i].name);
                *out_glyph = GL_WEB;
                return;
            }
        }
    }

    char *disp = NULL;
    if (app && *app) disp = g_strdup(app);
    else if (node && *node) disp = g_strdup(node);
    else if (bin && *bin) {
        disp = g_strdup(bin);
        disp[0] = (char)toupper((unsigned char)disp[0]);
    } else if (info->name && *info->name) disp = g_strdup(info->name);
    else disp = g_strdup("Application");

    *out_name  = disp;
    *out_key   = g_ascii_strdown(disp, -1);
    *out_glyph = pick_app_glyph(disp, bin, icon);
}

/* -- placement (zero-fork BSPWM socket IPC) -------------------------------- */
static void anchor(gboolean pointer) {
    bar_anchor(pointer, &S.mon, S.edge, &S.off);
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
    } else if (strcmp(S.side, "left") == 0) {
        x = S.mon.x + 8;
    } else {
        x = S.mon.x + S.mon.w - WIDTH - 8;
    }
    x = MAX(S.mon.x + 8, MIN(x, S.mon.x + S.mon.w - WIDTH - 8));
    int y = strcmp(S.edge, "top") == 0 ? S.mon.y + S.off + GAP : S.mon.y + S.mon.h - S.off - GAP - h;
    int cx, cy;
    gtk_window_get_position(GTK_WINDOW(S.win), &cx, &cy);
    if (cx != x || cy != y) gtk_window_move(GTK_WINDOW(S.win), x, y);
}

static void fit(void) {
    GtkRequisition nat;
    gtk_widget_get_preferred_size(S.box, NULL, &nat);
    int max_h = MAX(240, S.mon.h - S.off - GAP * 3);
    int h = CLAMP(nat.height, 1, max_h);
    gtk_window_resize(GTK_WINDOW(S.win), WIDTH, h);
    place(h);
}

static gboolean fit_idle_cb(gpointer _d) {
    S.fit_idle_id = 0;
    if (!S.closing) fit();
    return G_SOURCE_REMOVE;
}

static void schedule_fit(void) {
    if (S.closing || !S.shown) return;
    if (!S.fit_idle_id) S.fit_idle_id = g_idle_add(fit_idle_cb, NULL);
}

static gboolean fit_tick_cb(GtkWidget *_w, GdkFrameClock *clock, gpointer _d) {
    if (S.closing) { S.fit_tick_id = 0; return G_SOURCE_REMOVE; }
    fit();
    if (gdk_frame_clock_get_frame_time(clock) < S.fit_until_us) return G_SOURCE_CONTINUE;
    S.fit_tick_id = 0;
    return G_SOURCE_REMOVE;
}

static void fit_follow(int ms) {
    if (S.closing || !S.shown) return;
    S.fit_until_us = MAX(S.fit_until_us, g_get_monotonic_time() + (gint64)ms * 1000);
    if (!S.fit_tick_id) S.fit_tick_id = gtk_widget_add_tick_callback(S.box, fit_tick_cb, NULL, NULL);
    fit();
}

/* -- Master UI updates (dirty-checked) ------------------------------------- */
static const char *master_glyph(const MasterCtrl *m) {
    if (m->is_mic) return m->muted ? GL_MIC_MUTE : GL_MIC;
    if (m->muted) return GL_VOL_MUTE;
    if (m->value > 66) return GL_VOL_HIGH;
    if (m->value > 33) return GL_VOL_MED;
    return GL_VOL_LOW;
}

static void update_master_ui(MasterCtrl *m, gboolean sync_scale) {
    const char *gl = master_glyph(m);
    if (!m->ui_inited || m->last_glyph != gl) {
        gtk_button_set_label(GTK_BUTTON(m->icon_btn), gl);
        m->last_glyph = gl;
    }

    char title_buf[64];
    if (m->muted) g_snprintf(title_buf, sizeof title_buf, "%s \xc2\xb7 Muted", m->base_name);
    else g_strlcpy(title_buf, m->base_name, sizeof title_buf);
    if (!m->ui_inited || strcmp(m->last_title, title_buf) != 0) {
        gtk_label_set_text(GTK_LABEL(m->name_lbl), title_buf);
        g_strlcpy(m->last_title, title_buf, sizeof m->last_title);
    }

    if (!m->ui_inited || m->last_shown_pct != m->value) {
        char pct_buf[16];
        g_snprintf(pct_buf, sizeof pct_buf, "%d%%", m->value);
        gtk_label_set_text(GTK_LABEL(m->num_lbl), pct_buf);
        m->last_shown_pct = m->value;
    }

    if (!m->ui_inited || m->last_muted != m->muted) {
        set_class(m->block, "muted", m->muted);
        m->last_muted = m->muted;
    }

    gboolean is_zero = m->value <= 0;
    if (!m->ui_inited || m->last_zero != is_zero) {
        set_class(m->scale, "zero", is_zero);
        m->last_zero = is_zero;
    }

    if (sync_scale && !m->dragging) {
        double target = (double)CLAMP(m->value, 0, 100);
        if (!m->ui_inited || fabs(gtk_range_get_value(GTK_RANGE(m->scale)) - target) >= 0.5) {
            S.syncing = TRUE;
            gtk_range_set_value(GTK_RANGE(m->scale), target);
            S.syncing = FALSE;
        }
    }
    m->ui_inited = TRUE;
}

/* -- AppRow UI updates (dirty-checked, updates inline & page row if present) */
static void update_app_row_ui(AppRowUI *ui, gboolean sync_scale) {
    if (!ui || !ui->grp) return;
    AppGroup *g = ui->grp;

    const char *gl = g->muted ? GL_VOL_MUTE : (g->glyph ? g->glyph : GL_MUSIC_BOX);
    if (!ui->ui_inited || ui->last_glyph != gl) {
        gtk_button_set_label(GTK_BUTTON(ui->icon_btn), gl);
        char tip[96];
        g_snprintf(tip, sizeof tip, "%s %s", g->muted ? "Unmute" : "Mute", g->display_name);
        gtk_widget_set_tooltip_text(ui->icon_btn, tip);
        ui->last_glyph = gl;
    }

    if (!ui->ui_inited || strcmp(ui->last_name, g->display_name) != 0) {
        gtk_label_set_text(GTK_LABEL(ui->name_lbl), g->display_name);
        gtk_widget_set_tooltip_text(ui->name_lbl, g->display_name);
        g_strlcpy(ui->last_name, g->display_name, sizeof ui->last_name);
    }

    if (!ui->ui_inited || ui->last_shown_pct != g->value) {
        char pct_buf[16];
        g_snprintf(pct_buf, sizeof pct_buf, "%d%%", g->value);
        gtk_label_set_text(GTK_LABEL(ui->num_lbl), pct_buf);
        ui->last_shown_pct = g->value;
    }

    if (!ui->ui_inited || ui->last_muted != g->muted) {
        set_class(ui->row, "muted", g->muted);
        ui->last_muted = g->muted;
    }

    gboolean is_zero = g->value <= 0;
    if (!ui->ui_inited || ui->last_zero != is_zero) {
        set_class(ui->scale, "zero", is_zero);
        ui->last_zero = is_zero;
    }

    if (sync_scale && !g->dragging) {
        double target = (double)CLAMP(g->value, 0, 100);
        if (!ui->ui_inited || fabs(gtk_range_get_value(GTK_RANGE(ui->scale)) - target) >= 0.5) {
            S.syncing = TRUE;
            gtk_range_set_value(GTK_RANGE(ui->scale), target);
            S.syncing = FALSE;
        }
    }
    ui->ui_inited = TRUE;
}

static void update_group_ui(AppGroup *g, gboolean sync_scale, AppRowUI *skip_scale_ui) {
    if (g->inline_ui) update_app_row_ui(g->inline_ui, sync_scale && g->inline_ui != skip_scale_ui);
    if (g->page_ui)   update_app_row_ui(g->page_ui,   sync_scale && g->page_ui != skip_scale_ui);
}

/* -- PulseAudio zero-fork commands (master & per-app batch) ---------------- */
static void send_master_volume(MasterCtrl *m, int pct, gboolean unmute) {
    if (!S.pa_ready || !S.pa_ctx || m->pa_idx == PA_INVALID_INDEX) return;
    cvolume_apply_pct(&m->raw_vol, m->channels ? m->channels : 2, pct);
    if (m->is_mic) {
        if (unmute)
            op_unref(PA.context_set_source_mute_by_index(S.pa_ctx, m->pa_idx, 0, NULL, NULL));
        op_unref(PA.context_set_source_volume_by_index(S.pa_ctx, m->pa_idx, &m->raw_vol, NULL, NULL));
    } else {
        if (unmute)
            op_unref(PA.context_set_sink_mute_by_index(S.pa_ctx, m->pa_idx, 0, NULL, NULL));
        op_unref(PA.context_set_sink_volume_by_index(S.pa_ctx, m->pa_idx, &m->raw_vol, NULL, NULL));
    }
}

static void send_master_mute(MasterCtrl *m, gboolean muted) {
    if (!S.pa_ready || !S.pa_ctx || m->pa_idx == PA_INVALID_INDEX) return;
    int v = muted ? 1 : 0;
    if (m->is_mic)
        op_unref(PA.context_set_source_mute_by_index(S.pa_ctx, m->pa_idx, v, NULL, NULL));
    else
        op_unref(PA.context_set_sink_mute_by_index(S.pa_ctx, m->pa_idx, v, NULL, NULL));
}

static gboolean master_cooldown_cb(gpointer data) {
    MasterCtrl *m = data;
    if (S.closing) { m->send_timer = 0; return G_SOURCE_REMOVE; }
    if (m->has_pending) {
        int pct = m->pending_val;
        gboolean unm = m->pending_unmute;
        m->has_pending = FALSE;
        m->pending_unmute = FALSE;
        send_master_volume(m, pct, unm);
        return G_SOURCE_CONTINUE;
    }
    m->send_timer = 0;
    return G_SOURCE_REMOVE;
}

static void schedule_master_send(MasterCtrl *m, int pct, gboolean was_muted) {
    m->suppress_until_us = g_get_monotonic_time() + ECHO_SUPPRESS_US;
    if (!m->send_timer) {
        send_master_volume(m, pct, was_muted);
        m->send_timer = g_timeout_add(SEND_COALESCE_MS, master_cooldown_cb, m);
    } else {
        m->pending_val = pct;
        m->has_pending = TRUE;
        if (was_muted) m->pending_unmute = TRUE;
    }
}

/* Dispatch volume to EVERY sink-input belonging to AppGroup in one batch. */
static void send_group_volume(AppGroup *g, int pct, gboolean unmute) {
    if (!S.pa_ready || !S.pa_ctx || !S.raw_streams) return;
    for (guint i = 0; i < S.raw_streams->len; i++) {
        RawStream *rs = g_ptr_array_index(S.raw_streams, i);
        if (strcmp(rs->app_key, g->app_key) != 0) continue;
        cvolume_apply_pct(&rs->raw_vol, rs->channels ? rs->channels : 2, pct);
        rs->value = pct;
        if (unmute || rs->muted) {
            rs->muted = FALSE;
            op_unref(PA.context_set_sink_input_mute(S.pa_ctx, rs->pa_idx, 0, NULL, NULL));
        }
        op_unref(PA.context_set_sink_input_volume(S.pa_ctx, rs->pa_idx, &rs->raw_vol, NULL, NULL));
    }
}

static void send_group_mute(AppGroup *g, gboolean muted) {
    if (!S.pa_ready || !S.pa_ctx || !S.raw_streams) return;
    int v = muted ? 1 : 0;
    for (guint i = 0; i < S.raw_streams->len; i++) {
        RawStream *rs = g_ptr_array_index(S.raw_streams, i);
        if (strcmp(rs->app_key, g->app_key) != 0) continue;
        rs->muted = muted;
        op_unref(PA.context_set_sink_input_mute(S.pa_ctx, rs->pa_idx, v, NULL, NULL));
    }
}

static gboolean group_cooldown_cb(gpointer data) {
    AppGroup *g = data;
    if (S.closing) { g->send_timer = 0; return G_SOURCE_REMOVE; }
    if (g->has_pending) {
        int pct = g->pending_val;
        gboolean unm = g->pending_unmute;
        g->has_pending = FALSE;
        g->pending_unmute = FALSE;
        send_group_volume(g, pct, unm);
        return G_SOURCE_CONTINUE;
    }
    g->send_timer = 0;
    return G_SOURCE_REMOVE;
}

static void schedule_group_send(AppGroup *g, int pct, gboolean was_muted) {
    g->suppress_until_us = g_get_monotonic_time() + ECHO_SUPPRESS_US;
    if (!g->send_timer) {
        send_group_volume(g, pct, was_muted);
        g->send_timer = g_timeout_add(SEND_COALESCE_MS, group_cooldown_cb, g);
    } else {
        g->pending_val = pct;
        g->has_pending = TRUE;
        if (was_muted) g->pending_unmute = TRUE;
    }
}

/* -- Master control signal handlers ---------------------------------------- */
static void nudge_master(MasterCtrl *m, int delta) {
    int next = CLAMP(CLAMP(m->value, 0, 100) + delta, 0, 100);
    gboolean was_muted = m->muted;
    if (next == m->value && !was_muted) return;
    m->value = next;
    m->muted = FALSE;
    update_master_ui(m, TRUE);
    schedule_master_send(m, next, was_muted);
}

static void on_master_scale_changed(GtkRange *range, gpointer data) {
    if (S.syncing || S.closing) return;
    MasterCtrl *m = data;
    int pct = CLAMP((int)lround(gtk_range_get_value(range)), 0, 100);
    if (pct == m->value && !m->muted) return;
    gboolean was_muted = m->muted;
    m->value = pct;
    m->muted = FALSE;
    update_master_ui(m, FALSE);
    schedule_master_send(m, pct, was_muted);
}

static gboolean on_master_press(GtkWidget *_w, GdkEventButton *ev, gpointer data) {
    if (ev->button == 1) {
        MasterCtrl *m = data;
        m->dragging = TRUE;
        m->suppress_until_us = g_get_monotonic_time() + 1000000;
        if (S.kb_active) { S.kb_active = FALSE; update_focus_visuals(); }
    }
    return FALSE;
}

static gboolean on_master_release(GtkWidget *_w, GdkEventButton *_ev, gpointer data) {
    MasterCtrl *m = data;
    if (m->dragging) {
        m->dragging = FALSE;
        m->suppress_until_us = g_get_monotonic_time() + ECHO_SUPPRESS_US;
        if (m->send_timer) { g_source_remove(m->send_timer); m->send_timer = 0; }
        if (m->has_pending) {
            int pct = m->pending_val;
            gboolean unm = m->pending_unmute;
            m->has_pending = FALSE;
            m->pending_unmute = FALSE;
            send_master_volume(m, pct, unm);
        }
    }
    return FALSE;
}

static gboolean on_master_scroll(GtkWidget *_w, GdkEventScroll *ev, gpointer data) {
    MasterCtrl *m = data;
    int dir = 0;
    if (ev->direction == GDK_SCROLL_UP) dir = 1;
    else if (ev->direction == GDK_SCROLL_DOWN) dir = -1;
    else if (ev->direction == GDK_SCROLL_SMOOTH) {
        if (ev->delta_y < -0.05) dir = 1;
        else if (ev->delta_y > 0.05) dir = -1;
    }
    if (dir != 0) nudge_master(m, dir * ((ev->state & GDK_SHIFT_MASK) ? 1 : 5));
    return TRUE;
}

static void on_master_mute(GtkButton *_b, gpointer data) {
    MasterCtrl *m = data;
    m->muted = !m->muted;
    m->suppress_until_us = g_get_monotonic_time() + ECHO_SUPPRESS_US;
    update_master_ui(m, FALSE);
    send_master_mute(m, m->muted);
}

/* -- AppRow signal handlers ------------------------------------------------ */
static void nudge_group(AppGroup *g, int delta) {
    if (!g) return;
    int next = CLAMP(CLAMP(g->value, 0, 100) + delta, 0, 100);
    gboolean was_muted = g->muted;
    if (next == g->value && !was_muted) return;
    g->value = next;
    g->muted = FALSE;
    update_group_ui(g, TRUE, NULL);
    schedule_group_send(g, next, was_muted);
}

static void on_app_scale_changed(GtkRange *range, gpointer data) {
    if (S.syncing || S.closing) return;
    AppRowUI *ui = data;
    AppGroup *g = ui->grp;
    int pct = CLAMP((int)lround(gtk_range_get_value(range)), 0, 100);
    if (pct == g->value && !g->muted) return;
    gboolean was_muted = g->muted;
    g->value = pct;
    g->muted = FALSE;
    update_group_ui(g, TRUE, ui);
    schedule_group_send(g, pct, was_muted);
}

static gboolean on_app_press(GtkWidget *_w, GdkEventButton *ev, gpointer data) {
    if (ev->button == 1) {
        AppRowUI *ui = data;
        ui->grp->dragging = TRUE;
        ui->grp->suppress_until_us = g_get_monotonic_time() + 1000000;
        if (S.kb_active) { S.kb_active = FALSE; update_focus_visuals(); }
    }
    return FALSE;
}

static gboolean on_app_release(GtkWidget *_w, GdkEventButton *_ev, gpointer data) {
    AppRowUI *ui = data;
    AppGroup *g = ui->grp;
    if (g->dragging) {
        g->dragging = FALSE;
        g->suppress_until_us = g_get_monotonic_time() + ECHO_SUPPRESS_US;
        if (g->send_timer) { g_source_remove(g->send_timer); g->send_timer = 0; }
        if (g->has_pending) {
            int pct = g->pending_val;
            gboolean unm = g->pending_unmute;
            g->has_pending = FALSE;
            g->pending_unmute = FALSE;
            send_group_volume(g, pct, unm);
        }
    }
    return FALSE;
}

static gboolean on_app_scroll(GtkWidget *_w, GdkEventScroll *ev, gpointer data) {
    AppRowUI *ui = data;
    /* On the scrollable All Apps page, let wheel scroll the list when overflow
     * exists unless Shift is held. On the non-scrollable main view, wheel
     * adjusts the app's volume directly. */
    if (ui->in_scroll && !(ev->state & GDK_SHIFT_MASK) && S.all_apps_scroll) {
        GtkAdjustment *va = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(S.all_apps_scroll));
        if (va && gtk_adjustment_get_upper(va) > gtk_adjustment_get_page_size(va) + 2.0)
            return FALSE;
    }
    int dir = 0;
    if (ev->direction == GDK_SCROLL_UP) dir = 1;
    else if (ev->direction == GDK_SCROLL_DOWN) dir = -1;
    else if (ev->direction == GDK_SCROLL_SMOOTH) {
        if (ev->delta_y < -0.05) dir = 1;
        else if (ev->delta_y > 0.05) dir = -1;
    }
    if (dir != 0) nudge_group(ui->grp, dir * ((ev->state & GDK_SHIFT_MASK) ? 1 : 5));
    return TRUE;
}

static void on_app_mute(GtkButton *_b, gpointer data) {
    AppGroup *g = data;
    g->muted = !g->muted;
    g->suppress_until_us = g_get_monotonic_time() + ECHO_SUPPRESS_US;
    update_group_ui(g, FALSE, NULL);
    send_group_mute(g, g->muted);
}

static void on_gear_click(GtkButton *_b, gpointer _d) {
    panel_close();
    const char *argv[] = {"pavucontrol", NULL};
    spawn_fire(argv);
}

/* -- Widget builders ------------------------------------------------------- */
static void init_master_widgets(MasterCtrl *m, const char *name, const char *tip, gboolean is_mic) {
    m->is_mic = is_mic;
    m->pa_idx = PA_INVALID_INDEX;
    m->channels = 2;
    m->base_name = name;
    m->last_shown_pct = -1;

    m->block = gtk_box_new(GTK_ORIENTATION_VERTICAL, 8);
    add_class(m->block, "vol-block");

    GtkWidget *top = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    m->icon_btn = gtk_button_new_with_label(master_glyph(m));
    add_class(m->icon_btn, "vol-icon");
    gtk_widget_set_can_focus(m->icon_btn, FALSE);
    gtk_widget_set_valign(m->icon_btn, GTK_ALIGN_CENTER);
    gtk_widget_set_tooltip_text(m->icon_btn, tip);
    make_clickable(m->icon_btn);
    g_signal_connect(m->icon_btn, "clicked", G_CALLBACK(on_master_mute), m);

    m->name_lbl = gtk_label_new(name);
    add_class(m->name_lbl, "vol-name");
    gtk_label_set_xalign(GTK_LABEL(m->name_lbl), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(m->name_lbl), PANGO_ELLIPSIZE_END);
    gtk_widget_set_valign(m->name_lbl, GTK_ALIGN_CENTER);
    gtk_widget_set_hexpand(m->name_lbl, TRUE);

    m->num_lbl = gtk_label_new("0%");
    add_class(m->num_lbl, "vol-num");
    gtk_label_set_xalign(GTK_LABEL(m->num_lbl), 1.0f);
    gtk_label_set_width_chars(GTK_LABEL(m->num_lbl), 4);
    gtk_widget_set_valign(m->num_lbl, GTK_ALIGN_CENTER);

    gtk_box_pack_start(GTK_BOX(top), m->icon_btn, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(top), m->name_lbl, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(top), m->num_lbl, FALSE, FALSE, 0);

    m->scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 100.0, 1.0);
    add_class(m->scale, "vol-pill");
    gtk_scale_set_draw_value(GTK_SCALE(m->scale), FALSE);
    gtk_widget_set_can_focus(m->scale, FALSE);
    gtk_widget_set_hexpand(m->scale, TRUE);
    gtk_widget_add_events(m->scale, GDK_SCROLL_MASK | GDK_SMOOTH_SCROLL_MASK);
    make_clickable(m->scale);
    g_signal_connect(m->scale, "value-changed", G_CALLBACK(on_master_scale_changed), m);
    g_signal_connect(m->scale, "button-press-event", G_CALLBACK(on_master_press), m);
    g_signal_connect(m->scale, "button-release-event", G_CALLBACK(on_master_release), m);
    g_signal_connect(m->scale, "scroll-event", G_CALLBACK(on_master_scroll), m);

    gtk_box_pack_start(GTK_BOX(m->block), top, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(m->block), m->scale, FALSE, FALSE, 0);
}

/* Single-line, minimal app row: [icon] [App] [slim 8 px slider] [80%] */
static AppRowUI *create_app_row_ui(AppGroup *g, gboolean in_scroll) {
    AppRowUI *ui = g_new0(AppRowUI, 1);
    ui->grp = g;
    ui->in_scroll = in_scroll;
    ui->last_shown_pct = -1;

    ui->row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    add_class(ui->row, "app-row");

    ui->icon_btn = gtk_button_new_with_label(g->glyph ? g->glyph : GL_MUSIC_BOX);
    add_class(ui->icon_btn, "app-icon");
    gtk_widget_set_can_focus(ui->icon_btn, FALSE);
    gtk_widget_set_valign(ui->icon_btn, GTK_ALIGN_CENTER);
    make_clickable(ui->icon_btn);
    g_signal_connect(ui->icon_btn, "clicked", G_CALLBACK(on_app_mute), g);

    ui->name_lbl = gtk_label_new(g->display_name);
    add_class(ui->name_lbl, "app-name");
    gtk_label_set_xalign(GTK_LABEL(ui->name_lbl), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(ui->name_lbl), PANGO_ELLIPSIZE_END);
    gtk_label_set_width_chars(GTK_LABEL(ui->name_lbl), 9);
    gtk_label_set_max_width_chars(GTK_LABEL(ui->name_lbl), 9);
    gtk_widget_set_valign(ui->name_lbl, GTK_ALIGN_CENTER);

    ui->scale = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 100.0, 1.0);
    add_class(ui->scale, "app-pill");
    gtk_scale_set_draw_value(GTK_SCALE(ui->scale), FALSE);
    gtk_widget_set_can_focus(ui->scale, FALSE);
    gtk_widget_set_hexpand(ui->scale, TRUE);
    gtk_widget_set_valign(ui->scale, GTK_ALIGN_CENTER);
    gtk_widget_add_events(ui->scale, GDK_SCROLL_MASK | GDK_SMOOTH_SCROLL_MASK);
    make_clickable(ui->scale);
    g_signal_connect(ui->scale, "value-changed", G_CALLBACK(on_app_scale_changed), ui);
    g_signal_connect(ui->scale, "button-press-event", G_CALLBACK(on_app_press), ui);
    g_signal_connect(ui->scale, "button-release-event", G_CALLBACK(on_app_release), ui);
    g_signal_connect(ui->scale, "scroll-event", G_CALLBACK(on_app_scroll), ui);

    ui->num_lbl = gtk_label_new("0%");
    add_class(ui->num_lbl, "app-num");
    gtk_label_set_xalign(GTK_LABEL(ui->num_lbl), 1.0f);
    gtk_label_set_width_chars(GTK_LABEL(ui->num_lbl), 4);
    gtk_widget_set_valign(ui->num_lbl, GTK_ALIGN_CENTER);

    gtk_box_pack_start(GTK_BOX(ui->row), ui->icon_btn, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(ui->row), ui->name_lbl, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(ui->row), ui->scale, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(ui->row), ui->num_lbl, FALSE, FALSE, 0);

    update_app_row_ui(ui, TRUE);
    gtk_widget_show_all(ui->row);
    return ui;
}

static void destroy_app_row_ui(AppRowUI **ui_ptr) {
    if (!ui_ptr || !*ui_ptr) return;
    AppRowUI *ui = *ui_ptr;
    *ui_ptr = NULL;
    if (ui->row) gtk_widget_destroy(ui->row);
    g_free(ui);
}

static void free_raw_stream(gpointer data) {
    RawStream *rs = data;
    if (!rs) return;
    g_free(rs->app_key);
    g_free(rs->display_name);
    g_free(rs);
}

static void free_app_group(AppGroup *g) {
    if (!g) return;
    if (g->send_timer) { g_source_remove(g->send_timer); g->send_timer = 0; }
    destroy_app_row_ui(&g->inline_ui);
    destroy_app_row_ui(&g->page_ui);
    g_free(g->app_key);
    g_free(g->display_name);
    g_free(g);
}

static void free_sink_dev(gpointer data) {
    SinkDev *d = data;
    if (!d) return;
    g_free(d->name);
    g_free(d->desc);
    g_free(d);
}

/* -- Per-app aggregation & incremental UI reconciliation ------------------- */
static RawStream *find_raw_stream(uint32_t pa_idx, guint *pos_out) {
    for (guint i = 0; i < S.raw_streams->len; i++) {
        RawStream *rs = g_ptr_array_index(S.raw_streams, i);
        if (rs->pa_idx == pa_idx) {
            if (pos_out) *pos_out = i;
            return rs;
        }
    }
    return NULL;
}

static AppGroup *find_group(const char *app_key, guint *pos_out) {
    for (guint i = 0; i < S.groups->len; i++) {
        AppGroup *g = g_ptr_array_index(S.groups, i);
        if (strcmp(g->app_key, app_key) == 0) {
            if (pos_out) *pos_out = i;
            return g;
        }
    }
    return NULL;
}

static gint cmp_groups(gconstpointer a, gconstpointer b) {
    const AppGroup *ga = *(const AppGroup *const *)a;
    const AppGroup *gb = *(const AppGroup *const *)b;
    /* Active (uncorked) apps come before paused apps when >INLINE_APPS_MAX so
     * the 3 inline slots always surface what is actually playing. */
    if (S.groups->len > INLINE_APPS_MAX && ga->all_corked != gb->all_corked)
        return ga->all_corked ? 1 : -1;
    return g_ascii_strcasecmp(ga->display_name, gb->display_name);
}

static void reconcile_apps_ui(void) {
    gint64 now = g_get_monotonic_time();
    gboolean any_dragging = FALSE;

    /* 1. Reset stream counts & aggregate raw streams into per-app groups. */
    for (guint i = 0; i < S.groups->len; i++) {
        AppGroup *g = g_ptr_array_index(S.groups, i);
        g->stream_count = 0;
        if (g->dragging) any_dragging = TRUE;
    }

    for (guint i = 0; i < S.raw_streams->len; i++) {
        RawStream *rs = g_ptr_array_index(S.raw_streams, i);
        AppGroup *g = find_group(rs->app_key, NULL);
        if (!g) {
            g = g_new0(AppGroup, 1);
            g->app_key = g_strdup(rs->app_key);
            g->display_name = g_strdup(rs->display_name);
            g->glyph = rs->glyph;
            g->value = rs->value;
            g->muted = rs->muted;
            g->all_corked = rs->corked;
            g_ptr_array_add(S.groups, g);
        }
        g->stream_count++;
    }

    /* 2. Remove groups with 0 remaining streams. */
    gboolean layout_changed = FALSE;
    for (gint i = (gint)S.groups->len - 1; i >= 0; i--) {
        AppGroup *g = g_ptr_array_index(S.groups, (guint)i);
        if (g->stream_count == 0) {
            g_ptr_array_remove_index(S.groups, (guint)i);
            free_app_group(g);
            layout_changed = TRUE;
        }
    }

    /* 3. Compute aggregated volume / mute / corked per group. */
    for (guint i = 0; i < S.groups->len; i++) {
        AppGroup *g = g_ptr_array_index(S.groups, i);
        gboolean all_muted = TRUE;
        gboolean all_corked = TRUE;
        long sum_active = 0, cnt_active = 0;
        long sum_unmuted = 0, cnt_unmuted = 0;
        long sum_all = 0, cnt_all = 0;
        const char *best_name = g->display_name;
        const char *best_glyph = g->glyph;

        for (guint j = 0; j < S.raw_streams->len; j++) {
            RawStream *rs = g_ptr_array_index(S.raw_streams, j);
            if (strcmp(rs->app_key, g->app_key) != 0) continue;
            if (!rs->muted) all_muted = FALSE;
            if (!rs->corked) all_corked = FALSE;
            if (!rs->muted && !rs->corked) { sum_active += rs->value; cnt_active++; }
            if (!rs->muted) { sum_unmuted += rs->value; cnt_unmuted++; }
            sum_all += rs->value;
            cnt_all++;
            if (rs->display_name && *rs->display_name) best_name = rs->display_name;
            if (rs->glyph) best_glyph = rs->glyph;
        }

        if (strcmp(g->display_name, best_name) != 0) {
            g_free(g->display_name);
            g->display_name = g_strdup(best_name);
        }
        g->glyph = best_glyph;
        g->all_corked = all_corked;

        if (!g->dragging && now >= g->suppress_until_us) {
            g->muted = all_muted;
            if (cnt_active > 0)       g->value = (int)llround((double)sum_active / cnt_active);
            else if (cnt_unmuted > 0) g->value = (int)llround((double)sum_unmuted / cnt_unmuted);
            else if (cnt_all > 0)     g->value = (int)llround((double)sum_all / cnt_all);
        }
    }

    /* 4. Stable sort (frozen while dragging so a row never jumps under cursor). */
    if (!any_dragging && S.groups->len > 1) {
        g_ptr_array_sort(S.groups, cmp_groups);
    }

    /* 5. Reconcile inline main-page widgets (capped at INLINE_APPS_MAX). */
    guint n = S.groups->len;
    gboolean was_vis = gtk_widget_get_visible(S.apps_sec);
    gtk_widget_set_visible(S.apps_sec, n > 0);
    if (was_vis != (n > 0)) layout_changed = TRUE;

    for (guint i = 0; i < n; i++) {
        AppGroup *g = g_ptr_array_index(S.groups, i);
        if (i < INLINE_APPS_MAX) {
            if (!g->inline_ui) {
                g->inline_ui = create_app_row_ui(g, FALSE);
                gtk_box_pack_start(GTK_BOX(S.apps_inline_box), g->inline_ui->row, FALSE, FALSE, 0);
                layout_changed = TRUE;
            }
            gtk_box_reorder_child(GTK_BOX(S.apps_inline_box), g->inline_ui->row, (gint)i);
            update_app_row_ui(g->inline_ui, TRUE);
        } else if (g->inline_ui) {
            destroy_app_row_ui(&g->inline_ui);
            layout_changed = TRUE;
        }
    }

    gboolean overflow = n > INLINE_APPS_MAX;
    gboolean had_more = gtk_widget_get_visible(S.apps_more_btn);
    gtk_widget_set_visible(S.apps_more_btn, overflow);
    if (overflow) {
        char buf[32];
        g_snprintf(buf, sizeof buf, "All %u apps", n);
        gtk_label_set_text(GTK_LABEL(S.apps_more_lbl), buf);
    }
    if (had_more != overflow) layout_changed = TRUE;

    /* 6. Reconcile secondary All Apps page ONLY when open (lazy allocation). */
    if (S.on_apps_page && S.apps_page && S.all_apps_box) {
        if (n == 0) {
            show_main_page();
        } else {
            char cbuf[16];
            g_snprintf(cbuf, sizeof cbuf, "%u", n);
            gtk_label_set_text(GTK_LABEL(S.all_apps_count_lbl), cbuf);
            for (guint i = 0; i < n; i++) {
                AppGroup *g = g_ptr_array_index(S.groups, i);
                if (!g->page_ui) {
                    g->page_ui = create_app_row_ui(g, TRUE);
                    gtk_box_pack_start(GTK_BOX(S.all_apps_box), g->page_ui->row, FALSE, FALSE, 0);
                    layout_changed = TRUE;
                }
                gtk_box_reorder_child(GTK_BOX(S.all_apps_box), g->page_ui->row, (gint)i);
                update_app_row_ui(g->page_ui, TRUE);
            }
        }
    }

    update_focus_visuals();
    if (layout_changed) schedule_fit();
}

static void upsert_raw_stream(const pa_sink_input_info *info, gboolean is_full_sync) {
    if (should_ignore_stream(info)) {
        guint pos = 0;
        if (find_raw_stream(info->index, &pos)) {
            g_ptr_array_remove_index(S.raw_streams, pos);
            if (!is_full_sync) reconcile_apps_ui();
        }
        return;
    }

    char *key = NULL, *name = NULL;
    const char *glyph = NULL;
    resolve_app_identity(info, &key, &name, &glyph);

    RawStream *rs = find_raw_stream(info->index, NULL);
    if (!rs) {
        rs = g_new0(RawStream, 1);
        rs->pa_idx = info->index;
        g_ptr_array_add(S.raw_streams, rs);
    }
    g_free(rs->app_key);
    g_free(rs->display_name);
    rs->app_key = key;
    rs->display_name = name;
    rs->glyph = glyph;
    rs->raw_vol = info->volume;
    rs->channels = info->volume.channels ? info->volume.channels : 2;
    rs->value = cvolume_to_pct(&info->volume);
    rs->muted = info->mute != 0;
    rs->corked = info->corked != 0;
    if (is_full_sync) rs->seen_gen = S.stream_gen;

    if (!is_full_sync) reconcile_apps_ui();
}

static void remove_raw_stream_by_idx(uint32_t pa_idx) {
    guint pos = 0;
    if (!find_raw_stream(pa_idx, &pos)) return;
    g_ptr_array_remove_index(S.raw_streams, pos);
    reconcile_apps_ui();
}

/* -- Page switching (Main <-> All Apps, WifiPanel swap_page pattern) ------- */
static void on_more_apps_click(GtkButton *_b, gpointer _d);
static void on_back_click(GtkButton *_b, gpointer _d);

static void destroy_apps_page(void) {
    if (S.apps_page) {
        gtk_widget_destroy(S.apps_page);
        S.apps_page = NULL;
        S.all_apps_count_lbl = NULL;
        S.all_apps_scroll = NULL;
        S.all_apps_box = NULL;
    }
    if (S.groups) {
        for (guint i = 0; i < S.groups->len; i++) {
            AppGroup *g = g_ptr_array_index(S.groups, i);
            if (g->page_ui) {
                g_free(g->page_ui);
                g->page_ui = NULL;
            }
        }
    }
}

static void swap_page(GtkWidget **slot, GtkWidget *page, const char *name) {
    GtkWidget *old = *slot;
    *slot = page;
    gtk_widget_show_all(page);
    gtk_stack_add_named(GTK_STACK(S.stack), page, old ? NULL : name);
    gtk_stack_set_visible_child(GTK_STACK(S.stack), page);
    if (old) gtk_widget_destroy(old);
    fit_follow(PAGE_ANIM_MS + 40);
}

static void show_apps_page(void) {
    if (S.on_apps_page || !S.groups || S.groups->len == 0) return;
    S.on_apps_page = TRUE;
    S.focus_idx = 0;

    /* Clear any stale page_ui pointers before creating a fresh sub-page. */
    for (guint i = 0; i < S.groups->len; i++) {
        AppGroup *g = g_ptr_array_index(S.groups, i);
        if (g->page_ui) { g_free(g->page_ui); g->page_ui = NULL; }
    }

    GtkWidget *page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);

    GtkWidget *phead = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    add_class(phead, "page-head");
    GtkWidget *back = gtk_button_new_with_label(GL_BACK);
    add_class(back, "page-back");
    gtk_widget_set_can_focus(back, FALSE);
    gtk_widget_set_tooltip_text(back, "Back");
    make_clickable(back);
    g_signal_connect(back, "clicked", G_CALLBACK(on_back_click), NULL);

    GtkWidget *ptitle = gtk_label_new("Apps");
    add_class(ptitle, "page-title");
    gtk_label_set_xalign(GTK_LABEL(ptitle), 0.0f);
    gtk_widget_set_hexpand(ptitle, TRUE);

    char cbuf[16];
    g_snprintf(cbuf, sizeof cbuf, "%u", S.groups->len);
    S.all_apps_count_lbl = gtk_label_new(cbuf);
    add_class(S.all_apps_count_lbl, "apps-count");

    gtk_box_pack_start(GTK_BOX(phead), back, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(phead), ptitle, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(phead), S.all_apps_count_lbl, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(page), phead, FALSE, FALSE, 0);

    S.all_apps_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    add_class(S.all_apps_box, "apps-group");
    for (guint i = 0; i < S.groups->len; i++) {
        AppGroup *g = g_ptr_array_index(S.groups, i);
        g->page_ui = create_app_row_ui(g, TRUE);
        gtk_box_pack_start(GTK_BOX(S.all_apps_box), g->page_ui->row, FALSE, FALSE, 0);
    }

    S.all_apps_scroll = gtk_scrolled_window_new(NULL, NULL);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(S.all_apps_scroll),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    gtk_scrolled_window_set_propagate_natural_height(GTK_SCROLLED_WINDOW(S.all_apps_scroll), TRUE);
    gtk_scrolled_window_set_max_content_height(GTK_SCROLLED_WINDOW(S.all_apps_scroll), ALL_APPS_MAX_H);
    gtk_container_add(GTK_CONTAINER(S.all_apps_scroll), S.all_apps_box);
    gtk_box_pack_start(GTK_BOX(page), S.all_apps_scroll, FALSE, FALSE, 0);

    swap_page(&S.apps_page, page, "apps");
    update_focus_visuals();
}

static void on_stack_transition(GObject *_o, GParamSpec *_p, gpointer _d) {
    if (S.closing) return;
    if (gtk_stack_get_transition_running(GTK_STACK(S.stack))) {
        fit_follow(PAGE_ANIM_MS + 40);
    } else {
        if (!S.on_apps_page && S.apps_page) {
            destroy_apps_page();
            heap_trim();
        }
        fit();
    }
}

static void show_main_page(void) {
    if (!S.on_apps_page) return;
    S.on_apps_page = FALSE;
    S.focus_idx = 0;
    gtk_stack_set_visible_child(GTK_STACK(S.stack), S.main_page);
    update_focus_visuals();
    fit_follow(PAGE_ANIM_MS + 40);
}

static void on_more_apps_click(GtkButton *_b, gpointer _d) { show_apps_page(); }
static void on_back_click(GtkButton *_b, gpointer _d)      { show_main_page(); }

/* -- Output device switcher (inline footer drawer when >1 sink) ------------ */
static void on_sink_pick(GtkButton *btn, gpointer _d) {
    const char *name = g_object_get_data(G_OBJECT(btn), "sink-name");
    if (name && *name && S.pa_ready && S.pa_ctx) {
        op_unref(PA.context_set_default_sink(S.pa_ctx, name, NULL, NULL));
    }
    S.sinks_open = FALSE;
    gtk_revealer_set_reveal_child(GTK_REVEALER(S.sinks_rev), FALSE);
    gtk_label_set_text(GTK_LABEL(S.dev_chev_lbl), GL_CHEV_RT);
    fit_follow(180);
}

static void on_footer_click(GtkButton *_b, gpointer _d) {
    if (!S.sinks || S.sinks->len <= 1) return;
    S.sinks_open = !S.sinks_open;
    gtk_revealer_set_reveal_child(GTK_REVEALER(S.sinks_rev), S.sinks_open);
    gtk_label_set_text(GTK_LABEL(S.dev_chev_lbl), S.sinks_open ? GL_CHEV_DN : GL_CHEV_RT);
    fit_follow(180);
}

static void rebuild_sinks_drawer(void) {
    GList *kids = gtk_container_get_children(GTK_CONTAINER(S.sinks_box));
    for (GList *l = kids; l; l = l->next) gtk_widget_destroy(GTK_WIDGET(l->data));
    g_list_free(kids);

    gboolean multi = S.sinks && S.sinks->len > 1;
    set_class(S.foot_btn, "switchable", multi);
    gtk_widget_set_visible(S.dev_chev_lbl, multi);
    GdkWindow *fgw = gtk_widget_get_window(S.foot_btn);
    if (fgw) {
        if (multi && !S.ptr_cursor)
            S.ptr_cursor = gdk_cursor_new_from_name(gdk_window_get_display(fgw), "pointer");
        gdk_window_set_cursor(fgw, multi ? S.ptr_cursor : NULL);
    }
    if (!multi) {
        S.sinks_open = FALSE;
        gtk_revealer_set_reveal_child(GTK_REVEALER(S.sinks_rev), FALSE);
        return;
    }

    gtk_label_set_text(GTK_LABEL(S.dev_chev_lbl), S.sinks_open ? GL_CHEV_DN : GL_CHEV_RT);
    for (guint i = 0; i < S.sinks->len; i++) {
        SinkDev *d = g_ptr_array_index(S.sinks, i);
        gboolean active = S.default_sink_name && strcmp(d->name, S.default_sink_name) == 0;

        GtkWidget *btn = gtk_button_new();
        add_class(btn, "sink-row");
        if (active) add_class(btn, "active");
        gtk_widget_set_can_focus(btn, FALSE);
        make_clickable(btn);

        GtkWidget *row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
        GtkWidget *ic = gtk_label_new(d->is_headphones ? GL_HEADPHONES : GL_SPEAKER);
        add_class(ic, "sink-row-icon");
        GtkWidget *lbl = gtk_label_new(d->desc);
        add_class(lbl, "sink-row-name");
        gtk_label_set_xalign(GTK_LABEL(lbl), 0.0f);
        gtk_label_set_ellipsize(GTK_LABEL(lbl), PANGO_ELLIPSIZE_END);
        gtk_widget_set_hexpand(lbl, TRUE);

        gtk_box_pack_start(GTK_BOX(row), ic, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(row), lbl, TRUE, TRUE, 0);
        if (active) {
            GtkWidget *chk = gtk_label_new(GL_CHECK);
            add_class(chk, "sink-row-check");
            gtk_box_pack_end(GTK_BOX(row), chk, FALSE, FALSE, 0);
        }

        gtk_container_add(GTK_CONTAINER(btn), row);
        g_object_set_data_full(G_OBJECT(btn), "sink-name", g_strdup(d->name), g_free);
        g_signal_connect(btn, "clicked", G_CALLBACK(on_sink_pick), NULL);
        gtk_box_pack_start(GTK_BOX(S.sinks_box), btn, FALSE, FALSE, 0);
    }
    gtk_widget_show_all(S.sinks_box);
}

static gboolean sink_is_headphones(const pa_sink_info *info) {
    if (info->active_port && info->active_port->name) {
        if (str_contains_ci(info->active_port->name, "headphone") ||
            str_contains_ci(info->active_port->name, "headset"))
            return TRUE;
    }
    if (info->proplist) {
        const char *ff = PA.proplist_gets(info->proplist, "device.form_factor");
        if (ff && (str_contains_ci(ff, "headphone") || str_contains_ci(ff, "headset")))
            return TRUE;
    }
    return FALSE;
}

static void apply_default_sink_info(const pa_sink_info *info) {
    S.out.pa_idx = info->index;
    gint64 now = g_get_monotonic_time();
    if (!S.out.dragging && now >= S.out.suppress_until_us) {
        S.out.raw_vol = info->volume;
        S.out.channels = info->volume.channels ? info->volume.channels : 2;
        S.out.value = cvolume_to_pct(&info->volume);
        S.out.muted = info->mute != 0;
        update_master_ui(&S.out, TRUE);
    }
    const char *desc = (info->description && *info->description) ? info->description : info->name;
    gtk_label_set_text(GTK_LABEL(S.dev_lbl), desc && *desc ? desc : "No output device");
    gtk_label_set_text(GTK_LABEL(S.dev_icon_lbl),
                       sink_is_headphones(info) ? GL_HEADPHONES : GL_SPEAKER);
}

/* -- PulseAudio callbacks -------------------------------------------------- */
static void maybe_show_initial(void) {
    if (!S.shown && !S.closing &&
        S.init_sinks_done && S.init_source_done && S.init_streams_done) {
        show_panel_now();
    }
}

static void on_sink_list(pa_context *_c, const pa_sink_info *info, int eol, void *userdata) {
    GPtrArray *batch = userdata;
    if (S.closing) {
        if (eol != 0 && batch) g_ptr_array_free(batch, TRUE);
        return;
    }
    if (eol < 0) {
        if (batch) g_ptr_array_free(batch, TRUE);
        S.init_sinks_done = TRUE;
        maybe_show_initial();
        return;
    }
    if (eol > 0) {
        if (S.sinks) g_ptr_array_free(S.sinks, TRUE);
        S.sinks = batch;
        rebuild_sinks_drawer();
        if (S.sinks->len == 0) gtk_label_set_text(GTK_LABEL(S.dev_lbl), "No output device");
        S.init_sinks_done = TRUE;
        maybe_show_initial();
        schedule_fit();
        return;
    }
    if (!info || !batch) return;

    SinkDev *d = g_new0(SinkDev, 1);
    d->index = info->index;
    d->name = g_strdup(info->name ? info->name : "");
    d->desc = g_strdup(info->description && *info->description ? info->description : d->name);
    d->is_headphones = sink_is_headphones(info);
    g_ptr_array_add(batch, d);

    gboolean is_def = (S.default_sink_name && info->name &&
                       strcmp(info->name, S.default_sink_name) == 0) ||
                      (!S.default_sink_name && batch->len == 1);
    if (is_def) apply_default_sink_info(info);
}

static void fetch_sink_list(pa_context *c) {
    GPtrArray *batch = g_ptr_array_new_with_free_func(free_sink_dev);
    pa_operation *op = PA.context_get_sink_info_list(c, on_sink_list, batch);
    if (op) PA.operation_unref(op);
    else g_ptr_array_free(batch, TRUE);
}

static void on_sink_single(pa_context *_c, const pa_sink_info *info, int eol, void *_u) {
    if (S.closing || eol != 0 || !info) return;
    if (info->index == S.out.pa_idx ||
        (S.default_sink_name && info->name && strcmp(info->name, S.default_sink_name) == 0)) {
        apply_default_sink_info(info);
    }
}

static void on_source_info(pa_context *_c, const pa_source_info *info, int eol, void *_u) {
    if (S.closing) return;
    if (eol != 0) {
        S.init_source_done = TRUE;
        maybe_show_initial();
        return;
    }
    if (!info) return;
    S.mic.pa_idx = info->index;
    gint64 now = g_get_monotonic_time();
    if (!S.mic.dragging && now >= S.mic.suppress_until_us) {
        S.mic.raw_vol = info->volume;
        S.mic.channels = info->volume.channels ? info->volume.channels : 2;
        S.mic.value = cvolume_to_pct(&info->volume);
        S.mic.muted = info->mute != 0;
        update_master_ui(&S.mic, TRUE);
    }
}

static void on_sink_input_list(pa_context *_c, const pa_sink_input_info *info, int eol, void *userdata) {
    if (S.closing) return;
    uint32_t gen = GPOINTER_TO_UINT(userdata);
    if (gen != S.stream_gen) return;
    if (eol < 0) {
        S.init_streams_done = TRUE;
        maybe_show_initial();
        return;
    }
    if (eol > 0) {
        for (gint i = (gint)S.raw_streams->len - 1; i >= 0; i--) {
            RawStream *rs = g_ptr_array_index(S.raw_streams, (guint)i);
            if (rs->seen_gen != gen) g_ptr_array_remove_index(S.raw_streams, (guint)i);
        }
        reconcile_apps_ui();
        S.init_streams_done = TRUE;
        maybe_show_initial();
        return;
    }
    if (info) upsert_raw_stream(info, TRUE);
}

static void on_sink_input_single(pa_context *_c, const pa_sink_input_info *info, int eol, void *_u) {
    if (S.closing || eol != 0 || !info) return;
    upsert_raw_stream(info, FALSE);
}

static void on_server_info(pa_context *c, const pa_server_info *info, void *_u) {
    if (S.closing || !info) return;
    g_free(S.default_sink_name);
    g_free(S.default_source_name);
    S.default_sink_name = g_strdup(info->default_sink_name);
    S.default_source_name = g_strdup(info->default_source_name);

    fetch_sink_list(c);

    if (S.default_source_name && *S.default_source_name) {
        op_unref(PA.context_get_source_info_by_name(c, S.default_source_name, on_source_info, NULL));
    } else {
        S.init_source_done = TRUE;
        maybe_show_initial();
    }

    S.stream_gen++;
    op_unref(PA.context_get_sink_input_info_list(c, on_sink_input_list, GUINT_TO_POINTER(S.stream_gen)));
}

static void refresh_all_state(void) {
    if (!S.pa_ready || !S.pa_ctx || S.closing) return;
    op_unref(PA.context_get_server_info(S.pa_ctx, on_server_info, NULL));
}

static void on_pa_event(pa_context *c, int t, uint32_t idx, void *_u) {
    if (S.closing) return;
    int fac = t & PA_SUBSCRIPTION_EVENT_FACILITY_MASK;
    int ev  = t & PA_SUBSCRIPTION_EVENT_TYPE_MASK;

    if (fac == PA_SUBSCRIPTION_EVENT_SINK_INPUT) {
        if (ev == PA_SUBSCRIPTION_EVENT_REMOVE) {
            remove_raw_stream_by_idx(idx);
        } else {
            op_unref(PA.context_get_sink_input_info(c, idx, on_sink_input_single, NULL));
        }
    } else if (fac == PA_SUBSCRIPTION_EVENT_SINK) {
        if (ev == PA_SUBSCRIPTION_EVENT_CHANGE && idx == S.out.pa_idx) {
            op_unref(PA.context_get_sink_info_by_index(c, idx, on_sink_single, NULL));
        } else if (ev == PA_SUBSCRIPTION_EVENT_NEW || ev == PA_SUBSCRIPTION_EVENT_REMOVE) {
            fetch_sink_list(c);
        }
    } else if (fac == PA_SUBSCRIPTION_EVENT_SOURCE) {
        if (ev == PA_SUBSCRIPTION_EVENT_CHANGE && idx == S.mic.pa_idx) {
            op_unref(PA.context_get_source_info_by_index(c, idx, on_source_info, NULL));
        }
    } else if (fac == PA_SUBSCRIPTION_EVENT_SERVER) {
        refresh_all_state();
    }
}

static void on_pa_state(pa_context *c, void *_u) {
    if (S.closing) return;
    int st = PA.context_get_state(c);
    if (st == PA_CONTEXT_READY) {
        S.pa_ready = TRUE;
        PA.context_set_subscribe_callback(c, on_pa_event, NULL);
        op_unref(PA.context_subscribe(c,
                                      PA_SUBSCRIPTION_MASK_SINK |
                                      PA_SUBSCRIPTION_MASK_SOURCE |
                                      PA_SUBSCRIPTION_MASK_SINK_INPUT |
                                      PA_SUBSCRIPTION_MASK_SERVER,
                                      NULL, NULL));
        refresh_all_state();
    } else if (st == PA_CONTEXT_FAILED || st == PA_CONTEXT_TERMINATED) {
        S.pa_ready = FALSE;
        gtk_label_set_text(GTK_LABEL(S.dev_lbl), "Audio server unavailable");
        if (!S.shown) show_panel_now();
    }
}

static gboolean pa_load_symbols(void) {
    PA.h_pa = dlopen("libpulse.so.0", RTLD_LAZY | RTLD_LOCAL);
    PA.h_glib = dlopen("libpulse-mainloop-glib.so.0", RTLD_LAZY | RTLD_LOCAL);
    if (!PA.h_pa || !PA.h_glib) return FALSE;

#define SYM(h, field, name) do { \
    *(void **)(&PA.field) = dlsym((h), (name)); \
    if (!PA.field) return FALSE; \
} while (0)

    SYM(PA.h_glib, glib_mainloop_new,                  "pa_glib_mainloop_new");
    SYM(PA.h_glib, glib_mainloop_get_api,              "pa_glib_mainloop_get_api");
    SYM(PA.h_glib, glib_mainloop_free,                 "pa_glib_mainloop_free");
    SYM(PA.h_pa,   context_new,                        "pa_context_new");
    SYM(PA.h_pa,   context_unref,                      "pa_context_unref");
    SYM(PA.h_pa,   context_set_state_callback,         "pa_context_set_state_callback");
    SYM(PA.h_pa,   context_connect,                    "pa_context_connect");
    SYM(PA.h_pa,   context_disconnect,                 "pa_context_disconnect");
    SYM(PA.h_pa,   context_get_state,                  "pa_context_get_state");
    SYM(PA.h_pa,   context_set_subscribe_callback,     "pa_context_set_subscribe_callback");
    SYM(PA.h_pa,   context_subscribe,                  "pa_context_subscribe");
    SYM(PA.h_pa,   context_get_server_info,            "pa_context_get_server_info");
    SYM(PA.h_pa,   context_get_sink_info_by_index,     "pa_context_get_sink_info_by_index");
    SYM(PA.h_pa,   context_get_sink_info_list,         "pa_context_get_sink_info_list");
    SYM(PA.h_pa,   context_get_source_info_by_name,    "pa_context_get_source_info_by_name");
    SYM(PA.h_pa,   context_get_source_info_by_index,   "pa_context_get_source_info_by_index");
    SYM(PA.h_pa,   context_get_sink_input_info,        "pa_context_get_sink_input_info");
    SYM(PA.h_pa,   context_get_sink_input_info_list,   "pa_context_get_sink_input_info_list");
    SYM(PA.h_pa,   context_set_sink_volume_by_index,   "pa_context_set_sink_volume_by_index");
    SYM(PA.h_pa,   context_set_sink_mute_by_index,     "pa_context_set_sink_mute_by_index");
    SYM(PA.h_pa,   context_set_default_sink,           "pa_context_set_default_sink");
    SYM(PA.h_pa,   context_set_source_volume_by_index, "pa_context_set_source_volume_by_index");
    SYM(PA.h_pa,   context_set_source_mute_by_index,   "pa_context_set_source_mute_by_index");
    SYM(PA.h_pa,   context_set_sink_input_volume,      "pa_context_set_sink_input_volume");
    SYM(PA.h_pa,   context_set_sink_input_mute,        "pa_context_set_sink_input_mute");
    SYM(PA.h_pa,   operation_unref,                    "pa_operation_unref");
    SYM(PA.h_pa,   cvolume_avg,                        "pa_cvolume_avg");
    SYM(PA.h_pa,   cvolume_set,                        "pa_cvolume_set");
    SYM(PA.h_pa,   cvolume_scale,                      "pa_cvolume_scale");
    SYM(PA.h_pa,   proplist_gets,                      "pa_proplist_gets");
#undef SYM
    return TRUE;
}

static const char *resolve_pa_server(char buf[96]) {
    const char *ps = g_getenv("PULSE_SERVER");
    if (ps && *ps) return NULL;
    const char *rt = g_getenv("XDG_RUNTIME_DIR");
    if (rt && *rt) {
        char *sock = g_build_filename(rt, "pulse", "native", NULL);
        gboolean exists = g_file_test(sock, G_FILE_TEST_EXISTS);
        g_free(sock);
        if (exists) return NULL;
    }
    char raw[80];
    g_snprintf(raw, sizeof raw, "/run/user/%u/pulse/native", (unsigned)getuid());
    if (g_file_test(raw, G_FILE_TEST_EXISTS)) {
        g_snprintf(buf, 96, "unix:%s", raw);
        return buf;
    }
    return NULL;
}

static gboolean pa_init(void) {
    if (!pa_load_symbols()) return FALSE;
    S.pa_ml = PA.glib_mainloop_new(NULL);
    if (!S.pa_ml) return FALSE;
    S.pa_ctx = PA.context_new(PA.glib_mainloop_get_api(S.pa_ml), "Mira SoundPanel");
    if (!S.pa_ctx) return FALSE;
    PA.context_set_state_callback(S.pa_ctx, on_pa_state, NULL);
    char srv_buf[96];
    const char *srv = resolve_pa_server(srv_buf);
    if (PA.context_connect(S.pa_ctx, srv, 0, NULL) < 0) return FALSE;
    S.pa_ok = TRUE;
    return TRUE;
}

static void pa_shutdown(void) {
    if (S.pa_ctx) {
        PA.context_set_state_callback(S.pa_ctx, NULL, NULL);
        PA.context_set_subscribe_callback(S.pa_ctx, NULL, NULL);
        PA.context_disconnect(S.pa_ctx);
        PA.context_unref(S.pa_ctx);
        S.pa_ctx = NULL;
    }
    if (S.pa_ml) {
        PA.glib_mainloop_free(S.pa_ml);
        S.pa_ml = NULL;
    }
    S.pa_ready = FALSE;
}

/* -- Keyboard focus & navigation (page-aware) ------------------------------ */
static int inline_app_count(void) {
    if (!S.groups) return 0;
    return (int)MIN(S.groups->len, (guint)INLINE_APPS_MAX);
}

static gboolean has_more_btn(void) {
    return S.groups && S.groups->len > INLINE_APPS_MAX;
}

static int main_focus_count(void) {
    return 2 + inline_app_count() + (has_more_btn() ? 1 : 0);
}

static void update_focus_visuals(void) {
    if (!S.on_apps_page) {
        int n = main_focus_count();
        if (S.focus_idx >= n) S.focus_idx = MAX(0, n - 1);
        set_class(S.out.block, "kb-focus", S.kb_active && S.focus_idx == 0);
        set_class(S.mic.block, "kb-focus", S.kb_active && S.focus_idx == 1);
        int ic = inline_app_count();
        for (int i = 0; i < ic; i++) {
            AppGroup *g = g_ptr_array_index(S.groups, (guint)i);
            if (g->inline_ui)
                set_class(g->inline_ui->row, "kb-focus", S.kb_active && S.focus_idx == i + 2);
        }
        set_class(S.apps_more_btn, "kb-focus",
                  S.kb_active && has_more_btn() && S.focus_idx == 2 + ic);
    } else {
        int n = S.groups ? (int)S.groups->len : 0;
        if (S.focus_idx >= n) S.focus_idx = MAX(0, n - 1);
        for (int i = 0; i < n; i++) {
            AppGroup *g = g_ptr_array_index(S.groups, (guint)i);
            if (g->page_ui)
                set_class(g->page_ui->row, "kb-focus", S.kb_active && S.focus_idx == i);
        }
        if (S.kb_active && n > 0 && S.all_apps_scroll && S.all_apps_box) {
            AppGroup *g = g_ptr_array_index(S.groups, (guint)S.focus_idx);
            if (g->page_ui && g->page_ui->row) {
                GtkAdjustment *va = gtk_scrolled_window_get_vadjustment(GTK_SCROLLED_WINDOW(S.all_apps_scroll));
                GtkAllocation alloc;
                gtk_widget_get_allocation(g->page_ui->row, &alloc);
                int wx = 0, wy = 0;
                if (va && alloc.height > 0 &&
                    gtk_widget_translate_coordinates(g->page_ui->row, S.all_apps_box, 0, 0, &wx, &wy)) {
                    double cur = gtk_adjustment_get_value(va);
                    double page = gtk_adjustment_get_page_size(va);
                    if (wy < cur) gtk_adjustment_set_value(va, wy);
                    else if (wy + alloc.height > cur + page)
                        gtk_adjustment_set_value(va, wy + alloc.height - page);
                }
            }
        }
    }
}

static gboolean on_key(GtkWidget *_w, GdkEventKey *ev, gpointer _d) {
    guint k = ev->keyval;
    int step = (ev->state & GDK_SHIFT_MASK) ? 1 : 5;

    if (k == GDK_KEY_Escape) {
        if (S.on_apps_page) { show_main_page(); return TRUE; }
        if (S.sinks_open) {
            S.sinks_open = FALSE;
            gtk_revealer_set_reveal_child(GTK_REVEALER(S.sinks_rev), FALSE);
            gtk_label_set_text(GTK_LABEL(S.dev_chev_lbl), GL_CHEV_RT);
            fit_follow(180);
            return TRUE;
        }
        panel_close();
        return TRUE;
    }
    if (k == GDK_KEY_BackSpace && S.on_apps_page) {
        show_main_page();
        return TRUE;
    }

    int n = S.on_apps_page ? (S.groups ? (int)S.groups->len : 0) : main_focus_count();
    if (n <= 0) return FALSE;

    if (k == GDK_KEY_Down || k == GDK_KEY_Tab) {
        S.focus_idx = S.kb_active ? (S.focus_idx + 1) % n : (n > 1 ? 1 : 0);
        S.kb_active = TRUE;
        update_focus_visuals();
        return TRUE;
    }
    if (k == GDK_KEY_Up || k == GDK_KEY_ISO_Left_Tab) {
        S.focus_idx = S.kb_active ? (S.focus_idx + n - 1) % n : (n - 1);
        S.kb_active = TRUE;
        update_focus_visuals();
        return TRUE;
    }

    /* Main page "All N apps ›" button activation */
    if (!S.on_apps_page && has_more_btn() && S.kb_active && S.focus_idx == 2 + inline_app_count()) {
        if (k == GDK_KEY_Return || k == GDK_KEY_KP_Enter || k == GDK_KEY_space || k == GDK_KEY_Right) {
            show_apps_page();
            return TRUE;
        }
        return FALSE;
    }

    int delta = 0;
    if (k == GDK_KEY_Right || k == GDK_KEY_plus || k == GDK_KEY_equal) delta = step;
    else if (k == GDK_KEY_Left || k == GDK_KEY_minus) delta = -step;
    gboolean do_mute = (k == GDK_KEY_m || k == GDK_KEY_M || k == GDK_KEY_space);
    if (!delta && !do_mute) return FALSE;

    if (S.on_apps_page) {
        AppGroup *g = g_ptr_array_index(S.groups, (guint)CLAMP(S.focus_idx, 0, n - 1));
        if (do_mute) on_app_mute(NULL, g);
        else nudge_group(g, delta);
        return TRUE;
    }

    if (S.focus_idx <= 0) {
        if (do_mute) on_master_mute(NULL, &S.out);
        else nudge_master(&S.out, delta);
    } else if (S.focus_idx == 1) {
        if (do_mute) on_master_mute(NULL, &S.mic);
        else nudge_master(&S.mic, delta);
    } else {
        guint gi = (guint)(S.focus_idx - 2);
        if (S.groups && gi < S.groups->len) {
            AppGroup *g = g_ptr_array_index(S.groups, gi);
            if (do_mute) on_app_mute(NULL, g);
            else nudge_group(g, delta);
        }
    }
    return TRUE;
}

/* -- open / close & input -------------------------------------------------- */
static void show_panel_now(void) {
    if (S.shown || S.closing) return;
    S.shown = TRUE;
    if (S.show_timer) { g_source_remove(S.show_timer); S.show_timer = 0; }
    fit();
    /* Show and flush the panel before spawning misa-backdrop (WifiPanel trick)
     * so the popup hits the screen immediately without waiting on backdrop. */
    gtk_widget_show(S.win);
    gdk_window_raise(gtk_widget_get_window(S.win));
    gdk_display_flush(gdk_display_get_default());
    S.shade = backdrop(S.mon, 0, NULL);
    gdk_window_raise(gtk_widget_get_window(S.win));
    grab_soon(S.win, G_CALLBACK(panel_close), NULL);
}

static gboolean show_timeout_cb(gpointer _d) {
    S.show_timer = 0;
    show_panel_now();
    return G_SOURCE_REMOVE;
}

static void panel_close(void) {
    if (S.closing) return;
    S.closing = TRUE;
    if (S.show_timer)  { g_source_remove(S.show_timer);  S.show_timer = 0; }
    if (S.fit_idle_id) { g_source_remove(S.fit_idle_id); S.fit_idle_id = 0; }
    if (S.fit_tick_id) {
        gtk_widget_remove_tick_callback(S.box, S.fit_tick_id);
        S.fit_tick_id = 0;
    }
    if (S.out.send_timer) { g_source_remove(S.out.send_timer); S.out.send_timer = 0; }
    if (S.mic.send_timer) { g_source_remove(S.mic.send_timer); S.mic.send_timer = 0; }
    if (S.groups) {
        for (guint i = 0; i < S.groups->len; i++) {
            AppGroup *g = g_ptr_array_index(S.groups, i);
            if (g->send_timer) { g_source_remove(g->send_timer); g->send_timer = 0; }
        }
    }
    pa_shutdown();
    ungrab();
    gtk_widget_hide(S.win);
    drop_backdrop(S.shade);
    gtk_main_quit();
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

static void on_alloc(GtkWidget *_w, GdkRectangle *_a, gpointer _d) {
    if (S.shown) place(0);
}

static gboolean on_term(gpointer _d) { panel_close(); return G_SOURCE_REMOVE; }

/* -- UI construction ------------------------------------------------------- */
static char *build_css(const Palette *p) {
    GString *c = g_string_sized_new(4096);
    g_string_append_printf(c, "@define-color fg %s;\n@define-color bg %s;\n", p->fg, p->bg);
    g_string_append_printf(c, "* { font-family: \"%s\"; }\n", p->font);
    g_string_append(c,
        "window, stack { background: transparent; }\n"
        ".vol-panel {\n"
        "  background-color: @bg;\n"
        "  color: @fg;\n"
        "  border: 1px solid alpha(@fg, 0.12);\n"
        "  border-radius: 14px;\n"
        "  padding: 14px 16px 12px 16px;\n"
        "}\n"
        ".vol-title {\n"
        "  font-size: 15px;\n"
        "  font-weight: bold;\n"
        "  color: @fg;\n"
        "  padding-left: 2px;\n"
        "}\n"
        ".page-head { padding: 2px 2px 2px 0px; min-height: 26px; }\n"
        ".page-title {\n"
        "  font-size: 14px;\n"
        "  font-weight: 600;\n"
        "  color: @fg;\n"
        "}\n"
        "button {\n"
        "  background-image: none;\n"
        "  box-shadow: none;\n"
        "  text-shadow: none;\n"
        "  -gtk-icon-shadow: none;\n"
        "  border: none;\n"
        "  outline: none;\n"
        "  min-height: 0;\n"
        "  min-width: 0;\n"
        "}\n"
        "button.vol-gear, button.page-back {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 18px;\n"
        "  padding: 2px 6px;\n"
        "  border-radius: 8px;\n"
        "  background: transparent;\n"
        "  color: alpha(@fg, 0.55);\n"
        "}\n"
        "button.page-back {\n"
        "  font-size: 20px;\n"
        "  padding: 0px 3px;\n"
        "  color: @fg;\n"
        "}\n"
        "button.vol-gear:hover, button.page-back:hover {\n"
        "  color: @fg;\n"
        "  background-color: alpha(@fg, 0.08);\n"
        "}\n"
        /* Primary master channel cards */
        ".vol-block {\n"
        "  padding: 10px 12px 12px 12px;\n"
        "  border-radius: 12px;\n"
        "  border: 1px solid transparent;\n"
        "  background-color: alpha(@fg, 0.06);\n"
        "  transition: opacity 120ms ease-out;\n"
        "}\n"
        ".vol-block.muted { opacity: 0.40; }\n"
        ".vol-block.kb-focus {\n"
        "  border-color: alpha(@fg, 0.24);\n"
        "  background-color: alpha(@fg, 0.085);\n"
        "}\n"
        "button.vol-icon {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 18px;\n"
        "  min-width: 26px;\n"
        "  padding: 0px 2px;\n"
        "  border-radius: 8px;\n"
        "  background: transparent;\n"
        "  color: @fg;\n"
        "}\n"
        "button.vol-icon:hover, button.app-icon:hover {\n"
        "  background-color: alpha(@fg, 0.10);\n"
        "}\n"
        ".vol-name {\n"
        "  font-size: 13px;\n"
        "  font-weight: 600;\n"
        "  color: alpha(@fg, 0.72);\n"
        "}\n"
        ".vol-num {\n"
        "  font-size: 13px;\n"
        "  font-weight: bold;\n"
        "  color: @fg;\n"
        "  font-feature-settings: \"tnum\";\n"
        "}\n"
        "scale.vol-pill { padding: 0; margin: 0; }\n"
        "scale.vol-pill contents, scale.app-pill contents { padding: 0; margin: 0; }\n"
        "scale.vol-pill trough {\n"
        "  min-height: 26px;\n"
        "  border-radius: 13px;\n"
        "  background-color: alpha(@fg, 0.14);\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "  background-image: none;\n"
        "  padding: 0;\n"
        "  margin: 0;\n"
        "}\n"
        "scale.vol-pill highlight {\n"
        "  min-height: 26px;\n"
        "  border-radius: 13px;\n"
        "  background-color: @fg;\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "  background-image: none;\n"
        "  padding: 0;\n"
        "  margin: 0;\n"
        "}\n"
        "scale.vol-pill.zero highlight, scale.app-pill.zero highlight {\n"
        "  background-color: transparent;\n"
        "}\n"
        "scale.vol-pill slider, scale.app-pill slider {\n"
        "  min-width: 0px;\n"
        "  min-height: 0px;\n"
        "  margin: 0px;\n"
        "  padding: 0px;\n"
        "  background: transparent;\n"
        "  background-image: none;\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "}\n"
        /* Subordinate minimal Apps mixer (single container, single-line rows) */
        ".apps-label {\n"
        "  font-size: 11px;\n"
        "  font-weight: 600;\n"
        "  color: alpha(@fg, 0.40);\n"
        "  padding: 0px 4px;\n"
        "}\n"
        ".apps-count {\n"
        "  font-size: 11.5px;\n"
        "  font-weight: 600;\n"
        "  color: alpha(@fg, 0.42);\n"
        "  padding-right: 4px;\n"
        "  font-feature-settings: \"tnum\";\n"
        "}\n"
        ".apps-group {\n"
        "  background-color: alpha(@fg, 0.042);\n"
        "  border-radius: 12px;\n"
        "  padding: 4px 5px;\n"
        "}\n"
        ".app-row {\n"
        "  padding: 3px 6px;\n"
        "  border-radius: 8px;\n"
        "  border: 1px solid transparent;\n"
        "  transition: opacity 120ms ease-out, background-color 100ms ease-out;\n"
        "}\n"
        ".app-row:hover { background-color: alpha(@fg, 0.035); }\n"
        ".app-row.muted { opacity: 0.38; }\n"
        ".app-row.kb-focus {\n"
        "  border-color: alpha(@fg, 0.22);\n"
        "  background-color: alpha(@fg, 0.065);\n"
        "}\n"
        "button.app-icon {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 15px;\n"
        "  min-width: 22px;\n"
        "  padding: 2px 2px;\n"
        "  border-radius: 6px;\n"
        "  background: transparent;\n"
        "  color: alpha(@fg, 0.72);\n"
        "}\n"
        ".app-name {\n"
        "  font-size: 12px;\n"
        "  font-weight: 500;\n"
        "  color: alpha(@fg, 0.80);\n"
        "}\n"
        ".app-num {\n"
        "  font-size: 11.5px;\n"
        "  font-weight: 600;\n"
        "  color: alpha(@fg, 0.62);\n"
        "  font-feature-settings: \"tnum\";\n"
        "}\n"
        /* Visual 8 px capsule slider with 22 px comfortable click/drag hit area */
        "scale.app-pill { padding: 7px 0px; margin: 0; }\n"
        "scale.app-pill trough {\n"
        "  min-height: 8px;\n"
        "  border-radius: 4px;\n"
        "  background-color: alpha(@fg, 0.13);\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "  background-image: none;\n"
        "  padding: 0;\n"
        "  margin: 0;\n"
        "}\n"
        "scale.app-pill highlight {\n"
        "  min-height: 8px;\n"
        "  border-radius: 4px;\n"
        "  background-color: alpha(@fg, 0.78);\n"
        "  border: none;\n"
        "  box-shadow: none;\n"
        "  background-image: none;\n"
        "  padding: 0;\n"
        "  margin: 0;\n"
        "}\n"
        "button.apps-more {\n"
        "  background: transparent;\n"
        "  border-top: 1px solid alpha(@fg, 0.06);\n"
        "  border-radius: 0px 0px 8px 8px;\n"
        "  margin-top: 2px;\n"
        "  padding: 6px 8px 4px 8px;\n"
        "}\n"
        "button.apps-more:hover, button.apps-more.kb-focus {\n"
        "  background-color: alpha(@fg, 0.06);\n"
        "  border-radius: 8px;\n"
        "}\n"
        ".apps-more-lbl {\n"
        "  font-size: 11.5px;\n"
        "  font-weight: 500;\n"
        "  color: alpha(@fg, 0.56);\n"
        "}\n"
        "button.apps-more:hover .apps-more-lbl { color: @fg; }\n"
        ".apps-more-chev {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 14px;\n"
        "  color: alpha(@fg, 0.45);\n"
        "}\n"
        /* All Apps page scroll viewport (matches WifiPanel/BluetoothPanel) */
        "scrolledwindow, viewport { background: transparent; border: none; }\n"
        "scrollbar { opacity: 0; }\n"
        /* Output device switcher drawer & footer */
        ".sinks-box {\n"
        "  background-color: alpha(@fg, 0.05);\n"
        "  border-radius: 10px;\n"
        "  padding: 4px;\n"
        "  margin-bottom: 6px;\n"
        "}\n"
        "button.sink-row {\n"
        "  background: transparent;\n"
        "  border-radius: 8px;\n"
        "  padding: 6px 8px;\n"
        "}\n"
        "button.sink-row:hover { background-color: alpha(@fg, 0.07); }\n"
        "button.sink-row.active { background-color: alpha(@fg, 0.09); }\n"
        ".sink-row-icon, .sink-row-check {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 14px;\n"
        "  color: alpha(@fg, 0.65);\n"
        "}\n"
        "button.sink-row.active .sink-row-icon, button.sink-row.active .sink-row-check {\n"
        "  color: @fg;\n"
        "}\n"
        ".sink-row-name {\n"
        "  font-size: 12px;\n"
        "  color: alpha(@fg, 0.72);\n"
        "}\n"
        "button.sink-row.active .sink-row-name {\n"
        "  color: @fg;\n"
        "  font-weight: 600;\n"
        "}\n"
        "button.vol-foot {\n"
        "  padding: 3px 4px;\n"
        "  border-radius: 8px;\n"
        "  background: transparent;\n"
        "  color: alpha(@fg, 0.50);\n"
        "}\n"
        "button.vol-foot.switchable:hover { background-color: alpha(@fg, 0.06); }\n"
        ".vol-foot-icon, .vol-foot-chev {\n"
        "  font-family: \"Material Design Icons Desktop\";\n"
        "  font-size: 14px;\n"
        "  color: alpha(@fg, 0.50);\n"
        "}\n"
        ".vol-dev {\n"
        "  font-size: 12px;\n"
        "  color: alpha(@fg, 0.50);\n"
        "}\n"
        "tooltip, tooltip.background {\n"
        "  background-color: @bg;\n"
        "  color: @fg;\n"
        "  border: 1px solid alpha(@fg, 0.14);\n"
        "  border-radius: 8px;\n"
        "}\n");
    return g_string_free(c, FALSE);
}

static void build_ui(void) {
    g_object_set(gtk_settings_get_default(), "gtk-primary-button-warps-slider", TRUE, NULL);

    Palette p;
    palette_load(&p);
    char *css = build_css(&p);
    css_load(css);
    g_free(css);

    S.win = overlay_window("SoundPanel", 0, 0, WIDTH, 100);
    gtk_widget_set_size_request(S.win, WIDTH, -1);
    gtk_window_set_default_size(GTK_WINDOW(S.win), -1, -1);
    gtk_window_set_resizable(GTK_WINDOW(S.win), TRUE);
    gtk_widget_add_events(S.win, GDK_KEY_PRESS_MASK | GDK_BUTTON_PRESS_MASK);
    g_signal_connect(S.win, "key-press-event", G_CALLBACK(on_key), NULL);
    g_signal_connect(S.win, "button-press-event", G_CALLBACK(on_outside), NULL);
    g_signal_connect(S.win, "size-allocate", G_CALLBACK(on_alloc), NULL);

    /* -- Page 1: Main Sound View -- */
    S.main_page = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);

    GtkWidget *head = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget *title = gtk_label_new("Sound");
    add_class(title, "vol-title");
    gtk_label_set_xalign(GTK_LABEL(title), 0.0f);
    gtk_widget_set_hexpand(title, TRUE);

    GtkWidget *gear = gtk_button_new_with_label(GL_GEAR);
    add_class(gear, "vol-gear");
    gtk_widget_set_can_focus(gear, FALSE);
    gtk_widget_set_tooltip_text(gear, "Sound settings");
    make_clickable(gear);
    g_signal_connect(gear, "clicked", G_CALLBACK(on_gear_click), NULL);

    gtk_box_pack_start(GTK_BOX(head), title, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(head), gear, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.main_page), head, FALSE, FALSE, 0);

    init_master_widgets(&S.out, "Output", "Mute output", FALSE);
    init_master_widgets(&S.mic, "Microphone", "Mute microphone", TRUE);
    gtk_box_pack_start(GTK_BOX(S.main_page), S.out.block, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.main_page), S.mic.block, FALSE, FALSE, 0);

    /* Compact inline Apps section (hidden completely when 0 apps play). */
    S.apps_sec = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    GtkWidget *apps_lbl = gtk_label_new("Apps");
    add_class(apps_lbl, "apps-label");
    gtk_label_set_xalign(GTK_LABEL(apps_lbl), 0.0f);

    S.apps_group_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    add_class(S.apps_group_box, "apps-group");
    S.apps_inline_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);

    S.apps_more_btn = gtk_button_new();
    add_class(S.apps_more_btn, "apps-more");
    gtk_widget_set_can_focus(S.apps_more_btn, FALSE);
    make_clickable(S.apps_more_btn);
    g_signal_connect(S.apps_more_btn, "clicked", G_CALLBACK(on_more_apps_click), NULL);

    GtkWidget *more_row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    S.apps_more_lbl = gtk_label_new("");
    add_class(S.apps_more_lbl, "apps-more-lbl");
    gtk_label_set_xalign(GTK_LABEL(S.apps_more_lbl), 0.0f);
    gtk_widget_set_hexpand(S.apps_more_lbl, TRUE);
    GtkWidget *more_chev = gtk_label_new(GL_CHEV_RT);
    add_class(more_chev, "apps-more-chev");
    gtk_box_pack_start(GTK_BOX(more_row), S.apps_more_lbl, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(more_row), more_chev, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(S.apps_more_btn), more_row);

    gtk_box_pack_start(GTK_BOX(S.apps_group_box), S.apps_inline_box, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.apps_group_box), S.apps_more_btn, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.apps_sec), apps_lbl, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.apps_sec), S.apps_group_box, FALSE, FALSE, 0);
    gtk_widget_show_all(S.apps_sec);
    gtk_widget_hide(S.apps_more_btn);
    gtk_widget_set_no_show_all(S.apps_more_btn, TRUE);
    gtk_widget_hide(S.apps_sec);
    gtk_widget_set_no_show_all(S.apps_sec, TRUE);
    gtk_box_pack_start(GTK_BOX(S.main_page), S.apps_sec, FALSE, FALSE, 0);

    /* Footer wrapper: zero extra spacing when sink switcher drawer is collapsed. */
    GtkWidget *foot_wrap = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    S.sinks_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);
    add_class(S.sinks_box, "sinks-box");
    S.sinks_rev = gtk_revealer_new();
    gtk_revealer_set_transition_type(GTK_REVEALER(S.sinks_rev), GTK_REVEALER_TRANSITION_TYPE_SLIDE_DOWN);
    gtk_revealer_set_transition_duration(GTK_REVEALER(S.sinks_rev), 160);
    gtk_revealer_set_reveal_child(GTK_REVEALER(S.sinks_rev), FALSE);
    gtk_container_add(GTK_CONTAINER(S.sinks_rev), S.sinks_box);
    gtk_box_pack_start(GTK_BOX(foot_wrap), S.sinks_rev, FALSE, FALSE, 0);

    S.foot_btn = gtk_button_new();
    add_class(S.foot_btn, "vol-foot");
    gtk_widget_set_can_focus(S.foot_btn, FALSE);
    g_signal_connect(S.foot_btn, "clicked", G_CALLBACK(on_footer_click), NULL);

    GtkWidget *foot = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    S.dev_icon_lbl = gtk_label_new(GL_SPEAKER);
    add_class(S.dev_icon_lbl, "vol-foot-icon");
    S.dev_lbl = gtk_label_new("");
    add_class(S.dev_lbl, "vol-dev");
    gtk_label_set_xalign(GTK_LABEL(S.dev_lbl), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(S.dev_lbl), PANGO_ELLIPSIZE_END);
    gtk_label_set_max_width_chars(GTK_LABEL(S.dev_lbl), 36);
    gtk_widget_set_hexpand(S.dev_lbl, TRUE);
    S.dev_chev_lbl = gtk_label_new(GL_CHEV_RT);
    add_class(S.dev_chev_lbl, "vol-foot-chev");
    gtk_widget_set_no_show_all(S.dev_chev_lbl, TRUE);

    gtk_box_pack_start(GTK_BOX(foot), S.dev_icon_lbl, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(foot), S.dev_lbl, TRUE, TRUE, 0);
    gtk_box_pack_end(GTK_BOX(foot), S.dev_chev_lbl, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(S.foot_btn), foot);
    gtk_box_pack_start(GTK_BOX(foot_wrap), S.foot_btn, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(S.main_page), foot_wrap, FALSE, FALSE, 0);

    /* -- Stack container (main page always mounted; Apps sub-page built lazily
     * via swap_page() like WifiPanel/BluetoothPanel) -- */
    S.stack = gtk_stack_new();
    gtk_stack_set_transition_type(GTK_STACK(S.stack), GTK_STACK_TRANSITION_TYPE_SLIDE_LEFT_RIGHT);
    gtk_stack_set_transition_duration(GTK_STACK(S.stack), PAGE_ANIM_MS);
    gtk_stack_set_vhomogeneous(GTK_STACK(S.stack), FALSE);
    gtk_stack_set_interpolate_size(GTK_STACK(S.stack), TRUE);
    gtk_stack_add_named(GTK_STACK(S.stack), S.main_page, "main");
    gtk_stack_set_visible_child(GTK_STACK(S.stack), S.main_page);
    g_signal_connect(S.stack, "notify::transition-running", G_CALLBACK(on_stack_transition), NULL);

    S.box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    add_class(S.box, "vol-panel");
    gtk_box_pack_start(GTK_BOX(S.box), S.stack, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(S.win), S.box);
    gtk_widget_show_all(S.box);
    gtk_widget_realize(S.win);
}

/* -- single-instance toggle ------------------------------------------------ */
static int single(void) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    char *path = g_build_filename(run && *run ? run : "/tmp", "sound-panel.pid", NULL);
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
        if (pwrite(fd, t, n, 0) < 0) { /* lock holds */ }
    }
    return 1;
}

int main(int argc, char **argv) {
    strcpy(S.side, "right");
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pointer")) S.pointer = TRUE;
        else if (!strcmp(argv[i], "--side") && i + 1 < argc)
            g_strlcpy(S.side, argv[++i], sizeof S.side);
    }
    if (!single()) return 0;
    close_bar_popups("sound");

    mallopt(M_ARENA_MAX, 1);
    gtk_env_lean();
    gtk_init(&argc, &argv);
    g_unix_signal_add(SIGTERM, on_term, NULL);

    S.raw_streams = g_ptr_array_new_with_free_func(free_raw_stream);
    S.groups = g_ptr_array_new();
    S.sinks = g_ptr_array_new_with_free_func(free_sink_dev);

    gboolean pa_started = pa_init();

    build_ui();
    anchor(S.pointer);

    if (pa_started) {
        S.show_timer = g_timeout_add(INITIAL_SHOW_MS, show_timeout_cb, NULL);
    } else {
        gtk_label_set_text(GTK_LABEL(S.dev_lbl), "Audio server unavailable");
        show_panel_now();
    }

    gtk_main();

    if (S.groups) {
        for (guint i = 0; i < S.groups->len; i++)
            free_app_group(g_ptr_array_index(S.groups, i));
        g_ptr_array_free(S.groups, TRUE);
    }
    if (S.raw_streams) g_ptr_array_free(S.raw_streams, TRUE);
    if (S.sinks) g_ptr_array_free(S.sinks, TRUE);
    g_free(S.default_sink_name);
    g_free(S.default_source_name);
    if (S.ptr_cursor) g_object_unref(S.ptr_cursor);
    return 0;
}
