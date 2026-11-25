/* live-gl.h — Hyper-optimized GPU live-shader & Luma-Halftone Develop engine
 * for CardPicker (--layout gallery).
 *
 * Renders built-in theme wallpapers (.frag sidecar) into tiny 1:1 cell-res FBOs
 * (640x360 on 1080p/1440p/4K = 9x–36x fewer fragment invocations) while keeping
 * custom user wallpapers (no .frag) razor-sharp and static, and blends live->live,
 * live->static, static->live, and static->static directly on the GPU via Mira's
 * exact Luma-Halftone Develop transition at 60 FPS (dropping to 15/10 FPS when
 * idle on a live wallpaper, or 0 FPS on a static wallpaper).
 *
 * Zero extra build dependencies: loads OpenGL dynamically via dlopen(libGL.so.1)
 * inside GTK3's built-in GtkGLArea, with automatic GLSL 120 -> 150 Core adaptation
 * and seamless fallback to CPU Cairo if GL is unavailable.
 */
#ifndef MISA_UI_LIVE_GL_H
#define MISA_UI_LIVE_GL_H

#include <gtk/gtk.h>

typedef struct LiveGl LiveGl;

/* Returns newly allocated canonical `<stem>.frag` path if image has a live shader
 * sidecar (and MISA_WALL_LIVE != "0"), else NULL. Caller must g_free(). */
char *live_gl_find_sidecar(const char *image_path);

/* Recommended texture decode cap (2x cell grid) for a live shader wallpaper on
 * monitor (mon_w x mon_h). Static wallpapers decode at full monitor resolution. */
void live_gl_tex_cap(int mon_w, int mon_h, int *out_w, int *out_h);

/* Fast 16x16 stride luma auto-level range for Mira's Luma-Halftone Develop. */
void live_gl_scan_luma_rgb(const guchar *rgb, int w, int h, int rowstride, int ch,
                           float *out_max, float *out_span);

/* Create a GtkGLArea-backed renderer covering window (win_w x win_h) positioned
 * at monitor top offset `top_px` on monitor (mon_w x mon_h). Pass `corner_r > 0`
 * for antialiased rounded cards (used by Theme Picker showcase). */
LiveGl *live_gl_new(int mon_w, int mon_h, int top_px, int win_w, int win_h);
LiveGl *live_gl_new_rounded(int w, int h, float corner_r);
/* When `static_only` is TRUE (Theme Picker), each `.frag` wallpaper renders ONCE
 * at `fixed_time` (capturing dither + atmosphere particles) and rests at 0 FPS / 0% CPU. */
void live_gl_set_static_frame(LiveGl *lg, gboolean static_only, float fixed_time);
GtkWidget *live_gl_widget(LiveGl *lg);
gboolean live_gl_ready(const LiveGl *lg);
void live_gl_stop(LiveGl *lg);
void live_gl_destroy(LiveGl *lg);

/* Upload an RGB888 buffer (tightly packed, w*h*3 bytes) into the target slot.
 * If `animate` is TRUE, current target is promoted to source and a GPU
 * Luma-Halftone Develop transition begins (dur_ms). `frag_path` may be NULL for
 * static custom wallpapers (rendered untouched, zero dither, 0 FPS at idle). */
void live_gl_set_wallpaper(LiveGl *lg, const guchar *rgb, int w, int h,
                           const char *frag_path, float luma_max, float luma_span,
                           gboolean animate, int dur_ms);

/* Refine the current target texture (e.g. full-res arrived after quick thumb)
 * without restarting an in-flight transition. */
void live_gl_refine_target(LiveGl *lg, const guchar *rgb, int w, int h,
                           const char *frag_path, float luma_max, float luma_span);

/* Update subtle cursor parallax target (normalized [-1, 1]). */
void live_gl_set_mouse(LiveGl *lg, float nx, float ny);

/* Query transition status or attach a one-shot callback fired when the active
 * transition finishes (used by Enter / gallery_confirm for zero-flash apply). */
gboolean live_gl_trans_done(const LiveGl *lg);
gboolean live_gl_target_is_live(const LiveGl *lg);
void live_gl_set_finish_cb(LiveGl *lg, void (*cb)(void *user), void *user);

/* Render one crisp 1:1 dither + atmosphere thumbnail frame of a `.frag` shader
 * into an offscreen FBO (tw x th). Returns newly allocated top-down RGB888
 * bytes (`tw * th * 3`, caller frees with g_free()), or NULL on failure. */
guchar *live_gl_render_thumb(LiveGl *lg, const guchar *src_rgb, int src_w, int src_h,
                             const char *frag_path, int tw, int th, float fixed_time);

/* Pause/resume helper coordinating with `misa-wall --live-daemon` so the desktop
 * daemon drops to 0.00% CPU/GPU while CardPicker's fullscreen GPU preview is open. */
void live_gl_pause_desktop_daemon(gboolean pause);

#endif
