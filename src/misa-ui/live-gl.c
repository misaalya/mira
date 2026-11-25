/* live-gl.c — Hyper-optimized GPU live-shader & Luma-Halftone Develop engine
 * for CardPicker (--layout gallery).
 * Copyright (C) 2026 Misaalya · GPL-3.0-only
 */
#define _GNU_SOURCE
#include "live-gl.h"

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Minimal OpenGL constants (zero libgl-dev header dependency). */
typedef unsigned int GLenum;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef int GLint;
typedef int GLsizei;
typedef unsigned int GLuint;
typedef float GLfloat;
typedef char GLchar;
typedef ptrdiff_t GLsizeiptr;

#define GL_TRIANGLES             0x0004
#define GL_UNSIGNED_BYTE         0x1401
#define GL_FLOAT                 0x1406
#define GL_RGB                   0x1907
#define GL_RGBA                  0x1908
#define GL_VERSION               0x1F02
#define GL_TEXTURE_2D            0x0DE1
#define GL_TEXTURE_MAG_FILTER    0x2800
#define GL_TEXTURE_MIN_FILTER    0x2801
#define GL_TEXTURE_WRAP_S        0x2802
#define GL_TEXTURE_WRAP_T        0x2803
#define GL_NEAREST               0x2600
#define GL_LINEAR                0x2601
#define GL_CLAMP_TO_EDGE         0x812F
#define GL_TEXTURE0              0x84C0
#define GL_TEXTURE1              0x84C1
#define GL_ARRAY_BUFFER          0x8892
#define GL_STATIC_DRAW           0x88E4
#define GL_FRAGMENT_SHADER       0x8B30
#define GL_VERTEX_SHADER         0x8B31
#define GL_COMPILE_STATUS        0x8B81
#define GL_LINK_STATUS           0x8B82
#define GL_PACK_ALIGNMENT        0x0D05
#define GL_UNPACK_ALIGNMENT      0x0CF5
#define GL_FRAMEBUFFER           0x8D40
#define GL_FRAMEBUFFER_BINDING   0x8CA6
#define GL_COLOR_ATTACHMENT0     0x8CE0
#define GL_FRAMEBUFFER_COMPLETE  0x8CD5
#define GL_MAJOR_VERSION         0x821B
#define GL_SCISSOR_TEST          0x0C11
#define GL_DEPTH_TEST            0x0B71
#define GL_BLEND                 0x0BE2

typedef void *(*FnGetProc)(const unsigned char *);

typedef struct {
    void (*Viewport)(GLint, GLint, GLsizei, GLsizei);
    void (*Disable)(GLenum);
    void (*PixelStorei)(GLenum, GLint);
    void (*ReadPixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void *);
    void (*GetIntegerv)(GLenum, GLint *);
    const unsigned char *(*GetString)(GLenum);
    void (*GenTextures)(GLsizei, GLuint *);
    void (*DeleteTextures)(GLsizei, const GLuint *);
    void (*BindTexture)(GLenum, GLuint);
    void (*TexParameteri)(GLenum, GLenum, GLint);
    void (*TexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *);
    void (*ActiveTexture)(GLenum);
    void (*DrawArrays)(GLenum, GLint, GLsizei);
    GLuint (*CreateShader)(GLenum);
    void (*DeleteShader)(GLuint);
    void (*ShaderSource)(GLuint, GLsizei, const GLchar *const *, const GLint *);
    void (*CompileShader)(GLuint);
    void (*GetShaderiv)(GLuint, GLenum, GLint *);
    void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
    GLuint (*CreateProgram)(void);
    void (*DeleteProgram)(GLuint);
    void (*AttachShader)(GLuint, GLuint);
    void (*LinkProgram)(GLuint);
    void (*GetProgramiv)(GLuint, GLenum, GLint *);
    void (*UseProgram)(GLuint);
    GLint (*GetUniformLocation)(GLuint, const GLchar *);
    GLint (*GetAttribLocation)(GLuint, const GLchar *);
    void (*BindAttribLocation)(GLuint, GLuint, const GLchar *);
    void (*Uniform1f)(GLint, GLfloat);
    void (*Uniform2f)(GLint, GLfloat, GLfloat);
    void (*Uniform4f)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
    void (*Uniform1i)(GLint, GLint);
    void (*GenBuffers)(GLsizei, GLuint *);
    void (*DeleteBuffers)(GLsizei, const GLuint *);
    void (*BindBuffer)(GLenum, GLuint);
    void (*BufferData)(GLenum, GLsizeiptr, const void *, GLenum);
    void (*EnableVertexAttribArray)(GLuint);
    void (*VertexAttribPointer)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void *);
    void (*GenVertexArrays)(GLsizei, GLuint *);
    void (*DeleteVertexArrays)(GLsizei, const GLuint *);
    void (*BindVertexArray)(GLuint);
    void (*GenFramebuffers)(GLsizei, GLuint *);
    void (*DeleteFramebuffers)(GLsizei, const GLuint *);
    void (*BindFramebuffer)(GLenum, GLuint);
    void (*FramebufferTexture2D)(GLenum, GLenum, GLenum, GLuint, GLint);
    GLenum (*CheckFramebufferStatus)(GLenum);
} GlFns;

#define PROG_CACHE_MAX 12

typedef struct {
    char *path;
    GLuint prog;
    GLint u_tex, u_res, u_cover, u_time, u_mouse, u_cell;
    guint64 last_use;
} CachedProg;

typedef struct {
    GLuint img_tex;
    int img_w, img_h;
    float cover[4];
    GLuint fbo;
    GLuint fbo_tex;
    char *frag_path;
    int prog_idx;      /* -1 when static custom wallpaper */
    gboolean valid;
    gboolean fbo_dirty;
} Slot;

struct LiveGl {
    GtkWidget *area;
    GlFns gl;
    gboolean gl_ok;
    gboolean is_core;
    int mon_w, mon_h, top_px, win_w, win_h;
    int cell_px, fw, fh;

    GLuint vao, vbo;
    GLuint comp_prog;
    GLint c_src, c_tgt, c_src_cover, c_tgt_cover;
    GLint c_src_fbo, c_tgt_fbo, c_mon, c_win, c_top;
    GLint c_eased, c_lmax, c_lspan, c_corner_r;
    float corner_r;

    CachedProg cache[PROG_CACHE_MAX];
    guint64 use_seq;

    Slot slots[2];
    int cur_slot;      /* active target slot (0 or 1); source is cur_slot ^ 1 */

    GLuint thumb_fbo, thumb_fbo_tex, thumb_img_tex;
    int thumb_w, thumb_h;

    /* Transition & live pacing state */
    gint64 start_us;
    gint64 trans_start_us;
    gint64 last_fbo_us;
    int trans_dur_ms;
    float luma_max, luma_span;
    gboolean trans_done;
    gboolean has_slot;
    gboolean static_only;
    float fixed_time;
    float target_mx, target_my, smooth_mx, smooth_my;

    guint tick_id;     /* 60 FPS frame clock ONLY during 460ms transitions */
    guint live_timer;  /* 15/10 FPS timer ONLY when resting on a .frag wallpaper */
    int live_interval_ms;

    void (*finish_cb)(void *);
    void *finish_user;
};

static const char *VS_CORE =
    "#version 150 core\n"
    "in vec2 aPos;\n"
    "out vec2 vUv;\n"
    "void main() {\n"
    "    vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);\n"
    "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
    "}\n";

static const char *VS_COMPAT =
    "#version 120\n"
    "attribute vec2 aPos;\n"
    "varying vec2 vUv;\n"
    "void main() {\n"
    "    vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);\n"
    "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
    "}\n";

/* Unified GPU Mira Luma-Halftone Develop compositor:
 * Blends live FBO or crisp static Source -> live FBO or crisp static Target,
 * aligned pixel-for-pixel to full monitor coordinates below Polybar. */
static const char *COMP_FS_BODY =
    "uniform sampler2D uSrc;\n"
    "uniform sampler2D uTgt;\n"
    "uniform vec4 uSrcCover;\n"
    "uniform vec4 uTgtCover;\n"
    "uniform float uSrcFbo;\n"
    "uniform float uTgtFbo;\n"
    "uniform vec2 uMonSize;\n"
    "uniform vec2 uWinSize;\n"
    "uniform float uTop;\n"
    "uniform float uEased;\n"
    "uniform float uLumaMax;\n"
    "uniform float uLumaSpan;\n"
    "uniform float uCornerR;\n"
    "float bayer4(vec2 p) {\n"
    "    vec2 b0 = floor(mod(p, 2.0));\n"
    "    vec2 b1 = floor(mod(floor(p * 0.5), 2.0));\n"
    "    float q0 = abs(b0.x - b0.y) * 2.0 + b0.y;\n"
    "    float q1 = abs(b1.x - b1.y) * 2.0 + b1.y;\n"
    "    return (q0 * 4.0 + q1) / 15.0;\n"
    "}\n"
    "vec2 mapUv(vec2 monUv, vec4 cov, float isFbo) {\n"
    "    return isFbo > 0.5 ? vec2(monUv.x, 1.0 - monUv.y) : (monUv * cov.xy + cov.zw);\n"
    "}\n"
    "void main() {\n"
    "    float alphaOut = 1.0;\n"
    "    if (uCornerR > 0.5) {\n"
    "        vec2 halfSz = uWinSize * 0.5;\n"
    "        vec2 q = abs(vUv * uWinSize - halfSz) - (halfSz - vec2(uCornerR));\n"
    "        float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - uCornerR;\n"
    "        alphaOut = clamp(0.5 - d, 0.0, 1.0);\n"
    "        if (alphaOut <= 0.001) { OUT_COLOR = vec4(0.0); return; }\n"
    "    }\n"
    "    vec2 monPx = vec2(vUv.x * uWinSize.x, uTop + vUv.y * uWinSize.y);\n"
    "    vec2 monUv = clamp(monPx / max(uMonSize, vec2(1.0)), 0.0, 1.0);\n"
    "    vec3 tgt = SAMPLE_TEX(uTgt, mapUv(monUv, uTgtCover, uTgtFbo)).rgb;\n"
    "    vec3 col = tgt;\n"
    "    if (uEased < 0.9999) {\n"
    "        vec3 src = SAMPLE_TEX(uSrc, mapUv(monUv, uSrcCover, uSrcFbo)).rgb;\n"
    "        float b = bayer4(floor(monPx * 0.25));\n"
    "        float luma = dot(tgt, vec3(0.3008, 0.5859, 0.1133));\n"
    "        float invL = clamp((uLumaMax - luma) / max(uLumaSpan, 0.18), 0.0, 1.0);\n"
    "        float tau = 6.2831853;\n"
    "        float org = 0.5 + 0.275 * sin(tau * 1.35 * monUv.x + 0.4) * cos(tau * 1.15 * monUv.y - 0.3)\n"
    "                        + 0.225 * cos(tau * 2.15 * monUv.x - 0.9) * sin(tau * 1.75 * monUv.y + 0.7);\n"
    "        float th = 0.42 * b + 0.38 * invL + 0.20 * org;\n"
    "        float feather = 0.0586;\n"
    "        float head = uEased * (1.0 + feather);\n"
    "        float a = clamp((head - th) / feather, 0.0, 1.0);\n"
    "        col = mix(src, tgt, a);\n"
    "    }\n"
    "    OUT_COLOR = vec4(col * alphaOut, alphaOut);\n"
    "}\n";

static int compute_cell_px(int mh) {
    if (mh < 720) return 2;
    int c = (mh + 180) / 360;
    return c < 3 ? 3 : c;
}

static gboolean on_battery_power(void) {
    GDir *d = g_dir_open("/sys/class/power_supply", 0, NULL);
    if (!d) return FALSE;
    gboolean saw_ac = FALSE;
    const char *name;
    while ((name = g_dir_read_name(d))) {
        char *tp = g_build_filename("/sys/class/power_supply", name, "type", NULL);
        char *kind = NULL;
        if (g_file_get_contents(tp, &kind, NULL, NULL) && kind) {
            if (strncmp(kind, "Mains", 5) == 0) {
                saw_ac = TRUE;
                char *op = g_build_filename("/sys/class/power_supply", name, "online", NULL);
                char *on = NULL;
                if (g_file_get_contents(op, &on, NULL, NULL) && on && on[0] == '1') {
                    g_free(on); g_free(op); g_free(kind); g_free(tp);
                    g_dir_close(d);
                    return FALSE;
                }
                g_free(on); g_free(op);
            }
        }
        g_free(kind); g_free(tp);
    }
    g_dir_close(d);
    return saw_ac;
}

static int default_live_interval_ms(void) {
    const char *env = g_getenv("MISA_WALL_LIVE_FPS");
    if (env && *env) {
        int fps = atoi(env);
        if (fps < 6) fps = 6;
        if (fps > 60) fps = 60;
        return 1000 / fps;
    }
    return on_battery_power() ? 100 : 66; /* 10 FPS battery / 15 FPS AC */
}

static void compute_cover(int mon_w, int mon_h, int im_w, int im_h, float out[4]) {
    float vp_a = (float)mon_w / (float)(mon_h > 0 ? mon_h : 1);
    float im_a = (float)im_w / (float)(im_h > 0 ? im_h : 1);
    if (vp_a > im_a) {
        float sy = im_a / vp_a;
        out[0] = 1.0f; out[1] = sy; out[2] = 0.0f; out[3] = (1.0f - sy) * 0.5f;
    } else {
        float sx = vp_a / im_a;
        out[0] = sx; out[1] = 1.0f; out[2] = (1.0f - sx) * 0.5f; out[3] = 0.0f;
    }
}

char *live_gl_find_sidecar(const char *image_path) {
    if (!image_path || !*image_path) return NULL;
    const char *env = g_getenv("MISA_WALL_LIVE");
    if (env && strcmp(env, "0") == 0) return NULL;
    char *canon = realpath(image_path, NULL);
    const char *base = canon ? canon : image_path;
    const char *dot = strrchr(base, '.');
    const char *slash = strrchr(base, '/');
    size_t stem_len = (dot && (!slash || dot > slash)) ? (size_t)(dot - base) : strlen(base);
    char *frag = g_malloc(stem_len + 6);
    memcpy(frag, base, stem_len);
    memcpy(frag + stem_len, ".frag", 6);
    free(canon);
    if (g_file_test(frag, G_FILE_TEST_IS_REGULAR)) return frag;
    g_free(frag);
    return NULL;
}

void live_gl_tex_cap(int mon_w, int mon_h, int *out_w, int *out_h) {
    int c = compute_cell_px(mon_h);
    int fw = (mon_w + c - 1) / c;
    int fh = (mon_h + c - 1) / c;
    int cw = fw * 2, ch = fh * 2;
    if (cw < 960) cw = 960;
    if (cw > 1600) cw = 1600;
    if (ch < 540) ch = 540;
    if (ch > 960) ch = 960;
    *out_w = cw;
    *out_h = ch;
}

void live_gl_scan_luma_rgb(const guchar *rgb, int w, int h, int rowstride, int ch,
                           float *out_max, float *out_span) {
    int l_min = 255, l_max = 0;
    for (int y = 0; y < h; y += 16) {
        const guchar *row = rgb + (size_t)y * (size_t)rowstride;
        for (int x = 0; x < w; x += 16) {
            const guchar *p = row + (size_t)x * (size_t)ch;
            int l = (p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8;
            if (l < l_min) l_min = l;
            if (l > l_max) l_max = l;
        }
    }
    int span = l_max - l_min;
    if (span < 48) span = 48;
    *out_max = l_max / 255.0f;
    *out_span = span / 255.0f;
}

static void *gl_sym(void *h, FnGetProc gp, const char *name) {
    void *p = dlsym(h, name);
    if (!p && gp) p = gp((const unsigned char *)name);
    return p;
}

static gboolean load_gl_fns(GlFns *f) {
    static const char *libs[] = {"libGL.so.1", "libGL.so", "libGLX.so.0", NULL};
    void *h = NULL;
    for (int i = 0; libs[i] && !h; i++) h = dlopen(libs[i], RTLD_LAZY);
    if (!h) return FALSE;
    FnGetProc gp = (FnGetProc)dlsym(h, "glXGetProcAddressARB");
    if (!gp) gp = (FnGetProc)dlsym(h, "glXGetProcAddress");

#define REQ(field, sym) do { \
    *(void **)&f->field = gl_sym(h, gp, sym); \
    if (!f->field) return FALSE; \
} while (0)
#define OPT(field, sym, alt) do { \
    *(void **)&f->field = gl_sym(h, gp, sym); \
    if (!f->field && alt) *(void **)&f->field = gl_sym(h, gp, alt); \
} while (0)

    REQ(Viewport, "glViewport");
    REQ(Disable, "glDisable");
    REQ(PixelStorei, "glPixelStorei");
    REQ(ReadPixels, "glReadPixels");
    REQ(GetIntegerv, "glGetIntegerv");
    REQ(GetString, "glGetString");
    REQ(GenTextures, "glGenTextures");
    REQ(DeleteTextures, "glDeleteTextures");
    REQ(BindTexture, "glBindTexture");
    REQ(TexParameteri, "glTexParameteri");
    REQ(TexImage2D, "glTexImage2D");
    REQ(ActiveTexture, "glActiveTexture");
    REQ(DrawArrays, "glDrawArrays");
    REQ(CreateShader, "glCreateShader");
    REQ(DeleteShader, "glDeleteShader");
    REQ(ShaderSource, "glShaderSource");
    REQ(CompileShader, "glCompileShader");
    REQ(GetShaderiv, "glGetShaderiv");
    REQ(GetShaderInfoLog, "glGetShaderInfoLog");
    REQ(CreateProgram, "glCreateProgram");
    REQ(DeleteProgram, "glDeleteProgram");
    REQ(AttachShader, "glAttachShader");
    REQ(LinkProgram, "glLinkProgram");
    REQ(GetProgramiv, "glGetProgramiv");
    REQ(UseProgram, "glUseProgram");
    REQ(GetUniformLocation, "glGetUniformLocation");
    REQ(GetAttribLocation, "glGetAttribLocation");
    REQ(BindAttribLocation, "glBindAttribLocation");
    REQ(Uniform1f, "glUniform1f");
    REQ(Uniform2f, "glUniform2f");
    REQ(Uniform4f, "glUniform4f");
    REQ(Uniform1i, "glUniform1i");
    REQ(GenBuffers, "glGenBuffers");
    REQ(DeleteBuffers, "glDeleteBuffers");
    REQ(BindBuffer, "glBindBuffer");
    REQ(BufferData, "glBufferData");
    REQ(EnableVertexAttribArray, "glEnableVertexAttribArray");
    REQ(VertexAttribPointer, "glVertexAttribPointer");
    OPT(GenVertexArrays, "glGenVertexArrays", "glGenVertexArraysAPPLE");
    OPT(DeleteVertexArrays, "glDeleteVertexArrays", "glDeleteVertexArraysAPPLE");
    OPT(BindVertexArray, "glBindVertexArray", "glBindVertexArrayAPPLE");
    OPT(GenFramebuffers, "glGenFramebuffers", "glGenFramebuffersEXT");
    OPT(DeleteFramebuffers, "glDeleteFramebuffers", "glDeleteFramebuffersEXT");
    OPT(BindFramebuffer, "glBindFramebuffer", "glBindFramebufferEXT");
    OPT(FramebufferTexture2D, "glFramebufferTexture2D", "glFramebufferTexture2DEXT");
    OPT(CheckFramebufferStatus, "glCheckFramebufferStatus", "glCheckFramebufferStatusEXT");

#undef REQ
#undef OPT
    return f->GenFramebuffers && f->BindFramebuffer && f->FramebufferTexture2D && f->CheckFramebufferStatus;
}

static char *str_replace_all(const char *src, const char *from, const char *to) {
    GString *out = g_string_sized_new(strlen(src) + 64);
    size_t flen = strlen(from);
    const char *p = src, *hit;
    while ((hit = strstr(p, from))) {
        g_string_append_len(out, p, (gssize)(hit - p));
        g_string_append(out, to);
        p = hit + flen;
    }
    g_string_append(out, p);
    return g_string_free(out, FALSE);
}

/* Adapt our GLSL 120 `.frag` shaders cleanly to GLSL 150 Core when GtkGLArea
 * provides an OpenGL 3.2+ Core context (zero duplicate shader files). */
static char *adapt_frag_for_core(const char *raw) {
    char *a = str_replace_all(raw, "#version 120", "#version 150 core\nout vec4 _miraOut;");
    char *b = str_replace_all(a, "varying vec2 vUv;", "in vec2 vUv;");
    g_free(a);
    char *c = str_replace_all(b, "texture2D(", "texture(");
    g_free(b);
    char *d = str_replace_all(c, "gl_FragColor", "_miraOut");
    g_free(c);
    return d;
}

static GLuint compile_shader(GlFns *gl, GLenum kind, const char *src) {
    GLuint sh = gl->CreateShader(kind);
    if (!sh) return 0;
    gl->ShaderSource(sh, 1, &src, NULL);
    gl->CompileShader(sh);
    GLint ok = 0;
    gl->GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        GLsizei len = 0;
        gl->GetShaderInfoLog(sh, (GLsizei)sizeof log, &len, log);
        fprintf(stderr, "CardPicker live-gl shader compile error:\n%s\n", log);
        gl->DeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint link_program(LiveGl *lg, const char *fs_src) {
    GLuint vs = compile_shader(&lg->gl, GL_VERTEX_SHADER, lg->is_core ? VS_CORE : VS_COMPAT);
    if (!vs) return 0;
    GLuint fs = compile_shader(&lg->gl, GL_FRAGMENT_SHADER, fs_src);
    if (!fs) {
        lg->gl.DeleteShader(vs);
        return 0;
    }
    GLuint prog = lg->gl.CreateProgram();
    lg->gl.AttachShader(prog, vs);
    lg->gl.AttachShader(prog, fs);
    lg->gl.BindAttribLocation(prog, 0, "aPos");
    lg->gl.LinkProgram(prog);
    lg->gl.DeleteShader(vs);
    lg->gl.DeleteShader(fs);
    GLint ok = 0;
    lg->gl.GetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        lg->gl.DeleteProgram(prog);
        return 0;
    }
    return prog;
}

static int get_or_compile_frag(LiveGl *lg, const char *frag_path) {
    if (!frag_path || !*frag_path || !lg->gl_ok) return -1;
    for (int i = 0; i < PROG_CACHE_MAX; i++) {
        if (lg->cache[i].path && strcmp(lg->cache[i].path, frag_path) == 0) {
            lg->cache[i].last_use = ++lg->use_seq;
            return i;
        }
    }
    char *raw = NULL;
    if (!g_file_get_contents(frag_path, &raw, NULL, NULL) || !raw) return -1;
    char *src = lg->is_core ? adapt_frag_for_core(raw) : raw;
    GLuint prog = link_program(lg, src);
    if (lg->is_core) g_free(src);
    g_free(raw);
    if (!prog) return -1;

    /* Pick free slot or least-recently-used slot not currently bound by active slots. */
    int pick = -1;
    guint64 oldest = UINT64_MAX;
    for (int i = 0; i < PROG_CACHE_MAX; i++) {
        if (!lg->cache[i].prog) { pick = i; break; }
        if (lg->slots[0].prog_idx == i || lg->slots[1].prog_idx == i) continue;
        if (lg->cache[i].last_use < oldest) { oldest = lg->cache[i].last_use; pick = i; }
    }
    if (pick < 0) pick = 0;
    if (lg->cache[pick].prog) lg->gl.DeleteProgram(lg->cache[pick].prog);
    g_free(lg->cache[pick].path);

    CachedProg *cp = &lg->cache[pick];
    cp->path = g_strdup(frag_path);
    cp->prog = prog;
    cp->u_tex = lg->gl.GetUniformLocation(prog, "uTex");
    cp->u_res = lg->gl.GetUniformLocation(prog, "uRes");
    cp->u_cover = lg->gl.GetUniformLocation(prog, "uCover");
    cp->u_time = lg->gl.GetUniformLocation(prog, "uTime");
    cp->u_mouse = lg->gl.GetUniformLocation(prog, "uMouse");
    cp->u_cell = lg->gl.GetUniformLocation(prog, "uCell");
    cp->last_use = ++lg->use_seq;

    lg->gl.UseProgram(prog);
    if (cp->u_tex >= 0) lg->gl.Uniform1i(cp->u_tex, 0);
    if (cp->u_res >= 0) lg->gl.Uniform2f(cp->u_res, (float)lg->fw, (float)lg->fh);
    if (cp->u_cell >= 0) lg->gl.Uniform1f(cp->u_cell, 1.0f);
    return pick;
}

static gboolean init_slot_fbo(LiveGl *lg, Slot *s) {
    lg->gl.GenTextures(1, &s->fbo_tex);
    lg->gl.BindTexture(GL_TEXTURE_2D, s->fbo_tex);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    lg->gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, lg->fw, lg->fh, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    lg->gl.GenFramebuffers(1, &s->fbo);
    lg->gl.BindFramebuffer(GL_FRAMEBUFFER, s->fbo);
    lg->gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s->fbo_tex, 0);
    GLenum st = lg->gl.CheckFramebufferStatus(GL_FRAMEBUFFER);
    return st == GL_FRAMEBUFFER_COMPLETE;
}

static void render_slot_fbo(LiveGl *lg, Slot *s, float elapsed) {
    if (!s->valid || s->prog_idx < 0) return;
    CachedProg *cp = &lg->cache[s->prog_idx];
    if (!cp->prog) return;

    lg->gl.BindFramebuffer(GL_FRAMEBUFFER, s->fbo);
    lg->gl.Viewport(0, 0, lg->fw, lg->fh);
    lg->gl.UseProgram(cp->prog);
    lg->gl.ActiveTexture(GL_TEXTURE0);
    lg->gl.BindTexture(GL_TEXTURE_2D, s->img_tex);
    if (cp->u_res >= 0) lg->gl.Uniform2f(cp->u_res, (float)lg->fw, (float)lg->fh);
    if (cp->u_cell >= 0) lg->gl.Uniform1f(cp->u_cell, 1.0f);
    if (cp->u_cover >= 0)
        lg->gl.Uniform4f(cp->u_cover, s->cover[0], s->cover[1], s->cover[2], s->cover[3]);
    if (cp->u_time >= 0) lg->gl.Uniform1f(cp->u_time, elapsed);
    if (cp->u_mouse >= 0) lg->gl.Uniform2f(cp->u_mouse, lg->smooth_mx, lg->smooth_my);
    lg->gl.DrawArrays(GL_TRIANGLES, 0, 3);
    s->fbo_dirty = FALSE;
}

static float ease_develop(double t) {
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
    double smooth = t * t * (3.0 - 2.0 * t);
    double inv = 1.0 - t;
    double cubic_out = 1.0 - inv * inv * inv;
    return (float)(0.80 * smooth + 0.20 * cubic_out);
}

static void stop_timers(LiveGl *lg) {
    if (lg->tick_id && lg->area) {
        gtk_widget_remove_tick_callback(lg->area, lg->tick_id);
        lg->tick_id = 0;
    }
    if (lg->live_timer) {
        g_source_remove(lg->live_timer);
        lg->live_timer = 0;
    }
}

static gboolean on_live_timer(gpointer data) {
    LiveGl *lg = data;
    if (!lg->gl_ok || !lg->has_slot || lg->slots[lg->cur_slot].prog_idx < 0) {
        lg->live_timer = 0;
        return G_SOURCE_REMOVE;
    }
    lg->slots[lg->cur_slot].fbo_dirty = TRUE;
    gtk_gl_area_queue_render(GTK_GL_AREA(lg->area));
    return G_SOURCE_CONTINUE;
}

static void sync_pacing(LiveGl *lg);

static gboolean on_trans_tick(GtkWidget *_w, GdkFrameClock *_clk, gpointer data) {
    LiveGl *lg = data;
    if (!lg->gl_ok || lg->trans_done) {
        lg->tick_id = 0;
        sync_pacing(lg);
        return G_SOURCE_REMOVE;
    }
    /* Throttle FBO shader re-renders to 15 FPS during transitions (skip completely in static_only mode!). */
    gint64 now = g_get_monotonic_time();
    if (!lg->static_only && now - lg->last_fbo_us >= (gint64)lg->live_interval_ms * 1000) {
        lg->last_fbo_us = now;
        lg->slots[0].fbo_dirty = TRUE;
        lg->slots[1].fbo_dirty = TRUE;
    }
    gtk_gl_area_queue_render(GTK_GL_AREA(lg->area));

    double elapsed_ms = (now - lg->trans_start_us) / 1000.0;
    double dur = lg->trans_dur_ms > 0 ? (double)lg->trans_dur_ms : 1.0;
    if ((elapsed_ms + 16.0) / dur >= 1.0) {
        lg->trans_done = TRUE;
        lg->tick_id = 0;
        sync_pacing(lg);
        if (lg->finish_cb) {
            void (*cb)(void *) = lg->finish_cb;
            void *u = lg->finish_user;
            lg->finish_cb = NULL;
            lg->finish_user = NULL;
            cb(u);
        }
        return G_SOURCE_REMOVE;
    }
    return G_SOURCE_CONTINUE;
}

static void sync_pacing(LiveGl *lg) {
    if (!lg->gl_ok || !lg->has_slot) {
        stop_timers(lg);
        return;
    }
    if (!lg->trans_done) {
        if (lg->live_timer) { g_source_remove(lg->live_timer); lg->live_timer = 0; }
        if (!lg->tick_id)
            lg->tick_id = gtk_widget_add_tick_callback(lg->area, on_trans_tick, lg, NULL);
    } else if (!lg->static_only && lg->slots[lg->cur_slot].prog_idx >= 0) {
        if (lg->tick_id) { gtk_widget_remove_tick_callback(lg->area, lg->tick_id); lg->tick_id = 0; }
        if (!lg->live_timer)
            lg->live_timer = g_timeout_add((guint)lg->live_interval_ms, on_live_timer, lg);
    } else {
        /* Custom static wallpaper or static Theme Picker card: 0 FPS, 0.00% CPU, 0.00% GPU! */
        stop_timers(lg);
    }
}

static void on_realize(GtkGLArea *area, gpointer data) {
    LiveGl *lg = data;
    gtk_gl_area_make_current(area);
    if (gtk_gl_area_get_error(area) != NULL) return;
    if (!load_gl_fns(&lg->gl)) return;

    GLint maj = 0;
    lg->gl.GetIntegerv(GL_MAJOR_VERSION, &maj);
    if (maj == 0) {
        const char *ver = (const char *)lg->gl.GetString(GL_VERSION);
        if (ver) maj = atoi(ver);
    }
    lg->is_core = (maj >= 3);

    /* Save GtkGLArea's bound FBO before initializing our offscreen cell-FBOs. */
    GLint prev_fbo = 0;
    lg->gl.GetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

    GString *comp_fs = g_string_new(lg->is_core
        ? "#version 150 core\nin vec2 vUv;\nout vec4 _out;\n#define SAMPLE_TEX texture\n#define OUT_COLOR _out\n"
        : "#version 120\nvarying vec2 vUv;\n#define SAMPLE_TEX texture2D\n#define OUT_COLOR gl_FragColor\n");
    g_string_append(comp_fs, COMP_FS_BODY);
    lg->comp_prog = link_program(lg, comp_fs->str);
    g_string_free(comp_fs, TRUE);
    if (!lg->comp_prog) return;

    if (lg->is_core && lg->gl.GenVertexArrays && lg->gl.BindVertexArray) {
        lg->gl.GenVertexArrays(1, &lg->vao);
        lg->gl.BindVertexArray(lg->vao);
    }
    static const float verts[6] = {-1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f};
    lg->gl.GenBuffers(1, &lg->vbo);
    lg->gl.BindBuffer(GL_ARRAY_BUFFER, lg->vbo);
    lg->gl.BufferData(GL_ARRAY_BUFFER, (GLsizeiptr)sizeof verts, verts, GL_STATIC_DRAW);
    lg->gl.EnableVertexAttribArray(0);
    lg->gl.VertexAttribPointer(0, 2, GL_FLOAT, 0, 0, NULL);

    lg->c_src = lg->gl.GetUniformLocation(lg->comp_prog, "uSrc");
    lg->c_tgt = lg->gl.GetUniformLocation(lg->comp_prog, "uTgt");
    lg->c_src_cover = lg->gl.GetUniformLocation(lg->comp_prog, "uSrcCover");
    lg->c_tgt_cover = lg->gl.GetUniformLocation(lg->comp_prog, "uTgtCover");
    lg->c_src_fbo = lg->gl.GetUniformLocation(lg->comp_prog, "uSrcFbo");
    lg->c_tgt_fbo = lg->gl.GetUniformLocation(lg->comp_prog, "uTgtFbo");
    lg->c_mon = lg->gl.GetUniformLocation(lg->comp_prog, "uMonSize");
    lg->c_win = lg->gl.GetUniformLocation(lg->comp_prog, "uWinSize");
    lg->c_top = lg->gl.GetUniformLocation(lg->comp_prog, "uTop");
    lg->c_eased = lg->gl.GetUniformLocation(lg->comp_prog, "uEased");
    lg->c_lmax = lg->gl.GetUniformLocation(lg->comp_prog, "uLumaMax");
    lg->c_lspan = lg->gl.GetUniformLocation(lg->comp_prog, "uLumaSpan");
    lg->c_corner_r = lg->gl.GetUniformLocation(lg->comp_prog, "uCornerR");

    lg->gl.UseProgram(lg->comp_prog);
    if (lg->c_src >= 0) lg->gl.Uniform1i(lg->c_src, 0);
    if (lg->c_tgt >= 0) lg->gl.Uniform1i(lg->c_tgt, 1);
    if (lg->c_mon >= 0) lg->gl.Uniform2f(lg->c_mon, (float)lg->mon_w, (float)lg->mon_h);
    if (lg->c_win >= 0) lg->gl.Uniform2f(lg->c_win, (float)lg->win_w, (float)lg->win_h);
    if (lg->c_top >= 0) lg->gl.Uniform1f(lg->c_top, (float)lg->top_px);
    if (lg->c_corner_r >= 0) lg->gl.Uniform1f(lg->c_corner_r, lg->corner_r);

    if (!init_slot_fbo(lg, &lg->slots[0]) || !init_slot_fbo(lg, &lg->slots[1])) {
        lg->gl.BindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
        return;
    }
    lg->gl.BindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
    lg->gl.Disable(GL_DEPTH_TEST);
    lg->gl.Disable(GL_BLEND);

    /* Seed parallax immediately from current cursor position on monitor so
     * opening Super+Alt+W matches misa-wall's uMouse with zero visual nudge. */
    if (!lg->static_only) {
        GdkDisplay *dpy = gtk_widget_get_display(GTK_WIDGET(area));
        GdkSeat *seat = dpy ? gdk_display_get_default_seat(dpy) : NULL;
        GdkDevice *ptr = seat ? gdk_seat_get_pointer(seat) : NULL;
        if (ptr) {
            int px = 0, py = 0;
            gdk_device_get_position(ptr, NULL, &px, &py);
            GdkMonitor *mon = gdk_display_get_monitor_at_point(dpy, px, py);
            GdkRectangle mr = {0, 0, lg->mon_w, lg->mon_h};
            if (mon) gdk_monitor_get_geometry(mon, &mr);
            if (mr.width > 0 && mr.height > 0) {
                float nx = ((float)(px - mr.x) / (float)mr.width) * 2.0f - 1.0f;
                float ny = ((float)(py - mr.y) / (float)mr.height) * 2.0f - 1.0f;
                if (nx < -1.0f) nx = -1.0f; else if (nx > 1.0f) nx = 1.0f;
                if (ny < -1.0f) ny = -1.0f; else if (ny > 1.0f) ny = 1.0f;
                lg->target_mx = lg->smooth_mx = nx;
                lg->target_my = lg->smooth_my = ny;
            }
        }
    }
    lg->gl_ok = TRUE;
}

static void on_unrealize(GtkGLArea *area, gpointer data) {
    LiveGl *lg = data;
    stop_timers(lg);
    if (!lg->gl_ok) return;
    gtk_gl_area_make_current(area);
    if (gtk_gl_area_get_error(area) != NULL) return;
    for (int i = 0; i < 2; i++) {
        if (lg->slots[i].img_tex) lg->gl.DeleteTextures(1, &lg->slots[i].img_tex);
        if (lg->slots[i].fbo_tex) lg->gl.DeleteTextures(1, &lg->slots[i].fbo_tex);
        if (lg->slots[i].fbo && lg->gl.DeleteFramebuffers)
            lg->gl.DeleteFramebuffers(1, &lg->slots[i].fbo);
        g_clear_pointer(&lg->slots[i].frag_path, g_free);
    }
    if (lg->thumb_img_tex) lg->gl.DeleteTextures(1, &lg->thumb_img_tex);
    if (lg->thumb_fbo_tex) lg->gl.DeleteTextures(1, &lg->thumb_fbo_tex);
    if (lg->thumb_fbo && lg->gl.DeleteFramebuffers)
        lg->gl.DeleteFramebuffers(1, &lg->thumb_fbo);
    lg->thumb_img_tex = lg->thumb_fbo_tex = lg->thumb_fbo = 0;
    lg->thumb_w = lg->thumb_h = 0;
    for (int i = 0; i < PROG_CACHE_MAX; i++) {
        if (lg->cache[i].prog) lg->gl.DeleteProgram(lg->cache[i].prog);
        g_clear_pointer(&lg->cache[i].path, g_free);
    }
    if (lg->comp_prog) lg->gl.DeleteProgram(lg->comp_prog);
    if (lg->vbo) lg->gl.DeleteBuffers(1, &lg->vbo);
    if (lg->vao && lg->gl.DeleteVertexArrays) lg->gl.DeleteVertexArrays(1, &lg->vao);
    lg->gl_ok = FALSE;
}

static gboolean on_render(GtkGLArea *area, GdkGLContext *_ctx, gpointer data) {
    LiveGl *lg = data;
    if (!lg->gl_ok || !lg->has_slot) return TRUE;

    GLint gtk_fbo = 0;
    lg->gl.GetIntegerv(GL_FRAMEBUFFER_BINDING, &gtk_fbo);
    if (lg->is_core && lg->vao && lg->gl.BindVertexArray)
        lg->gl.BindVertexArray(lg->vao);
    else {
        lg->gl.BindBuffer(GL_ARRAY_BUFFER, lg->vbo);
        lg->gl.EnableVertexAttribArray(0);
        lg->gl.VertexAttribPointer(0, 2, GL_FLOAT, 0, 0, NULL);
    }
    lg->gl.Disable(GL_SCISSOR_TEST);
    lg->gl.Disable(GL_BLEND);

    lg->smooth_mx += (lg->target_mx - lg->smooth_mx) * 0.16f;
    lg->smooth_my += (lg->target_my - lg->smooth_my) * 0.16f;
    gint64 now = g_get_monotonic_time();
    float elapsed = lg->static_only ? lg->fixed_time : (float)((now - lg->start_us) / 1000000.0);

    Slot *tgt = &lg->slots[lg->cur_slot];
    Slot *src = &lg->slots[lg->cur_slot ^ 1];

    /* 1. Render active live cell-FBO(s) only when dirty (15 FPS) */
    if (tgt->valid && tgt->prog_idx >= 0 && tgt->fbo_dirty)
        render_slot_fbo(lg, tgt, elapsed);
    if (!lg->trans_done && src->valid && src->prog_idx >= 0 && src->fbo_dirty)
        render_slot_fbo(lg, src, elapsed);

    /* 2. Composite into GtkGLArea's bound FBO (GPU Luma-Halftone Develop or direct pass) */
    int scale = gtk_widget_get_scale_factor(GTK_WIDGET(area));
    if (scale < 1) scale = 1;
    lg->gl.BindFramebuffer(GL_FRAMEBUFFER, (GLuint)gtk_fbo);
    lg->gl.Viewport(0, 0, lg->win_w * scale, lg->win_h * scale);
    lg->gl.UseProgram(lg->comp_prog);

    float eased = 1.0f;
    if (!lg->trans_done && src->valid) {
        double dt_ms = (now - lg->trans_start_us) / 1000.0;
        double dur = lg->trans_dur_ms > 0 ? (double)lg->trans_dur_ms : 1.0;
        eased = ease_develop((dt_ms + 16.0) / dur);
    }

    lg->gl.ActiveTexture(GL_TEXTURE0);
    lg->gl.BindTexture(GL_TEXTURE_2D,
                       (src->valid && src->prog_idx >= 0) ? src->fbo_tex
                       : (src->valid ? src->img_tex : tgt->img_tex));
    lg->gl.ActiveTexture(GL_TEXTURE1);
    lg->gl.BindTexture(GL_TEXTURE_2D, tgt->prog_idx >= 0 ? tgt->fbo_tex : tgt->img_tex);

    if (lg->c_src_cover >= 0 && src->valid)
        lg->gl.Uniform4f(lg->c_src_cover, src->cover[0], src->cover[1], src->cover[2], src->cover[3]);
    if (lg->c_tgt_cover >= 0)
        lg->gl.Uniform4f(lg->c_tgt_cover, tgt->cover[0], tgt->cover[1], tgt->cover[2], tgt->cover[3]);
    if (lg->c_src_fbo >= 0)
        lg->gl.Uniform1f(lg->c_src_fbo, (src->valid && src->prog_idx >= 0) ? 1.0f : 0.0f);
    if (lg->c_tgt_fbo >= 0)
        lg->gl.Uniform1f(lg->c_tgt_fbo, tgt->prog_idx >= 0 ? 1.0f : 0.0f);
    if (lg->c_eased >= 0) lg->gl.Uniform1f(lg->c_eased, eased);
    if (lg->c_lmax >= 0) lg->gl.Uniform1f(lg->c_lmax, lg->luma_max);
    if (lg->c_lspan >= 0) lg->gl.Uniform1f(lg->c_lspan, lg->luma_span);

    lg->gl.DrawArrays(GL_TRIANGLES, 0, 3);
    return TRUE;
}

static void upload_slot_texture(LiveGl *lg, Slot *s, const guchar *rgb, int w, int h,
                                const char *frag_path) {
    gtk_gl_area_make_current(GTK_GL_AREA(lg->area));
    if (gtk_gl_area_get_error(GTK_GL_AREA(lg->area)) != NULL) return;

    if (!s->img_tex) lg->gl.GenTextures(1, &s->img_tex);
    lg->gl.ActiveTexture(GL_TEXTURE0);
    lg->gl.BindTexture(GL_TEXTURE_2D, s->img_tex);
    lg->gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    lg->gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb);

    s->img_w = w;
    s->img_h = h;
    compute_cover(lg->mon_w, lg->mon_h, w, h, s->cover);
    g_free(s->frag_path);
    s->frag_path = frag_path ? g_strdup(frag_path) : NULL;
    s->prog_idx = frag_path ? get_or_compile_frag(lg, frag_path) : -1;
    s->valid = TRUE;
    s->fbo_dirty = TRUE;
}

static char *runtime_disp_path(const char *suffix);
static gint64 shared_epoch_us(void);

static LiveGl *live_gl_create_internal(int mon_w, int mon_h, int top_px, int win_w, int win_h, float corner_r) {
    LiveGl *lg = g_new0(LiveGl, 1);
    lg->mon_w = mon_w > 0 ? mon_w : 1920;
    lg->mon_h = mon_h > 0 ? mon_h : 1080;
    lg->top_px = top_px > 0 ? top_px : 0;
    lg->win_w = win_w > 0 ? win_w : lg->mon_w;
    lg->win_h = win_h > 0 ? win_h : lg->mon_h;
    lg->corner_r = corner_r > 0.0f ? corner_r : 0.0f;
    lg->cell_px = compute_cell_px(lg->mon_h);
    lg->fw = (lg->mon_w + lg->cell_px - 1) / lg->cell_px;
    lg->fh = (lg->mon_h + lg->cell_px - 1) / lg->cell_px;
    lg->slots[0].prog_idx = -1;
    lg->slots[1].prog_idx = -1;
    lg->trans_done = TRUE;
    lg->luma_max = 1.0f;
    lg->luma_span = 1.0f;
    lg->start_us = shared_epoch_us();
    lg->live_interval_ms = default_live_interval_ms();

    lg->area = gtk_gl_area_new();
    gtk_gl_area_set_has_alpha(GTK_GL_AREA(lg->area), lg->corner_r > 0.5f);
    gtk_gl_area_set_has_depth_buffer(GTK_GL_AREA(lg->area), FALSE);
    gtk_gl_area_set_has_stencil_buffer(GTK_GL_AREA(lg->area), FALSE);
    gtk_gl_area_set_auto_render(GTK_GL_AREA(lg->area), FALSE);
    gtk_widget_set_size_request(lg->area, lg->win_w, lg->win_h);
    g_signal_connect(lg->area, "realize", G_CALLBACK(on_realize), lg);
    g_signal_connect(lg->area, "unrealize", G_CALLBACK(on_unrealize), lg);
    g_signal_connect(lg->area, "render", G_CALLBACK(on_render), lg);
    return lg;
}

LiveGl *live_gl_new(int mon_w, int mon_h, int top_px, int win_w, int win_h) {
    return live_gl_create_internal(mon_w, mon_h, top_px, win_w, win_h, 0.0f);
}

LiveGl *live_gl_new_rounded(int w, int h, float corner_r) {
    return live_gl_create_internal(w, h, 0, w, h, corner_r);
}

void live_gl_set_static_frame(LiveGl *lg, gboolean static_only, float fixed_time) {
    if (!lg) return;
    lg->static_only = static_only;
    lg->fixed_time = fixed_time;
    sync_pacing(lg);
}

GtkWidget *live_gl_widget(LiveGl *lg) { return lg ? lg->area : NULL; }
gboolean live_gl_ready(const LiveGl *lg) { return lg && lg->gl_ok; }
gboolean live_gl_trans_done(const LiveGl *lg) { return !lg || lg->trans_done; }
gboolean live_gl_target_is_live(const LiveGl *lg) {
    return lg && lg->has_slot && lg->slots[lg->cur_slot].prog_idx >= 0;
}

void live_gl_set_finish_cb(LiveGl *lg, void (*cb)(void *), void *user) {
    if (!lg) return;
    lg->finish_cb = cb;
    lg->finish_user = user;
}

void live_gl_set_mouse(LiveGl *lg, float nx, float ny) {
    if (!lg) return;
    lg->target_mx = nx;
    lg->target_my = ny;
}

void live_gl_set_wallpaper(LiveGl *lg, const guchar *rgb, int w, int h,
                           const char *frag_path, float luma_max, float luma_span,
                           gboolean animate, int dur_ms) {
    if (!lg || !lg->gl_ok || !rgb || w <= 0 || h <= 0) return;

    if (animate && lg->has_slot) {
        /* Flip active slot: old target becomes live/static source, new uploads into other slot! */
        lg->cur_slot ^= 1;
        upload_slot_texture(lg, &lg->slots[lg->cur_slot], rgb, w, h, frag_path);
        lg->luma_max = luma_max;
        lg->luma_span = luma_span;
        lg->trans_dur_ms = dur_ms > 0 ? dur_ms : 460;
        lg->trans_start_us = g_get_monotonic_time();
        lg->last_fbo_us = 0;
        lg->trans_done = FALSE;
    } else {
        upload_slot_texture(lg, &lg->slots[lg->cur_slot], rgb, w, h, frag_path);
        lg->has_slot = TRUE;
        lg->trans_done = TRUE;
        lg->luma_max = luma_max;
        lg->luma_span = luma_span;
    }

    sync_pacing(lg);
    gtk_gl_area_queue_render(GTK_GL_AREA(lg->area));
}

void live_gl_refine_target(LiveGl *lg, const guchar *rgb, int w, int h,
                           const char *frag_path, float luma_max, float luma_span) {
    if (!lg || !lg->gl_ok || !rgb || w <= 0 || h <= 0) return;
    upload_slot_texture(lg, &lg->slots[lg->cur_slot], rgb, w, h, frag_path);
    lg->luma_max = luma_max;
    lg->luma_span = luma_span;
    sync_pacing(lg);
    gtk_gl_area_queue_render(GTK_GL_AREA(lg->area));
}

guchar *live_gl_render_thumb(LiveGl *lg, const guchar *src_rgb, int src_w, int src_h,
                             const char *frag_path, int tw, int th, float fixed_time) {
    if (!lg || !lg->gl_ok || !src_rgb || src_w <= 0 || src_h <= 0 ||
        !frag_path || !*frag_path || tw <= 0 || th <= 0)
        return NULL;

    gtk_gl_area_make_current(GTK_GL_AREA(lg->area));
    if (gtk_gl_area_get_error(GTK_GL_AREA(lg->area)) != NULL) return NULL;

    GLint prev_fbo = 0;
    lg->gl.GetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);

    int pidx = get_or_compile_frag(lg, frag_path);
    if (pidx < 0 || !lg->cache[pidx].prog) {
        lg->gl.BindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
        return NULL;
    }

    if (!lg->thumb_fbo || lg->thumb_w != tw || lg->thumb_h != th) {
        if (!lg->thumb_fbo_tex) lg->gl.GenTextures(1, &lg->thumb_fbo_tex);
        lg->gl.BindTexture(GL_TEXTURE_2D, lg->thumb_fbo_tex);
        lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        lg->gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        if (!lg->thumb_fbo) lg->gl.GenFramebuffers(1, &lg->thumb_fbo);
        lg->gl.BindFramebuffer(GL_FRAMEBUFFER, lg->thumb_fbo);
        lg->gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                    GL_TEXTURE_2D, lg->thumb_fbo_tex, 0);
        if (lg->gl.CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            lg->gl.BindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
            return NULL;
        }
        lg->thumb_w = tw;
        lg->thumb_h = th;
    }

    if (!lg->thumb_img_tex) lg->gl.GenTextures(1, &lg->thumb_img_tex);
    lg->gl.ActiveTexture(GL_TEXTURE0);
    lg->gl.BindTexture(GL_TEXTURE_2D, lg->thumb_img_tex);
    lg->gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    lg->gl.TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    lg->gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGB, src_w, src_h, 0, GL_RGB, GL_UNSIGNED_BYTE, src_rgb);

    if (lg->is_core && lg->vao && lg->gl.BindVertexArray)
        lg->gl.BindVertexArray(lg->vao);
    else {
        lg->gl.BindBuffer(GL_ARRAY_BUFFER, lg->vbo);
        lg->gl.EnableVertexAttribArray(0);
        lg->gl.VertexAttribPointer(0, 2, GL_FLOAT, 0, 0, NULL);
    }
    lg->gl.Disable(GL_SCISSOR_TEST);
    lg->gl.Disable(GL_DEPTH_TEST);
    lg->gl.Disable(GL_BLEND);

    CachedProg *cp = &lg->cache[pidx];
    float cov[4];
    compute_cover(tw, th, src_w, src_h, cov);

    lg->gl.BindFramebuffer(GL_FRAMEBUFFER, lg->thumb_fbo);
    lg->gl.Viewport(0, 0, tw, th);
    lg->gl.UseProgram(cp->prog);
    if (cp->u_tex >= 0) lg->gl.Uniform1i(cp->u_tex, 0);
    if (cp->u_res >= 0) lg->gl.Uniform2f(cp->u_res, (float)tw, (float)th);
    if (cp->u_cell >= 0) lg->gl.Uniform1f(cp->u_cell, 1.0f);
    if (cp->u_cover >= 0) lg->gl.Uniform4f(cp->u_cover, cov[0], cov[1], cov[2], cov[3]);
    if (cp->u_time >= 0) lg->gl.Uniform1f(cp->u_time, fixed_time);
    if (cp->u_mouse >= 0) lg->gl.Uniform2f(cp->u_mouse, 0.0f, 0.0f);
    lg->gl.DrawArrays(GL_TRIANGLES, 0, 3);

    size_t row_bytes = (size_t)tw * 3;
    guchar *raw = g_malloc(row_bytes * (size_t)th);
    lg->gl.PixelStorei(GL_PACK_ALIGNMENT, 1);
    lg->gl.ReadPixels(0, 0, tw, th, GL_RGB, GL_UNSIGNED_BYTE, raw);

    if (cp->u_res >= 0) lg->gl.Uniform2f(cp->u_res, (float)lg->fw, (float)lg->fh);
    lg->gl.BindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);

    guchar *out = g_malloc(row_bytes * (size_t)th);
    for (int y = 0; y < th; y++)
        memcpy(out + (size_t)y * row_bytes, raw + (size_t)(th - 1 - y) * row_bytes, row_bytes);
    g_free(raw);
    return out;
}

void live_gl_stop(LiveGl *lg) {
    if (!lg) return;
    stop_timers(lg);
    lg->finish_cb = NULL;
    lg->finish_user = NULL;
}

void live_gl_destroy(LiveGl *lg) {
    if (!lg) return;
    stop_timers(lg);
    g_free(lg);
}

static char *runtime_disp_path(const char *suffix) {
    const char *run = g_getenv("XDG_RUNTIME_DIR");
    if (!run || !*run) run = "/tmp";
    const char *raw = g_getenv("DISPLAY");
    if (!raw || !*raw) raw = ":0";
    GString *disp = g_string_new("");
    for (const char *p = raw; *p; p++)
        g_string_append_c(disp, g_ascii_isalnum(*p) ? *p : '_');
    if (disp->len == 0) g_string_append_c(disp, '0');
    char *out = g_strdup_printf("%s/misa-wall-live-%s.%s", run, disp->str, suffix);
    g_string_free(disp, TRUE);
    return out;
}

/* Shared session-wide CLOCK_MONOTONIC shader epoch so misa-wall and CardPicker
 * always compute the exact same continuous uTime (zero reset on open/enter/esc). */
static gint64 shared_epoch_us(void) {
    gint64 now = g_get_monotonic_time();
    char *path = runtime_disp_path("epoch");
    char *txt = NULL;
    if (g_file_get_contents(path, &txt, NULL, NULL) && txt) {
        gint64 ep = g_ascii_strtoll(txt, NULL, 10);
        g_free(txt);
        if (ep > 0 && now >= ep && (now - ep) < G_GINT64_CONSTANT(86400000000)) {
            g_free(path);
            return ep;
        }
    }
    char buf[32];
    g_snprintf(buf, sizeof buf, "%" G_GINT64_FORMAT "\n", now);
    g_file_set_contents(path, buf, -1, NULL);
    g_free(path);
    return now;
}

void live_gl_pause_desktop_daemon(gboolean pause) {
    char *path = runtime_disp_path("pause");
    if (pause) {
        char buf[32];
        g_snprintf(buf, sizeof buf, "%d\n", (int)getpid());
        g_file_set_contents(path, buf, -1, NULL);
    } else {
        char *txt = NULL;
        if (g_file_get_contents(path, &txt, NULL, NULL) && txt) {
            if (atoi(txt) == (int)getpid()) unlink(path);
            g_free(txt);
        }
    }
    g_free(path);
}
