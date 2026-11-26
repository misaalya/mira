//! misa-wall: hyper-optimized X11 wallpaper engine (`awww`-inspired zero-GLX
//! BitPack-LZ4 delta + MIT-SHM daemon & Mira's original Luma-Halftone Develop
//! AVX2 transition), written from scratch in crate-less Rust.
//!
//!   misa-wall [--no-transition|--instant] [--static] [--duration MS] [--fps N] <IMAGE|MEDIA>
//!   misa-wall pause | unpause | toggle | query | clear-cache | --stop-live
//!   misa-wall --ensure-cache <IMAGE|MEDIA>
//!   misa-wall --bake-frame <IMAGE> <FRAG> <OUT.ppm> [TIME_SEC]
//!   misa-wall --bake-loop  <IMAGE> <FRAG> <OUT.mwa> [CELL_W CELL_H FPS FRAMES]
//!   misa-wall --bake-media <MEDIA> <OUT.mwa> [MAX_W MAX_H MAX_FPS MAX_FRAMES]
//!   misa-wall --thumb      <MEDIA> <OUT.ppm> [W H]
//!   misa-wall --selftest

mod baker;
mod codec;
mod daemon;

use std::env;
use std::f64::consts::TAU;
use std::ffi::{c_char, c_int, c_long, c_uchar, c_uint, c_ulong, c_void, CString};
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub type Display = c_void;
pub type Visual = c_void;
pub type GC = *mut c_void;
type GdkPixbuf = c_void;
type XID = c_ulong;
pub type Window = XID;
type Drawable = XID;
pub type Pixmap = XID;
pub type Atom = c_ulong;

pub const ZPIXMAP: c_int = 2;
pub const PROP_MODE_REPLACE: c_int = 0;
const RETAIN_PERMANENT: c_int = 1;
pub const XA_PIXMAP: Atom = 20;
pub const ALL_PLANES: c_ulong = !0;
const GDK_COLORSPACE_RGB: c_int = 0;
const GDK_INTERP_NEAREST: c_int = 0;
const GDK_INTERP_BILINEAR: c_int = 2;
pub const IPC_PRIVATE: c_int = 0;
pub const IPC_CREAT: c_int = 0o1000;
pub const IPC_RMID: c_int = 0;
pub const SIGINT: c_int = 2;
pub const SIGTERM: c_int = 15;

// Fixed-point constants for Mira's Luma-Halftone Develop (14-bit threshold space).
const W_BAYER: i32 = 6881; // 42.0%
const W_LUMA: i32 = 6226; // 38.0%
const W_ORG: i32 = 3276; // 20.0%
const HALF_ORG: i32 = W_ORG / 2; // 1638
pub const MAX_THRESH: i32 = W_BAYER + W_LUMA + W_ORG; // 16383
pub const FEATHER: i32 = 960; // ~5.8% optical bloom width
const RECIP: i32 = (256 * 65536 + FEATHER / 2) / FEATHER; // 17476

const BAYER4_RAW: [[i32; 4]; 4] = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
];

// Default seamless shader animation bake profile: 48 frames @ 12 FPS = 4.0s loop at 640x360.
const DEFAULT_CELL_W: u32 = 640;
const DEFAULT_CELL_H: u32 = 360;
const DEFAULT_ANIM_FPS: u32 = 12;
const DEFAULT_ANIM_FRAMES: u32 = 48;

// Default user live media (GIF/WebP/APNG/video) bake caps: sharp & tiny PSS!
const DEFAULT_MEDIA_MAX_W: u32 = 960;
const DEFAULT_MEDIA_MAX_H: u32 = 540;
const DEFAULT_MEDIA_FPS: u32 = 15;
const DEFAULT_MEDIA_FRAMES: u32 = 240;

#[repr(C)]
pub struct XImage {
    pub width: c_int,
    pub height: c_int,
    pub xoffset: c_int,
    pub format: c_int,
    pub data: *mut c_char,
    pub byte_order: c_int,
    pub bitmap_unit: c_int,
    pub bitmap_bit_order: c_int,
    pub bitmap_pad: c_int,
    pub depth: c_int,
    pub bytes_per_line: c_int,
    pub bits_per_pixel: c_int,
    pub red_mask: c_ulong,
    pub green_mask: c_ulong,
    pub blue_mask: c_ulong,
    pub obdata: *mut c_char,
    pub create_image: *mut c_void,
    pub destroy_image: Option<unsafe extern "C" fn(*mut XImage) -> c_int>,
    pub get_pixel: *mut c_void,
    pub put_pixel: *mut c_void,
    pub sub_image: *mut c_void,
    pub add_pixel: *mut c_void,
}

#[repr(C)]
pub struct XShmSegmentInfo {
    pub shmseg: c_ulong,
    pub shmid: c_int,
    pub shmaddr: *mut c_char,
    pub read_only: c_int,
}

#[repr(C)]
struct XRRMonitorInfo {
    name: Atom,
    primary: c_int,
    automatic: c_int,
    noutput: c_int,
    x: c_int,
    y: c_int,
    width: c_int,
    height: c_int,
    mwidth: c_int,
    mheight: c_int,
    outputs: *mut XID,
}

#[derive(Clone, Copy)]
pub struct MonRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[link(name = "X11")]
extern "C" {
    pub fn XOpenDisplay(name: *const c_char) -> *mut Display;
    pub fn XDefaultScreen(d: *mut Display) -> c_int;
    pub fn XDefaultRootWindow(d: *mut Display) -> Window;
    pub fn XDefaultVisual(d: *mut Display, scr: c_int) -> *mut Visual;
    pub fn XDefaultDepth(d: *mut Display, scr: c_int) -> c_int;
    pub fn XDisplayWidth(d: *mut Display, scr: c_int) -> c_int;
    pub fn XDisplayHeight(d: *mut Display, scr: c_int) -> c_int;
    pub fn XInternAtom(d: *mut Display, name: *const c_char, only_if_exists: c_int) -> Atom;
    pub fn XCreatePixmap(d: *mut Display, draw: Drawable, w: c_uint, h: c_uint, depth: c_uint) -> Pixmap;
    pub fn XFreePixmap(d: *mut Display, pmap: Pixmap) -> c_int;
    pub fn XCreateGC(d: *mut Display, draw: Drawable, mask: c_ulong, vals: *mut c_void) -> GC;
    pub fn XFreeGC(d: *mut Display, gc: GC) -> c_int;
    pub fn XSetWindowBackgroundPixmap(d: *mut Display, w: Window, pmap: Pixmap) -> c_int;
    fn XClearWindow(d: *mut Display, w: Window) -> c_int;
    fn XChangeProperty(
        d: *mut Display,
        w: Window,
        prop: Atom,
        typ: Atom,
        fmt: c_int,
        mode: c_int,
        data: *const c_uchar,
        nelements: c_int,
    ) -> c_int;
    pub fn XGetWindowProperty(
        d: *mut Display,
        w: Window,
        prop: Atom,
        offset: c_long,
        len: c_long,
        del: c_int,
        req_type: Atom,
        actual_type: *mut Atom,
        actual_fmt: *mut c_int,
        nitems: *mut c_ulong,
        bytes_after: *mut c_ulong,
        prop_ret: *mut *mut c_uchar,
    ) -> c_int;
    pub fn XFree(data: *mut c_void) -> c_int;
    fn XGetGeometry(
        d: *mut Display,
        draw: Drawable,
        root_ret: *mut Window,
        x_ret: *mut c_int,
        y_ret: *mut c_int,
        w_ret: *mut c_uint,
        h_ret: *mut c_uint,
        bw_ret: *mut c_uint,
        depth_ret: *mut c_uint,
    ) -> c_int;
    fn XSetCloseDownMode(d: *mut Display, mode: c_int) -> c_int;
    fn XKillClient(d: *mut Display, resource: XID) -> c_int;
    pub fn XSync(d: *mut Display, discard: c_int) -> c_int;
    fn XCloseDisplay(d: *mut Display) -> c_int;
    pub fn XSetErrorHandler(
        handler: Option<extern "C" fn(*mut Display, *mut c_void) -> c_int>,
    ) -> *mut c_void;
    pub fn XCreateImage(
        d: *mut Display,
        vis: *mut Visual,
        depth: c_uint,
        fmt: c_int,
        offset: c_int,
        data: *mut c_char,
        w: c_uint,
        h: c_uint,
        pad: c_int,
        bpl: c_int,
    ) -> *mut XImage;
    pub fn XPutImage(
        d: *mut Display,
        draw: Drawable,
        gc: GC,
        img: *mut XImage,
        sx: c_int,
        sy: c_int,
        dx: c_int,
        dy: c_int,
        w: c_uint,
        h: c_uint,
    ) -> c_int;
    fn XGetSubImage(
        d: *mut Display,
        draw: Drawable,
        x: c_int,
        y: c_int,
        w: c_uint,
        h: c_uint,
        plane_mask: c_ulong,
        fmt: c_int,
        dest_img: *mut XImage,
        dx: c_int,
        dy: c_int,
    ) -> *mut XImage;
    pub fn XGetImage(
        d: *mut Display,
        draw: Drawable,
        x: c_int,
        y: c_int,
        w: c_uint,
        h: c_uint,
        plane_mask: c_ulong,
        fmt: c_int,
    ) -> *mut XImage;
}

#[link(name = "Xext")]
extern "C" {
    pub fn XShmQueryExtension(d: *mut Display) -> c_int;
    pub fn XShmCreateImage(
        d: *mut Display,
        vis: *mut Visual,
        depth: c_uint,
        fmt: c_int,
        data: *mut c_char,
        shminfo: *mut XShmSegmentInfo,
        w: c_uint,
        h: c_uint,
    ) -> *mut XImage;
    pub fn XShmAttach(d: *mut Display, shminfo: *mut XShmSegmentInfo) -> c_int;
    pub fn XShmDetach(d: *mut Display, shminfo: *mut XShmSegmentInfo) -> c_int;
    pub fn XShmPutImage(
        d: *mut Display,
        draw: Drawable,
        gc: GC,
        img: *mut XImage,
        sx: c_int,
        sy: c_int,
        dx: c_int,
        dy: c_int,
        w: c_uint,
        h: c_uint,
        send_event: c_int,
    ) -> c_int;
    fn XShmGetImage(
        d: *mut Display,
        draw: Drawable,
        img: *mut XImage,
        x: c_int,
        y: c_int,
        plane_mask: c_ulong,
    ) -> c_int;
}

#[link(name = "Xrandr")]
extern "C" {
    fn XRRGetMonitors(
        d: *mut Display,
        w: Window,
        get_active: c_int,
        nmonitors: *mut c_int,
    ) -> *mut XRRMonitorInfo;
    fn XRRFreeMonitors(monitors: *mut XRRMonitorInfo);
}

// Lazy dlopen bindings for libgdk_pixbuf-2.0 + libgobject-2.0 so the persistent
// `--daemon` process NEVER maps GdkPixbuf/GLib/libwebp/libpng into RAM!
const RTLD_LAZY: c_int = 1;
const RTLD_LOCAL: c_int = 0;

type FnGdkPixbufGetFileInfo =
    unsafe extern "C" fn(*const c_char, *mut c_int, *mut c_int) -> *mut c_void;
type FnGdkPixbufNewFromFile =
    unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> *mut GdkPixbuf;
type FnGdkPixbufNewFromFileAtScale =
    unsafe extern "C" fn(*const c_char, c_int, c_int, c_int, *mut *mut c_void) -> *mut GdkPixbuf;
type FnGdkPixbufNew = unsafe extern "C" fn(c_int, c_int, c_int, c_int, c_int) -> *mut GdkPixbuf;
type FnGdkPixbufScale = unsafe extern "C" fn(
    *const GdkPixbuf,
    *mut GdkPixbuf,
    c_int,
    c_int,
    c_int,
    c_int,
    f64,
    f64,
    f64,
    f64,
    c_int,
);
type FnGdkPixbufGetInt = unsafe extern "C" fn(*const GdkPixbuf) -> c_int;
type FnGdkPixbufGetPixels = unsafe extern "C" fn(*const GdkPixbuf) -> *const c_uchar;
type FnGObjectUnref = unsafe extern "C" fn(*mut c_void);

struct PixbufApi {
    get_file_info: FnGdkPixbufGetFileInfo,
    new_from_file: FnGdkPixbufNewFromFile,
    new_from_file_at_scale: FnGdkPixbufNewFromFileAtScale,
    new_buf: FnGdkPixbufNew,
    scale: FnGdkPixbufScale,
    get_width: FnGdkPixbufGetInt,
    get_height: FnGdkPixbufGetInt,
    get_rowstride: FnGdkPixbufGetInt,
    get_n_channels: FnGdkPixbufGetInt,
    get_pixels: FnGdkPixbufGetPixels,
    unref: FnGObjectUnref,
}

unsafe fn load_pixbuf_api() -> Option<&'static PixbufApi> {
    static API: std::sync::OnceLock<Option<PixbufApi>> = std::sync::OnceLock::new();
    API.get_or_init(|| {
        let mut lib_pb = null_mut();
        for soname in ["libgdk_pixbuf-2.0.so.0", "libgdk_pixbuf-2.0.so"] {
            let cs = CString::new(soname).unwrap();
            lib_pb = dlopen(cs.as_ptr(), RTLD_LAZY | RTLD_LOCAL);
            if !lib_pb.is_null() {
                break;
            }
        }
        if lib_pb.is_null() {
            return None;
        }
        let mut lib_go = null_mut();
        for soname in ["libgobject-2.0.so.0", "libgobject-2.0.so"] {
            let cs = CString::new(soname).unwrap();
            lib_go = dlopen(cs.as_ptr(), RTLD_LAZY | RTLD_LOCAL);
            if !lib_go.is_null() {
                break;
            }
        }
        if lib_go.is_null() {
            lib_go = lib_pb;
        }
        macro_rules! sym {
            ($lib:expr, $name:expr, $ty:ty) => {{
                let cs = CString::new($name).unwrap();
                let p = dlsym($lib, cs.as_ptr());
                if p.is_null() {
                    return None;
                }
                std::mem::transmute::<*mut c_void, $ty>(p)
            }};
        }
        Some(PixbufApi {
            get_file_info: sym!(lib_pb, "gdk_pixbuf_get_file_info", FnGdkPixbufGetFileInfo),
            new_from_file: sym!(lib_pb, "gdk_pixbuf_new_from_file", FnGdkPixbufNewFromFile),
            new_from_file_at_scale: sym!(lib_pb, "gdk_pixbuf_new_from_file_at_scale", FnGdkPixbufNewFromFileAtScale),
            new_buf: sym!(lib_pb, "gdk_pixbuf_new", FnGdkPixbufNew),
            scale: sym!(lib_pb, "gdk_pixbuf_scale", FnGdkPixbufScale),
            get_width: sym!(lib_pb, "gdk_pixbuf_get_width", FnGdkPixbufGetInt),
            get_height: sym!(lib_pb, "gdk_pixbuf_get_height", FnGdkPixbufGetInt),
            get_rowstride: sym!(lib_pb, "gdk_pixbuf_get_rowstride", FnGdkPixbufGetInt),
            get_n_channels: sym!(lib_pb, "gdk_pixbuf_get_n_channels", FnGdkPixbufGetInt),
            get_pixels: sym!(lib_pb, "gdk_pixbuf_get_pixels", FnGdkPixbufGetPixels),
            unref: sym!(lib_go, "g_object_unref", FnGObjectUnref),
        })
    })
    .as_ref()
}

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn shmget(key: c_int, size: usize, shmflg: c_int) -> c_int;
    fn shmat(shmid: c_int, shmaddr: *const c_void, shmflg: c_int) -> *mut c_void;
    fn shmdt(shmaddr: *const c_void) -> c_int;
    fn shmctl(shmid: c_int, cmd: c_int, buf: *mut c_void) -> c_int;
    pub fn signal(sig: c_int, handler: extern "C" fn(c_int)) -> usize;
    pub fn kill(pid: c_int, sig: c_int) -> c_int;
    pub fn getpid() -> c_int;
    fn _exit(status: c_int) -> !;
}

pub static STOP: AtomicBool = AtomicBool::new(false);
pub static X_ERR: AtomicBool = AtomicBool::new(false);

pub extern "C" fn on_sig(_sig: c_int) {
    STOP.store(true, Ordering::Relaxed);
}

pub extern "C" fn ignore_x_err(_d: *mut Display, _e: *mut c_void) -> c_int {
    X_ERR.store(true, Ordering::Relaxed);
    0
}

pub fn ease_develop(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    let inv = 1.0 - t;
    let cubic_out = 1.0 - inv * inv * inv;
    0.80 * smooth + 0.20 * cubic_out
}

/// Precompute the per-pixel 14-bit threshold field for each monitor once so
/// every animation frame runs at memory bandwidth with zero trig/luma overhead.
pub fn build_threshold_field(tgt: &[u8], sw: usize, sh: usize, monitors: &[MonRect]) -> Vec<u16> {
    let mut bayer = [[0i32; 4]; 4];
    for by in 0..4 {
        for bx in 0..4 {
            bayer[by][bx] = (BAYER4_RAW[by][bx] * W_BAYER) / 15;
        }
    }

    let mut thresh = vec![0u16; sw * sh];

    for &m in monitors {
        let mx = m.x.max(0) as usize;
        let my = m.y.max(0) as usize;
        let mw = (m.w.min(sw as i32 - mx as i32)).max(0) as usize;
        let mh = (m.h.min(sh as i32 - my as i32)).max(0) as usize;
        if mw == 0 || mh == 0 {
            continue;
        }

        let mut l_min = 255i32;
        let mut l_max = 0i32;
        let mut ly = 0usize;
        while ly < mh {
            let row_base = (my + ly) * sw + mx;
            let mut lx = 0usize;
            while lx < mw {
                let off = (row_base + lx) * 4;
                let tb = tgt[off] as i32;
                let tg = tgt[off + 1] as i32;
                let tr = tgt[off + 2] as i32;
                let luma = (tr * 77 + tg * 150 + tb * 29) >> 8;
                if luma < l_min {
                    l_min = luma;
                }
                if luma > l_max {
                    l_max = luma;
                }
                lx += 16;
            }
            ly += 16;
        }
        let span = (l_max - l_min).max(48);
        let luma_scale = (W_LUMA * 256 + span / 2) / span;

        let inv_w = if mw > 1 { 1.0 / (mw as f64 - 1.0) } else { 0.0 };
        let inv_h = if mh > 1 { 1.0 / (mh as f64 - 1.0) } else { 0.0 };

        let mut sa = Vec::with_capacity(mw);
        let mut cc = Vec::with_capacity(mw);
        for lx in 0..mw {
            let u = (lx as f64) * inv_w;
            sa.push(((TAU * 1.35 * u + 0.4).sin() * 256.0).round() as i32);
            cc.push(((TAU * 2.15 * u - 0.9).cos() * 256.0).round() as i32);
        }

        let mut cb = Vec::with_capacity(mh);
        let mut sd = Vec::with_capacity(mh);
        for ly in 0..mh {
            let v = (ly as f64) * inv_h;
            cb.push((0.55 * (TAU * 1.15 * v - 0.3).cos() * (HALF_ORG as f64)).round() as i32);
            sd.push((0.45 * (TAU * 1.75 * v + 0.7).sin() * (HALF_ORG as f64)).round() as i32);
        }

        for ly in 0..mh {
            let by = (ly >> 2) & 3;
            let row_bayer = bayer[by];
            let k_cb = cb[ly];
            let k_sd = sd[ly];
            let row_idx = (my + ly) * sw + mx;

            for lx in 0..mw {
                let px_idx = row_idx + lx;
                let off = px_idx * 4;
                let tb = tgt[off] as i32;
                let tg = tgt[off + 1] as i32;
                let tr = tgt[off + 2] as i32;
                let luma = (tr * 77 + tg * 150 + tb * 29) >> 8;
                let inv_l = (l_max - luma).clamp(0, span);
                let luma_term = (inv_l * luma_scale) >> 8;
                let org_term = HALF_ORG + ((sa[lx] * k_cb + cc[lx] * k_sd) >> 8);
                let th = row_bayer[(lx >> 2) & 3] + luma_term + org_term;
                thresh[px_idx] = th.clamp(0, MAX_THRESH) as u16;
            }
        }
    }

    thresh
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn render_develop_avx2(
    cur: &mut [u8],
    src: &[u8],
    tgt: &[u8],
    thresh: &[u16],
    head: i32,
    prev_head: i32,
) {
    use std::arch::x86_64::*;

    let n = thresh.len();
    let c_ptr = cur.as_mut_ptr();
    let s_ptr = src.as_ptr();
    let t_ptr = tgt.as_ptr();
    let th_ptr = thresh.as_ptr();

    let head_v = _mm256_set1_epi32(head);
    let prev_lim = _mm256_set1_epi32(if prev_head < 0 { -1 } else { prev_head - FEATHER });
    let settle_lim = _mm256_set1_epi32(head - FEATHER);
    let zero = _mm256_setzero_si256();
    let feather_v = _mm256_set1_epi32(FEATHER);
    let recip_v = _mm256_set1_epi32(RECIP);
    let mask_ff = _mm256_set1_epi32(0xff);

    let mut i = 0usize;
    while i + 8 <= n {
        let th16 = _mm_loadu_si128(th_ptr.add(i) as *const __m128i);
        let th = _mm256_cvtepu16_epi32(th16);

        let reached = _mm256_cmpgt_epi32(head_v, th);
        let unsettled = _mm256_cmpgt_epi32(th, prev_lim);
        let active = _mm256_and_si256(reached, unsettled);
        if _mm256_testz_si256(active, active) != 0 {
            i += 8;
            continue;
        }

        let off = i * 4;
        let not_done = _mm256_cmpgt_epi32(th, settle_lim);
        if _mm256_testz_si256(not_done, not_done) != 0 {
            let t_px = _mm256_loadu_si256(t_ptr.add(off) as *const __m256i);
            _mm256_storeu_si256(c_ptr.add(off) as *mut __m256i, t_px);
            i += 8;
            continue;
        }

        let s_px = _mm256_loadu_si256(s_ptr.add(off) as *const __m256i);
        let t_px = _mm256_loadu_si256(t_ptr.add(off) as *const __m256i);

        let diff = _mm256_min_epi32(_mm256_max_epi32(_mm256_sub_epi32(head_v, th), zero), feather_v);
        let alpha = _mm256_srli_epi32(_mm256_mullo_epi32(diff, recip_v), 16);

        let sb = _mm256_and_si256(s_px, mask_ff);
        let sg = _mm256_and_si256(_mm256_srli_epi32(s_px, 8), mask_ff);
        let sr = _mm256_and_si256(_mm256_srli_epi32(s_px, 16), mask_ff);

        let tb = _mm256_and_si256(t_px, mask_ff);
        let tg = _mm256_and_si256(_mm256_srli_epi32(t_px, 8), mask_ff);
        let tr = _mm256_and_si256(_mm256_srli_epi32(t_px, 16), mask_ff);

        let ob = _mm256_add_epi32(sb, _mm256_srai_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(tb, sb), alpha), 8));
        let og = _mm256_add_epi32(sg, _mm256_srai_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(tg, sg), alpha), 8));
        let or = _mm256_add_epi32(sr, _mm256_srai_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(tr, sr), alpha), 8));

        let out = _mm256_or_si256(
            _mm256_or_si256(ob, _mm256_slli_epi32(og, 8)),
            _mm256_slli_epi32(or, 16),
        );
        _mm256_storeu_si256(c_ptr.add(off) as *mut __m256i, out);
        i += 8;
    }

    if i < n {
        render_develop_scalar_range(cur, src, tgt, thresh, i, n, head, prev_head);
    }
}

fn render_develop_scalar_range(
    cur: &mut [u8],
    src: &[u8],
    tgt: &[u8],
    thresh: &[u16],
    start: usize,
    end: usize,
    head: i32,
    prev_head: i32,
) {
    let prev_lim = if prev_head < 0 { -1 } else { prev_head - FEATHER };
    for i in start..end {
        let th = thresh[i] as i32;
        if th >= head || th <= prev_lim {
            continue;
        }
        let off = i * 4;
        let diff = head - th;
        if diff >= FEATHER {
            cur[off] = tgt[off];
            cur[off + 1] = tgt[off + 1];
            cur[off + 2] = tgt[off + 2];
            cur[off + 3] = 0;
        } else {
            let a = (diff * RECIP) >> 16;
            let sb = src[off] as i32;
            let sg = src[off + 1] as i32;
            let sr = src[off + 2] as i32;
            let tb = tgt[off] as i32;
            let tg = tgt[off + 1] as i32;
            let tr = tgt[off + 2] as i32;
            cur[off] = (sb + (((tb - sb) * a) >> 8)) as u8;
            cur[off + 1] = (sg + (((tg - sg) * a) >> 8)) as u8;
            cur[off + 2] = (sr + (((tr - sr) * a) >> 8)) as u8;
            cur[off + 3] = 0;
        }
    }
}

pub fn render_develop_frame(
    cur: &mut [u8],
    src: &[u8],
    tgt: &[u8],
    thresh: &[u16],
    head: i32,
    prev_head: i32,
    use_avx2: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if use_avx2 {
        unsafe {
            render_develop_avx2(cur, src, tgt, thresh, head, prev_head);
        }
        return;
    }
    let _ = use_avx2;
    render_develop_scalar_range(cur, src, tgt, thresh, 0, thresh.len(), head, prev_head);
}

pub unsafe fn query_monitors(dpy: *mut Display, root: Window, sw: i32, sh: i32) -> Vec<MonRect> {
    let mut n: c_int = 0;
    let raw = XRRGetMonitors(dpy, root, 1, &mut n);
    let mut out = Vec::new();
    if !raw.is_null() && n > 0 {
        for i in 0..(n as usize) {
            let m = &*raw.add(i);
            if m.width > 0 && m.height > 0 {
                out.push(MonRect {
                    x: m.x,
                    y: m.y,
                    w: m.width,
                    h: m.height,
                });
            }
        }
        XRRFreeMonitors(raw);
    }
    if out.is_empty() {
        out.push(MonRect { x: 0, y: 0, w: sw, h: sh });
    }
    out
}

pub unsafe fn decode_rgb_image_scaled(path: &str, max_w: i32, max_h: i32) -> Option<(i32, i32, Vec<u8>)> {
    let api = load_pixbuf_api()?;
    let c_path = CString::new(path).ok()?;
    let mut iw: c_int = 0;
    let mut ih: c_int = 0;
    let pb = if max_w > 0
        && max_h > 0
        && !(api.get_file_info)(c_path.as_ptr(), &mut iw, &mut ih).is_null()
        && iw > 0
        && ih > 0
        && (iw > max_w || ih > max_h)
    {
        (api.new_from_file_at_scale)(c_path.as_ptr(), max_w, max_h, 1, null_mut())
    } else {
        (api.new_from_file)(c_path.as_ptr(), null_mut())
    };
    if pb.is_null() {
        return None;
    }
    let w = (api.get_width)(pb);
    let h = (api.get_height)(pb);
    let rs = (api.get_rowstride)(pb) as usize;
    let ch = (api.get_n_channels)(pb) as usize;
    let px = (api.get_pixels)(pb);
    if w <= 0 || h <= 0 || (ch != 3 && ch != 4) || px.is_null() {
        (api.unref)(pb);
        return None;
    }
    let w_us = w as usize;
    let h_us = h as usize;
    let row_rgb = w_us * 3;
    let mut out = vec![0u8; row_rgb * h_us];
    if ch == 3 {
        for y in 0..h_us {
            let src = std::slice::from_raw_parts(px.add(y * rs), row_rgb);
            out[y * row_rgb..(y + 1) * row_rgb].copy_from_slice(src);
        }
    } else {
        for y in 0..h_us {
            let s_row = px.add(y * rs);
            let d_row = out.as_mut_ptr().add(y * row_rgb);
            for x in 0..w_us {
                *d_row.add(x * 3) = *s_row.add(x * 4);
                *d_row.add(x * 3 + 1) = *s_row.add(x * 4 + 1);
                *d_row.add(x * 3 + 2) = *s_row.add(x * 4 + 2);
            }
        }
    }
    (api.unref)(pb);
    Some((w, h, out))
}

pub unsafe fn decode_and_scale(
    path: &str,
    sw: i32,
    sh: i32,
    monitors: &[MonRect],
) -> Option<Vec<u8>> {
    let api = load_pixbuf_api()?;
    let c_path = CString::new(path).ok()?;
    let mut max_w = 1i32;
    let mut max_h = 1i32;
    for m in monitors {
        if m.w > max_w {
            max_w = m.w;
        }
        if m.h > max_h {
            max_h = m.h;
        }
    }

    let mut iw: c_int = 0;
    let mut ih: c_int = 0;
    let src = if !(api.get_file_info)(c_path.as_ptr(), &mut iw, &mut ih).is_null()
        && iw > 0
        && ih > 0
    {
        let s = (max_w as f64 / iw as f64).max(max_h as f64 / ih as f64);
        if s < 1.0 {
            let tw = (iw as f64 * s).ceil() as c_int;
            let th = (ih as f64 * s).ceil() as c_int;
            (api.new_from_file_at_scale)(c_path.as_ptr(), tw.max(max_w), th.max(max_h), 0, null_mut())
        } else {
            (api.new_from_file)(c_path.as_ptr(), null_mut())
        }
    } else {
        (api.new_from_file)(c_path.as_ptr(), null_mut())
    };
    if src.is_null() {
        if baker::is_video_ext(Path::new(path)) {
            return baker::decode_video_static_bgrx(path, sw, sh, monitors);
        }
        return None;
    }

    let pw = (api.get_width)(src);
    let ph = (api.get_height)(src);
    if pw <= 0 || ph <= 0 {
        (api.unref)(src);
        return None;
    }

    let mut target = vec![0u8; (sw as usize) * (sh as usize) * 4];

    for m in monitors {
        let k = (m.w as f64 / pw as f64).max(m.h as f64 / ph as f64);
        let ox = (pw as f64 * k - m.w as f64) * 0.5;
        let oy = (ph as f64 * k - m.h as f64) * 0.5;
        let dst = (api.new_buf)(GDK_COLORSPACE_RGB, 0, 8, m.w, m.h);
        if dst.is_null() {
            continue;
        }
        let interp = if (k - 1.0).abs() < 1e-6 {
            GDK_INTERP_NEAREST
        } else {
            GDK_INTERP_BILINEAR
        };
        (api.scale)(src, dst, 0, 0, m.w, m.h, -ox, -oy, k, k, interp);

        let rs = (api.get_rowstride)(dst) as usize;
        let ch = (api.get_n_channels)(dst) as usize;
        let px = (api.get_pixels)(dst);

        for y in 0..m.h {
            let dy = m.y + y;
            if dy < 0 || dy >= sh {
                continue;
            }
            let src_row = px.add((y as usize) * rs);
            let dst_row_off = ((dy as usize) * (sw as usize) + (m.x.max(0) as usize)) * 4;
            let max_cols = (m.w.min(sw - m.x.max(0))).max(0) as usize;
            let mut s_ptr = src_row;
            let mut d_ptr = target.as_mut_ptr().add(dst_row_off);
            for _ in 0..max_cols {
                *d_ptr = *s_ptr.add(2);
                *d_ptr.add(1) = *s_ptr.add(1);
                *d_ptr.add(2) = *s_ptr;
                *d_ptr.add(3) = 0;
                s_ptr = s_ptr.add(ch);
                d_ptr = d_ptr.add(4);
            }
        }
        (api.unref)(dst);
    }

    (api.unref)(src);
    Some(target)
}

pub unsafe fn get_root_pixmap(dpy: *mut Display, root: Window, atom: Atom) -> Pixmap {
    if atom == 0 {
        return 0;
    }
    let mut actual_type: Atom = 0;
    let mut actual_fmt: c_int = 0;
    let mut nitems: c_ulong = 0;
    let mut bytes_after: c_ulong = 0;
    let mut prop: *mut c_uchar = null_mut();
    let rc = XGetWindowProperty(
        dpy, root, atom, 0, 1, 0, XA_PIXMAP,
        &mut actual_type, &mut actual_fmt, &mut nitems, &mut bytes_after, &mut prop,
    );
    let mut pmap: Pixmap = 0;
    if rc == 0 && !prop.is_null() {
        if actual_type == XA_PIXMAP && actual_fmt == 32 && nitems >= 1 {
            pmap = *(prop as *const Pixmap);
        }
        XFree(prop as *mut c_void);
    }
    pmap
}

pub unsafe fn pixmap_matches(dpy: *mut Display, pmap: Pixmap, sw: i32, sh: i32, depth: i32) -> bool {
    if pmap == 0 {
        return false;
    }
    X_ERR.store(false, Ordering::Relaxed);
    let mut r: Window = 0;
    let (mut x, mut y) = (0, 0);
    let (mut w, mut h, mut bw, mut d) = (0u32, 0u32, 0u32, 0u32);
    let ok = XGetGeometry(dpy, pmap, &mut r, &mut x, &mut y, &mut w, &mut h, &mut bw, &mut d);
    XSync(dpy, 0);
    ok != 0 && !X_ERR.load(Ordering::Relaxed) && w == sw as u32 && h == sh as u32 && d == depth as u32
}

/// Publish a persistent `_XROOTPMAP_ID` on a dedicated `RETAIN_PERMANENT` X11
/// connection so the root wallpaper survives after the daemon exits.
pub unsafe fn publish_persistent_root_pixmap(
    bgrx: &[u8],
    sw: i32,
    sh: i32,
    monitors: &[MonRect],
) -> bool {
    let dpy = XOpenDisplay(null());
    if dpy.is_null() {
        return false;
    }
    XSetErrorHandler(Some(ignore_x_err));
    let scr = XDefaultScreen(dpy);
    let root = XDefaultRootWindow(dpy);
    let vis = XDefaultVisual(dpy, scr);
    let depth = XDefaultDepth(dpy, scr);

    let c_rootpmap = CString::new("_XROOTPMAP_ID").unwrap();
    let c_esetroot = CString::new("ESETROOT_PMAP_ID").unwrap();
    let atom_rootpmap = XInternAtom(dpy, c_rootpmap.as_ptr(), 0);
    let atom_esetroot = XInternAtom(dpy, c_esetroot.as_ptr(), 0);
    let old_eset_pmap = get_root_pixmap(dpy, root, atom_esetroot);

    let mut buf = bgrx.to_vec();
    let ximg = XCreateImage(
        dpy, vis, depth as c_uint, ZPIXMAP, 0,
        buf.as_mut_ptr() as *mut c_char, sw as c_uint, sh as c_uint, 32, sw * 4,
    );
    if ximg.is_null() {
        XCloseDisplay(dpy);
        return false;
    }
    let gc = XCreateGC(dpy, root, 0, null_mut());
    XSetCloseDownMode(dpy, RETAIN_PERMANENT);
    let pmap = XCreatePixmap(dpy, root, sw as c_uint, sh as c_uint, depth as c_uint);
    for m in monitors {
        let mx = m.x.max(0);
        let my = m.y.max(0);
        let mw = m.w.min(sw - mx);
        let mh = m.h.min(sh - my);
        if mw > 0 && mh > 0 {
            XPutImage(dpy, pmap, gc, ximg, mx, my, mx, my, mw as c_uint, mh as c_uint);
        }
    }
    XSetWindowBackgroundPixmap(dpy, root, pmap);
    XChangeProperty(
        dpy, root, atom_rootpmap, XA_PIXMAP, 32, PROP_MODE_REPLACE,
        &pmap as *const Pixmap as *const c_uchar, 1,
    );
    XChangeProperty(
        dpy, root, atom_esetroot, XA_PIXMAP, 32, PROP_MODE_REPLACE,
        &pmap as *const Pixmap as *const c_uchar, 1,
    );
    XClearWindow(dpy, root);
    XSync(dpy, 0);
    if old_eset_pmap != 0 && old_eset_pmap != pmap {
        XKillClient(dpy, old_eset_pmap);
        XSync(dpy, 0);
    }
    XFreeGC(dpy, gc);
    (*ximg).data = null_mut();
    if let Some(destroy) = (*ximg).destroy_image {
        destroy(ximg);
    }
    XCloseDisplay(dpy);
    true
}

fn pidfile_static_path() -> String {
    let run = env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    format!("{run}/misa-wall.pid")
}

unsafe fn preempt_previous() -> String {
    let pfile = pidfile_static_path();
    let self_pid = getpid();
    if let Ok(txt) = fs::read_to_string(&pfile) {
        if let Ok(old_pid) = txt.trim().parse::<c_int>() {
            if old_pid > 1 && old_pid != self_pid {
                let comm_path = format!("/proc/{old_pid}/comm");
                if let Ok(comm) = fs::read_to_string(comm_path) {
                    if comm.trim() == "misa-wall" && kill(old_pid, SIGTERM) == 0 {
                        for _ in 0..20 {
                            if kill(old_pid, 0) != 0 {
                                break;
                            }
                            thread::sleep(Duration::from_millis(2));
                        }
                    }
                }
            }
        }
    }
    let _ = fs::write(&pfile, format!("{self_pid}\n"));
    pfile
}

fn write_fehbg(path: &Path) {
    let Ok(home) = env::var("HOME") else {
        return;
    };
    let abs = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let s = abs.to_string_lossy().replace('\'', "'\\''");
    let content = format!("#!/bin/sh\nfeh --no-fehbg --bg-fill '{s}'\n");
    let fehbg = Path::new(&home).join(".fehbg");
    let _ = fs::write(&fehbg, content);
}

fn ensure_daemon_running() -> bool {
    if daemon::is_daemon_alive() {
        return true;
    }
    // Clean up any stale daemon process before spawning a fresh one.
    unsafe { daemon::stop_daemon(); }
    let Ok(exe) = env::current_exe() else {
        return false;
    };
    let mut cmd = Command::new(exe);
    cmd.arg("--daemon")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if cmd.spawn().is_err() {
        return false;
    }
    for _ in 0..100 {
        if daemon::is_daemon_alive() {
            return true;
        }
        thread::sleep(Duration::from_millis(10));
    }
    false
}

/// Ensure a valid `.mwa` BitPack-LZ4 cache exists for `(img_path, frag_path)`.
/// On cache hit: 0 ms, 0 OpenGL! On cache miss: runs one-shot `--bake-loop`
/// subprocess that renders & packs the loop and exits immediately to free Mesa.
fn ensure_mwa_cache(img_path: &Path, frag_path: &Path) -> Option<String> {
    let fps: u32 = env::var("MISA_WALL_BAKE_FPS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_ANIM_FPS)
        .clamp(6, 20);
    let frames: u32 = env::var("MISA_WALL_BAKE_FRAMES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_ANIM_FRAMES)
        .clamp(24, 96);

    let hash = codec::compute_cache_hash(img_path, frag_path, DEFAULT_CELL_W, DEFAULT_CELL_H, fps, frames)?;
    let out_mwa = codec::cache_path_for_hash(hash);
    if out_mwa.is_file() {
        // Quick header sanity check via mmap open.
        if codec::MwaFile::open(&out_mwa).map(|m| m.src_hash == hash).unwrap_or(false) {
            return Some(out_mwa.to_string_lossy().into_owned());
        }
        let _ = fs::remove_file(&out_mwa);
    }

    // Spawn transient one-shot baker subprocess so parent/daemon NEVER loads libGL!
    let exe = env::current_exe().ok()?;
    let status = Command::new(exe)
        .arg("--bake-loop")
        .arg(img_path)
        .arg(frag_path)
        .arg(&out_mwa)
        .arg(DEFAULT_CELL_W.to_string())
        .arg(DEFAULT_CELL_H.to_string())
        .arg(fps.to_string())
        .arg(frames.to_string())
        .stdin(Stdio::null())
        .status()
        .ok()?;
    if status.success() && out_mwa.is_file() {
        Some(out_mwa.to_string_lossy().into_owned())
    } else {
        None
    }
}

fn ensure_media_mwa_cache(media_path: &Path) -> Option<String> {
    let max_w: u32 = env::var("MISA_WALL_MEDIA_MAX_W")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MEDIA_MAX_W)
        .clamp(320, 1920);
    let max_h: u32 = env::var("MISA_WALL_MEDIA_MAX_H")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MEDIA_MAX_H)
        .clamp(180, 1080);
    let max_fps: u32 = env::var("MISA_WALL_MEDIA_FPS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MEDIA_FPS)
        .clamp(6, 20);
    let max_frames: u32 = env::var("MISA_WALL_MEDIA_FRAMES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_MEDIA_FRAMES)
        .clamp(16, 480);
    let filter_hint = env::var("MISA_WALL_MEDIA_FILTER").unwrap_or_default();

    let hash = codec::compute_media_cache_hash(media_path, max_w, max_h, max_fps, max_frames, &filter_hint)?;
    let out_mwa = codec::cache_path_for_hash(hash);
    if out_mwa.is_file() {
        if codec::MwaFile::open(&out_mwa).map(|m| m.src_hash == hash).unwrap_or(false) {
            return Some(out_mwa.to_string_lossy().into_owned());
        }
        let _ = fs::remove_file(&out_mwa);
    }

    let exe = env::current_exe().ok()?;
    let status = Command::new(exe)
        .arg("--bake-media")
        .arg(media_path)
        .arg(&out_mwa)
        .arg(max_w.to_string())
        .arg(max_h.to_string())
        .arg(max_fps.to_string())
        .arg(max_frames.to_string())
        .stdin(Stdio::null())
        .status()
        .ok()?;
    if status.success() && out_mwa.is_file() {
        Some(out_mwa.to_string_lossy().into_owned())
    } else {
        None
    }
}

/// Resolve & ensure `.mwa` cache for either a sidecar `.frag` shader wallpaper
/// or a user-supplied animated media file (GIF/WebP/APNG/video).
fn ensure_any_anim_cache(canon_img: &Path) -> Option<(String, String)> {
    if let Some(frag) = daemon::find_sidecar_shader(canon_img) {
        let mwa = ensure_mwa_cache(canon_img, &frag)?;
        return Some((mwa, frag.to_string_lossy().into_owned()));
    }
    if baker::detect_media_anim(canon_img).is_some() {
        let mwa = ensure_media_mwa_cache(canon_img)?;
        return Some((mwa, String::new()));
    }
    None
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  misa-wall [--no-transition|--instant] [--static] [--duration MS] [--fps 1..240] <image|media>\n  misa-wall pause | unpause | toggle | query | clear-cache | --stop-live\n  misa-wall --ensure-cache <image|media>\n  misa-wall --bake-frame <img> <frag> <out.ppm> [t_sec]\n  misa-wall --bake-loop  <img> <frag> <out.mwa> [w h fps frames]\n  misa-wall --bake-media <media> <out.mwa> [max_w max_h max_fps max_frames]\n  misa-wall --thumb      <media> <out.ppm> [w h]\n  misa-wall --selftest"
    );
    std::process::exit(2);
}

fn main() {
    let mut instant = false;
    let mut force_static = false;
    let mut duration_ms: u64 = env::var("MISA_WALL_DURATION_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(540);
    let mut fps: u32 = env::var("MISA_WALL_FPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let mut img_path: Option<String> = None;

    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }

    // Fast IPC / utility subcommands (awww-style CLI).
    match args[0].as_str() {
        "--daemon" => std::process::exit(daemon::run_daemon()),
        "--stop-live" | "kill" | "stop" => {
            unsafe { daemon::stop_daemon(); }
            return;
        }
        "pause" => match daemon::ipc_send("PAUSE\n") {
            Ok(r) => { print!("{r}"); return; }
            Err(e) => { eprintln!("misa-wall: {e}"); std::process::exit(1); }
        },
        "unpause" | "resume" => match daemon::ipc_send("UNPAUSE\n") {
            Ok(r) => { print!("{r}"); return; }
            Err(e) => { eprintln!("misa-wall: {e}"); std::process::exit(1); }
        },
        "toggle" => match daemon::ipc_send("TOGGLE\n") {
            Ok(r) => { print!("{r}"); return; }
            Err(e) => { eprintln!("misa-wall: {e}"); std::process::exit(1); }
        },
        "query" | "status" => match daemon::ipc_send("QUERY\n") {
            Ok(r) => { print!("{r}"); return; }
            Err(_) => { println!("mode=static (no daemon running)"); return; }
        },
        "clear-cache" => {
            let n = codec::clear_cache_dir();
            println!("cleared {n} cached animation(s)");
            return;
        }
        "--selftest" => match codec::run_selftest() {
            Ok(()) => { println!("misa-wall codec selftest: OK"); return; }
            Err(e) => { eprintln!("misa-wall codec selftest FAILED: {e}"); std::process::exit(1); }
        },
        _ => {}
    }

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-transition" | "--instant" => instant = true,
            "--static" => force_static = true,
            "--ensure-cache" if i + 1 < args.len() => {
                let p = Path::new(&args[i + 1]);
                if !p.is_file() {
                    std::process::exit(1);
                }
                let canon = fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
                if let Some((mwa_path, _)) = ensure_any_anim_cache(&canon) {
                    println!("{mwa_path}");
                    return;
                }
                std::process::exit(1);
            }
            "--bake-frame" if i + 3 < args.len() => {
                let ip = &args[i + 1];
                let fp = &args[i + 2];
                let out = &args[i + 3];
                let t_sec: f32 = if i + 4 < args.len() {
                    args[i + 4].parse().unwrap_or(4.2)
                } else {
                    4.2
                };
                std::process::exit(baker::bake_shader_frame(ip, fp, out, t_sec));
            }
            "--bake-loop" if i + 3 < args.len() => {
                let ip = &args[i + 1];
                let fp = &args[i + 2];
                let out = Path::new(&args[i + 3]);
                let fw: u32 = args.get(i + 4).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_CELL_W);
                let fh: u32 = args.get(i + 5).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_CELL_H);
                let bfps: u32 = args.get(i + 6).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_ANIM_FPS);
                let frames: u32 = args.get(i + 7).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_ANIM_FRAMES);
                let Some(hash) = codec::compute_cache_hash(Path::new(ip), Path::new(fp), fw, fh, bfps, frames) else {
                    eprintln!("misa-wall: cannot read {ip} or {fp}");
                    std::process::exit(1);
                };
                std::process::exit(baker::bake_shader_loop(ip, fp, out, hash, fw, fh, bfps, frames));
            }
            "--bake-media" if i + 2 < args.len() => {
                let mp = &args[i + 1];
                let out = Path::new(&args[i + 2]);
                let mw: u32 = args.get(i + 3).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MEDIA_MAX_W);
                let mh: u32 = args.get(i + 4).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MEDIA_MAX_H);
                let mfps: u32 = args.get(i + 5).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MEDIA_FPS);
                let mframes: u32 = args.get(i + 6).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MEDIA_FRAMES);
                let filter_hint = env::var("MISA_WALL_MEDIA_FILTER").unwrap_or_default();
                let force_nearest = filter_hint.eq_ignore_ascii_case("nearest");
                let Some(hash) = codec::compute_media_cache_hash(Path::new(mp), mw, mh, mfps, mframes, &filter_hint) else {
                    eprintln!("misa-wall: cannot read {mp}");
                    std::process::exit(1);
                };
                std::process::exit(baker::bake_media_loop(mp, out, hash, mw, mh, mfps, mframes, force_nearest));
            }
            "--thumb" if i + 2 < args.len() => {
                let mp = &args[i + 1];
                let out = &args[i + 2];
                let tw: u32 = args.get(i + 3).and_then(|s| s.parse().ok()).unwrap_or(176);
                let th: u32 = args.get(i + 4).and_then(|s| s.parse().ok()).unwrap_or(99);
                std::process::exit(baker::bake_media_thumb(mp, out, tw, th));
            }
            "--duration" if i + 1 < args.len() => {
                duration_ms = args[i + 1].parse().unwrap_or(540);
                if duration_ms == 0 {
                    instant = true;
                }
                i += 1;
            }
            "--fps" if i + 1 < args.len() => {
                fps = args[i + 1].parse().unwrap_or(60).clamp(1, 240);
                i += 1;
            }
            "-h" | "--help" => usage(),
            s if s.starts_with('-') => usage(),
            s => img_path = Some(s.to_string()),
        }
        i += 1;
    }

    let Some(path_str) = img_path else {
        usage();
    };
    let path = Path::new(&path_str);
    if !path.is_file() {
        eprintln!("misa-wall: file not found: {path_str}");
        std::process::exit(1);
    }
    let canon_img = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let canon_str = canon_img.to_string_lossy().into_owned();

    if !baker::is_video_ext(&canon_img) {
        write_fehbg(&canon_img);
    }

    let t_ms = if instant { 0 } else { duration_ms };

    // 1. Animated wallpaper path (.frag shader sidecar OR user GIF/WebP/APNG/video!)
    let anim_cached = if force_static {
        None
    } else {
        ensure_any_anim_cache(&canon_img)
    };
    if let Some((mwa_path, frag_str)) = anim_cached {
        if ensure_daemon_running() {
            let frag_field = if frag_str.is_empty() { "-" } else { frag_str.as_str() };
            let cmd = format!("SET_ANIM\t{canon_str}\t{frag_field}\t{mwa_path}\t{t_ms}\n");
            if let Ok(resp) = daemon::ipc_send(&cmd) {
                if resp.starts_with("OK") {
                    return;
                }
            }
        }
        unsafe { daemon::stop_daemon(); }
    } else if daemon::is_daemon_alive() {
        // Switching from animated daemon -> static wallpaper seamlessly!
        let cmd = format!("SET_STATIC\t{canon_str}\t{t_ms}\n");
        if let Ok(resp) = daemon::ipc_send(&cmd) {
            if resp.starts_with("OK") {
                return;
            }
        }
        unsafe { daemon::stop_daemon(); }
    } else {
        unsafe { daemon::stop_daemon(); }
    }

    // 2. Direct one-shot static wallpaper path (0.0 MiB resident RAM after exit!).
    unsafe {
        let dpy = XOpenDisplay(null());
        if dpy.is_null() {
            eprintln!("misa-wall: cannot open X display");
            std::process::exit(1);
        }
        XSetErrorHandler(Some(ignore_x_err));
        let scr = XDefaultScreen(dpy);
        let root = XDefaultRootWindow(dpy);
        let vis = XDefaultVisual(dpy, scr);
        let depth = XDefaultDepth(dpy, scr);
        let sw = XDisplayWidth(dpy, scr);
        let sh = XDisplayHeight(dpy, scr);
        if sw <= 0 || sh <= 0 {
            std::process::exit(1);
        }

        let monitors = query_monitors(dpy, root, sw, sh);
        let Some(target_buf) = decode_and_scale(&canon_str, sw, sh, &monitors) else {
            eprintln!("misa-wall: failed to decode image: {canon_str}");
            std::process::exit(1);
        };

        signal(SIGTERM, on_sig);
        signal(SIGINT, on_sig);
        let pfile = preempt_previous();

        let c_rootpmap = CString::new("_XROOTPMAP_ID").unwrap();
        let c_esetroot = CString::new("ESETROOT_PMAP_ID").unwrap();
        let atom_rootpmap = XInternAtom(dpy, c_rootpmap.as_ptr(), 0);
        let atom_esetroot = XInternAtom(dpy, c_esetroot.as_ptr(), 0);

        let old_root_pmap = get_root_pixmap(dpy, root, atom_rootpmap);
        let old_eset_pmap = get_root_pixmap(dpy, root, atom_esetroot);

        let buf_len = (sw as usize) * (sh as usize) * 4;
        let mut shminfo = XShmSegmentInfo {
            shmseg: 0,
            shmid: -1,
            shmaddr: null_mut(),
            read_only: 0,
        };
        let mut use_shm = XShmQueryExtension(dpy) != 0;
        let mut ximg: *mut XImage = null_mut();
        let mut _fallback_vec: Vec<u8>;

        if use_shm {
            ximg = XShmCreateImage(
                dpy, vis, depth as c_uint, ZPIXMAP, null_mut(), &mut shminfo, sw as c_uint, sh as c_uint,
            );
            if !ximg.is_null() {
                let sz = ((*ximg).bytes_per_line as usize) * (sh as usize);
                shminfo.shmid = shmget(IPC_PRIVATE, sz.max(buf_len), IPC_CREAT | 0o600);
                if shminfo.shmid >= 0 {
                    let addr = shmat(shminfo.shmid, null(), 0);
                    if addr != (-1isize as *mut c_void) && !addr.is_null() {
                        shminfo.shmaddr = addr as *mut c_char;
                        (*ximg).data = shminfo.shmaddr;
                        X_ERR.store(false, Ordering::Relaxed);
                        if XShmAttach(dpy, &mut shminfo) != 0 {
                            XSync(dpy, 0);
                        } else {
                            X_ERR.store(true, Ordering::Relaxed);
                        }
                        shmctl(shminfo.shmid, IPC_RMID, null_mut());
                        if X_ERR.load(Ordering::Relaxed) {
                            shmdt(shminfo.shmaddr as *const c_void);
                            shminfo.shmaddr = null_mut();
                            use_shm = false;
                        }
                    } else {
                        shmctl(shminfo.shmid, IPC_RMID, null_mut());
                        use_shm = false;
                    }
                } else {
                    use_shm = false;
                }
            } else {
                use_shm = false;
            }
        }

        if !use_shm {
            _fallback_vec = vec![0u8; buf_len];
            ximg = XCreateImage(
                dpy, vis, depth as c_uint, ZPIXMAP, 0,
                _fallback_vec.as_mut_ptr() as *mut c_char, sw as c_uint, sh as c_uint, 32, sw * 4,
            );
            if ximg.is_null() {
                std::process::exit(1);
            }
        }

        let cur_slice: &mut [u8] =
            std::slice::from_raw_parts_mut((*ximg).data as *mut u8, buf_len);

        let mut have_old = false;
        if !instant && duration_ms > 0 && pixmap_matches(dpy, old_root_pmap, sw, sh, depth) {
            X_ERR.store(false, Ordering::Relaxed);
            if use_shm {
                if XShmGetImage(dpy, old_root_pmap, ximg, 0, 0, ALL_PLANES) != 0 {
                    XSync(dpy, 0);
                    have_old = !X_ERR.load(Ordering::Relaxed);
                }
            } else {
                let got = XGetSubImage(
                    dpy, old_root_pmap, 0, 0, sw as c_uint, sh as c_uint,
                    ALL_PLANES, ZPIXMAP, ximg, 0, 0,
                );
                XSync(dpy, 0);
                have_old = !got.is_null() && !X_ERR.load(Ordering::Relaxed);
            }
            if have_old {
                let mut k = 3usize;
                while k < buf_len {
                    cur_slice[k] = 0;
                    k += 4;
                }
            }
        }

        let gc = XCreateGC(dpy, root, 0, null_mut());
        XSetCloseDownMode(dpy, RETAIN_PERMANENT);

        let blit_to_pixmap = |pmap: Pixmap| {
            for m in &monitors {
                let mx = m.x.max(0);
                let my = m.y.max(0);
                let mw = m.w.min(sw - mx);
                let mh = m.h.min(sh - my);
                if mw <= 0 || mh <= 0 {
                    continue;
                }
                if use_shm {
                    XShmPutImage(dpy, pmap, gc, ximg, mx, my, mx, my, mw as c_uint, mh as c_uint, 0);
                } else {
                    XPutImage(dpy, pmap, gc, ximg, mx, my, mx, my, mw as c_uint, mh as c_uint);
                }
            }
        };

        let publish_pixmap = |pmap: Pixmap| {
            XSetWindowBackgroundPixmap(dpy, root, pmap);
            XChangeProperty(
                dpy, root, atom_rootpmap, XA_PIXMAP, 32, PROP_MODE_REPLACE,
                &pmap as *const Pixmap as *const c_uchar, 1,
            );
            XChangeProperty(
                dpy, root, atom_esetroot, XA_PIXMAP, 32, PROP_MODE_REPLACE,
                &pmap as *const Pixmap as *const c_uchar, 1,
            );
            XClearWindow(dpy, root);
            XSync(dpy, 0);
        };

        let identical = have_old && cur_slice == target_buf.as_slice();
        if !have_old || identical {
            cur_slice.copy_from_slice(&target_buf);
            let pmap = XCreatePixmap(dpy, root, sw as c_uint, sh as c_uint, depth as c_uint);
            blit_to_pixmap(pmap);
            publish_pixmap(pmap);
            if old_eset_pmap != 0 && old_eset_pmap != pmap {
                XKillClient(dpy, old_eset_pmap);
                XSync(dpy, 0);
            }
        } else {
            let src_buf = cur_slice.to_vec();
            let thresh = build_threshold_field(&target_buf, sw as usize, sh as usize, &monitors);
            let pmaps = [
                XCreatePixmap(dpy, root, sw as c_uint, sh as c_uint, depth as c_uint),
                XCreatePixmap(dpy, root, sw as c_uint, sh as c_uint, depth as c_uint),
            ];
            #[cfg(target_arch = "x86_64")]
            let use_avx2 = std::is_x86_feature_detected!("avx2");
            #[cfg(not(target_arch = "x86_64"))]
            let use_avx2 = false;

            let frame_dur = Duration::from_nanos(1_000_000_000 / (fps as u64));
            let total_dur = Duration::from_millis(duration_ms.max(1));
            let start = Instant::now();
            let mut idx = 0usize;
            let mut freed_old = false;
            let mut prev_head: i32 = -1;

            loop {
                let t0 = Instant::now();
                let elapsed = t0.duration_since(start) + frame_dur;
                let progress = (elapsed.as_secs_f64() / total_dur.as_secs_f64()).min(1.0);
                let finish = progress >= 1.0;
                let eased = ease_develop(progress);
                let head = if finish {
                    MAX_THRESH + FEATHER
                } else {
                    (eased * ((MAX_THRESH + FEATHER) as f64)).round() as i32
                };

                render_develop_frame(cur_slice, &src_buf, &target_buf, &thresh, head, prev_head, use_avx2);
                prev_head = head;

                blit_to_pixmap(pmaps[idx]);
                publish_pixmap(pmaps[idx]);

                if !freed_old {
                    freed_old = true;
                    if old_eset_pmap != 0 && old_eset_pmap != pmaps[0] && old_eset_pmap != pmaps[1] {
                        XKillClient(dpy, old_eset_pmap);
                        XSync(dpy, 0);
                    }
                }

                if finish || STOP.load(Ordering::Relaxed) {
                    XFreePixmap(dpy, pmaps[idx ^ 1]);
                    break;
                }
                idx ^= 1;
                if let Some(rem) = frame_dur.checked_sub(t0.elapsed()) {
                    thread::sleep(rem);
                }
            }
        }

        XFreeGC(dpy, gc);
        if use_shm {
            XShmDetach(dpy, &mut shminfo);
            XSync(dpy, 0);
            shmdt(shminfo.shmaddr as *const c_void);
            (*ximg).data = null_mut();
            if let Some(destroy) = (*ximg).destroy_image {
                destroy(ximg);
            }
        } else {
            XSync(dpy, 0);
        }

        let self_pid = getpid();
        if let Ok(txt) = fs::read_to_string(&pfile) {
            if txt.trim().parse::<c_int>().ok() == Some(self_pid) {
                let _ = fs::remove_file(&pfile);
            }
        }
        _exit(0);
    }
}
