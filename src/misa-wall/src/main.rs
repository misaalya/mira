//! misa-wall: hyper-optimized X11 wallpaper engine with Mira's original
//! **Luma-Halftone Develop** transition, written from scratch in crate-less Rust.
//!
//!   misa-wall [--no-transition|--instant] [--duration MS] [--fps N] <IMAGE>
//!
//! How Mira's Luma-Halftone Develop works:
//!   Instead of a directional wipe/sweep, a plain crossfade, or per-channel RGB
//!   stepping, the incoming wallpaper crystallizes organically across the entire
//!   screen simultaneously — like an analog photograph developing through a
//!   crisp halftone screen:
//!     1. 4x4 ordered Bayer halftone lattice at 4x4 screen-pixel pitch (`42%`)
//!        giving every region an unmistakable, tactile pixel-stipple bloom.
//!     2. Auto-leveled luminance photo-develop (`38%`): highlights of the
//!        incoming wallpaper (moonlight, mist, skies, accents) sparkle into
//!        view first everywhere on screen, followed by midtones and deep
//!        shadows — auto-normalized to the target's brightness range so dark
//!        and bright wallpapers alike begin developing on frame 1.
//!     3. Non-directional organic pocket drift (`20%`): soft low-frequency
//!        harmonic undulations prevent flat solid zones from flipping in rigid
//!        lockstep, with zero directional sweep or hard edge.
//!     4. Narrow optical feather (`~5.8%`) so each 4x4 pixel cell blooms
//!        smoothly over ~2-3 frames with zero flicker at 60 FPS.
//!
//! Performance engineering:
//!   - Zero external crates (`cargo build --release --offline`).
//!   - Precomputed per-pixel `u16` threshold field (built once per transition)
//!     so the per-frame kernel does zero trig, zero luma, and zero Bayer math.
//!   - Explicit x86_64 AVX2 branchless SIMD kernel (8 BGRX pixels / iteration)
//!     that skips already-settled and not-yet-reached 8-pixel blocks without
//!     even reading source/target pixels from memory, plus a portable scalar
//!     fallback.
//!   - Zero-copy X11 MIT-SHM (`XShmCreateImage`, `XShmGetImage`, `XShmPutImage`)
//!     with automatic fallback to standard XImage if SHM is unavailable.
//!   - Double-buffered X11 root Pixmaps (`pmap[0]`, `pmap[1]`) so picom (`glx`
//!     + `use-damage`) re-binds the root texture cleanly at 60 FPS with zero
//!     tearing and zero per-frame Pixmap allocation/deallocation.
//!   - Reads the current desktop wallpaper directly from `_XROOTPMAP_ID` and
//!     exits completely when done: 0.0 MB resident RAM and 0.0% CPU at idle.

mod live;

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
type GC = *mut c_void;
type GdkPixbuf = c_void;
type XID = c_ulong;
pub type Window = XID;
type Drawable = XID;
type Pixmap = XID;
pub type Atom = c_ulong;

const ZPIXMAP: c_int = 2;
const PROP_MODE_REPLACE: c_int = 0;
const RETAIN_PERMANENT: c_int = 1;
const XA_PIXMAP: Atom = 20;
const ALL_PLANES: c_ulong = !0;
const GDK_COLORSPACE_RGB: c_int = 0;
const GDK_INTERP_NEAREST: c_int = 0;
const GDK_INTERP_BILINEAR: c_int = 2;
const IPC_PRIVATE: c_int = 0;
const IPC_CREAT: c_int = 0o1000;
const IPC_RMID: c_int = 0;
pub const SIGINT: c_int = 2;
pub const SIGTERM: c_int = 15;

// Fixed-point constants for Mira's Luma-Halftone Develop (14-bit threshold space).
const W_BAYER: i32 = 6881; // 42.0%
const W_LUMA: i32 = 6226;  // 38.0%
const W_ORG: i32 = 3276;   // 20.0%
const HALF_ORG: i32 = W_ORG / 2; // 1638
const MAX_THRESH: i32 = W_BAYER + W_LUMA + W_ORG; // 16383
const FEATHER: i32 = 960;  // ~5.8% optical bloom width
const RECIP: i32 = (256 * 65536 + FEATHER / 2) / FEATHER; // 17476

const BAYER4_RAW: [[i32; 4]; 4] = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
];

#[repr(C)]
struct XImage {
    width: c_int,
    height: c_int,
    xoffset: c_int,
    format: c_int,
    data: *mut c_char,
    byte_order: c_int,
    bitmap_unit: c_int,
    bitmap_bit_order: c_int,
    bitmap_pad: c_int,
    depth: c_int,
    bytes_per_line: c_int,
    bits_per_pixel: c_int,
    red_mask: c_ulong,
    green_mask: c_ulong,
    blue_mask: c_ulong,
    obdata: *mut c_char,
    create_image: *mut c_void,
    destroy_image: Option<unsafe extern "C" fn(*mut XImage) -> c_int>,
    get_pixel: *mut c_void,
    put_pixel: *mut c_void,
    sub_image: *mut c_void,
    add_pixel: *mut c_void,
}

#[repr(C)]
struct XShmSegmentInfo {
    shmseg: c_ulong,
    shmid: c_int,
    shmaddr: *mut c_char,
    read_only: c_int,
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
    fn XDefaultVisual(d: *mut Display, scr: c_int) -> *mut Visual;
    fn XDefaultDepth(d: *mut Display, scr: c_int) -> c_int;
    pub fn XDisplayWidth(d: *mut Display, scr: c_int) -> c_int;
    pub fn XDisplayHeight(d: *mut Display, scr: c_int) -> c_int;
    pub fn XInternAtom(d: *mut Display, name: *const c_char, only_if_exists: c_int) -> Atom;
    fn XCreatePixmap(d: *mut Display, draw: Drawable, w: c_uint, h: c_uint, depth: c_uint) -> Pixmap;
    fn XFreePixmap(d: *mut Display, pmap: Pixmap) -> c_int;
    fn XCreateGC(d: *mut Display, draw: Drawable, mask: c_ulong, vals: *mut c_void) -> GC;
    fn XFreeGC(d: *mut Display, gc: GC) -> c_int;
    fn XSetWindowBackgroundPixmap(d: *mut Display, w: Window, pmap: Pixmap) -> c_int;
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
    pub fn XSetErrorHandler(
        handler: Option<extern "C" fn(*mut Display, *mut c_void) -> c_int>,
    ) -> *mut c_void;
    fn XCreateImage(
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
    fn XPutImage(
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
    fn XGetImage(
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
    fn XShmQueryExtension(d: *mut Display) -> c_int;
    fn XShmCreateImage(
        d: *mut Display,
        vis: *mut Visual,
        depth: c_uint,
        fmt: c_int,
        data: *mut c_char,
        shminfo: *mut XShmSegmentInfo,
        w: c_uint,
        h: c_uint,
    ) -> *mut XImage;
    fn XShmAttach(d: *mut Display, shminfo: *mut XShmSegmentInfo) -> c_int;
    fn XShmDetach(d: *mut Display, shminfo: *mut XShmSegmentInfo) -> c_int;
    fn XShmPutImage(
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

#[link(name = "gdk_pixbuf-2.0")]
extern "C" {
    fn gdk_pixbuf_get_file_info(
        filename: *const c_char,
        width: *mut c_int,
        height: *mut c_int,
    ) -> *mut c_void;
    fn gdk_pixbuf_new_from_file(filename: *const c_char, error: *mut *mut c_void) -> *mut GdkPixbuf;
    fn gdk_pixbuf_new_from_file_at_scale(
        filename: *const c_char,
        width: c_int,
        height: c_int,
        preserve_aspect_ratio: c_int,
        error: *mut *mut c_void,
    ) -> *mut GdkPixbuf;
    fn gdk_pixbuf_new(
        colorspace: c_int,
        has_alpha: c_int,
        bits_per_sample: c_int,
        width: c_int,
        height: c_int,
    ) -> *mut GdkPixbuf;
    fn gdk_pixbuf_scale(
        src: *const GdkPixbuf,
        dest: *mut GdkPixbuf,
        dest_x: c_int,
        dest_y: c_int,
        dest_width: c_int,
        dest_height: c_int,
        offset_x: f64,
        offset_y: f64,
        scale_x: f64,
        scale_y: f64,
        interp_type: c_int,
    );
    fn gdk_pixbuf_get_width(pb: *const GdkPixbuf) -> c_int;
    fn gdk_pixbuf_get_height(pb: *const GdkPixbuf) -> c_int;
    fn gdk_pixbuf_get_rowstride(pb: *const GdkPixbuf) -> c_int;
    fn gdk_pixbuf_get_n_channels(pb: *const GdkPixbuf) -> c_int;
    fn gdk_pixbuf_get_pixels(pb: *const GdkPixbuf) -> *const c_uchar;
}

#[link(name = "gobject-2.0")]
extern "C" {
    fn g_object_unref(obj: *mut c_void);
}

extern "C" {
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
static X_ERR: AtomicBool = AtomicBool::new(false);

pub extern "C" fn on_sig(_sig: c_int) {
    STOP.store(true, Ordering::Relaxed);
}

pub extern "C" fn ignore_x_err(_d: *mut Display, _e: *mut c_void) -> c_int {
    X_ERR.store(true, Ordering::Relaxed);
    0
}

fn ease_develop(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    let smooth = t * t * (3.0 - 2.0 * t);
    let inv = 1.0 - t;
    let cubic_out = 1.0 - inv * inv * inv;
    0.80 * smooth + 0.20 * cubic_out
}

/// Precompute the per-pixel 14-bit threshold field for each monitor once so
/// every animation frame runs at memory bandwidth with zero trig/luma overhead.
fn build_threshold_field(tgt: &[u8], sw: usize, sh: usize, monitors: &[MonRect]) -> Vec<u16> {
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

        // Fast 16x16 stride sample to auto-level luma across dark & bright images.
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

        // Active iff (th < head) AND (th > prev_head - FEATHER).
        let reached = _mm256_cmpgt_epi32(head_v, th);
        let unsettled = _mm256_cmpgt_epi32(th, prev_lim);
        let active = _mm256_and_si256(reached, unsettled);
        if _mm256_testz_si256(active, active) != 0 {
            i += 8;
            continue;
        }

        let off = i * 4;
        // Fast path: all 8 pixels have now reached 100% target (th <= head - FEATHER).
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

fn render_develop_frame(
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

/// Decode an image file directly at a capped resolution into tightly packed RGB
/// bytes (3 bytes/px, stripping unused alpha) so the live shader GPU texture
/// fits in L2 texture cache with minimal decode RAM and VRAM footprint.
pub unsafe fn decode_rgb_image_scaled(path: &str, max_w: i32, max_h: i32) -> Option<(i32, i32, Vec<u8>)> {
    let c_path = CString::new(path).ok()?;
    let mut iw: c_int = 0;
    let mut ih: c_int = 0;
    let pb = if max_w > 0
        && max_h > 0
        && !gdk_pixbuf_get_file_info(c_path.as_ptr(), &mut iw, &mut ih).is_null()
        && iw > 0
        && ih > 0
        && (iw > max_w || ih > max_h)
    {
        gdk_pixbuf_new_from_file_at_scale(c_path.as_ptr(), max_w, max_h, 1, null_mut())
    } else {
        gdk_pixbuf_new_from_file(c_path.as_ptr(), null_mut())
    };
    if pb.is_null() {
        return None;
    }
    let w = gdk_pixbuf_get_width(pb);
    let h = gdk_pixbuf_get_height(pb);
    let rs = gdk_pixbuf_get_rowstride(pb) as usize;
    let ch = gdk_pixbuf_get_n_channels(pb) as usize;
    let px = gdk_pixbuf_get_pixels(pb);
    if w <= 0 || h <= 0 || (ch != 3 && ch != 4) || px.is_null() {
        g_object_unref(pb);
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
    g_object_unref(pb);
    Some((w, h, out))
}

unsafe fn decode_and_scale(
    path: &str,
    sw: i32,
    sh: i32,
    monitors: &[MonRect],
) -> Option<Vec<u8>> {
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
    let src = if !gdk_pixbuf_get_file_info(c_path.as_ptr(), &mut iw, &mut ih).is_null()
        && iw > 0
        && ih > 0
    {
        let s = (max_w as f64 / iw as f64).max(max_h as f64 / ih as f64);
        if s < 1.0 {
            let tw = (iw as f64 * s).ceil() as c_int;
            let th = (ih as f64 * s).ceil() as c_int;
            gdk_pixbuf_new_from_file_at_scale(c_path.as_ptr(), tw.max(max_w), th.max(max_h), 0, null_mut())
        } else {
            gdk_pixbuf_new_from_file(c_path.as_ptr(), null_mut())
        }
    } else {
        gdk_pixbuf_new_from_file(c_path.as_ptr(), null_mut())
    };
    if src.is_null() {
        return None;
    }

    let pw = gdk_pixbuf_get_width(src);
    let ph = gdk_pixbuf_get_height(src);
    if pw <= 0 || ph <= 0 {
        g_object_unref(src);
        return None;
    }

    let mut target = vec![0u8; (sw as usize) * (sh as usize) * 4];

    for m in monitors {
        let k = (m.w as f64 / pw as f64).max(m.h as f64 / ph as f64);
        let ox = (pw as f64 * k - m.w as f64) * 0.5;
        let oy = (ph as f64 * k - m.h as f64) * 0.5;
        let dst = gdk_pixbuf_new(GDK_COLORSPACE_RGB, 0, 8, m.w, m.h);
        if dst.is_null() {
            continue;
        }
        let interp = if (k - 1.0).abs() < 1e-6 {
            GDK_INTERP_NEAREST
        } else {
            GDK_INTERP_BILINEAR
        };
        gdk_pixbuf_scale(src, dst, 0, 0, m.w, m.h, -ox, -oy, k, k, interp);

        let rs = gdk_pixbuf_get_rowstride(dst) as usize;
        let ch = gdk_pixbuf_get_n_channels(dst) as usize;
        let px = gdk_pixbuf_get_pixels(dst);

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
                *d_ptr = *s_ptr.add(2);       // B
                *d_ptr.add(1) = *s_ptr.add(1);// G
                *d_ptr.add(2) = *s_ptr;       // R
                *d_ptr.add(3) = 0;            // pad = 0 (matches Xorg depth-24 top byte)
                s_ptr = s_ptr.add(ch);
                d_ptr = d_ptr.add(4);
            }
        }
        g_object_unref(dst);
    }

    g_object_unref(src);
    Some(target)
}

unsafe fn get_root_pixmap(dpy: *mut Display, root: Window, atom: Atom) -> Pixmap {
    if atom == 0 {
        return 0;
    }
    let mut actual_type: Atom = 0;
    let mut actual_fmt: c_int = 0;
    let mut nitems: c_ulong = 0;
    let mut bytes_after: c_ulong = 0;
    let mut prop: *mut c_uchar = null_mut();
    let rc = XGetWindowProperty(
        dpy,
        root,
        atom,
        0,
        1,
        0,
        XA_PIXMAP,
        &mut actual_type,
        &mut actual_fmt,
        &mut nitems,
        &mut bytes_after,
        &mut prop,
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

unsafe fn pixmap_matches(dpy: *mut Display, pmap: Pixmap, sw: i32, sh: i32, depth: i32) -> bool {
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

/// Read the current static desktop wallpaper from `_XROOTPMAP_ID` into a compact
/// RGB888 buffer (downsampled cleanly by integer stride on large multi-monitor
/// desktops) for seamless Static -> Live GPU transitions inside `--live-daemon`.
pub unsafe fn capture_root_pixmap_rgb(
    dpy: *mut Display,
    root: Window,
    sw: i32,
    sh: i32,
) -> Option<(i32, i32, Vec<u8>)> {
    let c_rootpmap = CString::new("_XROOTPMAP_ID").ok()?;
    let atom_rootpmap = XInternAtom(dpy, c_rootpmap.as_ptr(), 0);
    let pmap = get_root_pixmap(dpy, root, atom_rootpmap);
    let scr = XDefaultScreen(dpy);
    let depth = XDefaultDepth(dpy, scr);
    if !pixmap_matches(dpy, pmap, sw, sh, depth) {
        return None;
    }
    X_ERR.store(false, Ordering::Relaxed);
    let img = XGetImage(dpy, pmap, 0, 0, sw as c_uint, sh as c_uint, ALL_PLANES, ZPIXMAP);
    XSync(dpy, 0);
    if img.is_null() || X_ERR.load(Ordering::Relaxed) || (*img).data.is_null() {
        if !img.is_null() {
            if let Some(destroy) = (*img).destroy_image {
                destroy(img);
            }
        }
        return None;
    }
    let bpp = ((*img).bits_per_pixel / 8).max(3) as usize;
    let bpl = (*img).bytes_per_line as usize;
    let step = ((sw / 1280).max(sh / 720)).max(1) as usize;
    let ow = ((sw as usize) + step - 1) / step;
    let oh = ((sh as usize) + step - 1) / step;
    let mut rgb = vec![0u8; ow * oh * 3];
    let raw = (*img).data as *const u8;
    for oy in 0..oh {
        let sy = (oy * step).min((sh - 1) as usize);
        let s_row = raw.add(sy * bpl);
        let d_off = oy * ow * 3;
        for ox in 0..ow {
            let sx = (ox * step).min((sw - 1) as usize);
            let sp = s_row.add(sx * bpp);
            // X11 depth-24 ZPixmap on little-endian is BGRX
            rgb[d_off + ox * 3] = *sp.add(2);
            rgb[d_off + ox * 3 + 1] = *sp.add(1);
            rgb[d_off + ox * 3 + 2] = *sp;
        }
    }
    if let Some(destroy) = (*img).destroy_image {
        destroy(img);
    }
    Some((ow as i32, oh as i32, rgb))
}

fn pidfile_path() -> String {
    let run = env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    format!("{run}/misa-wall.pid")
}

unsafe fn preempt_previous() -> String {
    let pfile = pidfile_path();
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
    // Compatible with `WallSync` (which parses `feh --no-fehbg --bg... '...'`)
    // and executable as `~/.fehbg`.
    let content = format!("#!/bin/sh\nfeh --no-fehbg --bg-fill '{s}'\n");
    let fehbg = Path::new(&home).join(".fehbg");
    let _ = fs::write(&fehbg, content);
}

fn usage() -> ! {
    eprintln!("usage: misa-wall [--no-transition|--instant] [--static] [--duration MS] [--fps 1..240] <image>");
    std::process::exit(2);
}

fn main() {
    let mut instant = false;
    let mut force_static = false;
    let mut live_daemon_mode = false;
    let mut live_frag: Option<String> = None;
    let mut live_trans_ms: u64 = 0;
    let mut live_ready_token: Option<String> = None;
    let mut duration_ms: u64 = env::var("MISA_WALL_DURATION_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(540);
    let mut fps: u32 = env::var("MISA_WALL_FPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let live_fps_override: Option<u32> = env::var("MISA_WALL_LIVE_FPS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .map(|v| v.clamp(6, 60));
    let mut img_path: Option<String> = None;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-transition" | "--instant" => instant = true,
            "--static" => force_static = true,
            "--stop-live" => {
                unsafe { live::stop_live_daemon(); }
                return;
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
                std::process::exit(live::bake_shader_frame(ip, fp, out, t_sec));
            }
            "--live-daemon" if i + 2 < args.len() => {
                live_daemon_mode = true;
                img_path = Some(args[i + 1].clone());
                live_frag = Some(args[i + 2].clone());
                i += 2;
            }
            "--transition-ms" if i + 1 < args.len() => {
                live_trans_ms = args[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            "--ready-token" if i + 1 < args.len() => {
                live_ready_token = Some(args[i + 1].clone());
                i += 1;
            }
            "--duration" if i + 1 < args.len() => {
                duration_ms = args[i + 1].parse().unwrap_or(540);
                if duration_ms == 0 {
                    instant = true;
                }
                i += 1;
            }
            // Legacy compatibility: accept --step / --direction silently.
            "--step" if i + 1 < args.len() => {
                if args[i + 1].parse::<u8>().ok() == Some(255) {
                    instant = true;
                }
                i += 1;
            }
            "--direction" if i + 1 < args.len() => {
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

    if live_daemon_mode {
        let Some(ip) = img_path else { usage(); };
        let Some(fp) = live_frag else { usage(); };
        let code = live::run_live_daemon(
            &ip,
            &fp,
            live_fps_override,
            live_trans_ms,
            live_ready_token.as_deref(),
        );
        std::process::exit(code);
    }

    let Some(path_str) = img_path else {
        usage();
    };
    let path = Path::new(&path_str);
    if !path.is_file() {
        eprintln!("misa-wall: file not found: {path_str}");
        std::process::exit(1);
    }

    let sidecar_frag = if force_static {
        None
    } else {
        live::find_sidecar_shader(path)
    };

    write_fehbg(path);

    unsafe {
        // Fast path for live shader wallpapers (.frag sidecar):
        // 1. If the exact same live wallpaper is already running, return in < 1ms
        //    with zero flicker, zero animation reset, and zero CPU work!
        // 2. Otherwise, launch the new live daemon FIRST (handling live->live or
        //    static->live GPU Luma-Halftone Develop seamlessly when !instant) and
        //    wait until its first frame is mapped & swapped before quietly syncing
        //    `_XROOTPMAP_ID` underneath — zero undithered static flash on theme switch
        //    or WallSelect Enter!
        let mut live_already_launched = false;
        let mut defer_stop_live = false;
        if let Some(ref frag) = sidecar_frag {
            if live::is_live_running_for(path, frag) {
                _exit(0);
            }
            if let Ok(exe) = env::current_exe() {
                let token = format!("{}", getpid());
                let mut cmd = Command::new(exe);
                cmd.arg("--live-daemon")
                    .arg(&path_str)
                    .arg(frag)
                    .arg("--ready-token")
                    .arg(&token);
                if !instant && duration_ms > 0 {
                    cmd.arg("--transition-ms").arg(duration_ms.to_string());
                }
                cmd.stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null());
                if let Ok(mut child) = cmd.spawn() {
                    if live::wait_for_ready(&mut child, &token, 4000) {
                        live_already_launched = true;
                        // Live GLX window is now mapped & opaque on screen: update
                        // _XROOTPMAP_ID quietly in 1 instant frame underneath it.
                        instant = true;
                    } else if matches!(child.try_wait(), Ok(Some(_))) {
                        // GLX unavailable / daemon exited early: fall back cleanly.
                        live::stop_live_daemon();
                    } else {
                        // Daemon is still initializing on slow software GL: don't kill it.
                        instant = true;
                    }
                } else {
                    live::stop_live_daemon();
                }
            }
        } else if instant {
            // Switching to a custom static wallpaper via WallSelect/instant:
            // publish _XROOTPMAP_ID first, THEN stop old live daemon so there
            // is zero black/stale flash between them.
            defer_stop_live = true;
        } else {
            live::stop_live_daemon();
        }
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

        // 1. Decode and cover-scale the target wallpaper FIRST (before stopping
        //    any running transition) so rapid switches never pause mid-flight.
        let monitors = query_monitors(dpy, root, sw, sh);
        let Some(target_buf) = decode_and_scale(&path_str, sw, sh, &monitors) else {
            eprintln!("misa-wall: failed to decode image: {path_str}");
            std::process::exit(1);
        };

        // 2. Preempt any in-flight misa-wall transition cleanly.
        signal(SIGTERM, on_sig);
        signal(SIGINT, on_sig);
        let pfile = preempt_previous();

        let c_rootpmap = CString::new("_XROOTPMAP_ID").unwrap();
        let c_esetroot = CString::new("ESETROOT_PMAP_ID").unwrap();
        let atom_rootpmap = XInternAtom(dpy, c_rootpmap.as_ptr(), 0);
        let atom_esetroot = XInternAtom(dpy, c_esetroot.as_ptr(), 0);

        let old_root_pmap = get_root_pixmap(dpy, root, atom_rootpmap);
        let old_eset_pmap = get_root_pixmap(dpy, root, atom_esetroot);

        // 3. Set up MIT-SHM XImage (or standard XImage fallback).
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
                dpy,
                vis,
                depth as c_uint,
                ZPIXMAP,
                null_mut(),
                &mut shminfo,
                sw as c_uint,
                sh as c_uint,
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
                dpy,
                vis,
                depth as c_uint,
                ZPIXMAP,
                0,
                _fallback_vec.as_mut_ptr() as *mut c_char,
                sw as c_uint,
                sh as c_uint,
                32,
                sw * 4,
            );
            if ximg.is_null() {
                std::process::exit(1);
            }
        }

        let cur_slice: &mut [u8] =
            std::slice::from_raw_parts_mut((*ximg).data as *mut u8, buf_len);

        // 4. Read the previous wallpaper from `_XROOTPMAP_ID` so we transition
        //    smoothly from whatever is currently on screen.
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
                    dpy,
                    old_root_pmap,
                    0,
                    0,
                    sw as c_uint,
                    sh as c_uint,
                    ALL_PLANES,
                    ZPIXMAP,
                    ximg,
                    0,
                    0,
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
                    XShmPutImage(
                        dpy,
                        pmap,
                        gc,
                        ximg,
                        mx,
                        my,
                        mx,
                        my,
                        mw as c_uint,
                        mh as c_uint,
                        0,
                    );
                } else {
                    XPutImage(
                        dpy,
                        pmap,
                        gc,
                        ximg,
                        mx,
                        my,
                        mx,
                        my,
                        mw as c_uint,
                        mh as c_uint,
                    );
                }
            }
        };

        let publish_pixmap = |pmap: Pixmap| {
            XSetWindowBackgroundPixmap(dpy, root, pmap);
            XChangeProperty(
                dpy,
                root,
                atom_rootpmap,
                XA_PIXMAP,
                32,
                PROP_MODE_REPLACE,
                &pmap as *const Pixmap as *const c_uchar,
                1,
            );
            XChangeProperty(
                dpy,
                root,
                atom_esetroot,
                XA_PIXMAP,
                32,
                PROP_MODE_REPLACE,
                &pmap as *const Pixmap as *const c_uchar,
                1,
            );
            XClearWindow(dpy, root);
            XSync(dpy, 0);
        };

        let identical = have_old && cur_slice == target_buf.as_slice();

        if !have_old || identical {
            // Instant set (first boot, --no-transition, or unchanged wallpaper).
            cur_slice.copy_from_slice(&target_buf);
            let pmap = XCreatePixmap(dpy, root, sw as c_uint, sh as c_uint, depth as c_uint);
            blit_to_pixmap(pmap);
            publish_pixmap(pmap);
            if old_eset_pmap != 0 && old_eset_pmap != pmap {
                XKillClient(dpy, old_eset_pmap);
                XSync(dpy, 0);
            }
        } else {
            // Double-buffered 60 FPS Mira Luma-Halftone Develop transition.
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

                render_develop_frame(
                    cur_slice,
                    &src_buf,
                    &target_buf,
                    &thresh,
                    head,
                    prev_head,
                    use_avx2,
                );
                prev_head = head;

                blit_to_pixmap(pmaps[idx]);
                publish_pixmap(pmaps[idx]);

                if !freed_old {
                    freed_old = true;
                    if old_eset_pmap != 0
                        && old_eset_pmap != pmaps[0]
                        && old_eset_pmap != pmaps[1]
                    {
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

        // Clean up temporary resources (leaving only the active root Pixmap).
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

        if defer_stop_live {
            live::stop_live_daemon();
        }
        let _ = live_already_launched;
        _exit(0);
    }
}
