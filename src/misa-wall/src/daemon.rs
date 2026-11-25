//! Zero-OpenGL, ultra-lightweight X11 animated wallpaper daemon for `misa-wall` v3.
//!
//! Architecture & optimization highlights:
//! - ZERO `libGL.so` / ZERO Mesa / ZERO `libgdk_pixbuf` in the persistent daemon
//!   (pure `X11 + Xext + Xrandr + Xrender + libc`).
//! - Hardware `XRender` nearest-neighbor 3x scaling from a tiny `640x360x4`
//!   (921 KB, ~0.45 MiB PSS) MIT-SHM buffer directly onto the desktop window —
//!   eliminating the 8.3 MB `1920x1080` SHM buffer, the `691 KB` `cell_canvas`
//!   copy, and 100% of CPU pixel-expansion work during animated playback!
//! - Zero-PSS `pread` streaming of MWA3 Planar-Delta BitPack+LZ4 frames directly
//!   into the `640x360` BGRX SHM buffer (no multi-MB `.mwa` file resident in PSS!).
//! - Coarse `16x9` tile dirty bitmasks per frame: uploads & composites ONLY the
//!   changed tile rectangles intersected with each monitor's unoccluded region,
//!   slashing Xorg & Picom XDamage/texture-upload bandwidth!
//! - Decoupled DPMS polling + per-window opaqueness cache: zero periodic full
//!   window-tree scans when no windows are moving.
//! - Automatic MIT-SHM detachment + scratch buffer reclamation + `POSIX_FADV_DONTNEED`
//!   after 2s of pause/occlusion, waking cleanly in < 1 ms.

use std::collections::HashMap;
use std::env;
use std::ffi::{c_char, c_int, c_long, c_uchar, c_uint, c_ulong, c_ushort, c_void, CStr, CString};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use crate::codec::{
    lz4_decompress, merge_tile_rects, unpack_planar_delta_bgrx,
    unpack_planar_delta_bgrx_unchecked, FrameEntry, MwaFile, TILE_ROWS,
};
use crate::{
    build_threshold_field, decode_and_scale, ease_develop, get_root_pixmap, getpid, ignore_x_err,
    on_sig, pixmap_matches, publish_persistent_root_pixmap, query_monitors, render_develop_frame,
    signal, Atom, Display, MonRect, Pixmap, Visual, Window, XCreateGC, XCreateImage, XCreatePixmap,
    XDefaultDepth, XDefaultRootWindow, XDefaultScreen, XDefaultVisual, XDisplayHeight,
    XDisplayWidth, XFree, XFreeGC, XFreePixmap, XGetImage, XGetWindowProperty, XImage, XInternAtom,
    XOpenDisplay, XPutImage, XSetErrorHandler, XSetWindowBackgroundPixmap, XShmAttach,
    XShmCreateImage, XShmDetach, XShmPutImage, XShmQueryExtension, XShmSegmentInfo, XSync,
    ALL_PLANES, FEATHER, GC, IPC_CREAT, IPC_PRIVATE, IPC_RMID, MAX_THRESH, PROP_MODE_REPLACE,
    SIGINT, SIGTERM, STOP, XA_PIXMAP, X_ERR, ZPIXMAP,
};

const XA_ATOM: Atom = 4;
const XA_CARDINAL: Atom = 6;
const XA_WINDOW: Atom = 33;

const INPUT_OUTPUT: c_uint = 1;
const IS_VIEWABLE: c_int = 2;
const CW_BACK_PIXMAP: c_ulong = 1 << 0;
const CW_BORDER_PIXEL: c_ulong = 1 << 3;
const CW_OVERRIDE_REDIRECT: c_ulong = 1 << 9;
const CW_EVENT_MASK: c_ulong = 1 << 11;
const STRUCTURE_NOTIFY_MASK: i64 = 1 << 17;
const SUBSTRUCTURE_NOTIFY_MASK: i64 = 1 << 19;
const PROPERTY_CHANGE_MASK: i64 = 1 << 22;

const SHAPE_SET: c_int = 0;
const SHAPE_INPUT: c_int = 2;

const PICT_OP_SRC: c_int = 1;

const LOCK_EX: c_int = 2;
const LOCK_NB: c_int = 4;
const POLLIN: i16 = 0x0001;
const RTLD_LAZY: c_int = 1;
const RTLD_LOCAL: c_int = 0;

const GRID_X: usize = 32;
const GRID_Y: usize = 18;

type Picture = c_ulong;
type XFixed = c_int;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct XTransform {
    matrix: [[XFixed; 3]; 3],
}

#[repr(C)]
struct XSetWindowAttributes {
    background_pixmap: c_ulong,
    background_pixel: c_ulong,
    border_pixmap: c_ulong,
    border_pixel: c_ulong,
    bit_gravity: c_int,
    win_gravity: c_int,
    backing_store: c_int,
    backing_planes: c_ulong,
    backing_pixel: c_ulong,
    save_under: c_int,
    event_mask: i64,
    do_not_propagate_mask: i64,
    override_redirect: c_int,
    colormap: c_ulong,
    cursor: c_ulong,
}

#[repr(C)]
struct XClassHint {
    res_name: *mut c_char,
    res_class: *mut c_char,
}

#[repr(C)]
struct XWindowAttributes {
    x: c_int,
    y: c_int,
    width: c_int,
    height: c_int,
    border_width: c_int,
    depth: c_int,
    visual: *mut Visual,
    root: Window,
    class: c_int,
    bit_gravity: c_int,
    win_gravity: c_int,
    backing_store: c_int,
    backing_planes: c_ulong,
    backing_pixel: c_ulong,
    save_under: c_int,
    colormap: c_ulong,
    map_installed: c_int,
    map_state: c_int,
    all_event_masks: c_long,
    your_event_mask: c_long,
    do_not_propagate_mask: c_long,
    override_redirect: c_int,
    screen: *mut c_void,
}

#[repr(C)]
struct XEvent {
    pad: [c_long; 24],
}

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: i16,
    revents: i16,
}

#[link(name = "X11")]
extern "C" {
    fn XCloseDisplay(d: *mut Display) -> c_int;
    fn XConnectionNumber(d: *mut Display) -> c_int;
    fn XCreateWindow(
        d: *mut Display,
        parent: Window,
        x: c_int,
        y: c_int,
        w: c_uint,
        h: c_uint,
        bw: c_uint,
        depth: c_int,
        class: c_uint,
        vis: *mut Visual,
        mask: c_ulong,
        attrs: *mut XSetWindowAttributes,
    ) -> Window;
    fn XDestroyWindow(d: *mut Display, w: Window) -> c_int;
    fn XMapWindow(d: *mut Display, w: Window) -> c_int;
    fn XUnmapWindow(d: *mut Display, w: Window) -> c_int;
    fn XLowerWindow(d: *mut Display, w: Window) -> c_int;
    fn XResizeWindow(d: *mut Display, w: Window, w_px: c_uint, h_px: c_uint) -> c_int;
    fn XSelectInput(d: *mut Display, w: Window, mask: i64) -> c_int;
    fn XPending(d: *mut Display) -> c_int;
    fn XNextEvent(d: *mut Display, ev: *mut XEvent) -> c_int;
    fn XFlush(d: *mut Display) -> c_int;
    fn XSetClassHint(d: *mut Display, w: Window, ch: *mut XClassHint) -> c_int;
    fn XGetClassHint(d: *mut Display, w: Window, ch: *mut XClassHint) -> c_int;
    fn XGetWindowAttributes(d: *mut Display, w: Window, attr: *mut XWindowAttributes) -> c_int;
    fn XTranslateCoordinates(
        d: *mut Display,
        src: Window,
        dst: Window,
        sx: c_int,
        sy: c_int,
        dx: *mut c_int,
        dy: *mut c_int,
        child: *mut Window,
    ) -> c_int;
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
    fn XClearWindow(d: *mut Display, w: Window) -> c_int;
}

#[link(name = "Xext")]
extern "C" {
    fn XShapeCombineRectangles(
        d: *mut Display,
        dest: Window,
        dest_kind: c_int,
        x_off: c_int,
        y_off: c_int,
        rects: *const c_void,
        n_rects: c_int,
        op: c_int,
        ordering: c_int,
    );
}

#[link(name = "Xrender")]
extern "C" {
    fn XRenderQueryExtension(d: *mut Display, ev_base: *mut c_int, err_base: *mut c_int) -> c_int;
    fn XRenderFindVisualFormat(d: *mut Display, vis: *const Visual) -> *mut c_void;
    fn XRenderCreatePicture(
        d: *mut Display,
        drawable: c_ulong,
        fmt: *const c_void,
        valuemask: c_ulong,
        attrs: *const c_void,
    ) -> Picture;
    fn XRenderFreePicture(d: *mut Display, pic: Picture);
    fn XRenderSetPictureFilter(
        d: *mut Display,
        pic: Picture,
        filter: *const c_char,
        params: *const XFixed,
        nparams: c_int,
    );
    fn XRenderSetPictureTransform(d: *mut Display, pic: Picture, transform: *const XTransform);
    fn XRenderComposite(
        d: *mut Display,
        op: c_int,
        src: Picture,
        mask: Picture,
        dst: Picture,
        src_x: c_int,
        src_y: c_int,
        mask_x: c_int,
        mask_y: c_int,
        dst_x: c_int,
        dst_y: c_int,
        width: c_uint,
        height: c_uint,
    );
}

extern "C" {
    fn setsid() -> c_int;
    fn close(fd: c_int) -> c_int;
    fn flock(fd: c_int, op: c_int) -> c_int;
    fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
    fn malloc_trim(pad: usize) -> c_int;
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn shmget(key: c_int, size: usize, shmflg: c_int) -> c_int;
    fn shmat(shmid: c_int, shmaddr: *const c_void, shmflg: c_int) -> *mut c_void;
    fn shmdt(shmaddr: *const c_void) -> c_int;
    fn shmctl(shmid: c_int, cmd: c_int, buf: *mut c_void) -> c_int;
    fn kill(pid: c_int, sig: c_int) -> c_int;
}

type FnDPMSInfo = unsafe extern "C" fn(*mut Display, *mut c_ushort, *mut c_uchar) -> c_int;

unsafe fn load_dpms_info() -> Option<FnDPMSInfo> {
    let cs = CString::new("libXext.so.6").unwrap();
    let lib = dlopen(cs.as_ptr(), RTLD_LAZY | RTLD_LOCAL);
    if lib.is_null() {
        return None;
    }
    let p = dlsym(lib, CString::new("DPMSInfo").unwrap().as_ptr());
    if p.is_null() {
        None
    } else {
        Some(std::mem::transmute(p))
    }
}

unsafe fn query_dpms_off(dpy: *mut Display, dpms_info: Option<FnDPMSInfo>) -> bool {
    if let Some(dpms) = dpms_info {
        let mut level: c_ushort = 0;
        let mut state: c_uchar = 0;
        if dpms(dpy, &mut level, &mut state) != 0 && state != 0 && level != 0 {
            return true;
        }
    }
    false
}

pub fn disp_safe() -> String {
    let raw = env::var("DISPLAY").unwrap_or_else(|_| ":0".to_string());
    let s: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() {
        "0".to_string()
    } else {
        s
    }
}

fn runtime_path(suffix: &str) -> String {
    let run = env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    let disp = disp_safe();
    format!("{run}/misa-wall-live-{disp}.{suffix}")
}

pub fn socket_path() -> PathBuf {
    let run = env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_string());
    let disp = disp_safe();
    PathBuf::from(format!("{run}/misa-wall-{disp}.sock"))
}

pub fn pidfile_path() -> String {
    runtime_path("pid")
}

pub fn state_path() -> String {
    runtime_path("state")
}

pub fn lock_path() -> String {
    runtime_path("lock")
}

pub fn pause_path() -> String {
    runtime_path("pause")
}

pub fn find_sidecar_shader(image_path: &Path) -> Option<PathBuf> {
    if env::var("MISA_WALL_LIVE").ok().as_deref() == Some("0") {
        return None;
    }
    let canon = fs::canonicalize(image_path).unwrap_or_else(|_| image_path.to_path_buf());
    let sidecar = canon.with_extension("frag");
    if sidecar.is_file() {
        Some(sidecar)
    } else {
        None
    }
}

pub fn ipc_send(cmd: &str) -> Result<String, String> {
    let sp = socket_path();
    let mut stream = UnixStream::connect(&sp).map_err(|e| format!("connect {}: {e}", sp.display()))?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    stream
        .write_all(cmd.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let mut out = String::new();
    stream
        .read_to_string(&mut out)
        .map_err(|e| format!("read: {e}"))?;
    Ok(out)
}

pub fn is_daemon_alive() -> bool {
    ipc_send("PING\n").map(|s| s.trim() == "PONG").unwrap_or(false)
}

pub unsafe fn stop_daemon() {
    if ipc_send("KILL\n").is_ok() {
        for _ in 0..40 {
            if !is_daemon_alive() {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    let pfile = pidfile_path();
    let sfile = state_path();
    let self_pid = getpid();
    let target_disp = env::var("DISPLAY").unwrap_or_else(|_| ":0".to_string());

    let mut pids: Vec<c_int> = Vec::new();
    if let Ok(txt) = fs::read_to_string(&pfile) {
        if let Ok(old_pid) = txt.trim().parse::<c_int>() {
            if old_pid > 1 && old_pid != self_pid {
                pids.push(old_pid);
            }
        }
    }
    if let Ok(entries) = fs::read_dir("/proc") {
        for e in entries.flatten() {
            let Ok(name) = e.file_name().into_string() else { continue; };
            let Ok(pid) = name.parse::<c_int>() else { continue; };
            if pid <= 1 || pid == self_pid || pids.contains(&pid) {
                continue;
            }
            let Ok(comm) = fs::read_to_string(format!("/proc/{pid}/comm")) else { continue; };
            if comm.trim() != "misa-wall" {
                continue;
            }
            let Ok(cmd) = fs::read(format!("/proc/{pid}/cmdline")) else { continue; };
            let is_daemon = cmd.windows(8).any(|w| w == b"--daemon")
                || cmd.windows(13).any(|w| w == b"--live-daemon");
            if !is_daemon {
                continue;
            }
            let Ok(env_raw) = fs::read(format!("/proc/{pid}/environ")) else { continue; };
            let disp_ok = env_raw.split(|&b| b == 0).any(|entry| {
                entry
                    .strip_prefix(b"DISPLAY=")
                    .map(|v| v == target_disp.as_bytes())
                    .unwrap_or(false)
            });
            if disp_ok {
                pids.push(pid);
            }
        }
    }
    for &pid in &pids {
        let _ = kill(pid, SIGTERM);
    }
    for &pid in &pids {
        for _ in 0..40 {
            if kill(pid, 0) != 0 {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
    let _ = fs::remove_file(&pfile);
    let _ = fs::remove_file(&sfile);
    let _ = fs::remove_file(socket_path());
}

unsafe fn close_inherited_fds() {
    if let Ok(entries) = fs::read_dir("/proc/self/fd") {
        let mut to_close = Vec::new();
        for e in entries.flatten() {
            if let Ok(name) = e.file_name().into_string() {
                if let Ok(fd) = name.parse::<c_int>() {
                    if fd >= 3 {
                        to_close.push(fd);
                    }
                }
            }
        }
        for fd in to_close {
            let _ = close(fd);
        }
    }
}

fn on_battery_power() -> bool {
    let Ok(entries) = fs::read_dir("/sys/class/power_supply") else {
        return false;
    };
    let mut saw_ac = false;
    for e in entries.flatten() {
        let p = e.path();
        let Ok(kind) = fs::read_to_string(p.join("type")) else { continue; };
        if kind.trim() == "Mains" {
            saw_ac = true;
            if let Ok(online) = fs::read_to_string(p.join("online")) {
                if online.trim() == "1" {
                    return false;
                }
            }
        }
    }
    saw_ac
}

fn detect_terminal_opaque() -> bool {
    let ghostty_cfg = env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| env::var("HOME").ok().map(|h| PathBuf::from(h).join(".config")))
        .map(|c| c.join("ghostty").join("config"));
    if let Some(p) = ghostty_cfg {
        if let Ok(txt) = fs::read_to_string(&p) {
            let mut last_op: Option<f32> = None;
            for line in txt.lines() {
                let l = line.trim();
                if l.starts_with('#') {
                    continue;
                }
                if let Some(rest) = l.strip_prefix("background-opacity") {
                    if let Some(val) = rest.split('=').nth(1) {
                        if let Ok(v) = val.trim().parse::<f32>() {
                            last_op = Some(v);
                        }
                    }
                }
            }
            if let Some(v) = last_op {
                return v >= 0.99;
            }
        }
    }
    let mira_dir = env::var("MIRA_CONFIG_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| env::var("HOME").ok().map(|h| PathBuf::from(h).join(".config/bspwm")));
    if let Some(d) = mira_dir {
        if let Ok(txt) = fs::read_to_string(d.join("config/picom/picom-rules.conf")) {
            for line in txt.lines() {
                if line.contains("#-blur-switch") && line.contains("true") {
                    return false;
                }
            }
        }
    }
    true
}

unsafe fn external_picker_paused(ppath: &str) -> bool {
    let Ok(txt) = fs::read_to_string(ppath) else {
        return false;
    };
    let Ok(pid) = txt.trim().parse::<c_int>() else {
        return false;
    };
    if pid > 1 && kill(pid, 0) == 0 {
        true
    } else {
        let _ = fs::remove_file(ppath);
        false
    }
}

unsafe fn window_has_atom(dpy: *mut Display, w: Window, prop_atom: Atom, needle: Atom) -> bool {
    if prop_atom == 0 || needle == 0 {
        return false;
    }
    let mut act_type: Atom = 0;
    let mut act_fmt: c_int = 0;
    let mut nitems: c_ulong = 0;
    let mut after: c_ulong = 0;
    let mut prop: *mut c_uchar = null_mut();
    let rc = XGetWindowProperty(
        dpy, w, prop_atom, 0, 16, 0, XA_ATOM,
        &mut act_type, &mut act_fmt, &mut nitems, &mut after, &mut prop,
    );
    let mut found = false;
    if rc == 0 && !prop.is_null() {
        if act_fmt == 32 {
            let atoms = std::slice::from_raw_parts(prop as *const Atom, nitems as usize);
            found = atoms.contains(&needle);
        }
        XFree(prop as *mut c_void);
    }
    found
}

unsafe fn window_is_opaque_opacity(dpy: *mut Display, w: Window, atom_opacity: Atom) -> bool {
    if atom_opacity == 0 {
        return true;
    }
    let mut act_type: Atom = 0;
    let mut act_fmt: c_int = 0;
    let mut nitems: c_ulong = 0;
    let mut after: c_ulong = 0;
    let mut prop: *mut c_uchar = null_mut();
    let rc = XGetWindowProperty(
        dpy, w, atom_opacity, 0, 1, 0, XA_CARDINAL,
        &mut act_type, &mut act_fmt, &mut nitems, &mut after, &mut prop,
    );
    let mut opaque = true;
    if rc == 0 && !prop.is_null() {
        if act_fmt == 32 && nitems >= 1 {
            let val = *(prop as *const c_ulong) as u32;
            opaque = val >= 0xFE00_0000;
        }
        XFree(prop as *mut c_void);
    }
    opaque
}

/// Classify whether a mapped client window is opaque (cached by `Window` ID).
unsafe fn window_is_effectively_opaque_cached(
    dpy: *mut Display,
    w: Window,
    depth: c_int,
    is_fs: bool,
    atom_opacity: Atom,
    term_opaque: bool,
    cache: &mut HashMap<Window, bool>,
) -> bool {
    if is_fs {
        return true;
    }
    if let Some(&cached) = cache.get(&w) {
        return cached;
    }
    let opaque = (|| {
        if !window_is_opaque_opacity(dpy, w, atom_opacity) {
            return false;
        }
        let mut ch = XClassHint {
            res_name: null_mut(),
            res_class: null_mut(),
        };
        let mut cls_low = String::new();
        let mut name_low = String::new();
        if XGetClassHint(dpy, w, &mut ch) != 0 {
            if !ch.res_class.is_null() {
                cls_low = CStr::from_ptr(ch.res_class).to_string_lossy().to_ascii_lowercase();
                XFree(ch.res_class as *mut c_void);
            }
            if !ch.res_name.is_null() {
                name_low = CStr::from_ptr(ch.res_name).to_string_lossy().to_ascii_lowercase();
                XFree(ch.res_name as *mut c_void);
            }
        }
        for pat in [
            "backdrop",
            "cardpicker",
            "rofi",
            "polybar",
            "soundpanel",
            "musicpanel",
            "notiftoast",
            "notifcenter",
            "mediapill",
            "windowswitch",
            "clippanel",
            "wifipanel",
            "bluetoothpanel",
            "x-shot",
            "misawalllive",
        ] {
            if cls_low.contains(pat) || name_low.contains(pat) {
                return false;
            }
        }
        for pat in ["ghostty", "alacritty", "kitty", "xterm", "wezterm", "st-256color"] {
            if cls_low.contains(pat) || name_low.contains(pat) {
                return term_opaque;
            }
        }
        depth == 24 || depth == 32
    })();
    if cache.len() >= 128 {
        cache.clear();
    }
    cache.insert(w, opaque);
    opaque
}

#[derive(Clone, Copy)]
struct MonState {
    visible: bool,
    mostly_covered: bool,
    exp_x0: i32,
    exp_y0: i32,
    exp_x1: i32,
    exp_y1: i32,
}

impl MonState {
    fn full(m: MonRect) -> Self {
        Self {
            visible: true,
            mostly_covered: false,
            exp_x0: m.x,
            exp_y0: m.y,
            exp_x1: m.x + m.w,
            exp_y1: m.y + m.h,
        }
    }
}

/// 32x18 union-coverage occlusion & exposed-rect solver across all monitors.
unsafe fn compute_monitor_states(
    dpy: *mut Display,
    root: Window,
    self_win: Window,
    monitors: &[MonRect],
    atom_stacking: Atom,
    atom_active: Atom,
    atom_state: Atom,
    atom_fs: Atom,
    atom_opacity: Atom,
    dpms_off: bool,
    term_opaque: bool,
    opaque_cache: &mut HashMap<Window, bool>,
    states: &mut Vec<MonState>,
) {
    states.clear();
    for &m in monitors {
        states.push(MonState::full(m));
    }

    if dpms_off {
        for st in states.iter_mut() {
            st.visible = false;
        }
        return;
    }

    let mut wins: Vec<Window> = Vec::new();
    let mut act_type: Atom = 0;
    let mut act_fmt: c_int = 0;
    let mut nitems: c_ulong = 0;
    let mut after: c_ulong = 0;
    let mut prop: *mut c_uchar = null_mut();
    if atom_stacking != 0 {
        let rc = XGetWindowProperty(
            dpy, root, atom_stacking, 0, 256, 0, XA_WINDOW,
            &mut act_type, &mut act_fmt, &mut nitems, &mut after, &mut prop,
        );
        if rc == 0 && !prop.is_null() {
            if act_fmt == 32 && nitems > 0 {
                let sl = std::slice::from_raw_parts(prop as *const Window, nitems as usize);
                wins.extend_from_slice(sl);
            }
            XFree(prop as *mut c_void);
        }
    }
    if wins.is_empty() && atom_active != 0 {
        let rc = XGetWindowProperty(
            dpy, root, atom_active, 0, 1, 0, XA_WINDOW,
            &mut act_type, &mut act_fmt, &mut nitems, &mut after, &mut prop,
        );
        if rc == 0 && !prop.is_null() {
            if act_fmt == 32 && nitems >= 1 {
                wins.push(*(prop as *const Window));
            }
            XFree(prop as *mut c_void);
        }
    }

    let n_mon = monitors.len();
    let mut grids: Vec<[u32; GRID_Y]> = vec![[0u32; GRID_Y]; n_mon];
    let mut cov_areas: Vec<i64> = vec![0; n_mon];

    for &w in wins.iter().rev() {
        if w == 0 || w == root || w == self_win {
            continue;
        }
        let mut wa: XWindowAttributes = std::mem::zeroed();
        if XGetWindowAttributes(dpy, w, &mut wa) == 0 || wa.map_state != IS_VIEWABLE {
            continue;
        }
        let rw = wa.width;
        let rh = wa.height;
        if rw <= 0 || rh <= 0 {
            continue;
        }
        let is_fs = window_has_atom(dpy, w, atom_state, atom_fs);
        let is_opaque = window_is_effectively_opaque_cached(
            dpy, w, wa.depth, is_fs, atom_opacity, term_opaque, opaque_cache,
        );

        let mut rx: c_int = wa.x;
        let mut ry: c_int = wa.y;
        let mut child: Window = 0;
        let _ = XTranslateCoordinates(dpy, w, root, 0, 0, &mut rx, &mut ry, &mut child);

        let count_for_cov = is_fs || (rw > 64 && rh > 64);

        for (i, m) in monitors.iter().enumerate() {
            let ix0 = rx.max(m.x);
            let iy0 = ry.max(m.y);
            let ix1 = (rx + rw).min(m.x + m.w);
            let iy1 = (ry + rh).min(m.y + m.h);
            if ix1 <= ix0 || iy1 <= iy0 {
                continue;
            }
            if count_for_cov {
                cov_areas[i] = cov_areas[i].saturating_add((ix1 - ix0) as i64 * (iy1 - iy0) as i64);
            }
            if !is_opaque {
                continue;
            }

            let wa_x0 = m.x + 14;
            let wa_y0 = m.y + 54;
            let wa_x1 = m.x + m.w - 14;
            let wa_y1 = m.y + m.h - 14;
            let wa_w = (wa_x1 - wa_x0).max(1);
            let wa_h = (wa_y1 - wa_y0).max(1);

            let inset = if is_fs { 0 } else { 6 };
            let ox0 = if rx <= wa_x0 + 8 { wa_x0 } else { rx + inset };
            let oy0 = if ry <= wa_y0 + 8 { wa_y0 } else { ry + inset };
            let ox1 = if rx + rw >= wa_x1 - 8 { wa_x1 } else { rx + rw - inset };
            let oy1 = if ry + rh >= wa_y1 - 8 { wa_y1 } else { ry + rh - inset };
            if ox1 <= ox0 || oy1 <= oy0 {
                continue;
            }

            let gx0 = (((ox0 - wa_x0).max(0) as i64 * GRID_X as i64) / wa_w as i64).clamp(0, GRID_X as i64) as usize;
            let gx1 = ((((ox1 - wa_x0).max(0) as i64 * GRID_X as i64) + wa_w as i64 - 1) / wa_w as i64).clamp(0, GRID_X as i64) as usize;
            let gy0 = (((oy0 - wa_y0).max(0) as i64 * GRID_Y as i64) / wa_h as i64).clamp(0, GRID_Y as i64) as usize;
            let gy1 = ((((oy1 - wa_y0).max(0) as i64 * GRID_Y as i64) + wa_h as i64 - 1) / wa_h as i64).clamp(0, GRID_Y as i64) as usize;

            if gx1 > gx0 && gy1 > gy0 {
                let row_mask: u32 = if gx1 - gx0 >= 32 {
                    u32::MAX
                } else {
                    ((1u32 << (gx1 - gx0)) - 1) << gx0
                };
                for gy in gy0..gy1 {
                    grids[i][gy] |= row_mask;
                }
            }
        }
    }

    let full_row: u32 = if GRID_X >= 32 { u32::MAX } else { (1u32 << GRID_X) - 1 };

    for (i, m) in monitors.iter().enumerate() {
        let mut min_gx = GRID_X;
        let mut max_gx = 0usize;
        let mut min_gy = GRID_Y;
        let mut max_gy = 0usize;
        let mut uncovered = 0u32;

        for gy in 0..GRID_Y {
            let uncov_mask = (!grids[i][gy]) & full_row;
            if uncov_mask != 0 {
                uncovered += uncov_mask.count_ones();
                let lo = uncov_mask.trailing_zeros() as usize;
                let hi = 32 - uncov_mask.leading_zeros() as usize;
                if lo < min_gx { min_gx = lo; }
                if hi > max_gx { max_gx = hi; }
                if gy < min_gy { min_gy = gy; }
                if gy + 1 > max_gy { max_gy = gy + 1; }
            }
        }

        if uncovered <= 10 {
            states[i].visible = false;
            states[i].mostly_covered = true;
            states[i].exp_x1 = states[i].exp_x0;
        } else {
            states[i].visible = true;
            let mon_area = (m.w.max(1) as i64) * (m.h.max(1) as i64);
            states[i].mostly_covered = cov_areas[i] * 100 >= mon_area * 40;
            let x0 = if min_gx <= 1 {
                m.x
            } else {
                m.x + ((min_gx - 1) as i32 * m.w) / (GRID_X as i32)
            };
            let y0 = if min_gy <= 1 {
                m.y
            } else {
                m.y + ((min_gy - 1) as i32 * m.h) / (GRID_Y as i32)
            };
            let x1 = if max_gx >= GRID_X - 1 {
                m.x + m.w
            } else {
                m.x + (((max_gx + 1) as i32 * m.w) + GRID_X as i32 - 1) / (GRID_X as i32)
            };
            let y1 = if max_gy >= GRID_Y - 1 {
                m.y + m.h
            } else {
                m.y + (((max_gy + 1) as i32 * m.h) + GRID_Y as i32 - 1) / (GRID_Y as i32)
            };
            states[i].exp_x0 = x0.clamp(m.x, m.x + m.w);
            states[i].exp_y0 = y0.clamp(m.y, m.y + m.h);
            states[i].exp_x1 = x1.clamp(states[i].exp_x0, m.x + m.w);
            states[i].exp_y1 = y1.clamp(states[i].exp_y0, m.y + m.h);
        }
    }
}

// ============================================================================
// Per-monitor cell -> screen mapping (XRender hardware transform + CPU fallback)
// ============================================================================

struct MonExpand {
    rect: MonRect,
    exact_3x: bool,
    xform: XTransform,
    x_lut: Vec<u16>,
    y_lut: Vec<u16>,
}

impl MonExpand {
    fn new(rect: MonRect, cell_w: u32, cell_h: u32) -> Self {
        let cw = cell_w as i32;
        let ch = cell_h as i32;
        let exact_3x = rect.w == cw * 3 && rect.h == ch * 3;
        if exact_3x {
            // Homogeneous exact 1/3 scaling: src = dst / 3 (zero fixed-point drift!).
            let xform = XTransform {
                matrix: [[65536, 0, 0], [0, 65536, 0], [0, 0, 3 * 65536]],
            };
            return Self {
                rect,
                exact_3x: true,
                xform,
                x_lut: Vec::new(),
                y_lut: Vec::new(),
            };
        }

        let mw = rect.w.max(1) as f64;
        let mh = rect.h.max(1) as f64;
        let cw_f = cw.max(1) as f64;
        let ch_f = ch.max(1) as f64;
        let vp_a = mw / mh;
        let im_a = cw_f / ch_f;
        let (sx, sy, ox, oy) = if vp_a > im_a {
            let sy = im_a / vp_a;
            (1.0, sy, 0.0, (1.0 - sy) * 0.5)
        } else {
            let sx = vp_a / im_a;
            (sx, 1.0, (1.0 - sx) * 0.5, 0.0)
        };
        let to_fixed = |v: f64| -> XFixed { (v * 65536.0).round() as XFixed };
        let xform = XTransform {
            matrix: [
                [to_fixed(cw_f * sx / mw), 0, to_fixed(cw_f * ox)],
                [0, to_fixed(ch_f * sy / mh), to_fixed(ch_f * oy)],
                [0, 0, 65536],
            ],
        };

        let mut x_lut = Vec::with_capacity(rect.w as usize);
        for lx in 0..rect.w {
            let u = ((lx as f64 + 0.5) / mw) * sx + ox;
            let cx = ((u * cw_f).floor() as i32).clamp(0, cw - 1) as u16;
            x_lut.push(cx);
        }
        let mut y_lut = Vec::with_capacity(rect.h as usize);
        for ly in 0..rect.h {
            let v = ((ly as f64 + 0.5) / mh) * sy + oy;
            let cy = ((v * ch_f).floor() as i32).clamp(0, ch - 1) as u16;
            y_lut.push(cy);
        }
        Self {
            rect,
            exact_3x: false,
            xform,
            x_lut,
            y_lut,
        }
    }

    /// Map cell rect `(cx0, cy0, cx1, cy1)` to monitor screen rect `(sx0, sy0, sx1, sy1)`.
    #[inline]
    fn cell_rect_to_screen(
        &self,
        cx0: i32,
        cy0: i32,
        cx1: i32,
        cy1: i32,
        cell_w: i32,
        cell_h: i32,
    ) -> (i32, i32, i32, i32) {
        if self.exact_3x {
            let x0 = self.rect.x + cx0 * 3;
            let y0 = self.rect.y + cy0 * 3;
            let x1 = self.rect.x + cx1 * 3;
            let y1 = self.rect.y + cy1 * 3;
            return (
                x0.clamp(self.rect.x, self.rect.x + self.rect.w),
                y0.clamp(self.rect.y, self.rect.y + self.rect.h),
                x1.clamp(self.rect.x, self.rect.x + self.rect.w),
                y1.clamp(self.rect.y, self.rect.y + self.rect.h),
            );
        }
        let cw = cell_w.max(1);
        let ch = cell_h.max(1);
        let x0 = self.rect.x + ((cx0 * self.rect.w) / cw - 4).max(0);
        let y0 = self.rect.y + ((cy0 * self.rect.h) / ch - 4).max(0);
        let x1 = self.rect.x + (((cx1 * self.rect.w + cw - 1) / cw) + 4).min(self.rect.w);
        let y1 = self.rect.y + (((cy1 * self.rect.h + ch - 1) / ch) + 4).min(self.rect.h);
        (x0, y0, x1, y1)
    }

    /// Expand `cell_bgrx` (`cw * ch * 4` BGRX) inside screen rect `(dx0..dx1, dy0..dy1)`
    /// into full-screen `dst_bgrx` (`sw * sh * 4`). Used for Luma-Halftone transitions
    /// and non-XRender fallback.
    fn expand_bgrx_rect(
        &self,
        cell_bgrx: &[u8],
        cell_w: usize,
        dst_bgrx: &mut [u8],
        sw: usize,
        dx0: i32,
        dy0: i32,
        dx1: i32,
        dy1: i32,
    ) {
        if dx1 <= dx0 || dy1 <= dy0 {
            return;
        }
        let lx0 = (dx0 - self.rect.x).max(0) as usize;
        let ly0 = (dy0 - self.rect.y).max(0) as usize;
        let lx1 = (dx1 - self.rect.x).min(self.rect.w).max(0) as usize;
        let ly1 = (dy1 - self.rect.y).min(self.rect.h).max(0) as usize;
        if lx1 <= lx0 || ly1 <= ly0 {
            return;
        }

        let dst_u32 = dst_bgrx.as_mut_ptr() as *mut u32;
        let src_u32 = cell_bgrx.as_ptr() as *const u32;
        let mx = self.rect.x as usize;
        let my = self.rect.y as usize;

        if self.exact_3x {
            let cx0 = lx0 / 3;
            let cx1 = (lx1 + 2) / 3;
            let cy0 = ly0 / 3;
            let cy1 = (ly1 + 2) / 3;
            let copy_px = (cx1 - cx0) * 3;
            let sx_base = mx + cx0 * 3;
            unsafe {
                for cy in cy0..cy1 {
                    let s_row = src_u32.add(cy * cell_w);
                    let sy0 = my + cy * 3;
                    let d_row0 = dst_u32.add(sy0 * sw + sx_base);
                    let mut dp = d_row0;
                    for cx in cx0..cx1 {
                        let px = *s_row.add(cx);
                        *dp = px;
                        *dp.add(1) = px;
                        *dp.add(2) = px;
                        dp = dp.add(3);
                    }
                    std::ptr::copy_nonoverlapping(d_row0, dst_u32.add((sy0 + 1) * sw + sx_base), copy_px);
                    std::ptr::copy_nonoverlapping(d_row0, dst_u32.add((sy0 + 2) * sw + sx_base), copy_px);
                }
            }
            return;
        }

        let span = lx1 - lx0;
        let mut prev_cy = usize::MAX;
        let mut prev_sy = 0usize;
        unsafe {
            for ly in ly0..ly1 {
                let sy = my + ly;
                let cy = self.y_lut[ly] as usize;
                let d_row = dst_u32.add(sy * sw + mx + lx0);
                if cy == prev_cy {
                    std::ptr::copy_nonoverlapping(dst_u32.add(prev_sy * sw + mx + lx0), d_row, span);
                    continue;
                }
                let s_row = src_u32.add(cy * cell_w);
                for (k, &cx_u16) in self.x_lut[lx0..lx1].iter().enumerate() {
                    *d_row.add(k) = *s_row.add(cx_u16 as usize);
                }
                prev_cy = cy;
                prev_sy = sy;
            }
        }
    }
}

// ============================================================================
// Detachable MIT-SHM surface (used at tiny 640x360 during playback & sw*sh
// transiently during Luma-Halftone Develop transitions)
// ============================================================================

struct ShmSurface {
    use_shm: bool,
    shminfo: XShmSegmentInfo,
    ximg: *mut XImage,
    fallback: Vec<u8>,
    sw: i32,
    sh: i32,
}

impl ShmSurface {
    fn detached() -> Self {
        Self {
            use_shm: false,
            shminfo: XShmSegmentInfo {
                shmseg: 0,
                shmid: -1,
                shmaddr: null_mut(),
                read_only: 0,
            },
            ximg: null_mut(),
            fallback: Vec::new(),
            sw: 0,
            sh: 0,
        }
    }

    #[inline]
    fn is_attached(&self) -> bool {
        !self.ximg.is_null()
    }

    unsafe fn ensure_attached(
        &mut self,
        dpy: *mut Display,
        vis: *mut Visual,
        depth: i32,
        sw: i32,
        sh: i32,
    ) -> bool {
        if self.is_attached() && self.sw == sw && self.sh == sh {
            return true;
        }
        self.detach(dpy);
        self.sw = sw;
        self.sh = sh;
        let buf_len = (sw as usize) * (sh as usize) * 4;

        let mut try_shm = XShmQueryExtension(dpy) != 0;
        if try_shm {
            self.shminfo = XShmSegmentInfo {
                shmseg: 0,
                shmid: -1,
                shmaddr: null_mut(),
                read_only: 0,
            };
            self.ximg = XShmCreateImage(
                dpy, vis, depth as c_uint, ZPIXMAP, null_mut(), &mut self.shminfo,
                sw as c_uint, sh as c_uint,
            );
            if !self.ximg.is_null() {
                let sz = ((*self.ximg).bytes_per_line as usize) * (sh as usize);
                self.shminfo.shmid = shmget(IPC_PRIVATE, sz.max(buf_len), IPC_CREAT | 0o600);
                if self.shminfo.shmid >= 0 {
                    let addr = shmat(self.shminfo.shmid, null(), 0);
                    if addr != (-1isize as *mut c_void) && !addr.is_null() {
                        self.shminfo.shmaddr = addr as *mut c_char;
                        (*self.ximg).data = self.shminfo.shmaddr;
                        X_ERR.store(false, Ordering::Relaxed);
                        if XShmAttach(dpy, &mut self.shminfo) != 0 {
                            XSync(dpy, 0);
                        } else {
                            X_ERR.store(true, Ordering::Relaxed);
                        }
                        shmctl(self.shminfo.shmid, IPC_RMID, null_mut());
                        if X_ERR.load(Ordering::Relaxed) {
                            shmdt(self.shminfo.shmaddr as *const c_void);
                            self.shminfo.shmaddr = null_mut();
                            (*self.ximg).data = null_mut();
                            if let Some(destroy) = (*self.ximg).destroy_image {
                                destroy(self.ximg);
                            }
                            self.ximg = null_mut();
                            try_shm = false;
                        } else {
                            self.use_shm = true;
                            return true;
                        }
                    } else {
                        shmctl(self.shminfo.shmid, IPC_RMID, null_mut());
                        try_shm = false;
                    }
                } else {
                    try_shm = false;
                }
            } else {
                try_shm = false;
            }
        }

        if !try_shm {
            self.fallback = vec![0u8; buf_len];
            self.ximg = XCreateImage(
                dpy, vis, depth as c_uint, ZPIXMAP, 0,
                self.fallback.as_mut_ptr() as *mut c_char,
                sw as c_uint, sh as c_uint, 32, sw * 4,
            );
            self.use_shm = false;
            return !self.ximg.is_null();
        }
        false
    }

    unsafe fn detach(&mut self, dpy: *mut Display) {
        if !self.ximg.is_null() {
            if self.use_shm {
                XShmDetach(dpy, &mut self.shminfo);
                XSync(dpy, 0);
                if !self.shminfo.shmaddr.is_null() {
                    shmdt(self.shminfo.shmaddr as *const c_void);
                    self.shminfo.shmaddr = null_mut();
                }
            }
            (*self.ximg).data = null_mut();
            if let Some(destroy) = (*self.ximg).destroy_image {
                destroy(self.ximg);
            }
            self.ximg = null_mut();
        }
        self.use_shm = false;
        self.fallback.clear();
        self.fallback.shrink_to_fit();
    }

    #[inline]
    unsafe fn pixels(&self) -> &[u8] {
        let len = (self.sw as usize) * (self.sh as usize) * 4;
        std::slice::from_raw_parts((*self.ximg).data as *const u8, len)
    }

    #[inline]
    unsafe fn pixels_mut(&mut self) -> &mut [u8] {
        let len = (self.sw as usize) * (self.sh as usize) * 4;
        std::slice::from_raw_parts_mut((*self.ximg).data as *mut u8, len)
    }

    #[inline]
    unsafe fn put_rect(
        &self,
        dpy: *mut Display,
        drawable: Window,
        gc: GC,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
    ) {
        if w <= 0 || h <= 0 || self.ximg.is_null() {
            return;
        }
        if self.use_shm {
            XShmPutImage(dpy, drawable, gc, self.ximg, x, y, x, y, w as c_uint, h as c_uint, 0);
        } else {
            XPutImage(dpy, drawable, gc, self.ximg, x, y, x, y, w as c_uint, h as c_uint);
        }
    }
}

// ============================================================================
// XRender Hardware Nearest-Neighbor 3x Scaler (640x360 Pixmap -> Desktop Win)
// ============================================================================

struct XRenderScaler {
    fmt: *mut c_void,
    win_pic: Picture,
    cell_pmap: Pixmap,
    cell_gc: GC,
    cell_pic: Picture,
    cell_w: i32,
    cell_h: i32,
    last_xform: Option<XTransform>,
}

impl XRenderScaler {
    unsafe fn new(dpy: *mut Display, vis: *mut Visual, win: Window) -> Option<Self> {
        let mut ev = 0;
        let mut err = 0;
        if XRenderQueryExtension(dpy, &mut ev, &mut err) == 0 {
            return None;
        }
        let fmt = XRenderFindVisualFormat(dpy, vis);
        if fmt.is_null() {
            return None;
        }
        let win_pic = XRenderCreatePicture(dpy, win, fmt, 0, null());
        if win_pic == 0 {
            return None;
        }
        Some(Self {
            fmt,
            win_pic,
            cell_pmap: 0,
            cell_gc: null_mut(),
            cell_pic: 0,
            cell_w: 0,
            cell_h: 0,
            last_xform: None,
        })
    }

    unsafe fn ensure_cell_pixmap(
        &mut self,
        dpy: *mut Display,
        root: Window,
        depth: i32,
        cw: i32,
        ch: i32,
    ) -> bool {
        if self.cell_pmap != 0 && self.cell_w == cw && self.cell_h == ch {
            return true;
        }
        self.free_cell_pixmap(dpy);
        let pmap = XCreatePixmap(dpy, root, cw as c_uint, ch as c_uint, depth as c_uint);
        if pmap == 0 {
            return false;
        }
        let gc = XCreateGC(dpy, pmap, 0, null_mut());
        let pic = XRenderCreatePicture(dpy, pmap, self.fmt, 0, null());
        if pic == 0 {
            XFreeGC(dpy, gc);
            XFreePixmap(dpy, pmap);
            return false;
        }
        let f_nearest = CString::new("nearest").unwrap();
        XRenderSetPictureFilter(dpy, pic, f_nearest.as_ptr(), null(), 0);
        self.cell_pmap = pmap;
        self.cell_gc = gc;
        self.cell_pic = pic;
        self.cell_w = cw;
        self.cell_h = ch;
        self.last_xform = None;
        true
    }

    unsafe fn free_cell_pixmap(&mut self, dpy: *mut Display) {
        if self.cell_pic != 0 {
            XRenderFreePicture(dpy, self.cell_pic);
            self.cell_pic = 0;
        }
        if !self.cell_gc.is_null() {
            XFreeGC(dpy, self.cell_gc);
            self.cell_gc = null_mut();
        }
        if self.cell_pmap != 0 {
            XFreePixmap(dpy, self.cell_pmap);
            self.cell_pmap = 0;
        }
        self.cell_w = 0;
        self.cell_h = 0;
        self.last_xform = None;
    }

    #[inline]
    unsafe fn upload_cell_rect(&self, dpy: *mut Display, cell_shm: &ShmSurface, x: i32, y: i32, w: i32, h: i32) {
        if self.cell_pmap != 0 && w > 0 && h > 0 {
            cell_shm.put_rect(dpy, self.cell_pmap, self.cell_gc, x, y, w, h);
        }
    }

    #[inline]
    unsafe fn set_transform(&mut self, dpy: *mut Display, xf: XTransform) {
        if self.last_xform != Some(xf) {
            XRenderSetPictureTransform(dpy, self.cell_pic, &xf);
            self.last_xform = Some(xf);
        }
    }

    #[inline]
    unsafe fn composite_to_pic(
        &mut self,
        dpy: *mut Display,
        dst_pic: Picture,
        ex: &MonExpand,
        lx: i32,
        ly: i32,
        w: i32,
        h: i32,
    ) {
        if self.cell_pic == 0 || dst_pic == 0 || w <= 0 || h <= 0 {
            return;
        }
        self.set_transform(dpy, ex.xform);
        XRenderComposite(
            dpy,
            PICT_OP_SRC,
            self.cell_pic,
            0,
            dst_pic,
            lx,
            ly,
            0,
            0,
            ex.rect.x + lx,
            ex.rect.y + ly,
            w as c_uint,
            h as c_uint,
        );
    }

    #[inline]
    unsafe fn composite_to_win(&mut self, dpy: *mut Display, ex: &MonExpand, lx: i32, ly: i32, w: i32, h: i32) {
        let wp = self.win_pic;
        self.composite_to_pic(dpy, wp, ex, lx, ly, w, h);
    }

    /// Update `_XROOTPMAP_ID` in hardware straight from `cell_pic` (0 full-res CPU copy!).
    unsafe fn sync_root_pixmap(
        &mut self,
        dpy: *mut Display,
        root: Window,
        depth: i32,
        sw: i32,
        sh: i32,
        expands: &[MonExpand],
        atom_rootpmap: Atom,
        atom_esetroot: Atom,
        owned_pmap: &mut Pixmap,
    ) {
        let new_pmap = XCreatePixmap(dpy, root, sw as c_uint, sh as c_uint, depth as c_uint);
        if new_pmap == 0 {
            return;
        }
        let p_pic = XRenderCreatePicture(dpy, new_pmap, self.fmt, 0, null());
        if p_pic != 0 {
            for ex in expands {
                self.composite_to_pic(dpy, p_pic, ex, 0, 0, ex.rect.w, ex.rect.h);
            }
            XRenderFreePicture(dpy, p_pic);
        }
        XSetWindowBackgroundPixmap(dpy, root, new_pmap);
        XChangeProperty(
            dpy, root, atom_rootpmap, XA_PIXMAP, 32, PROP_MODE_REPLACE,
            &new_pmap as *const Pixmap as *const c_uchar, 1,
        );
        XChangeProperty(
            dpy, root, atom_esetroot, XA_PIXMAP, 32, PROP_MODE_REPLACE,
            &new_pmap as *const Pixmap as *const c_uchar, 1,
        );
        XClearWindow(dpy, root);
        if *owned_pmap != 0 && *owned_pmap != new_pmap {
            XFreePixmap(dpy, *owned_pmap);
        }
        *owned_pmap = new_pmap;
    }

    unsafe fn destroy(&mut self, dpy: *mut Display) {
        self.free_cell_pixmap(dpy);
        if self.win_pic != 0 {
            XRenderFreePicture(dpy, self.win_pic);
            self.win_pic = 0;
        }
    }
}

// ============================================================================
// Active animation + transition state inside the zero-GLX, zero-mmap daemon
// ============================================================================

struct AnimState {
    img_path: String,
    frag_path: String,
    mwa_path: String,
    mwa: MwaFile,
    comp_scratch: Vec<u8>,
    delta_scratch: Vec<u8>,
    frame_idx: usize,
    validated_loop: bool,
    canvas_synced: bool, // true when cell_shm currently holds frame `frame_idx`
    expands: Vec<MonExpand>,
}

impl AnimState {
    fn open(
        img_path: String,
        frag_path: String,
        mwa_path: String,
        monitors: &[MonRect],
    ) -> Result<Self, String> {
        let mwa = MwaFile::open(Path::new(&mwa_path))?;
        let comp_scratch = vec![0u8; mwa.max_comp_delta.max(16)];
        let delta_scratch = vec![0u8; mwa.max_raw_delta.max(16)];
        let expands = monitors
            .iter()
            .map(|&r| MonExpand::new(r, mwa.cell_w, mwa.cell_h))
            .collect();
        Ok(Self {
            img_path,
            frag_path,
            mwa_path,
            mwa,
            comp_scratch,
            delta_scratch,
            frame_idx: 0,
            validated_loop: false,
            canvas_synced: false,
            expands,
        })
    }

    fn rebuild_expands(&mut self, monitors: &[MonRect]) {
        self.expands = monitors
            .iter()
            .map(|&r| MonExpand::new(r, self.mwa.cell_w, self.mwa.cell_h))
            .collect();
    }

    fn reset_base_into(&mut self, dst_bgrx: &mut [u8]) -> Result<(), &'static str> {
        self.mwa.decode_base_bgrx(dst_bgrx)?;
        self.frame_idx = 0;
        self.canvas_synced = true;
        Ok(())
    }

    fn shrink_scratch(&mut self) {
        self.comp_scratch.clear();
        self.comp_scratch.shrink_to_fit();
        self.delta_scratch.clear();
        self.delta_scratch.shrink_to_fit();
        self.canvas_synced = false;
    }

    /// Stream next delta via `pread` and apply directly into `dst_bgrx` (`640x360` SHM)!
    fn advance_frame_into(&mut self, dst_bgrx: &mut [u8]) -> Result<FrameEntry, &'static str> {
        if !self.canvas_synced {
            self.reset_base_into(dst_bgrx)?;
            return Ok(FrameEntry {
                data_offset: 0,
                comp_len: 1,
                raw_len: 1,
                dirty_x0: 0,
                dirty_y0: 0,
                dirty_x1: self.mwa.cell_w as u16,
                dirty_y1: self.mwa.cell_h as u16,
                tile_mask: [u16::MAX; TILE_ROWS],
            });
        }
        let n = self.mwa.frame_count as usize;
        let next_idx = (self.frame_idx + 1) % n;
        let entry = self.mwa.entries[next_idx];
        if !entry.is_empty() {
            let cl = entry.comp_len as usize;
            if self.comp_scratch.len() < cl {
                self.comp_scratch.resize(cl, 0);
            }
            let read_n = self.mwa.read_delta_into(next_idx, &mut self.comp_scratch)?;
            let raw_len = entry.raw_len as usize;
            if self.delta_scratch.len() < raw_len {
                self.delta_scratch.resize(raw_len, 0);
            }
            let got = lz4_decompress(&self.comp_scratch[..read_n], &mut self.delta_scratch[..raw_len])?;
            if got != raw_len {
                return Err("delta lz4 size mismatch");
            }
            if self.validated_loop {
                unsafe {
                    unpack_planar_delta_bgrx_unchecked(dst_bgrx, &self.delta_scratch[..raw_len]);
                }
            } else {
                unpack_planar_delta_bgrx(dst_bgrx, &self.delta_scratch[..raw_len])?;
            }
        }
        self.frame_idx = next_idx;
        if next_idx == 0 {
            self.validated_loop = true;
        }
        Ok(entry)
    }

    fn expand_bgrx_full(&self, cell_bgrx: &[u8], dst_bgrx: &mut [u8], sw: usize) {
        let cw = self.mwa.cell_w as usize;
        for ex in &self.expands {
            ex.expand_bgrx_rect(
                cell_bgrx,
                cw,
                dst_bgrx,
                sw,
                ex.rect.x,
                ex.rect.y,
                ex.rect.x + ex.rect.w,
                ex.rect.y + ex.rect.h,
            );
        }
    }
}

struct ActiveTransition {
    src_bgrx: Vec<u8>,
    tgt_bgrx: Vec<u8>,
    thresh: Vec<u16>,
    start: Instant,
    dur: Duration,
    prev_head: i32,
    use_avx2: bool,
    exit_static_after: bool,
}

unsafe fn capture_current_root_bgrx(
    dpy: *mut Display,
    root: Window,
    atom_rootpmap: Atom,
    sw: i32,
    sh: i32,
    depth: i32,
) -> Option<Vec<u8>> {
    let pmap = get_root_pixmap(dpy, root, atom_rootpmap);
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
    let len = (sw as usize) * (sh as usize) * 4;
    let bpl = (*img).bytes_per_line as usize;
    let row_bytes = (sw as usize) * 4;
    let mut out = vec![0u8; len];
    let raw = (*img).data as *const u8;
    for y in 0..(sh as usize) {
        let s = std::slice::from_raw_parts(raw.add(y * bpl), row_bytes);
        out[y * row_bytes..(y + 1) * row_bytes].copy_from_slice(s);
    }
    if let Some(destroy) = (*img).destroy_image {
        destroy(img);
    }
    let mut k = 3usize;
    while k < len {
        out[k] = 0;
        k += 4;
    }
    Some(out)
}

unsafe fn sync_daemon_root_pixmap_from_shm(
    dpy: *mut Display,
    root: Window,
    gc: GC,
    shm: &ShmSurface,
    depth: i32,
    sw: i32,
    sh: i32,
    atom_rootpmap: Atom,
    atom_esetroot: Atom,
    owned_pmap: &mut Pixmap,
) {
    let new_pmap = XCreatePixmap(dpy, root, sw as c_uint, sh as c_uint, depth as c_uint);
    if new_pmap == 0 {
        return;
    }
    shm.put_rect(dpy, new_pmap, gc, 0, 0, sw, sh);
    XSetWindowBackgroundPixmap(dpy, root, new_pmap);
    XChangeProperty(
        dpy, root, atom_rootpmap, XA_PIXMAP, 32, PROP_MODE_REPLACE,
        &new_pmap as *const Pixmap as *const c_uchar, 1,
    );
    XChangeProperty(
        dpy, root, atom_esetroot, XA_PIXMAP, 32, PROP_MODE_REPLACE,
        &new_pmap as *const Pixmap as *const c_uchar, 1,
    );
    XClearWindow(dpy, root);
    if *owned_pmap != 0 && *owned_pmap != new_pmap {
        XFreePixmap(dpy, *owned_pmap);
    }
    *owned_pmap = new_pmap;
}

#[derive(Clone, Copy)]
struct DirtyAccum {
    bbox: (u16, u16, u16, u16),
    tiles: [u16; TILE_ROWS],
}

impl DirtyAccum {
    fn from_entry(e: &FrameEntry) -> Self {
        Self {
            bbox: (e.dirty_x0, e.dirty_y0, e.dirty_x1, e.dirty_y1),
            tiles: e.tile_mask,
        }
    }

    fn merge(&mut self, e: &FrameEntry) {
        self.bbox.0 = self.bbox.0.min(e.dirty_x0);
        self.bbox.1 = self.bbox.1.min(e.dirty_y0);
        self.bbox.2 = self.bbox.2.max(e.dirty_x1);
        self.bbox.3 = self.bbox.3.max(e.dirty_y1);
        for r in 0..TILE_ROWS {
            self.tiles[r] |= e.tile_mask[r];
        }
    }
}

/// Main persistent zero-GLX, zero-mmap, XRender-accelerated `misa-wall --daemon`.
pub fn run_daemon() -> i32 {
    unsafe {
        let _ = setsid();
        close_inherited_fds();
        signal(SIGTERM, on_sig);
        signal(SIGINT, on_sig);
    }

    let lpath = lock_path();
    let Ok(mut lock_file) = OpenOptions::new().create(true).write(true).open(&lpath) else {
        return 1;
    };
    unsafe {
        if flock(lock_file.as_raw_fd(), LOCK_EX | LOCK_NB) != 0 {
            return 0;
        }
    }
    let self_pid = unsafe { getpid() };
    let _ = lock_file.set_len(0);
    let _ = write!(lock_file, "{self_pid}\n");

    let spath = socket_path();
    let _ = fs::remove_file(&spath);
    let Ok(listener) = UnixListener::bind(&spath) else {
        return 1;
    };
    let _ = listener.set_nonblocking(true);
    let sock_fd = listener.as_raw_fd();

    let pfile = pidfile_path();
    let sfile = state_path();
    let ppath = pause_path();
    let _ = fs::write(&pfile, format!("{self_pid}\n"));

    unsafe {
        let dpy = XOpenDisplay(null());
        if dpy.is_null() {
            let _ = fs::remove_file(&spath);
            return 1;
        }
        XSetErrorHandler(Some(ignore_x_err));
        let x_fd = XConnectionNumber(dpy);
        let scr = XDefaultScreen(dpy);
        let root = XDefaultRootWindow(dpy);
        let vis = XDefaultVisual(dpy, scr);
        let depth = XDefaultDepth(dpy, scr);
        let mut sw = XDisplayWidth(dpy, scr);
        let mut sh = XDisplayHeight(dpy, scr);
        if sw <= 0 || sh <= 0 {
            XCloseDisplay(dpy);
            let _ = fs::remove_file(&spath);
            return 1;
        }

        let intern = |s: &str| {
            let cs = CString::new(s).unwrap();
            XInternAtom(dpy, cs.as_ptr(), 0)
        };
        let atom_wtype = intern("_NET_WM_WINDOW_TYPE");
        let atom_desktop = intern("_NET_WM_WINDOW_TYPE_DESKTOP");
        let atom_state = intern("_NET_WM_STATE");
        let atom_below = intern("_NET_WM_STATE_BELOW");
        let atom_sticky = intern("_NET_WM_STATE_STICKY");
        let atom_skip_tb = intern("_NET_WM_STATE_SKIP_TASKBAR");
        let atom_skip_pg = intern("_NET_WM_STATE_SKIP_PAGER");
        let atom_active = intern("_NET_ACTIVE_WINDOW");
        let atom_stacking = intern("_NET_CLIENT_LIST_STACKING");
        let atom_fs = intern("_NET_WM_STATE_FULLSCREEN");
        let atom_opacity = intern("_NET_WM_WINDOW_OPACITY");
        let atom_rootpmap = intern("_XROOTPMAP_ID");
        let atom_esetroot = intern("ESETROOT_PMAP_ID");

        let mut swa = XSetWindowAttributes {
            background_pixmap: 0,
            background_pixel: 0,
            border_pixmap: 0,
            border_pixel: 0,
            bit_gravity: 0,
            win_gravity: 0,
            backing_store: 0,
            backing_planes: 0,
            backing_pixel: 0,
            save_under: 0,
            event_mask: STRUCTURE_NOTIFY_MASK,
            do_not_propagate_mask: 0,
            override_redirect: 1,
            colormap: 0,
            cursor: 0,
        };
        let win = XCreateWindow(
            dpy, root, 0, 0, sw as c_uint, sh as c_uint, 0, depth, INPUT_OUTPUT, vis,
            CW_BACK_PIXMAP | CW_BORDER_PIXEL | CW_OVERRIDE_REDIRECT | CW_EVENT_MASK,
            &mut swa,
        );
        let c_name = CString::new("misa-wall-live").unwrap();
        let c_cls = CString::new("MisaWallLive").unwrap();
        let mut ch = XClassHint {
            res_name: c_name.as_ptr() as *mut c_char,
            res_class: c_cls.as_ptr() as *mut c_char,
        };
        XSetClassHint(dpy, win, &mut ch);
        XChangeProperty(
            dpy, win, atom_wtype, XA_ATOM, 32, PROP_MODE_REPLACE,
            &atom_desktop as *const Atom as *const c_uchar, 1,
        );
        let wstates = [atom_below, atom_sticky, atom_skip_tb, atom_skip_pg];
        XChangeProperty(
            dpy, win, atom_state, XA_ATOM, 32, PROP_MODE_REPLACE,
            wstates.as_ptr() as *const c_uchar, wstates.len() as c_int,
        );
        XShapeCombineRectangles(dpy, win, SHAPE_INPUT, 0, 0, null(), 0, SHAPE_SET, 0);
        let gc = XCreateGC(dpy, win, 0, null_mut());
        let mut win_mapped = false;

        XSelectInput(
            dpy, root,
            SUBSTRUCTURE_NOTIFY_MASK | STRUCTURE_NOTIFY_MASK | PROPERTY_CHANGE_MASK,
        );
        XSync(dpy, 0);

        let dpms_info = load_dpms_info();
        let mut xrender = XRenderScaler::new(dpy, vis, win);
        let mut monitors = query_monitors(dpy, root, sw, sh);
        let mut mon_states: Vec<MonState> = monitors.iter().map(|&m| MonState::full(m)).collect();
        let mut opaque_cache: HashMap<Window, bool> = HashMap::with_capacity(32);

        // Tiny 640x360x4 (921 KB!) MIT-SHM for hardware-scaled playback!
        let mut cell_shm = ShmSurface::detached();
        // Transient full-screen SHM used ONLY during 540ms transitions (detached immediately after!)
        // or as CPU fallback when XRender is unavailable.
        let mut full_shm = ShmSurface::detached();
        let mut owned_root_pmap: Pixmap = 0;

        let mut anim: Option<AnimState> = None;
        let mut trans: Option<ActiveTransition> = None;
        let mut user_paused = false;
        let mut pause_since: Option<Instant> = None;
        let mut mon_full_repaint: Vec<bool> = vec![true; monitors.len()];
        let mut mon_next_due: Vec<Instant> = vec![Instant::now(); monitors.len()];
        let mut mon_dirty: Vec<Option<DirtyAccum>> = vec![None; monitors.len()];
        let mut pmap_full_upload = true;
        let mut pmap_dirty: Option<DirtyAccum> = None;
        let mut tile_rects_scratch: Vec<(i32, i32, i32, i32)> = Vec::with_capacity(16);

        let mut need_occl_check = true;
        let mut dpms_off = query_dpms_off(dpy, dpms_info);
        let mut term_opaque = detect_terminal_opaque();
        let mut last_dpms_check = Instant::now();
        let mut last_batt_check = Instant::now() - Duration::from_secs(60);
        let mut batt_mode = on_battery_power();
        let pause_all_on_work = env::var("MISA_WALL_PAUSE_ALL_ON_WORK").ok().as_deref() == Some("1");
        let fps_override: Option<u32> = env::var("MISA_WALL_LIVE_FPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v.clamp(2, 30));
        let work_fps_override: Option<u32> = env::var("MISA_WALL_WORK_FPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .map(|v| v.clamp(1, 20));
        let mut next_canvas_at = Instant::now();

        #[cfg(target_arch = "x86_64")]
        let use_avx2 = std::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        let mut exit_after_static_handover = false;

        while !STOP.load(Ordering::Relaxed) {
            let t0 = Instant::now();

            // 1. Drain X11 root events.
            let mut need_lower = false;
            let mut need_resize = false;
            while XPending(dpy) > 0 {
                let mut ev = XEvent { pad: [0; 24] };
                XNextEvent(dpy, &mut ev);
                let ev_type = (ev.pad[0] & 0xFFFF_FFFF) as c_int;
                match ev_type {
                    17 => {
                        // DestroyNotify: evict destroyed window from opaqueness cache.
                        let dead_w = ev.pad[5] as Window;
                        opaque_cache.remove(&dead_w);
                        need_occl_check = true;
                    }
                    18 | 19 | 26 => {
                        need_lower = true;
                        need_occl_check = true;
                    }
                    22 => {
                        need_lower = true;
                        need_resize = true;
                        need_occl_check = true;
                    }
                    28 => {
                        let prop_atom = ev.pad[5] as Atom;
                        if prop_atom == atom_stacking || prop_atom == atom_active {
                            need_occl_check = true;
                        } else if prop_atom == atom_opacity || prop_atom == atom_state {
                            opaque_cache.clear();
                            need_occl_check = true;
                        }
                    }
                    _ => {}
                }
            }

            if need_resize {
                let nsw = XDisplayWidth(dpy, scr);
                let nsh = XDisplayHeight(dpy, scr);
                if nsw > 0 && nsh > 0 && (nsw != sw || nsh != sh) {
                    sw = nsw;
                    sh = nsh;
                    XResizeWindow(dpy, win, sw as c_uint, sh as c_uint);
                    monitors = query_monitors(dpy, root, sw, sh);
                    mon_states = monitors.iter().map(|&m| MonState::full(m)).collect();
                    mon_full_repaint = vec![true; monitors.len()];
                    mon_next_due = vec![t0; monitors.len()];
                    mon_dirty = vec![None; monitors.len()];
                    full_shm.detach(dpy);
                    trans = None;
                    if let Some(ref mut a) = anim {
                        a.rebuild_expands(&monitors);
                    }
                    if let Some(ref mut xr) = xrender {
                        xr.last_xform = None;
                    }
                    opaque_cache.clear();
                    need_occl_check = true;
                }
            }
            if need_lower && win_mapped {
                XLowerWindow(dpy, win);
            }

            // 2. Accept & handle pending IPC socket commands (non-blocking).
            loop {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
                        let mut cmd_buf = String::new();
                        let _ = stream.read_to_string(&mut cmd_buf);
                        let line = cmd_buf.lines().next().unwrap_or("").trim();
                        let reply: String = if line == "PING" {
                            "PONG\n".into()
                        } else if line == "PAUSE" {
                            user_paused = true;
                            pause_since.get_or_insert(t0);
                            "OK\n".into()
                        } else if line == "UNPAUSE" {
                            user_paused = false;
                            pause_since = None;
                            for f in &mut mon_full_repaint { *f = true; }
                            for d in &mut mon_next_due { *d = t0; }
                            next_canvas_at = t0;
                            "OK\n".into()
                        } else if line == "TOGGLE" {
                            user_paused = !user_paused;
                            if user_paused {
                                pause_since.get_or_insert(t0);
                            } else {
                                pause_since = None;
                                for f in &mut mon_full_repaint { *f = true; }
                                for d in &mut mon_next_due { *d = t0; }
                                next_canvas_at = t0;
                            }
                            format!("OK paused={}\n", if user_paused { 1 } else { 0 })
                        } else if line == "KILL" {
                            STOP.store(true, Ordering::Relaxed);
                            "OK\n".into()
                        } else if line == "QUERY" {
                            let vis_n = mon_states.iter().filter(|s| s.visible).count();
                            let rss_kib = fs::read_to_string("/proc/self/smaps_rollup")
                                .ok()
                                .and_then(|s| {
                                    s.lines().find(|l| l.starts_with("Pss:")).map(|l| l.trim().to_string())
                                })
                                .unwrap_or_else(|| "Pss: ?".into());
                            let scaler = if xrender.is_some() { "xrender-hw" } else { "cpu" };
                            if let Some(ref a) = anim {
                                let base_fps = fps_override.unwrap_or(if batt_mode { a.mwa.fps.min(10) } else { a.mwa.fps });
                                let w_fps = work_fps_override.unwrap_or(if batt_mode { 4 } else { 5 }).min(base_fps);
                                let mon_fps: String = mon_states
                                    .iter()
                                    .map(|s| {
                                        if user_paused || !s.visible {
                                            "0".to_string()
                                        } else if s.mostly_covered {
                                            w_fps.to_string()
                                        } else {
                                            base_fps.to_string()
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join(",");
                                format!(
                                    "mode=anim\nscaler={}\nimg={}\nfrag={}\nmwa={}\ncache_kb={}\ncell={}x{}\nframes={}\nframe={}\nfps={}\nmon_fps={}\npaused={}\nvisible_monitors={}/{}\nwin_mapped={}\ncell_shm={}\nfull_shm={}\n{}\n",
                                    scaler,
                                    a.img_path,
                                    a.frag_path,
                                    a.mwa_path,
                                    a.mwa.file_size() / 1024,
                                    a.mwa.cell_w,
                                    a.mwa.cell_h,
                                    a.mwa.frame_count,
                                    a.frame_idx,
                                    a.mwa.fps,
                                    mon_fps,
                                    if user_paused { 1 } else { 0 },
                                    vis_n,
                                    monitors.len(),
                                    if win_mapped { 1 } else { 0 },
                                    if cell_shm.is_attached() { 1 } else { 0 },
                                    if full_shm.is_attached() { 1 } else { 0 },
                                    rss_kib,
                                )
                            } else {
                                format!(
                                    "mode=idle\nscaler={}\npaused={}\nvisible_monitors={}/{}\nwin_mapped={}\ncell_shm={}\nfull_shm={}\n{}\n",
                                    scaler,
                                    if user_paused { 1 } else { 0 },
                                    vis_n,
                                    monitors.len(),
                                    if win_mapped { 1 } else { 0 },
                                    if cell_shm.is_attached() { 1 } else { 0 },
                                    if full_shm.is_attached() { 1 } else { 0 },
                                    rss_kib,
                                )
                            }
                        } else if let Some(rest) = line.strip_prefix("SET_ANIM\t") {
                            let parts: Vec<&str> = rest.split('\t').collect();
                            if parts.len() >= 4 {
                                let ip = parts[0].to_string();
                                let fp = parts[1].to_string();
                                let mp = parts[2].to_string();
                                let t_ms: u64 = parts[3].parse().unwrap_or(0);
                                term_opaque = detect_terminal_opaque();
                                opaque_cache.clear();

                                let same = anim
                                    .as_ref()
                                    .map(|a| a.mwa_path == mp && a.img_path == ip && a.frag_path == fp)
                                    .unwrap_or(false);
                                if same && trans.is_none() {
                                    "OK\n".into()
                                } else {
                                    // Snapshot current pixels ONLY if transition requested.
                                    let prev_bgrx: Option<Vec<u8>> = if t_ms > 0 {
                                        if let Some(ref old_a) = anim {
                                            let mut b = vec![0u8; (sw as usize) * (sh as usize) * 4];
                                            if old_a.canvas_synced
                                                && cell_shm.is_attached()
                                                && cell_shm.sw == old_a.mwa.cell_w as i32
                                                && cell_shm.sh == old_a.mwa.cell_h as i32
                                            {
                                                old_a.expand_bgrx_full(cell_shm.pixels(), &mut b, sw as usize);
                                            } else {
                                                let mut tmp_cell = vec![0u8; (old_a.mwa.cell_w as usize) * (old_a.mwa.cell_h as usize) * 4];
                                                if old_a.mwa.decode_base_bgrx(&mut tmp_cell).is_ok() {
                                                    old_a.expand_bgrx_full(&tmp_cell, &mut b, sw as usize);
                                                }
                                            }
                                            Some(b)
                                        } else {
                                            capture_current_root_bgrx(dpy, root, atom_rootpmap, sw, sh, depth)
                                        }
                                    } else {
                                        None
                                    };

                                    match AnimState::open(ip.clone(), fp.clone(), mp.clone(), &monitors) {
                                        Ok(mut new_anim) => {
                                            let cw = new_anim.mwa.cell_w as i32;
                                            let ch = new_anim.mwa.cell_h as i32;
                                            let mut ok_decode = false;
                                            if cell_shm.ensure_attached(dpy, vis, depth, cw, ch) {
                                                ok_decode = new_anim.reset_base_into(cell_shm.pixels_mut()).is_ok();
                                            }
                                            if !ok_decode {
                                                "ERR decode base frame failed\n".into()
                                            } else {
                                                let _ = fs::write(&sfile, format!("{ip}\n{fp}\n"));
                                                if let Some(ref mut xr) = xrender {
                                                    if xr.ensure_cell_pixmap(dpy, root, depth, cw, ch) {
                                                        xr.upload_cell_rect(dpy, &cell_shm, 0, 0, cw, ch);
                                                    }
                                                }

                                                let mut started_trans = false;
                                                if let Some(src_bgrx) = prev_bgrx {
                                                    let mut tgt_bgrx = vec![0u8; (sw as usize) * (sh as usize) * 4];
                                                    new_anim.expand_bgrx_full(cell_shm.pixels(), &mut tgt_bgrx, sw as usize);
                                                    if src_bgrx != tgt_bgrx
                                                        && full_shm.ensure_attached(dpy, vis, depth, sw, sh)
                                                    {
                                                        full_shm.pixels_mut().copy_from_slice(&src_bgrx);
                                                        if !win_mapped {
                                                            XMapWindow(dpy, win);
                                                            XLowerWindow(dpy, win);
                                                            win_mapped = true;
                                                        }
                                                        full_shm.put_rect(dpy, win, gc, 0, 0, sw, sh);
                                                        XSync(dpy, 0);
                                                        let thresh = build_threshold_field(
                                                            &tgt_bgrx, sw as usize, sh as usize, &monitors,
                                                        );
                                                        trans = Some(ActiveTransition {
                                                            src_bgrx,
                                                            tgt_bgrx,
                                                            thresh,
                                                            start: Instant::now(),
                                                            dur: Duration::from_millis(t_ms.max(1)),
                                                            prev_head: -1,
                                                            use_avx2,
                                                            exit_static_after: false,
                                                        });
                                                        started_trans = true;
                                                    }
                                                }

                                                if !started_trans {
                                                    trans = None;
                                                    if !win_mapped {
                                                        XMapWindow(dpy, win);
                                                        XLowerWindow(dpy, win);
                                                        win_mapped = true;
                                                    }
                                                    if let Some(ref mut xr) = xrender {
                                                        for ex in &new_anim.expands {
                                                            xr.composite_to_win(dpy, ex, 0, 0, ex.rect.w, ex.rect.h);
                                                        }
                                                        xr.sync_root_pixmap(
                                                            dpy, root, depth, sw, sh, &new_anim.expands,
                                                            atom_rootpmap, atom_esetroot, &mut owned_root_pmap,
                                                        );
                                                        full_shm.detach(dpy);
                                                    } else if full_shm.ensure_attached(dpy, vis, depth, sw, sh) {
                                                        new_anim.expand_bgrx_full(cell_shm.pixels(), full_shm.pixels_mut(), sw as usize);
                                                        full_shm.put_rect(dpy, win, gc, 0, 0, sw, sh);
                                                        sync_daemon_root_pixmap_from_shm(
                                                            dpy, root, gc, &full_shm, depth, sw, sh,
                                                            atom_rootpmap, atom_esetroot, &mut owned_root_pmap,
                                                        );
                                                    }
                                                    XSync(dpy, 0);
                                                }

                                                anim = Some(new_anim);
                                                pause_since = None;
                                                let now_i = Instant::now();
                                                for f in &mut mon_full_repaint { *f = false; }
                                                for d in &mut mon_next_due { *d = now_i; }
                                                for md in &mut mon_dirty { *md = None; }
                                                pmap_full_upload = false;
                                                pmap_dirty = None;
                                                need_occl_check = true;
                                                next_canvas_at = now_i;
                                                let _ = malloc_trim(0);
                                                "OK\n".into()
                                            }
                                        }
                                        Err(e) => format!("ERR {e}\n"),
                                    }
                                }
                            } else {
                                "ERR bad SET_ANIM args\n".into()
                            }
                        } else if let Some(rest) = line.strip_prefix("SET_STATIC\t") {
                            let parts: Vec<&str> = rest.split('\t').collect();
                            if parts.len() >= 2 {
                                let ip = parts[0];
                                let t_ms: u64 = parts[1].parse().unwrap_or(0);
                                if let Some(tgt_bgrx) = decode_and_scale(ip, sw, sh, &monitors) {
                                    let prev_bgrx: Option<Vec<u8>> = if t_ms > 0 {
                                        if let Some(ref old_a) = anim {
                                            let mut b = vec![0u8; (sw as usize) * (sh as usize) * 4];
                                            if old_a.canvas_synced && cell_shm.is_attached() {
                                                old_a.expand_bgrx_full(cell_shm.pixels(), &mut b, sw as usize);
                                            } else {
                                                let mut tmp_cell = vec![0u8; (old_a.mwa.cell_w as usize) * (old_a.mwa.cell_h as usize) * 4];
                                                if old_a.mwa.decode_base_bgrx(&mut tmp_cell).is_ok() {
                                                    old_a.expand_bgrx_full(&tmp_cell, &mut b, sw as usize);
                                                }
                                            }
                                            Some(b)
                                        } else {
                                            capture_current_root_bgrx(dpy, root, atom_rootpmap, sw, sh, depth)
                                        }
                                    } else {
                                        None
                                    };
                                    anim = None;
                                    cell_shm.detach(dpy);
                                    let _ = fs::remove_file(&sfile);
                                    if t_ms > 0
                                        && prev_bgrx.as_ref().map(|s| s != &tgt_bgrx).unwrap_or(false)
                                        && full_shm.ensure_attached(dpy, vis, depth, sw, sh)
                                    {
                                        let src_bgrx = prev_bgrx.unwrap();
                                        full_shm.pixels_mut().copy_from_slice(&src_bgrx);
                                        if !win_mapped {
                                            XMapWindow(dpy, win);
                                            XLowerWindow(dpy, win);
                                            win_mapped = true;
                                        }
                                        full_shm.put_rect(dpy, win, gc, 0, 0, sw, sh);
                                        XSync(dpy, 0);
                                        let thresh = build_threshold_field(
                                            &tgt_bgrx, sw as usize, sh as usize, &monitors,
                                        );
                                        trans = Some(ActiveTransition {
                                            src_bgrx,
                                            tgt_bgrx,
                                            thresh,
                                            start: Instant::now(),
                                            dur: Duration::from_millis(t_ms.max(1)),
                                            prev_head: -1,
                                            use_avx2,
                                            exit_static_after: true,
                                        });
                                        "OK\n".into()
                                    } else {
                                        let _ = publish_persistent_root_pixmap(&tgt_bgrx, sw, sh, &monitors);
                                        exit_after_static_handover = true;
                                        STOP.store(true, Ordering::Relaxed);
                                        "OK\n".into()
                                    }
                                } else {
                                    "ERR decode failed\n".into()
                                }
                            } else {
                                "ERR bad SET_STATIC args\n".into()
                            }
                        } else {
                            "ERR unknown cmd\n".into()
                        };
                        let _ = stream.write_all(reply.as_bytes());
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(_) => break,
                }
            }

            if STOP.load(Ordering::Relaxed) {
                break;
            }

            // 3. Active 60 FPS AVX2 Luma-Halftone Develop transition!
            if let Some(mut tr) = trans.take() {
                if full_shm.ensure_attached(dpy, vis, depth, sw, sh) {
                    let frame_dur = Duration::from_nanos(16_666_667);
                    let elapsed = t0.duration_since(tr.start) + frame_dur;
                    let progress = (elapsed.as_secs_f64() / tr.dur.as_secs_f64()).min(1.0);
                    let finish = progress >= 1.0;
                    let eased = ease_develop(progress);
                    let head = if finish {
                        MAX_THRESH + FEATHER
                    } else {
                        (eased * ((MAX_THRESH + FEATHER) as f64)).round() as i32
                    };

                    render_develop_frame(
                        full_shm.pixels_mut(),
                        &tr.src_bgrx,
                        &tr.tgt_bgrx,
                        &tr.thresh,
                        head,
                        tr.prev_head,
                        tr.use_avx2,
                    );
                    tr.prev_head = head;
                    full_shm.put_rect(dpy, win, gc, 0, 0, sw, sh);
                    XFlush(dpy);

                    if finish {
                        if tr.exit_static_after {
                            let _ = publish_persistent_root_pixmap(&tr.tgt_bgrx, sw, sh, &monitors);
                            exit_after_static_handover = true;
                            STOP.store(true, Ordering::Relaxed);
                            break;
                        }
                        sync_daemon_root_pixmap_from_shm(
                            dpy, root, gc, &full_shm, depth, sw, sh,
                            atom_rootpmap, atom_esetroot, &mut owned_root_pmap,
                        );
                        XSync(dpy, 0);
                        // Immediately free the 8.3 MB transition SHM when using XRender!
                        if xrender.is_some() {
                            full_shm.detach(dpy);
                        }
                        let now_i = Instant::now();
                        next_canvas_at = now_i;
                        for d in &mut mon_next_due { *d = now_i; }
                        for md in &mut mon_dirty { *md = None; }
                        drop(tr);
                        let _ = malloc_trim(0);
                    } else {
                        trans = Some(tr);
                        if let Some(rem) = frame_dur.checked_sub(t0.elapsed()) {
                            thread::sleep(rem);
                        }
                        continue;
                    }
                }
            }

            // 4. Lightweight DPMS poll (1 X11 call every 2s); run full window-tree
            //    occlusion scan ONLY on X11 events or DPMS state change!
            if t0.duration_since(last_dpms_check) >= Duration::from_secs(2) {
                last_dpms_check = t0;
                let now_dpms = query_dpms_off(dpy, dpms_info);
                if now_dpms != dpms_off {
                    dpms_off = now_dpms;
                    need_occl_check = true;
                }
            }

            if need_occl_check {
                let prev_states = mon_states.clone();
                need_occl_check = false;
                compute_monitor_states(
                    dpy, root, win, &monitors,
                    atom_stacking, atom_active, atom_state, atom_fs, atom_opacity,
                    dpms_off, term_opaque, &mut opaque_cache, &mut mon_states,
                );
                for i in 0..mon_states.len() {
                    let cur = mon_states[i];
                    let prev = prev_states.get(i).copied();
                    let expanded = match prev {
                        None => true,
                        Some(p) => (!p.visible && cur.visible)
                            || cur.exp_x0 < p.exp_x0
                            || cur.exp_y0 < p.exp_y0
                            || cur.exp_x1 > p.exp_x1
                            || cur.exp_y1 > p.exp_y1,
                    };
                    if cur.visible && expanded {
                        mon_full_repaint[i] = true;
                        mon_next_due[i] = t0;
                    } else if !cur.visible {
                        mon_dirty[i] = None;
                    }
                }
            }

            // 5. Pause / full-occlusion -> 0.00% CPU & full SHM + scratch reclamation!
            let picker_paused = external_picker_paused(&ppath);
            let any_occluded = mon_states.iter().any(|s| !s.visible);
            let vis_count = mon_states.iter().filter(|s| s.visible).count();
            let effectively_paused = anim.is_none()
                || user_paused
                || picker_paused
                || vis_count == 0
                || (pause_all_on_work && any_occluded);

            if effectively_paused {
                let ps = *pause_since.get_or_insert(t0);
                let any_res = cell_shm.is_attached() || full_shm.is_attached() || win_mapped;
                if any_res && t0.duration_since(ps) >= Duration::from_secs(2) {
                    // Sync _XROOTPMAP_ID to the exact current frame BEFORE unmapping
                    // `win` so picom frees the entire redirected GLX wallpaper texture
                    // (e.g. 16.6 MB VRAM) with zero visual jump or flicker!
                    if win_mapped {
                        if let Some(ref a) = anim {
                            if let Some(ref mut xr) = xrender {
                                let cw = a.mwa.cell_w as i32;
                                let ch = a.mwa.cell_h as i32;
                                if cell_shm.is_attached()
                                    && a.canvas_synced
                                    && xr.ensure_cell_pixmap(dpy, root, depth, cw, ch)
                                {
                                    xr.upload_cell_rect(dpy, &cell_shm, 0, 0, cw, ch);
                                    xr.sync_root_pixmap(
                                        dpy, root, depth, sw, sh, &a.expands,
                                        atom_rootpmap, atom_esetroot, &mut owned_root_pmap,
                                    );
                                }
                            } else if full_shm.is_attached() {
                                sync_daemon_root_pixmap_from_shm(
                                    dpy, root, gc, &full_shm, depth, sw, sh,
                                    atom_rootpmap, atom_esetroot, &mut owned_root_pmap,
                                );
                            }
                        }
                        XUnmapWindow(dpy, win);
                        XSync(dpy, 0);
                        win_mapped = false;
                        if let Some(ref mut xr) = xrender {
                            xr.free_cell_pixmap(dpy);
                        }
                    }
                    cell_shm.detach(dpy);
                    full_shm.detach(dpy);
                    if let Some(ref mut a) = anim {
                        a.shrink_scratch();
                        a.mwa.advise_dontneed();
                    }
                    let _ = malloc_trim(0);
                }

                let timeout_ms: c_int = if picker_paused {
                    150
                } else if !cell_shm.is_attached() && !full_shm.is_attached() && !win_mapped {
                    if user_paused || anim.is_none() { -1 } else { 5000 }
                } else {
                    2000
                };
                let mut pfds = [
                    PollFd { fd: x_fd, events: POLLIN, revents: 0 },
                    PollFd { fd: sock_fd, events: POLLIN, revents: 0 },
                ];
                let _ = poll(pfds.as_mut_ptr(), 2, timeout_ms);
                continue;
            }

            if pause_since.take().is_some() {
                if let Some(ref a) = anim {
                    a.mwa.advise_sequential();
                }
                pmap_full_upload = true;
                for i in 0..monitors.len() {
                    if mon_states[i].visible {
                        mon_full_repaint[i] = true;
                        mon_next_due[i] = t0;
                    }
                }
                next_canvas_at = t0;
            }

            // 6. Per-monitor independent adaptive FPS!
            if t0.duration_since(last_batt_check) >= Duration::from_secs(15) {
                last_batt_check = t0;
                batt_mode = on_battery_power();
            }

            let base_fps = anim.as_ref().map(|a| a.mwa.fps).unwrap_or(12);
            let full_fps = fps_override.unwrap_or(if batt_mode { base_fps.min(10) } else { base_fps });
            let work_fps = work_fps_override.unwrap_or(if batt_mode { 4 } else { 5 }).min(full_fps);
            let full_dur = Duration::from_nanos(1_000_000_000 / (full_fps.max(1) as u64));
            let work_dur = Duration::from_nanos(1_000_000_000 / (work_fps.max(1) as u64));

            let any_full_vis = mon_states.iter().any(|s| s.visible && !s.mostly_covered);
            let canvas_dur = if any_full_vis { full_dur } else { work_dur };

            // 7. Advance 640x360 BGRX SHM directly & hardware-composite ONLY dirty tiles!
            let any_mon_due = (0..monitors.len())
                .any(|i| mon_states[i].visible && (mon_full_repaint[i] || t0 >= mon_next_due[i]));
            let canvas_due = t0 >= next_canvas_at;

            if canvas_due || any_mon_due {
                if let Some(ref mut a) = anim {
                    let cw = a.mwa.cell_w as i32;
                    let ch = a.mwa.cell_h as i32;
                    let was_detached = !cell_shm.is_attached() || cell_shm.sw != cw || cell_shm.sh != ch;
                    if cell_shm.ensure_attached(dpy, vis, depth, cw, ch) {
                        if was_detached {
                            a.canvas_synced = false;
                            pmap_full_upload = true;
                            for i in 0..monitors.len() {
                                if mon_states[i].visible {
                                    mon_full_repaint[i] = true;
                                }
                            }
                        }

                        if canvas_due || !a.canvas_synced {
                            if let Ok(e) = a.advance_frame_into(cell_shm.pixels_mut()) {
                                if !e.is_empty() {
                                    if let Some(ref mut pd) = pmap_dirty {
                                        pd.merge(&e);
                                    } else {
                                        pmap_dirty = Some(DirtyAccum::from_entry(&e));
                                    }
                                    for i in 0..monitors.len() {
                                        if mon_states[i].visible {
                                            if let Some(ref mut md) = mon_dirty[i] {
                                                md.merge(&e);
                                            } else {
                                                mon_dirty[i] = Some(DirtyAccum::from_entry(&e));
                                            }
                                        }
                                    }
                                }
                            }
                            next_canvas_at = t0 + canvas_dur;
                        }

                        let mut any_blitted = false;
                        let mut remapped_now = false;
                        if !win_mapped {
                            XMapWindow(dpy, win);
                            XLowerWindow(dpy, win);
                            win_mapped = true;
                            remapped_now = true;
                        }

                        if let Some(ref mut xr) = xrender {
                            // Hardware XRender path: upload ONLY dirty tiles to 640x360 Pixmap!
                            if xr.ensure_cell_pixmap(dpy, root, depth, cw, ch) {
                                if pmap_full_upload || remapped_now {
                                    xr.upload_cell_rect(dpy, &cell_shm, 0, 0, cw, ch);
                                    pmap_full_upload = false;
                                    pmap_dirty = None;
                                } else if let Some(pd) = pmap_dirty.take() {
                                    merge_tile_rects(&pd.tiles, pd.bbox, cw, ch, &mut tile_rects_scratch);
                                    for &(cx0, cy0, cx1, cy1) in &tile_rects_scratch {
                                        xr.upload_cell_rect(dpy, &cell_shm, cx0, cy0, cx1 - cx0, cy1 - cy0);
                                    }
                                }

                                if remapped_now {
                                    // Initialize every monitor's full rect atomically with XMapWindow
                                    // so picom never sees an uninitialized black pixel anywhere!
                                    for (i, ex) in a.expands.iter().enumerate() {
                                        xr.composite_to_win(dpy, ex, 0, 0, ex.rect.w, ex.rect.h);
                                        mon_full_repaint[i] = false;
                                        mon_dirty[i] = None;
                                        mon_next_due[i] = t0 + if mon_states[i].mostly_covered { work_dur } else { full_dur };
                                    }
                                    any_blitted = true;
                                } else {
                                for (i, ex) in a.expands.iter().enumerate() {
                                    let st = mon_states[i];
                                    if !st.visible {
                                        mon_dirty[i] = None;
                                        continue;
                                    }
                                    if !mon_full_repaint[i] && t0 < mon_next_due[i] {
                                        continue;
                                    }

                                    if mon_full_repaint[i] {
                                        let lx0 = (st.exp_x0 - ex.rect.x).max(0);
                                        let ly0 = (st.exp_y0 - ex.rect.y).max(0);
                                        let lx1 = (st.exp_x1 - ex.rect.x).min(ex.rect.w);
                                        let ly1 = (st.exp_y1 - ex.rect.y).min(ex.rect.h);
                                        if lx1 > lx0 && ly1 > ly0 {
                                            xr.composite_to_win(dpy, ex, lx0, ly0, lx1 - lx0, ly1 - ly0);
                                            any_blitted = true;
                                        }
                                    } else if let Some(md) = mon_dirty[i] {
                                        merge_tile_rects(&md.tiles, md.bbox, cw, ch, &mut tile_rects_scratch);
                                        for &(cx0, cy0, cx1, cy1) in &tile_rects_scratch {
                                            let (fx0, fy0, fx1, fy1) =
                                                ex.cell_rect_to_screen(cx0, cy0, cx1, cy1, cw, ch);
                                            let dx0 = st.exp_x0.max(fx0);
                                            let dy0 = st.exp_y0.max(fy0);
                                            let dx1 = st.exp_x1.min(fx1);
                                            let dy1 = st.exp_y1.min(fy1);
                                            if dx1 > dx0 && dy1 > dy0 {
                                                let lx0 = (dx0 - ex.rect.x).max(0);
                                                let ly0 = (dy0 - ex.rect.y).max(0);
                                                let lx1 = (dx1 - ex.rect.x).min(ex.rect.w);
                                                let ly1 = (dy1 - ex.rect.y).min(ex.rect.h);
                                                if lx1 > lx0 && ly1 > ly0 {
                                                    xr.composite_to_win(dpy, ex, lx0, ly0, lx1 - lx0, ly1 - ly0);
                                                    any_blitted = true;
                                                }
                                            }
                                        }
                                    }

                                    mon_full_repaint[i] = false;
                                    mon_dirty[i] = None;
                                    mon_next_due[i] = t0 + if st.mostly_covered { work_dur } else { full_dur };
                                }
                                }
                            }
                        } else if full_shm.ensure_attached(dpy, vis, depth, sw, sh) {
                            // CPU fallback when XRender extension is absent.
                            let cell_slice = cell_shm.pixels();
                            if remapped_now {
                                a.expand_bgrx_full(cell_slice, full_shm.pixels_mut(), sw as usize);
                                full_shm.put_rect(dpy, win, gc, 0, 0, sw, sh);
                                for i in 0..a.expands.len() {
                                    mon_full_repaint[i] = false;
                                    mon_dirty[i] = None;
                                    mon_next_due[i] = t0 + if mon_states[i].mostly_covered { work_dur } else { full_dur };
                                }
                                any_blitted = true;
                            } else {
                            for (i, ex) in a.expands.iter().enumerate() {
                                let st = mon_states[i];
                                if !st.visible {
                                    mon_dirty[i] = None;
                                    continue;
                                }
                                if !mon_full_repaint[i] && t0 < mon_next_due[i] {
                                    continue;
                                }
                                if mon_full_repaint[i] {
                                    ex.expand_bgrx_rect(
                                        cell_slice, cw as usize, full_shm.pixels_mut(), sw as usize,
                                        st.exp_x0, st.exp_y0, st.exp_x1, st.exp_y1,
                                    );
                                    full_shm.put_rect(dpy, win, gc, st.exp_x0, st.exp_y0, st.exp_x1 - st.exp_x0, st.exp_y1 - st.exp_y0);
                                    any_blitted = true;
                                } else if let Some(md) = mon_dirty[i] {
                                    merge_tile_rects(&md.tiles, md.bbox, cw, ch, &mut tile_rects_scratch);
                                    for &(cx0, cy0, cx1, cy1) in &tile_rects_scratch {
                                        let (fx0, fy0, fx1, fy1) =
                                            ex.cell_rect_to_screen(cx0, cy0, cx1, cy1, cw, ch);
                                        let dx0 = st.exp_x0.max(fx0);
                                        let dy0 = st.exp_y0.max(fy0);
                                        let dx1 = st.exp_x1.min(fx1);
                                        let dy1 = st.exp_y1.min(fy1);
                                        if dx1 > dx0 && dy1 > dy0 {
                                            ex.expand_bgrx_rect(
                                                cell_slice, cw as usize, full_shm.pixels_mut(), sw as usize,
                                                dx0, dy0, dx1, dy1,
                                            );
                                            full_shm.put_rect(dpy, win, gc, dx0, dy0, dx1 - dx0, dy1 - dy0);
                                            any_blitted = true;
                                        }
                                    }
                                }
                                mon_full_repaint[i] = false;
                                mon_dirty[i] = None;
                                mon_next_due[i] = t0 + if st.mostly_covered { work_dur } else { full_dur };
                            }
                            }
                        }

                        if any_blitted {
                            XFlush(dpy);
                        }
                    }
                }
            }

            // 8. Event-driven sleep on [x11_fd, sock_fd] until soonest due tick!
            let now = Instant::now();
            let mut soonest = next_canvas_at;
            for i in 0..monitors.len() {
                if mon_states[i].visible && mon_next_due[i] < soonest {
                    soonest = mon_next_due[i];
                }
            }
            if let Some(rem) = soonest.checked_duration_since(now) {
                let ms = rem.as_millis().min(500) as c_int;
                if ms > 0 {
                    let mut pfds = [
                        PollFd { fd: x_fd, events: POLLIN, revents: 0 },
                        PollFd { fd: sock_fd, events: POLLIN, revents: 0 },
                    ];
                    let _ = poll(pfds.as_mut_ptr(), 2, ms);
                } else {
                    thread::sleep(rem);
                }
            }
        }

        // Clean shutdown: preserve last frame on persistent _XROOTPMAP_ID.
        if !exit_after_static_handover && owned_root_pmap != 0 {
            if let Some(ref a) = anim {
                let mut bgrx = vec![0u8; (sw as usize) * (sh as usize) * 4];
                if a.canvas_synced && cell_shm.is_attached() {
                    a.expand_bgrx_full(cell_shm.pixels(), &mut bgrx, sw as usize);
                } else {
                    let mut tmp_cell = vec![0u8; (a.mwa.cell_w as usize) * (a.mwa.cell_h as usize) * 4];
                    if a.mwa.decode_base_bgrx(&mut tmp_cell).is_ok() {
                        a.expand_bgrx_full(&tmp_cell, &mut bgrx, sw as usize);
                    }
                }
                let _ = publish_persistent_root_pixmap(&bgrx, sw, sh, &monitors);
            }
            XFreePixmap(dpy, owned_root_pmap);
        } else if owned_root_pmap != 0 {
            XFreePixmap(dpy, owned_root_pmap);
        }

        if let Some(ref mut xr) = xrender {
            xr.destroy(dpy);
        }
        cell_shm.detach(dpy);
        full_shm.detach(dpy);
        XFreeGC(dpy, gc);
        XDestroyWindow(dpy, win);
        XSync(dpy, 0);
        XCloseDisplay(dpy);
    }

    if let Ok(txt) = fs::read_to_string(&pfile) {
        if txt.trim().parse::<c_int>().ok() == Some(self_pid) {
            let _ = fs::remove_file(&pfile);
        }
    }
    let _ = fs::remove_file(&spath);
    drop(lock_file);
    0
}
