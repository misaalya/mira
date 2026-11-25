//! Scroll - Hyprland-style scrolling columns for bspwm.
//!
//! Every window on a "scroll" desktop becomes a column on a horizontal strip.
//! Columns outside the viewport are hidden, so the strip can be arbitrarily
//! long while every visible column keeps the full height of the screen.
//!
//! Mirrors the scrolling layout of the misa Hyprland config: new columns to
//! the right of the focused one, fullscreen_on_one_column, focus_fit_method =
//! 1 (scroll the minimum needed to fit the focused column), follow_focus and
//! wrap_focus. Tunables live in config/scroll.conf.
//!
//! The daemon is one thread and one poll() loop. It talks to bspwm over the
//! same UNIX socket bspc uses instead of starting a bspc process per request,
//! folds a burst of events into one layout pass per desktop, and reads the
//! whole desktop (rectangle, gap, focus, every window's geometry) with a
//! single tree query per pass. Its own geometry changes are recognised from
//! the events they echo, so only a resize/move made by someone else costs a
//! query.
//!
//! (Formerly Python; the runtime files - state.json, cmd.fifo, pid, on-<desk>,
//! the Layout hand-over files - are unchanged.)

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;
use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void, CString};
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const USAGE: &str = "Scroll - Hyprland-style scrolling columns for bspwm.

CLI:
  Scroll init                 enable on every desktop, start the daemon
  Scroll on|off|toggle        for the focused desktop
  Scroll left|right           move focus along the strip
  Scroll swap-left|right      move the focused column
  Scroll width SPEC           resize the focused column: grow|shrink (step),
                              next|prev (preset cycle), +N|-N, or N (fraction)
  Scroll float                take the focused column off the strip (floating
                              above it), or put a floating window back in
  Scroll sync                 re-place the focused desktop
  Scroll status               what every desktop is doing
  Scroll daemon               run the event loop (internal)
";

const EVENTS: [&str; 18] = [
    "node_add", "node_remove", "node_transfer", "node_swap", "node_focus", "node_activate",
    "node_state", "node_geometry", "desktop_focus", "desktop_rename",
    "desktop_remove", "desktop_transfer", "desktop_swap", "monitor_geometry",
    "monitor_add", "monitor_remove", "monitor_swap", "pointer_action",
];

/// A window may differ from what the strip asked for by a pixel or two
/// (rounding, toolkit constraints); only a bigger gap is someone else's doing.
const GEO_TOLERANCE: i64 = 2;
/// Events of one burst (node_add + node_focus + ...) arrive within a few ms;
/// waiting this long before a layout pass folds them into one.
const COALESCE: Duration = Duration::from_millis(4);
/// ...but never hold a pass back longer than this (a drag streams events).
const COALESCE_MAX: Duration = Duration::from_millis(30);
const SAVE_DELAY: Duration = Duration::from_secs(1);

// --- libc / Xlib --------------------------------------------------------

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: i16,
    revents: i16,
}
const POLLIN: i16 = 0x1;

#[repr(C)]
struct Tm {
    sec: c_int,
    min: c_int,
    hour: c_int,
    mday: c_int,
    mon: c_int,
    year: c_int,
    wday: c_int,
    yday: c_int,
    isdst: c_int,
    gmtoff: c_long,
    zone: *const c_char,
}

extern "C" {
    fn poll(fds: *mut PollFd, n: c_ulong, timeout: c_int) -> c_int;
    fn mkfifo(path: *const c_char, mode: c_uint) -> c_int;
    fn signal(sig: c_int, handler: usize) -> usize;
    fn setsid() -> c_int;
    fn getpid() -> c_int;
    fn time(t: *mut i64) -> i64;
    fn localtime_r(t: *const i64, tm: *mut Tm) -> *mut Tm;
}
const SIGINT: c_int = 2;
const SIGTERM: c_int = 15;

type XDisplay = c_void;
type XWindow = c_ulong;

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
    event_mask: c_long,
    do_not_propagate_mask: c_long,
    override_redirect: c_int,
    colormap: c_ulong,
    cursor: c_ulong,
}

#[link(name = "X11")]
extern "C" {
    fn XOpenDisplay(name: *const c_char) -> *mut XDisplay;
    fn XDefaultRootWindow(d: *mut XDisplay) -> XWindow;
    fn XQueryPointer(
        d: *mut XDisplay, w: XWindow, root: *mut XWindow, child: *mut XWindow,
        rx: *mut c_int, ry: *mut c_int, wx: *mut c_int, wy: *mut c_int, mask: *mut c_uint,
    ) -> c_int;
    fn XCreateWindow(
        d: *mut XDisplay, parent: XWindow, x: c_int, y: c_int, w: c_uint, h: c_uint,
        border: c_uint, depth: c_int, class: c_uint, visual: *mut c_void, mask: c_ulong,
        attrs: *mut XSetWindowAttributes,
    ) -> XWindow;
    fn XConnectionNumber(d: *mut XDisplay) -> c_int;
    fn XPending(d: *mut XDisplay) -> c_int;
    fn XNextEvent(d: *mut XDisplay, ev: *mut c_void) -> c_int;
    fn XMapRaised(d: *mut XDisplay, w: XWindow) -> c_int;
    fn XUnmapWindow(d: *mut XDisplay, w: XWindow) -> c_int;
    fn XMoveResizeWindow(d: *mut XDisplay, w: XWindow, x: c_int, y: c_int, ww: c_uint, h: c_uint) -> c_int;
    fn XFlush(d: *mut XDisplay) -> c_int;
}

static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn on_stop(_sig: c_int) {
    STOP.store(true, Ordering::SeqCst);
}

// --- paths, config, log -------------------------------------------------

struct Paths {
    runtime_dir: PathBuf,
    state: PathBuf,
    fifo: PathBuf,
    pid: PathBuf,
    log: PathBuf,
    layout_dir: PathBuf,
    self_exe: PathBuf,
    self_dir: PathBuf,
}

fn uid() -> u32 {
    use std::os::unix::fs::MetadataExt;
    fs::metadata("/proc/self").map(|m| m.uid()).unwrap_or(0)
}

fn paths() -> Paths {
    let runtime = std::env::var("XDG_RUNTIME_DIR").ok().filter(|s| !s.is_empty()).unwrap_or("/tmp".into());
    let rd = Path::new(&runtime).join(format!("bspwm-scroll-{}", uid()));
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("Scroll"));
    let dir = exe.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    Paths {
        state: rd.join("state.json"),
        fifo: rd.join("cmd.fifo"),
        pid: rd.join("pid"),
        log: rd.join("log"),
        layout_dir: Path::new(&runtime).join(format!("bspwm-layout-{}", uid())),
        runtime_dir: rd,
        self_exe: exe,
        self_dir: dir,
    }
}

#[derive(Clone)]
struct Conf {
    column_width: f64,
    explicit_column_widths: Vec<f64>,
    resize_step: f64,
    min_column_width: f64,
    settle_ms: f64,
    edge_peek: f64,
    edge_click: f64,
    float_width: f64,
    float_height: f64,
}

/// Python's float(): surrounding blanks allowed.
fn pyfloat(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<f64>().ok()
}

fn load_conf(dir: &Path) -> Conf {
    let mut c = Conf {
        column_width: 0.6,
        explicit_column_widths: vec![0.333, 0.5, 0.667, 1.0],
        resize_step: 0.05,
        min_column_width: 0.1,
        settle_ms: 200.0,
        edge_peek: 40.0,
        edge_click: 4.0,
        float_width: 0.5,
        float_height: 0.6,
    };
    let text = match fs::read_to_string(dir.join("..").join("config").join("scroll.conf")) {
        Ok(t) => t,
        Err(_) => return c,
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((key, val)) = line.split_once('=') else { continue };
        let (key, val) = (key.trim(), val.trim());
        if key == "explicit_column_widths" {
            let mut w = Vec::new();
            let mut bad = false;
            for v in val.split(',').filter(|v| !v.trim().is_empty()) {
                match pyfloat(v) {
                    Some(f) => w.push(f),
                    None => bad = true,
                }
            }
            if !bad && !w.is_empty() {
                w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                c.explicit_column_widths = w;
            }
            continue;
        }
        let Some(v) = pyfloat(val) else { continue };
        match key {
            "column_width" => c.column_width = v,
            "resize_step" => c.resize_step = v,
            "min_column_width" => c.min_column_width = v,
            "settle_ms" => c.settle_ms = v,
            "edge_peek" => c.edge_peek = v,
            "edge_click" => c.edge_click = v,
            "float_width" => c.float_width = v,
            "float_height" => c.float_height = v,
            _ => {}
        }
    }
    c
}

fn log_to(path: &Path, msg: &str) {
    let mut tm = Tm { sec: 0, min: 0, hour: 0, mday: 0, mon: 0, year: 0, wday: 0, yday: 0, isdst: 0, gmtoff: 0, zone: std::ptr::null() };
    unsafe {
        let t = time(std::ptr::null_mut());
        localtime_r(&t, &mut tm);
    }
    if let Ok(mut f) = fs::OpenOptions::new().append(true).create(true).open(path) {
        let _ = writeln!(f, "{:02}:{:02}:{:02} {}", tm.hour, tm.min, tm.sec, msg);
    }
}

fn hexid(wid: u64) -> String {
    format!("0x{:08X}", wid)
}

fn parse_hex(s: &str) -> Option<u64> {
    let t = s.trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    if t.is_empty() {
        return None;
    }
    u64::from_str_radix(t, 16).ok()
}

/// Python's round(x, 4).
fn round4(x: f64) -> f64 {
    format!("{:.4}", x).parse().unwrap_or(x)
}

// --- JSON ------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn num(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            Json::Bool(b) => Some(*b as i64 as f64),
            _ => None,
        }
    }
    fn int(&self, key: &str) -> i64 {
        self.get(key).and_then(Json::num).unwrap_or(0.0) as i64
    }
    fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Num(n) => *n != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::Arr(a) => !a.is_empty(),
            Json::Obj(o) => !o.is_empty(),
        }
    }
    fn is_obj(&self) -> bool {
        matches!(self, Json::Obj(_))
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn hex4(&mut self) -> Option<u32> {
        let h = self.s.get(self.i..self.i + 4)?;
        self.i += 4;
        u32::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok()
    }
    fn string(&mut self) -> Option<String> {
        if self.s.get(self.i) != Some(&b'"') {
            return None;
        }
        self.i += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = *self.s.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.s.get(self.i)?;
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xD800..0xDC00).contains(&u) && self.s.get(self.i..self.i + 2) == Some(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                match self.hex4() {
                                    Some(lo) if (0xDC00..0xE000).contains(&lo) => {
                                        u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                                    }
                                    _ => self.i = save,
                                }
                            }
                            let ch = char::from_u32(u).unwrap_or('\u{FFFD}');
                            let mut b = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut b).as_bytes());
                        }
                        other => out.push(other),
                    }
                }
                c if c < 0x20 => return None, // strict: no raw control characters
                c => out.push(c),
            }
        }
        Some(String::from_utf8_lossy(&out).into_owned())
    }
    fn value(&mut self, depth: usize) -> Option<Json> {
        if depth > 512 {
            return None;
        }
        self.ws();
        let c = *self.s.get(self.i)?;
        match c {
            b'{' => {
                self.i += 1;
                let mut m = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Some(Json::Obj(m));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return None;
                    }
                    self.i += 1;
                    let v = self.value(depth + 1)?;
                    m.push((k, v));
                    self.ws();
                    match self.s.get(self.i)? {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            return Some(Json::Obj(m));
                        }
                        _ => return None,
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Some(Json::Arr(a));
                }
                loop {
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match self.s.get(self.i)? {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            return Some(Json::Arr(a));
                        }
                        _ => return None,
                    }
                }
            }
            b'"' => self.string().map(Json::Str),
            b't' if self.s[self.i..].starts_with(b"true") => {
                self.i += 4;
                Some(Json::Bool(true))
            }
            b'f' if self.s[self.i..].starts_with(b"false") => {
                self.i += 5;
                Some(Json::Bool(false))
            }
            b'n' if self.s[self.i..].starts_with(b"null") => {
                self.i += 4;
                Some(Json::Null)
            }
            _ => {
                let st = self.i;
                while self.i < self.s.len() && matches!(self.s[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    self.i += 1;
                }
                std::str::from_utf8(&self.s[st..self.i]).ok()?.parse::<f64>().ok().map(Json::Num)
            }
        }
    }
}

fn json_loads(text: &str) -> Option<Json> {
    let t = text.trim();
    let mut p = Parser { s: t.as_bytes(), i: 0 };
    let v = p.value(0)?;
    if p.i != t.len() {
        return None;
    }
    Some(v)
}

/// JSON string, non-ASCII escaped (as Python's encode_basestring_ascii).
fn quote(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"),
            '\u{c}' => o.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    o.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Python's repr() of a float.
fn float_repr(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e16 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

// --- bspwm IPC ----------------------------------------------------------------

/// Where bspc would connect: $BSPWM_SOCKET, else bspwm's own template
/// (SOCKET_PATH_TPL, "/tmp/bspwm<host>_<display>_<screen>-socket").
fn socket_path() -> Option<String> {
    if let Ok(p) = std::env::var("BSPWM_SOCKET") {
        if !p.is_empty() {
            return Some(p);
        }
    }
    let disp = std::env::var("DISPLAY").unwrap_or_default();
    let (host, rest) = match disp.rfind(':') {
        Some(i) => (&disp[..i], &disp[i + 1..]),
        None => ("", ""),
    };
    let (num, screen) = rest.split_once('.').unwrap_or((rest, ""));
    let num: i64 = num.trim().parse().ok()?;
    let screen: i64 = if screen.is_empty() { 0 } else { screen.trim().parse().ok()? };
    Some(format!("/tmp/bspwm{}_{}_{}-socket", host, num, screen))
}

fn poll_fds(fds: &mut [PollFd], timeout: Option<Duration>) -> c_int {
    let ms = match timeout {
        None => -1,
        Some(d) => {
            let us = d.as_micros();
            ((us + 999) / 1000).min(i32::MAX as u128) as c_int
        }
    };
    unsafe { poll(fds.as_mut_ptr(), fds.len() as c_ulong, ms) }
}

/// The connection to bspwm: one-shot requests, plus (daemon) the event
/// subscription. While a request waits for its reply the subscription is
/// read too: bspwm is single-threaded, and if it is busy writing a burst of
/// events into a full subscriber buffer it never reads our request, so
/// waiting on the reply alone deadlocks both. Events read that way wait in
/// `buf` for the main loop.
struct Ipc {
    path: Option<String>,
    sub: Option<UnixStream>,
    buf: Vec<u8>,
    log: PathBuf,
}

impl Ipc {
    fn drain_sub(&mut self) {
        let Some(sock) = self.sub.as_mut() else { return };
        let mut chunk = [0u8; 65536];
        loop {
            match sock.read(&mut chunk) {
                Ok(0) => {
                    self.sub = None;
                    return;
                }
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == ErrorKind::WouldBlock => return,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(_) => {
                    self.sub = None;
                    return;
                }
            }
        }
    }

    /// One bspwm message, exactly what `bspc ARGS` sends. Returns (ok, text).
    fn bspc(&mut self, args: &[&str]) -> (bool, String) {
        let mut msg = Vec::new();
        for a in args {
            msg.extend_from_slice(a.as_bytes());
            msg.push(0);
        }
        let Some(path) = self.path.clone() else { return bspc_cli(args) };
        let Ok(mut s) = UnixStream::connect(&path) else { return bspc_cli(args) };
        let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
        if let Err(e) = s.write_all(&msg) {
            log_to(&self.log, &format!("bspc {:?}: {}", args, e));
            return (false, String::new());
        }
        if s.set_nonblocking(true).is_err() {
            return (false, String::new());
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut data = Vec::new();
        let mut chunk = [0u8; 65536];
        loop {
            let now = Instant::now();
            if now >= deadline {
                log_to(&self.log, &format!("bspc {:?}: no reply", args));
                return (false, String::new());
            }
            let mut fds = vec![PollFd { fd: s.as_raw_fd(), events: POLLIN, revents: 0 }];
            if let Some(sub) = &self.sub {
                fds.push(PollFd { fd: sub.as_raw_fd(), events: POLLIN, revents: 0 });
            }
            if poll_fds(&mut fds, Some(deadline - now)) < 0 {
                continue; // EINTR
            }
            if fds.len() > 1 && fds[1].revents != 0 {
                self.drain_sub();
            }
            if fds[0].revents != 0 {
                match s.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => data.extend_from_slice(&chunk[..n]),
                    Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => continue,
                    Err(e) => {
                        log_to(&self.log, &format!("bspc {:?}: {}", args, e));
                        return (false, String::new());
                    }
                }
            }
        }
        if data.first() == Some(&0x07) {
            // bspwm's FAILURE_MESSAGE
            return (false, String::from_utf8_lossy(&data[1..]).into_owned());
        }
        (true, String::from_utf8_lossy(&data).into_owned())
    }

    fn query(&mut self, args: &[&str]) -> String {
        let mut a = vec!["query"];
        a.extend_from_slice(args);
        let (ok, out) = self.bspc(&a);
        if ok { out.trim().to_string() } else { String::new() }
    }

    fn tree(&mut self, args: &[&str]) -> Option<Json> {
        let mut a = vec!["query", "-T"];
        a.extend_from_slice(args);
        let (ok, out) = self.bspc(&a);
        if !ok || out.trim().is_empty() {
            return None;
        }
        json_loads(&out)
    }
}

/// Fallback when the socket cannot be reached (unusual DISPLAY forms).
fn bspc_cli(args: &[&str]) -> (bool, String) {
    match Command::new("bspc").args(args).stdin(Stdio::null()).stderr(Stdio::null()).output() {
        Ok(o) => (o.status.success(), String::from_utf8_lossy(&o.stdout).into_owned()),
        Err(_) => (false, String::new()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
struct Rect {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

fn rect_of(j: Option<&Json>) -> Rect {
    match j {
        Some(r) => Rect { x: r.int("x"), y: r.int("y"), w: r.int("width"), h: r.int("height") },
        None => Rect::default(),
    }
}

#[derive(Clone)]
struct Client {
    state: String,
    fr: Rect,
    hidden: bool,
    layer: String,
}

/// Client nodes of a tree, left to right.
fn leaves(root: &Json) -> Vec<(u64, Client)> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(n) = stack.pop() {
        if !n.is_obj() {
            continue;
        }
        if let Some(c) = n.get("client").filter(|c| c.truthy()) {
            out.push((
                n.int("id") as u64,
                Client {
                    state: c.get("state").and_then(Json::str).unwrap_or("").to_string(),
                    fr: rect_of(c.get("floatingRectangle")),
                    hidden: n.get("hidden").map(Json::truthy).unwrap_or(false),
                    layer: c.get("layer").and_then(Json::str).unwrap_or("").to_string(),
                },
            ));
        }
        if let Some(s) = n.get("secondChild") {
            stack.push(s);
        }
        if let Some(f) = n.get("firstChild") {
            stack.push(f);
        }
    }
    out
}

fn find<'a>(clients: &'a [(u64, Client)], wid: u64) -> Option<&'a Client> {
    clients.iter().find(|(w, _)| *w == wid).map(|(_, c)| c)
}

fn tiled(state: &str) -> bool {
    state == "tiled" || state == "pseudo_tiled"
}

/// Buttons 1-3: a drag with the pointer modifier is still going on.
const BUTTONS_HELD: u32 = (1 << 8) | (1 << 9) | (1 << 10);

/// bspwm has no query for the pointer: XQueryPointer, on a connection opened
/// on the first drag only.
struct Pointer {
    tried: bool,
    dpy: *mut XDisplay,
    root: XWindow,
}

impl Pointer {
    fn get(&mut self) -> Option<(i64, i64, u32)> {
        if !self.tried {
            self.tried = true;
            unsafe {
                self.dpy = XOpenDisplay(std::ptr::null());
                if !self.dpy.is_null() {
                    self.root = XDefaultRootWindow(self.dpy);
                }
            }
        }
        if self.dpy.is_null() {
            return None;
        }
        let (mut r, mut c) = (0, 0);
        let (mut rx, mut ry, mut wx, mut wy) = (0, 0, 0, 0);
        let mut m: c_uint = 0;
        let ok = unsafe { XQueryPointer(self.dpy, self.root, &mut r, &mut c, &mut rx, &mut ry, &mut wx, &mut wy, &mut m) };
        if ok == 0 {
            return None; // the pointer is on another X screen
        }
        Some((rx as i64, ry as i64, m))
    }
}

/// Detached helper process (own session). The daemon reaps it later
/// (`Daemon::reap`): with SIGCHLD ignored, std's spawn would panic on an
/// exec failure (its waitpid gets ECHILD).
fn spawn(argv: &[&str]) -> Option<std::process::Child> {
    let mut cmd = Command::new(argv[0]);
    cmd.args(&argv[1..]).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    unsafe {
        cmd.pre_exec(|| {
            setsid();
            Ok(())
        });
    }
    cmd.spawn().ok()
}

// --- the strip ------------------------------------------------------------------

#[derive(Default)]
struct Strip {
    order: Vec<u64>,
    cam: i64,
    widths: Vec<(u64, f64)>,
    last: Option<u64>,
    pending: Vec<(u64, bool)>, // adopt on the next layout pass
    fresh: HashSet<u64>,       // columns that still have to be made floating
    /// A window that arrived while this desktop was not focused (sent from
    /// another desktop, opened in the background). It may sit off screen,
    /// hidden: the camera shows it, and it takes the focus once the desktop
    /// is focused - it never just vanishes into the strip.
    reveal: Option<u64>,
}

impl Strip {
    fn width_of(&self, wid: u64) -> Option<f64> {
        self.widths.iter().find(|(w, _)| *w == wid).map(|(_, v)| *v)
    }
    fn set_width(&mut self, wid: u64, v: f64) {
        match self.widths.iter_mut().find(|(w, _)| *w == wid) {
            Some(e) => e.1 = v,
            None => self.widths.push((wid, v)),
        }
    }
    fn fraction(&self, wid: u64, conf: &Conf) -> f64 {
        self.width_of(wid).unwrap_or(conf.column_width)
    }
    fn pos(&self, wid: Option<u64>) -> Option<usize> {
        let w = wid?;
        self.order.iter().position(|x| *x == w)
    }
    /// Pixel widths and strip offsets of the columns, left to right.
    fn spans(&self, rect: &Rect, gap: i64, conf: &Conf) -> (Vec<i64>, Vec<i64>, i64) {
        let widths: Vec<i64> = if self.order.len() == 1 && self.width_of(self.order[0]).is_none() {
            vec![rect.w - gap] // Hyprland: fullscreen_on_one_column (until explicitly resized)
        } else {
            self.order
                .iter()
                .map(|w| ((rect.w as f64 * self.fraction(*w, conf)).round_ties_even() as i64).max(1))
                .collect()
        };
        let mut lefts = Vec::with_capacity(widths.len());
        let mut total = 0;
        for w in &widths {
            lefts.push(total);
            total += w;
        }
        (widths, lefts, total)
    }
    fn insert(&mut self, wid: u64) {
        match self.pos(self.last) {
            Some(i) => self.order.insert(i + 1, wid),
            None => self.order.push(wid),
        }
    }
}

/// Width fraction asked for by a `width` spec, or None if it is invalid.
///   grow|shrink   +/- resize_step              (super + ctrl + Right/Left)
///   next|prev     cycle explicit_column_widths (super + period/comma)
///   +N | -N       relative change, N a fraction
///   N             absolute fraction
fn new_fraction(cur: f64, spec: &str, conf: &Conf, min_fraction: f64) -> Option<f64> {
    let presets = &conf.explicit_column_widths;
    let eps = 0.005; // a mouse-resized column sits between presets
    let new = match spec {
        "next" => *presets.iter().find(|p| **p > cur + eps).unwrap_or(&presets[0]),
        "prev" => *presets.iter().rev().find(|p| **p < cur - eps).unwrap_or(&presets[presets.len() - 1]),
        "grow" => cur + conf.resize_step,
        "shrink" => cur - conf.resize_step,
        _ => {
            let val = pyfloat(spec)?;
            if spec.starts_with('+') || spec.starts_with('-') { cur + val } else { val }
        }
    };
    Some(round4(min_fraction.max(clamp1(new))))
}

/// Python's min(1.0, x) (NaN keeps 1.0).
fn clamp1(x: f64) -> f64 {
    if x < 1.0 { x } else { 1.0 }
}

/// node_geometry payload "WxH+X+Y" (bspwm writes "+-N" for negatives).
fn parse_rect(text: &str) -> Option<(i64, i64, i64, i64)> {
    let p: Vec<&str> = text.split('+').collect();
    if p.len() != 3 {
        return None;
    }
    let (w, h) = p[0].split_once('x')?;
    if h.contains('x') {
        return None;
    }
    Some((p[1].trim().parse().ok()?, p[2].trim().parse().ok()?, w.trim().parse().ok()?, h.trim().parse().ok()?))
}

/// Invisible click zones on the screen edges. When the focused column fills
/// the screen there is no neighbouring column on screen to click, so
/// `edge_peek` has nothing to show. These two thin input-only windows catch a
/// click on the far left / right edge instead; the daemon then focuses the
/// column on that side and the strip scrolls.
struct Zones {
    dpy: *mut XDisplay,
    win: [XWindow; 2],
    fd: RawFd,
    shown: [bool; 2],
}
const LEFT: usize = 0;
const RIGHT: usize = 1;

impl Zones {
    fn new() -> Option<Zones> {
        unsafe {
            let dpy = XOpenDisplay(std::ptr::null());
            if dpy.is_null() {
                return None;
            }
            let root = XDefaultRootWindow(dpy);
            let mut a: XSetWindowAttributes = std::mem::zeroed();
            a.override_redirect = 1;
            a.event_mask = 1 << 2; // ButtonPressMask
            // InputOnly (class 2), depth/visual from the parent;
            // CWOverrideRedirect | CWEventMask.
            let mut win = [0; 2];
            for w in win.iter_mut() {
                *w = XCreateWindow(dpy, root, 0, 0, 1, 1, 0, 0, 2, std::ptr::null_mut(), (1 << 9) | (1 << 11), &mut a);
            }
            Some(Zones { dpy, win, fd: XConnectionNumber(dpy), shown: [false, false] })
        }
    }
    fn show(&mut self, side: usize, x: i64, y: i64, w: i64, h: i64) {
        unsafe {
            XMoveResizeWindow(self.dpy, self.win[side], x as c_int, y as c_int, w as c_uint, h as c_uint);
            XMapRaised(self.dpy, self.win[side]);
            self.shown[side] = true;
            XFlush(self.dpy);
        }
    }
    fn hide(&mut self, side: usize) {
        if self.shown[side] {
            unsafe {
                XUnmapWindow(self.dpy, self.win[side]);
                self.shown[side] = false;
                XFlush(self.dpy);
            }
        }
    }
    /// Sides (LEFT/RIGHT) that got a primary-button press.
    fn clicks(&mut self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut buf = [0u64; 24]; // sizeof(XEvent)
        unsafe {
            while XPending(self.dpy) != 0 {
                XNextEvent(self.dpy, buf.as_mut_ptr() as *mut c_void);
                let p = buf.as_ptr() as *const u8;
                let kind = *(p as *const c_int);
                if kind != 4 {
                    continue; // ButtonPress only
                }
                let win = *(p.add(32) as *const c_ulong);
                let button = *(p.add(84) as *const c_uint);
                if button == 1 {
                    if let Some(i) = self.win.iter().position(|w| *w == win) {
                        out.push(i);
                    }
                }
            }
        }
        out
    }
}

// --- the daemon ---------------------------------------------------------------------

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
enum Kind {
    Save,
    Settle,
}

type Geo = (i64, i64, i64, i64);

struct Daemon {
    p: Paths,
    conf: Conf,
    min_fraction: f64,
    ipc: Ipc,
    strips: Vec<(String, Strip)>,      // desktop name -> Strip (in creation order)
    owner: HashMap<u64, String>,       // window id -> desktop name, for every column
    names: HashMap<String, String>,    // desktop id -> name (cache)
    geo: HashMap<u64, Geo>,            // window id -> geometry the strip last asked for
    expect: HashMap<u64, Vec<Geo>>,    // window id -> geometry echoes still to come
    rects: HashMap<String, (Rect, i64)>, // desktop name -> (rect, gap) of the last pass
    ffp: Option<String>,               // focus_follows_pointer before the first strip
    dirty: Vec<(String, Option<u64>)>, // desktop name -> focus hint
    dirty_since: Instant,
    timers: BinaryHeap<Reverse<(Instant, u64, Kind, u64)>>,
    settle_at: HashMap<u64, Instant>,
    grabbed: HashSet<u64>,             // columns moved/resized with a mouse button held
    save_at: Option<Instant>,
    seq: u64,
    floated: HashMap<u64, (Option<f64>, Option<Geo>)>, // width fraction / rect it had on/off the strip
    zones: Option<Zones>,
    zone_desk: Option<String>,
    pointer: Pointer,
    children: Vec<std::process::Child>, // helpers not reaped yet
    cur_desk: Option<String>,          // name of the focused desktop
    mons: Option<Vec<Rect>>,           // monitor rectangles (cache)
}

impl Daemon {
    fn new(p: Paths, conf: Conf) -> Daemon {
        let min_fraction = conf.min_column_width.max(0.05).min(1.0);
        let ipc = Ipc { path: socket_path(), sub: None, buf: Vec::new(), log: p.log.clone() };
        let mut d = Daemon {
            p, conf, min_fraction, ipc,
            strips: Vec::new(), owner: HashMap::new(), names: HashMap::new(), geo: HashMap::new(),
            expect: HashMap::new(), rects: HashMap::new(), ffp: None, dirty: Vec::new(),
            dirty_since: Instant::now(), timers: BinaryHeap::new(), settle_at: HashMap::new(), grabbed: HashSet::new(),
            save_at: None, seq: 0, floated: HashMap::new(), zones: None, zone_desk: None,
            pointer: Pointer { tried: false, dpy: std::ptr::null_mut(), root: 0 },
            children: Vec::new(), cur_desk: None, mons: None,
        };
        d.load();
        d.cur_desk = d.focused().0;
        d
    }

    fn log(&self, msg: &str) {
        log_to(&self.p.log, msg);
    }

    fn strip(&self, name: &str) -> Option<&Strip> {
        self.strips.iter().find(|(n, _)| n == name).map(|(_, s)| s)
    }
    fn strip_mut(&mut self, name: &str) -> Option<&mut Strip> {
        self.strips.iter_mut().find(|(n, _)| n == name).map(|(_, s)| s)
    }
    fn has(&self, name: &str) -> bool {
        self.strips.iter().any(|(n, _)| n == name)
    }

    // --- state file ---

    fn load(&mut self) {
        let st = fs::read_to_string(&self.p.state).ok().and_then(|t| json_loads(&t)).unwrap_or(Json::Obj(vec![]));
        self.ffp = st.get("ffp").and_then(Json::str).map(str::to_string);
        let known_s = self.ipc.query(&["-D", "--names"]);
        let known: HashSet<&str> = known_s.split_whitespace().collect();
        let mut loaded = Vec::new();
        if let Some(Json::Obj(desks)) = st.get("desktops") {
            for (name, d) in desks {
                if !known.contains(name.as_str()) {
                    continue;
                }
                let parse = || -> Option<Strip> {
                    let mut s = Strip::default();
                    if let Some(o) = d.get("order") {
                        let Json::Arr(a) = o else { return None };
                        for w in a {
                            s.order.push(parse_hex(w.str()?)?);
                        }
                    }
                    if let Some(Json::Obj(ws)) = d.get("widths") {
                        for (w, v) in ws {
                            s.widths.push((parse_hex(w)?, v.num()?));
                        }
                    }
                    s.last = match d.get("last") {
                        Some(l) if l.truthy() => Some(parse_hex(l.str()?)?),
                        _ => None,
                    };
                    s.cam = d.get("cam").and_then(Json::num).unwrap_or(0.0) as i64;
                    Some(s)
                };
                if let Some(s) = parse() {
                    loaded.push((name.clone(), s));
                }
            }
        }
        for (name, s) in loaded {
            for w in &s.order {
                self.owner.insert(*w, name.clone());
            }
            self.strips.retain(|(n, _)| *n != name);
            self.strips.push((name.clone(), s));
            self.mark(&name, None);
        }
        // Strips restored from state.json skip enable(): hover focus off here.
        if !self.strips.is_empty() {
            self.ipc.bspc(&["config", "focus_follows_pointer", "false"]);
        }
    }

    fn save(&mut self) {
        self.save_at = None;
        let mut out = String::from("{\"desktops\":{");
        for (i, (name, s)) in self.strips.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let order: Vec<String> = s.order.iter().map(|w| quote(&hexid(*w))).collect();
            let widths: Vec<String> = s.widths.iter().map(|(w, v)| format!("{}:{}", quote(&hexid(*w)), float_repr(*v))).collect();
            let last = match s.last {
                Some(l) if l != 0 => quote(&hexid(l)),
                _ => "null".into(),
            };
            out.push_str(&format!(
                "{}:{{\"order\":[{}],\"cam\":{},\"widths\":{{{}}},\"last\":{}}}",
                quote(name), order.join(","), s.cam, widths.join(","), last
            ));
        }
        out.push('}');
        if let Some(f) = &self.ffp {
            out.push_str(&format!(",\"ffp\":{}", quote(f)));
        }
        out.push('}');
        let tmp = self.p.state.with_extension("json.tmp");
        if let Err(e) = fs::write(&tmp, out).and_then(|_| fs::rename(&tmp, &self.p.state)) {
            self.log(&format!("save: {}", e));
        }
    }

    fn save_later(&mut self) {
        if self.save_at.is_none() {
            let when = Instant::now() + SAVE_DELAY;
            self.save_at = Some(when);
            self.timer(when, Kind::Save, 0);
        }
    }

    // --- timers ---

    fn timer(&mut self, when: Instant, kind: Kind, key: u64) {
        self.seq += 1;
        self.timers.push(Reverse((when, self.seq, kind, key)));
    }

    fn run_timers(&mut self) {
        let now = Instant::now();
        while let Some(Reverse((when, _, _, _))) = self.timers.peek() {
            if *when > now {
                break;
            }
            let Reverse((when, _, kind, key)) = self.timers.pop().unwrap();
            match kind {
                Kind::Save if self.save_at == Some(when) => self.save(),
                Kind::Settle if self.settle_at.get(&key) == Some(&when) => {
                    self.settle_at.remove(&key);
                    self.settle(key);
                }
                _ => {}
            }
        }
    }

    fn next_timeout(&self) -> Option<Duration> {
        if !self.dirty.is_empty() {
            return Some(COALESCE);
        }
        let Reverse((when, ..)) = self.timers.peek()?;
        Some(when.saturating_duration_since(Instant::now()))
    }

    fn settle_later(&mut self, wid: u64) {
        let when = Instant::now() + Duration::from_secs_f64(self.conf.settle_ms / 1000.0);
        self.settle_at.insert(wid, when);
        self.timer(when, Kind::Settle, wid);
    }

    // --- helpers ---

    fn desk_name(&mut self, desk_id: &str) -> String {
        if let Some(n) = self.names.get(desk_id) {
            return n.clone();
        }
        let name = self.ipc.query(&["-D", "-d", desk_id, "--names"]);
        if !name.is_empty() {
            self.names.insert(desk_id.to_string(), name.clone());
        }
        name
    }

    fn mark(&mut self, name: &str, focus: Option<u64>) {
        if !self.has(name) {
            return;
        }
        if self.dirty.is_empty() {
            self.dirty_since = Instant::now();
        }
        match self.dirty.iter_mut().find(|(n, _)| n == name) {
            Some(e) => {
                if focus.is_some() {
                    e.1 = focus;
                }
            }
            None => self.dirty.push((name.to_string(), focus)),
        }
    }

    fn drop_col(&mut self, name: &str, wid: u64) -> Option<u64> {
        let mut fallback = None;
        if let Some(s) = self.strip_mut(name) {
            if let Some(i) = s.order.iter().position(|w| *w == wid) {
                s.order.remove(i);
                s.widths.retain(|(w, _)| *w != wid);
                s.fresh.remove(&wid);
                if !s.order.is_empty() {
                    let neighbor = s.order[if i > 0 { i - 1 } else { 0 }];
                    if s.last == Some(wid) || s.pos(s.last).is_none() {
                        s.last = Some(neighbor);
                    }
                    fallback = s.last;
                } else {
                    s.last = None;
                }
            } else if s.pos(s.last).is_some() {
                fallback = s.last;
            }
            s.pending.retain(|(w, _)| *w != wid);
            if s.reveal == Some(wid) {
                s.reveal = None;
            }
        }
        self.owner.remove(&wid);
        self.geo.remove(&wid);
        self.expect.remove(&wid);
        self.settle_at.remove(&wid);
        self.grabbed.remove(&wid);
        self.save_later();
        fallback
    }

    fn flush(&mut self) {
        let dirty = std::mem::take(&mut self.dirty);
        for (name, focus) in dirty {
            self.layout(&name, focus);
        }
    }

    // --- layout ---

    /// Place one desktop's strip and follow the focused column.
    fn layout(&mut self, name: &str, focus: Option<u64>) {
        if !self.has(name) {
            return;
        }
        let Some(d) = self.ipc.tree(&["-d", name]) else { return };
        let root = d.get("root").filter(|r| r.truthy());
        let clients = root.map(leaves).unwrap_or_default();
        let desk_focus = d.get("focusedNodeId").and_then(Json::num).map(|v| v as u64).filter(|v| *v != 0);
        let mut changed = false;

        // Columns that are gone. One that someone turned tiled stays a column
        // and is floated back in place: a tiled window here would sit behind
        // the strip and fill the whole desktop. Fullscreen ones resume after.
        let order_now = self.strip(name).map(|s| s.order.clone()).unwrap_or_default();
        for wid in order_now {
            match find(&clients, wid) {
                None => {
                    self.drop_col(name, wid);
                    changed = true;
                }
                Some(c) if tiled(&c.state) => {
                    self.strip_mut(name).unwrap().fresh.insert(wid);
                }
                _ => {}
            }
        }

        // New windows: right of the focused column, in arrival order.
        let pending = std::mem::take(&mut self.strip_mut(name).unwrap().pending);
        for (wid, force) in pending {
            let Some(c) = find(&clients, wid) else { continue };
            if self.strip(name).unwrap().order.contains(&wid) {
                continue;
            }
            // Opened fullscreen (a browser closed while fullscreen comes
            // back that way): still a column, or it is outside the strip,
            // with no width to change, covering it like a screen of its own.
            if !force && !tiled(&c.state) && c.state != "fullscreen" {
                // Floated by a rule (dialogs): like Hyprland, floating windows
                // stay above the columns, which are floating too.
                if c.state == "floating" && c.layer == "normal" {
                    self.ipc.bspc(&["node", &hexid(wid), "-l", "above"]);
                }
                continue;
            }
            let s = self.strip_mut(name).unwrap();
            s.insert(wid);
            if c.state != "floating" {
                s.fresh.insert(wid);
            }
            // The new window's own focus event usually came before it was
            // adopted; unless it opened in the background next to a column
            // that holds the focus, the chain continues from it.
            if desk_focus == Some(wid) || s.pos(desk_focus).is_none() {
                s.last = Some(wid);
            }
            self.owner.insert(wid, name.to_string());
            changed = true;
        }

        // Any other tiled, fullscreen, or orphaned hidden column on this desktop
        // joins the strip as well (never leave a hidden column stranded off-strip).
        for (wid, c) in &clients {
            let s = self.strip_mut(name).unwrap();
            if !s.order.contains(wid)
                && (tiled(&c.state) || c.state == "fullscreen" || (c.state == "floating" && c.hidden))
            {
                s.insert(*wid);
                if c.state != "floating" {
                    s.fresh.insert(*wid);
                }
                if desk_focus == Some(*wid) {
                    s.last = Some(*wid);
                }
                self.owner.insert(*wid, name.to_string());
                changed = true;
            }
        }

        let s = self.strip_mut(name).unwrap();
        if s.pos(s.reveal).is_none() {
            s.reveal = None;
        }
        let mut focus = focus;
        if s.reveal.is_some() {
            focus = s.reveal; // on screen (another monitor): show what arrived
        } else if s.pos(focus).is_none() {
            focus = desk_focus;
        }
        // When a fullscreen or full-width column closes while its neighbours
        // were hidden=on, bspwm has no unhidden node to focus (desk_focus is
        // None). Fall back to s.last (updated to the adjacent column by
        // drop_col) or the rightmost column so the camera always returns to
        // the neighbour and unhides + focuses it instead of leaving an empty screen.
        if s.pos(focus).is_none() {
            focus = s.last;
        }
        if s.pos(focus).is_none() {
            focus = s.order.last().copied();
        }
        if s.pos(focus).is_some() && s.last != focus {
            s.last = focus;
            changed = true;
        }

        let n = s.order.len();
        let Some(root) = root.filter(|_| n > 0) else {
            if changed {
                self.save_later();
            }
            return;
        };
        let rect = rect_of(root.get("rectangle"));
        let gap = (d.get("windowGap").and_then(Json::num).unwrap_or(0.0) as i64).max(0);
        self.rects.insert(name.to_string(), (rect, gap));
        let conf = self.conf.clone();
        let s = self.strip_mut(name).unwrap();
        let (widths, lefts, total) = s.spans(&rect, gap, &conf);

        let mut cam = s.cam;
        let fi = s.pos(focus);
        if let Some(i) = fi {
            // focus_fit_method = 1: move the camera the least amount needed.
            let left = lefts[i];
            let right = left + widths[i];
            if right - cam > rect.w {
                cam = right - rect.w;
            }
            if left - cam < 0 {
                cam = left;
            }
            // edge_peek: keep a sliver of the neighbouring column on screen,
            // never at the price of cropping the focused column itself.
            let peek = conf.edge_peek as i64;
            if peek > 0 {
                if i < n - 1 && right + peek - cam > rect.w {
                    let want = right + peek - rect.w;
                    if left - want >= 0 {
                        cam = want;
                    }
                } else if i > 0 && left - peek - cam < 0 {
                    let want = left - peek;
                    if right - want <= rect.w {
                        cam = want;
                    }
                }
            }
        }
        cam = cam.min((total - rect.w).max(0)).max(0);
        if cam != s.cam {
            s.cam = cam;
            changed = true;
        }

        // Edge click zones: a side needs one when a column lies beyond it but
        // none of it is on screen (so there is nothing to click there).
        let mut need = (false, false);
        if conf.edge_click > 0.0 {
            if let Some(i) = fi {
                need = (
                    i > 0 && lefts[i - 1] + widths[i - 1] - gap - cam <= 0,
                    i < n - 1 && lefts[i + 1] - cam >= rect.w,
                );
            }
        }
        self.sync_zones(name, rect, gap, need);

        let order = self.strip(name).unwrap().order.clone();
        // A fullscreen column covers the whole monitor. Once another column
        // has the focus it leaves fullscreen (as in Hyprland), or the focused
        // column sits invisible behind it: a browser that reopens fullscreen
        // made the terminals beside it look gone, reachable only by alt-tab.
        let held = self.strip(name).unwrap().last;
        for wid in &order {
            if Some(*wid) != held && held.is_some() {
                if let Some(c) = find(&clients, *wid) {
                    if c.state == "fullscreen" {
                        self.ipc.bspc(&["node", &hexid(*wid), "-t", "floating"]);
                    }
                }
            }
        }
        for (i, wid) in order.iter().enumerate() {
            let Some(c) = find(&clients, *wid) else { continue };
            if c.state == "fullscreen" && Some(*wid) == held {
                continue;
            }
            let mut args: Vec<String> = Vec::new();
            if c.layer != "normal" {
                args.extend(["-l".into(), "normal".into()]);
            }
            let s = self.strip_mut(name).unwrap();
            if s.fresh.remove(wid) {
                args.extend(["-t".into(), "floating".into()]);
            }
            let fr = c.fr;
            let x = rect.x + lefts[i] - cam;
            // A column half off screen shows that half on a monitor right
            // next to this one: only peek where nothing is.
            let spills = (x < rect.x && self.neighbour(&rect, true)) || (x + widths[i] - gap > rect.x + rect.w && self.neighbour(&rect, false));
            if x + widths[i] - gap > rect.x && x < rect.x + rect.w && !spills {
                let want = (x, rect.y, widths[i] - gap, rect.h - gap);
                self.geo.insert(*wid, want);
                let mut echoes = Vec::new();
                let (dx, dy) = (want.0 - fr.x, want.1 - fr.y);
                let (dw, dh) = (want.2 - fr.w, want.3 - fr.h);
                if dx != 0 || dy != 0 {
                    args.extend(["-v".into(), dx.to_string(), dy.to_string()]);
                    echoes.push((want.0, want.1, fr.w, fr.h));
                }
                if dw != 0 || dh != 0 {
                    args.extend(["-z".into(), "bottom_right".into(), dw.to_string(), dh.to_string()]);
                    echoes.push(want);
                }
                if c.hidden {
                    args.extend(["-g".into(), "hidden=off".into()]);
                }
                if !echoes.is_empty() {
                    self.expect.insert(*wid, echoes);
                }
            } else if !c.hidden {
                // If this is a newly adopted column that starts off-screen, still
                // place its coordinates onto this desktop's monitor so bspwm never
                // transfers it to another monitor when switching to floating.
                let want = (rect.x, rect.y, widths[i] - gap, rect.h - gap);
                let (dx, dy) = (want.0 - fr.x, want.1 - fr.y);
                let (dw, dh) = (want.2 - fr.w, want.3 - fr.h);
                if dx != 0 || dy != 0 {
                    args.extend(["-v".into(), dx.to_string(), dy.to_string()]);
                }
                if dw != 0 || dh != 0 {
                    args.extend(["-z".into(), "bottom_right".into(), dw.to_string(), dh.to_string()]);
                }
                args.extend(["-g".into(), "hidden=on".into()]);
            }
            if !args.is_empty() {
                let id = hexid(*wid);
                let mut a: Vec<&str> = vec!["node", &id];
                a.extend(args.iter().map(String::as_str));
                self.ipc.bspc(&a);
            }
        }
        // If bspwm lost focus on this desktop because the closed column's
        // neighbours were hidden=on (or desk_focus is now hidden), restore
        // focus to the active column once it is unhidden.
        let need_refocus = match desk_focus {
            None => true,
            Some(df) => find(&clients, df).map_or(true, |c| c.hidden),
        };
        if need_refocus {
            if let Some(target) = held.filter(|h| order.contains(h)) {
                if self.cur_desk.as_deref() == Some(name) {
                    self.ipc.bspc(&["node", &hexid(target), "-f"]);
                }
            }
        }
        if changed {
            self.save_later();
        }
    }

    /// A column's geometry no longer matches what was placed. A width change
    /// (mouse resize, `bspc node -z`) becomes the column's new width. A move
    /// (super + drag) drops the column into the slot of the column under the
    /// pointer, as in Hyprland; dropped on its own slot it slides back.
    /// Nothing happens while a mouse button is still held.
    fn settle(&mut self, wid: u64) {
        let Some(name) = self.owner.get(&wid).cloned() else { return };
        if !self.has(&name) {
            return;
        }
        let Some(want) = self.geo.get(&wid).copied() else { return };
        let Some(node) = self.ipc.tree(&["-n", &hexid(wid)]) else { return };
        let Some(c) = node.get("client").filter(|c| c.truthy()) else { return };
        if node.get("hidden").map(Json::truthy).unwrap_or(false)
            || c.get("state").and_then(Json::str) != Some("floating")
        {
            return;
        }
        let fr = rect_of(c.get("floatingRectangle"));
        let off = (fr.x - want.0).abs().max((fr.y - want.1).abs()).max((fr.w - want.2).abs()).max((fr.h - want.3).abs());
        if off <= GEO_TOLERANCE {
            return;
        }
        let ptr = self.pointer.get();
        if let Some((_, _, mask)) = ptr {
            if mask & BUTTONS_HELD != 0 {
                self.settle_later(wid);
                return;
            }
        }
        if !self.grabbed.remove(&wid) {
            // Not the user's doing: back into its slot, width unchanged.
            self.layout(&name, None);
            return;
        }
        let conf = self.conf.clone();
        let min_fraction = self.min_fraction;
        if let Some(&(rect, gap)) = self.rects.get(&name) {
            let s = self.strip_mut(&name).unwrap();
            if s.order.len() > 1 && s.order.contains(&wid) {
                if (fr.w - want.2).abs() > GEO_TOLERANCE {
                    let frac = (fr.w + gap) as f64 / rect.w as f64;
                    s.set_width(wid, round4(min_fraction.max(clamp1(frac))));
                    self.save_later();
                } else {
                    // Drop point: the pointer, else the window's centre.
                    let (px, py) = match ptr {
                        Some((x, y, _)) => (x, y),
                        None => (fr.x + fr.w.div_euclid(2), fr.y + fr.h.div_euclid(2)),
                    };
                    if rect.x <= px && px < rect.x + rect.w && rect.y <= py && py < rect.y + rect.h {
                        let (widths, lefts, _) = s.spans(&rect, gap, &conf);
                        let sx = px - rect.x + s.cam;
                        let mut j = s.order.len() - 1;
                        for (k, left) in lefts.iter().enumerate() {
                            if sx < left + widths[k] {
                                j = k;
                                break;
                            }
                        }
                        let i = s.order.iter().position(|w| *w == wid).unwrap();
                        if j != i {
                            let w = s.order.remove(i);
                            s.order.insert(j, w);
                            s.last = Some(wid);
                            self.save_later();
                        }
                    }
                }
            }
        }
        self.layout(&name, Some(wid));
    }

    /// Is there another monitor directly left (or right) of `r`?
    fn neighbour(&mut self, r: &Rect, left: bool) -> bool {
        if self.mons.is_none() {
            let ids = self.ipc.query(&["-M"]);
            let mut v = Vec::new();
            for id in ids.split_whitespace() {
                if let Some(m) = self.ipc.tree(&["-m", id]) {
                    v.push(rect_of(m.get("rectangle")));
                }
            }
            self.mons = Some(v);
        }
        self.mons.as_ref().unwrap().iter().any(|m| {
            let edge = if left { m.x + m.w == r.x } else { m.x == r.x + r.w };
            edge && m.y < r.y + r.h && r.y < m.y + m.h
        })
    }

    /// The focused desktop's strip: a window waiting to be revealed takes
    /// the focus now (the desktop is already on screen: no switch).
    fn reveal_now(&mut self, name: &str) {
        let Some(s) = self.strip_mut(name) else { return };
        let Some(w) = s.reveal.take() else { return };
        if s.order.contains(&w) || s.pending.iter().any(|(p, _)| *p == w) {
            // `last` only once it is a column: a pending one is still
            // inserted right of the column focused before it came.
            if s.order.contains(&w) {
                s.last = Some(w);
            }
            self.mark(name, Some(w));
            self.ipc.bspc(&["node", &hexid(w), "-f"]);
        }
    }

    fn remark_all(&mut self) {
        self.mons = None;
        let names: Vec<String> = self.strips.iter().map(|(n, _)| n.clone()).collect();
        for n in names {
            self.mark(&n, None);
        }
    }

    // --- events ---

    fn on_event(&mut self, line: &str) {
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 3 {
            return;
        }
        let hex = |s: &str| parse_hex(s);
        match p[0] {
            "node_geometry" => {
                // Every pointer-motion step of a drag lands here: no query,
                // and the strip's own placements are recognised by their echo.
                if p.len() < 5 {
                    return;
                }
                let Some(wid) = hex(p[3]) else { return };
                if !self.owner.contains_key(&wid) {
                    return;
                }
                if let (Some(r), Some(exp)) = (parse_rect(p[4]), self.expect.get_mut(&wid)) {
                    if let Some(k) = exp.iter().position(|e| {
                        (r.0 - e.0).abs().max((r.1 - e.1).abs()).max((r.2 - e.2).abs()).max((r.3 - e.3).abs()) <= GEO_TOLERANCE
                    }) {
                        exp.remove(k);
                        return;
                    }
                }
                self.settle_later(wid);
            }
            "pointer_action" if p.len() >= 6 => {
                // pointer_action MON DESK NODE move|resize_* begin|end: the
                // user moving/resizing a column with the mouse. Any other
                // change of a column's geometry is the app's own doing (a
                // browser restoring "maximized" puts itself back at full
                // size) and is undone: it once hid the columns beside it.
                if p[5] == "begin" {
                    if let Some(w) = hex(p[3]) {
                        self.grabbed.insert(w);
                    }
                }
            }
            "node_focus" | "node_activate" if p.len() >= 4 => {
                let name = self.desk_name(p[2]);
                if self.has(&name) {
                    let Some(w) = hex(p[3]) else { return };
                    self.mark(&name, Some(w));
                }
            }
            "node_add" if p.len() >= 5 => {
                let name = self.desk_name(p[2]);
                if self.has(&name) {
                    let Some(w) = hex(p[4]) else { return };
                    let away = self.cur_desk.as_deref() != Some(name.as_str());
                    let s = self.strip_mut(&name).unwrap();
                    s.pending.push((w, false));
                    if away {
                        s.reveal = Some(w); // opened on a desktop you are not looking at
                    }
                    self.mark(&name, None);
                }
            }
            "node_remove" if p.len() >= 4 => {
                let Some(wid) = hex(p[3]) else { return };
                self.floated.remove(&wid);
                if let Some(name) = self.owner.get(&wid).cloned() {
                    let next = self.drop_col(&name, wid);
                    self.mark(&name, next);
                } else {
                    let name = self.desk_name(p[2]);
                    if self.has(&name) {
                        self.mark(&name, None);
                    }
                }
            }
            "node_state" if p.len() >= 6 => {
                // A column leaving floating, any window turning tiled, or
                // fullscreen on/off; the strip's own `-t floating` on a new
                // column echoes "tiled off" + "floating on".
                if (p[4] == "floating" && p[5] == "off") || p[4] == "fullscreen" || (tiled(p[4]) && p[5] == "on") {
                    let Some(wid) = hex(p[3]) else { return };
                    let name = match self.owner.get(&wid).cloned() {
                        Some(n) if !n.is_empty() => n,
                        _ => self.desk_name(p[2]),
                    };
                    if self.has(&name) {
                        self.mark(&name, None);
                    }
                }
            }
            "node_transfer" if p.len() >= 7 => {
                // node_transfer SRC_MON SRC_DESK SRC_NODE DST_MON DST_DESK
                // DST_NODE: the moved window is SRC_NODE.
                let Some(wid) = hex(p[3]) else { return };
                let mut src = self.owner.get(&wid).cloned();
                if src.is_none() {
                    let sname = self.desk_name(p[2]);
                    if self.strip(&sname).map_or(false, |s| s.pending.iter().any(|(w, _)| *w == wid)) {
                        src = Some(sname);
                    }
                }
                let dst = self.desk_name(p[5]);
                let width = src.as_deref().and_then(|n| self.strip(n)).and_then(|s| s.width_of(wid));
                if let Some(src) = &src {
                    let next = self.drop_col(src, wid);
                    self.mark(src, next);
                }
                if self.has(&dst) {
                    let here = self.cur_desk.as_deref() == Some(dst.as_str());
                    let s = self.strip_mut(&dst).unwrap();
                    s.pending.push((wid, src.is_some()));
                    if let Some(w) = width {
                        s.set_width(wid, w); // the column keeps its width
                    }
                    // Sent there without following: the next look at that
                    // desktop starts on it (followed: it has the focus anyway).
                    if !here {
                        s.reveal = Some(wid);
                    }
                    self.mark(&dst, None);
                } else if src.is_some() {
                    // Leaving the strip for a tiling desktop: give the window
                    // back to the tree, and never leave it parked hidden.
                    self.ipc.bspc(&["node", p[3], "-g", "hidden=off", "-t", "tiled"]);
                }
            }
            "node_swap" if p.len() >= 7 => {
                // node_swap SRC_MON SRC_DESK SRC_NODE DST_MON DST_DESK DST_NODE:
                // across desktops each window lands on the other's desktop.
                if p[2] == p[5] {
                    let name = self.desk_name(p[2]);
                    self.mark(&name, None);
                    return;
                }
                let (da, db) = (self.desk_name(p[2]), self.desk_name(p[5]));
                for (node, to) in [(p[3], db.clone()), (p[6], da.clone())] {
                    let Some(w) = hex(node) else { continue };
                    if let Some(from) = self.owner.get(&w).cloned() {
                        self.drop_col(&from, w);
                    }
                    if self.has(&to) {
                        self.strip_mut(&to).unwrap().pending.push((w, true));
                    } else {
                        self.ipc.bspc(&["node", node, "-g", "hidden=off"]);
                    }
                }
                self.mark(&da, None);
                self.mark(&db, None);
            }
            "desktop_focus" => {
                let name = self.desk_name(p[2]);
                self.cur_desk = Some(name.clone());
                if self.has(&name) {
                    self.mark(&name, None);
                    self.reveal_now(&name);
                } else if self.zone_desk.is_some() {
                    self.hide_zones();
                }
            }
            "desktop_transfer" | "desktop_swap" | "monitor_add" | "monitor_remove" | "monitor_swap" => {
                // Desktops change monitor: every strip is placed again on the
                // rectangle it is on now (a desktop moved while not focused
                // would keep the old monitor's coordinates).
                self.names.clear();
                self.remark_all();
            }
            "desktop_rename" if p.len() >= 5 => {
                self.names.remove(p[2]);
                let (old, new) = (p[3], p[4]);
                if self.has(old) && !self.has(new) {
                    let i = self.strips.iter().position(|(n, _)| n == old).unwrap();
                    let (_, s) = self.strips.remove(i);
                    for w in &s.order {
                        self.owner.insert(*w, new.to_string());
                    }
                    self.strips.push((new.to_string(), s));
                    let _ = fs::rename(self.p.runtime_dir.join(format!("on-{}", old)), self.p.runtime_dir.join(format!("on-{}", new)));
                    self.save_later();
                }
            }
            "desktop_remove" => {
                if let Some(name) = self.names.remove(p[2]) {
                    if self.has(&name) {
                        let known = self.ipc.query(&["-D", "--names"]);
                        if !known.split_whitespace().any(|n| n == name) {
                            let i = self.strips.iter().position(|(n, _)| *n == name).unwrap();
                            let (_, s) = self.strips.remove(i);
                            for w in s.order {
                                self.owner.remove(&w);
                            }
                            self.unmark(&name);
                        }
                    }
                }
            }
            "monitor_geometry" => self.remark_all(),
            _ => {}
        }
    }

    fn unmark(&mut self, name: &str) {
        let _ = fs::remove_file(self.p.runtime_dir.join(format!("on-{}", name)));
        self.save_later();
    }

    // --- commands ---

    /// (desktop name, focused window id) of the focused desktop.
    fn focused(&mut self) -> (Option<String>, Option<u64>) {
        let Some(d) = self.ipc.tree(&["-d", "focused"]) else { return (None, None) };
        let name = d.get("name").and_then(Json::str).map(str::to_string);
        let f = d.get("focusedNodeId").and_then(Json::num).map(|v| v as u64).filter(|v| *v != 0);
        (name, f)
    }

    fn poke_bar(&mut self) {
        let layout = self.p.self_dir.join("Layout");
        if let Some(c) = spawn(&[layout.to_str().unwrap_or("Layout"), "poke"]) {
            self.children.push(c);
        }
    }

    fn reap(&mut self) {
        self.children.retain_mut(|c| matches!(c.try_wait(), Ok(None)));
    }

    fn enable(&mut self, name: &str) {
        if self.has(name) {
            return;
        }
        let Some(d) = self.ipc.tree(&["-d", name]) else { return };
        // A column sliding under a stationary pointer would otherwise make
        // bspwm focus it mid-scroll and drag the camera along with it.
        // Set every time: after a daemon restart `ffp` comes back from
        // state.json, and skipping this left hover focus on (2026-11-22).
        if self.ffp.is_none() {
            let (ok, val) = self.ipc.bspc(&["config", "focus_follows_pointer"]);
            let v = val.trim();
            self.ffp = Some(if ok && !v.is_empty() { v.to_string() } else { "true".into() });
        }
        self.ipc.bspc(&["config", "focus_follows_pointer", "false"]);
        // `hidden` is deliberately kept: a desktop coming from monocle hides
        // every window but one, and those must still become columns.
        let clients = d.get("root").filter(|r| r.truthy()).map(leaves).unwrap_or_default();
        let order: Vec<u64> = clients.iter().filter(|(_, c)| tiled(&c.state)).map(|(w, _)| *w).collect();
        // Entered through `Layout set scroll`: keep the order the windows had
        // in the layout that was just left (Layout writes it, once).
        let pref = self.take_order(&format!("{}.enter", name));
        let mut o: Vec<u64> = pref.iter().filter(|w| order.contains(w)).copied().collect();
        o.extend(order.iter().filter(|w| !pref.contains(w)));
        let s = Strip { fresh: o.iter().copied().collect(), order: o.clone(), ..Default::default() };
        self.strips.push((name.to_string(), s));
        for w in &o {
            self.owner.insert(*w, name.to_string());
        }
        // The tiling-layout engine must not rebuild this desktop underneath us.
        for suffix in [".layout", ".sig"] {
            let _ = fs::remove_file(self.p.layout_dir.join(format!("{}{}", name, suffix)));
        }
        let _ = fs::write(self.p.runtime_dir.join(format!("on-{}", name)), "");
        self.layout(name, None);
        self.save();
        self.poke_bar();
    }

    /// Window ids in the file Layout left in LAYOUT_DIR, which is removed.
    fn take_order(&self, fname: &str) -> Vec<u64> {
        let path = self.p.layout_dir.join(fname);
        let Ok(text) = fs::read_to_string(&path) else { return vec![] };
        let out = text.split_whitespace().filter_map(parse_hex).collect();
        let _ = fs::remove_file(&path);
        out
    }

    /// The strip order for Layout, so the next layout is built from it.
    fn put_order(&self, name: &str, order: &[u64]) {
        let mut text = String::from("scroll\n");
        for w in order {
            text.push_str(&hexid(*w));
            text.push('\n');
        }
        if let Err(e) = fs::create_dir_all(&self.p.layout_dir).and_then(|_| fs::write(self.p.layout_dir.join(format!("{}.order", name)), text)) {
            self.log(&format!("put_order: {}", e));
        }
    }

    fn disable(&mut self, name: &str) {
        let s = self.strips.iter().position(|(n, _)| n == name).map(|i| self.strips.remove(i).1);
        self.dirty.retain(|(n, _)| n != name);
        if self.zone_desk.as_deref() == Some(name) {
            self.hide_zones();
        }
        if let Some(s) = &s {
            self.put_order(name, &s.order);
            for w in &s.order {
                self.owner.remove(w);
                self.geo.remove(w);
                self.expect.remove(w);
                self.ipc.bspc(&["node", &hexid(*w), "-g", "hidden=off", "-t", "tiled"]);
            }
        }
        // The marker goes last: Layout waits for it to vanish and then builds
        // a tree over the windows, which must all be tiled again by then.
        let _ = fs::remove_file(self.p.runtime_dir.join(format!("on-{}", name)));
        if s.is_none() {
            return;
        }
        if self.strips.is_empty() {
            if let Some(f) = self.ffp.take() {
                self.ipc.bspc(&["config", "focus_follows_pointer", &f]);
            }
        }
        self.save();
        self.poke_bar();
    }

    /// Show or hide the edge click zones for desktop `name`.
    fn sync_zones(&mut self, name: &str, rect: Rect, gap: i64, need: (bool, bool)) {
        if !(need.0 || need.1 || self.zone_desk.as_deref() == Some(name)) {
            return; // nothing to show, nothing of ours to hide
        }
        let (fname, _) = self.focused();
        if fname.as_deref() != Some(name) {
            return; // another desktop is on screen; desktop_focus redoes this
        }
        self.zone_desk = Some(name.to_string());
        self.apply_zones(rect, gap, need);
    }

    fn apply_zones(&mut self, rect: Rect, gap: i64, need: (bool, bool)) {
        if self.zones.is_none() {
            if !(need.0 || need.1) {
                return;
            }
            match Zones::new() {
                Some(z) => self.zones = Some(z),
                None => {
                    self.log("zones: no X display");
                    self.conf.edge_click = 0.0;
                    return;
                }
            }
        }
        let w = (self.conf.edge_click as i64).max(1);
        // The columns stop `gap` short of the desktop edge, so the zones reach
        // that far outwards: the very last pixel of the screen counts too.
        let x0 = (rect.x - gap).max(0);
        let x1 = rect.x + rect.w - w;
        let z = self.zones.as_mut().unwrap();
        for (side, x, zw, on) in [(LEFT, x0, w + rect.x - x0, need.0), (RIGHT, x1, w + gap, need.1)] {
            if on {
                z.show(side, x, rect.y, zw, rect.h);
            } else {
                z.hide(side);
            }
        }
    }

    fn hide_zones(&mut self) {
        self.zone_desk = None;
        if let Some(z) = self.zones.as_mut() {
            z.hide(LEFT);
            z.hide(RIGHT);
        }
    }

    fn edge_click(&mut self, side: usize) {
        let (name, cur) = self.focused();
        let Some(name) = name else { return };
        let Some(s) = self.strip(&name) else { return };
        if s.order.is_empty() {
            return;
        }
        let cur = if s.pos(cur).is_some() { cur.unwrap() } else if s.pos(s.last).is_some() { s.last.unwrap() } else { s.order[0] };
        let j = s.pos(Some(cur)).unwrap() as i64 + if side == LEFT { -1 } else { 1 };
        if 0 <= j && (j as usize) < s.order.len() {
            let target = s.order[j as usize];
            self.layout(&name, Some(target));
            self.ipc.bspc(&["node", &hexid(target), "-f"]);
        }
    }

    fn step_focus(&mut self, name: &str, cur: Option<u64>, step: i64) {
        let Some(s) = self.strip(name) else { return };
        if s.order.is_empty() {
            return;
        }
        let n = s.order.len() as i64;
        let target = match s.pos(cur) {
            Some(i) => s.order[(i as i64 + step).rem_euclid(n) as usize],
            None => s.order[0],
        };
        // Place (and unhide) it first, so it never shows up at a stale spot.
        self.layout(name, Some(target));
        self.ipc.bspc(&["node", &hexid(target), "-f"]);
    }

    fn step_swap(&mut self, name: &str, cur: Option<u64>, step: i64) {
        let Some(s) = self.strip_mut(name) else { return };
        let Some(i) = s.pos(cur) else { return };
        if s.order.len() < 2 {
            return;
        }
        let j = (i as i64 + step).rem_euclid(s.order.len() as i64) as usize;
        s.order.swap(i, j);
        self.layout(name, cur);
        self.save_later();
    }

    fn step_width(&mut self, name: &str, cur: Option<u64>, spec: &str) {
        let conf = self.conf.clone();
        let min_fraction = self.min_fraction;
        let cur = {
            let Some(s) = self.strip(name) else { return };
            cur.filter(|c| s.order.contains(c))
                .or_else(|| s.last.filter(|c| s.order.contains(c)))
                .or_else(|| s.order.last().copied())
        };
        let Some(cur) = cur else { return };
        // A fullscreen column (or a single auto-full-width column) visually
        // occupies 1.0 of the screen: start resizing from 1.0 and leave
        // fullscreen first so Ctrl+Super+Left/Right always responds.
        let fs_check = self.ipc.tree(&["-n", &hexid(cur)]).and_then(|n| n.get("client").and_then(|c| c.get("state")).and_then(Json::str).map(|v| v == "fullscreen")).unwrap_or(false);
        let Some(s) = self.strip_mut(name) else { return };
        let lone_full = s.order.len() == 1 && s.width_of(cur).is_none();
        let cur_frac = if fs_check || lone_full { 1.0 } else { s.fraction(cur, &conf) };
        let Some(new) = new_fraction(cur_frac, spec, &conf, min_fraction) else { return };
        s.set_width(cur, new);
        if fs_check {
            s.fresh.remove(&cur);
            self.ipc.bspc(&["node", &hexid(cur), "-t", "floating"]);
        }
        self.layout(name, Some(cur));
        self.save_later();
    }

    /// super + v on a strip desktop (Hyprland's togglefloating). Every column
    /// is already a floating bspwm window, so `-t ~floating` would tile it
    /// underneath the strip. Instead a column leaves the strip and floats
    /// centred above it (layer "above"), and a floating window rejoins the
    /// strip right of the last focused column, with the width it had before.
    fn toggle_float(&mut self, name: &str, cur: Option<u64>) {
        if !self.has(name) {
            return;
        }
        let Some(cur) = cur else { return };
        let Some(node) = self.ipc.tree(&["-n", &hexid(cur)]) else { return };
        let Some(c) = node.get("client").filter(|c| c.truthy()) else { return };
        let state = c.get("state").and_then(Json::str).unwrap_or("").to_string();
        if state == "fullscreen" {
            return;
        }
        let fr = rect_of(c.get("floatingRectangle"));
        let s = self.strip(name).unwrap();
        if let Some(i) = s.pos(Some(cur)) {
            let frac = s.width_of(cur);
            let rest: Vec<u64> = s.order.iter().copied().filter(|w| *w != cur).collect();
            let back = if s.last.map_or(false, |l| rest.contains(&l)) {
                s.last
            } else if !rest.is_empty() {
                Some(if i > 0 { s.order[i - 1] } else { s.order[i + 1] })
            } else {
                None
            };
            self.drop_col(name, cur);
            self.strip_mut(name).unwrap().last = back;
            let mut want = self.floated.get(&cur).and_then(|f| f.1);
            if want.is_none() {
                if let Some(&(rect, _)) = self.rects.get(name) {
                    let w = (rect.w as f64 * self.conf.float_width) as i64;
                    let h = (rect.h as f64 * self.conf.float_height) as i64;
                    want = Some((rect.x + (rect.w - w).div_euclid(2), rect.y + (rect.h - h).div_euclid(2), w, h));
                }
            }
            let mut args: Vec<String> = vec!["-l".into(), "above".into(), "-g".into(), "hidden=off".into()];
            if state != "floating" {
                args.extend(["-t".into(), "floating".into()]);
            }
            if let Some(w) = want {
                args.extend([
                    "-v".into(), (w.0 - fr.x).to_string(), (w.1 - fr.y).to_string(),
                    "-z".into(), "bottom_right".into(), (w.2 - fr.w).to_string(), (w.3 - fr.h).to_string(),
                ]);
            }
            self.floated.insert(cur, (frac, None));
            let id = hexid(cur);
            let mut a: Vec<&str> = vec!["node", &id];
            a.extend(args.iter().map(String::as_str));
            self.ipc.bspc(&a);
            self.layout(name, back);
        } else {
            let frac = self.floated.get(&cur).and_then(|f| f.0);
            self.floated.insert(cur, (None, Some((fr.x, fr.y, fr.w, fr.h))));
            let s = self.strip_mut(name).unwrap();
            s.insert(cur);
            if let Some(f) = frac.filter(|f| *f != 0.0) {
                s.set_width(cur, f);
            }
            if state != "floating" {
                s.fresh.insert(cur);
            }
            s.last = Some(cur);
            self.owner.insert(cur, name.to_string());
            self.layout(name, Some(cur));
        }
        self.save_later();
    }

    fn run_cmd(&mut self, cmd: &str) {
        let (head, arg) = cmd.split_once(' ').unwrap_or((cmd, ""));
        let (verb, name, cur) = match head.split_once(':') {
            Some((v, n)) if matches!(v, "on" | "off" | "toggle") => (v, Some(n.to_string()), None),
            _ => {
                let (n, c) = self.focused();
                (head, n, c)
            }
        };
        let Some(name) = name.filter(|n| !n.is_empty()) else { return };
        match verb {
            "left" | "right" => self.step_focus(&name, cur, if verb == "left" { -1 } else { 1 }),
            "swap-left" | "swap-right" => self.step_swap(&name, cur, if verb == "swap-left" { -1 } else { 1 }),
            "width" => self.step_width(&name, cur, arg.trim()),
            "float" => self.toggle_float(&name, cur),
            "sync" => self.layout(&name, None),
            "on" => self.enable(&name),
            "off" => self.disable(&name),
            "toggle" => {
                if self.has(&name) {
                    self.disable(&name)
                } else {
                    self.enable(&name)
                }
            }
            _ => {}
        }
    }

    // --- main loop ---

    fn subscribe(&self) -> std::io::Result<UnixStream> {
        let path = self.ipc.path.clone().ok_or_else(|| std::io::Error::new(ErrorKind::NotFound, "no socket"))?;
        let mut s = UnixStream::connect(path)?;
        let mut msg = b"subscribe\0".to_vec();
        for e in EVENTS {
            msg.extend_from_slice(e.as_bytes());
            msg.push(0);
        }
        s.write_all(&msg)?;
        s.set_nonblocking(true)?;
        Ok(s)
    }

    fn run(&mut self) {
        let cpath = CString::new(self.p.fifo.as_os_str().as_encoded_bytes()).unwrap();
        if !self.p.fifo.exists() {
            unsafe { mkfifo(cpath.as_ptr(), 0o600) };
        }
        // Keep the read end open for the whole life of the daemon, plus a
        // writer that never writes: every keypress finds a reader at once,
        // and the reader never sees EOF when the last writer leaves.
        const O_NONBLOCK: i32 = 0o4000;
        let Ok(mut fifo) = fs::OpenOptions::new().read(true).custom_flags(O_NONBLOCK).open(&self.p.fifo) else {
            self.log("cannot open the command FIFO");
            return;
        };
        let _keepalive = fs::OpenOptions::new().write(true).custom_flags(O_NONBLOCK).open(&self.p.fifo);
        let mut lost_since: Option<Instant> = None;
        let mut sbuf: Vec<u8> = Vec::new();
        let mut cbuf: Vec<u8> = Vec::new();
        while !STOP.load(Ordering::SeqCst) {
            if self.ipc.sub.is_none() {
                match self.subscribe() {
                    Ok(s) => {
                        self.ipc.sub = Some(s);
                        self.ipc.buf.clear();
                        lost_since = None;
                        sbuf.clear();
                        let names: Vec<String> = self.strips.iter().map(|(n, _)| n.clone()).collect();
                        for n in names {
                            self.mark(&n, None);
                        }
                    }
                    Err(_) => {
                        // bspwm restarts (`bspc wm -r`) within a second and its
                        // bspwmrc starts a new daemon anyway; a session that is
                        // gone must not leave this one spinning.
                        let now = Instant::now();
                        let since = *lost_since.get_or_insert(now);
                        if now - since > Duration::from_secs(10) {
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(250));
                        continue;
                    }
                }
            }
            // Events a bspc() call already read are handled without waiting.
            let timeout = if !self.ipc.buf.is_empty() { Some(Duration::ZERO) } else { self.next_timeout() };
            let sub_fd = self.ipc.sub.as_ref().unwrap().as_raw_fd();
            let mut fds = vec![
                PollFd { fd: sub_fd, events: POLLIN, revents: 0 },
                PollFd { fd: fifo.as_raw_fd(), events: POLLIN, revents: 0 },
            ];
            if let Some(z) = &self.zones {
                fds.push(PollFd { fd: z.fd, events: POLLIN, revents: 0 });
            }
            let n = poll_fds(&mut fds, timeout);
            if n < 0 {
                continue; // EINTR (a signal: STOP is checked at the top)
            }
            let ready = n > 0;
            if fds.len() > 2 && fds[2].revents != 0 {
                let sides = self.zones.as_mut().map(Zones::clicks).unwrap_or_default();
                for side in sides {
                    self.edge_click(side);
                }
            }
            if fds[0].revents != 0 {
                self.ipc.drain_sub();
            }
            if !self.ipc.buf.is_empty() {
                sbuf.append(&mut self.ipc.buf);
                if let Some(end) = sbuf.iter().rposition(|b| *b == b'\n') {
                    let lines: Vec<u8> = sbuf.drain(..=end).collect();
                    for raw in lines.split(|b| *b == b'\n') {
                        if !raw.is_empty() {
                            self.on_event(&String::from_utf8_lossy(raw));
                        }
                    }
                }
            }
            if fds[1].revents != 0 {
                let mut b = [0u8; 4096];
                if let Ok(k) = fifo.read(&mut b) {
                    cbuf.extend_from_slice(&b[..k]);
                }
                if let Some(end) = cbuf.iter().rposition(|b| *b == b'\n') {
                    let lines: Vec<u8> = cbuf.drain(..=end).collect();
                    for raw in lines.split(|b| *b == b'\n') {
                        let cmd = String::from_utf8_lossy(raw).trim().to_string();
                        if !cmd.is_empty() {
                            self.run_cmd(&cmd);
                        }
                    }
                }
            }
            if !self.dirty.is_empty() && (!ready || self.dirty_since.elapsed() > COALESCE_MAX) {
                // Quiet for COALESCE (or held back long enough): lay it out.
                self.flush();
            }
            self.run_timers();
            self.reap();
        }
    }
}

// --- daemon process ------------------------------------------------------------

/// PID of a live daemon, or None.
fn read_pid(p: &Paths) -> Option<i32> {
    let pid: i32 = fs::read_to_string(&p.pid).ok()?.trim().parse().ok()?;
    let argv = fs::read(format!("/proc/{}/cmdline", pid)).ok()?;
    let args: Vec<&[u8]> = argv.split(|b| *b == 0).collect();
    let daemon = args.iter().any(|a| *a == b"daemon");
    let scroll = args.iter().any(|a| a.ends_with(b"Scroll"));
    if daemon && scroll { Some(pid) } else { None }
}

fn daemon(p: Paths) {
    let _ = fs::create_dir_all(&p.runtime_dir);
    let me = unsafe { getpid() };
    if let Some(pid) = read_pid(&p) {
        if pid != me {
            return;
        }
    }
    let _ = fs::write(&p.pid, me.to_string());
    unsafe {
        signal(SIGTERM, on_stop as usize);
        signal(SIGINT, on_stop as usize);
    }
    let conf = load_conf(&p.self_dir);
    let mut d = Daemon::new(p, conf);
    d.run();
    if d.save_at.is_some() {
        d.save();
    }
}

fn start_daemon(p: &Paths) -> Option<i32> {
    let _ = fs::create_dir_all(&p.runtime_dir);
    if let Some(pid) = read_pid(p) {
        return Some(pid);
    }
    let log = fs::OpenOptions::new().append(true).create(true).mode(0o644).open(&p.log).ok()?;
    let err = log.try_clone().ok()?;
    let mut cmd = Command::new(&p.self_exe);
    cmd.arg("daemon").stdin(Stdio::null()).stdout(log).stderr(err);
    unsafe {
        cmd.pre_exec(|| {
            setsid();
            Ok(())
        });
    }
    if let Err(e) = cmd.spawn() {
        eprintln!("Scroll: cannot start the daemon: {}", e);
        return None;
    }
    for _ in 0..150 {
        if let Some(pid) = read_pid(p) {
            return Some(pid);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}

fn send(p: &Paths, cmd: &str) -> bool {
    let _ = fs::create_dir_all(&p.runtime_dir);
    if !p.fifo.exists() {
        let c = CString::new(p.fifo.as_os_str().as_encoded_bytes()).unwrap();
        unsafe { mkfifo(c.as_ptr(), 0o600) };
    }
    const O_NONBLOCK: i32 = 0o4000;
    for _ in 0..5 {
        match fs::OpenOptions::new().write(true).custom_flags(O_NONBLOCK).open(&p.fifo) {
            Ok(mut f) => return f.write_all(format!("{}\n", cmd).as_bytes()).is_ok(),
            Err(_) => std::thread::sleep(Duration::from_millis(50)), // no reader yet
        }
    }
    false
}

// --- CLI ------------------------------------------------------------------------------

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let cmd = argv.get(1).map(String::as_str).unwrap_or("status");
    let p = paths();
    let code = match cmd {
        "daemon" => {
            daemon(p);
            0
        }
        "status" => {
            let st = fs::read_to_string(&p.state).ok().and_then(|t| json_loads(&t)).unwrap_or(Json::Obj(vec![]));
            println!("daemon: {}", if read_pid(&p).is_some() { "running" } else { "stopped" });
            if let Some(Json::Obj(desks)) = st.get("desktops") {
                let mut desks: Vec<&(String, Json)> = desks.iter().collect();
                let key = |n: &str| if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) { n.parse::<i64>().unwrap_or(0) } else { 0 };
                desks.sort_by_key(|(n, _)| key(n));
                for (name, d) in desks {
                    let cols = match d.get("order") {
                        Some(Json::Arr(a)) => a.len(),
                        _ => 0,
                    };
                    let cam = match d.get("cam") {
                        Some(Json::Num(n)) if n.fract() == 0.0 => format!("{}", *n as i64),
                        Some(Json::Num(n)) => float_repr(*n),
                        _ => "0".into(),
                    };
                    println!("desktop {:<3} {:>2} column(s)  cam={}", name, cols, cam);
                }
            }
            0
        }
        "init" => {
            start_daemon(&p);
            let mut ipc = Ipc { path: socket_path(), sub: None, buf: Vec::new(), log: p.log.clone() };
            for name in ipc.query(&["-D", "--names"]).split_whitespace() {
                send(&p, &format!("on:{}", name));
            }
            0
        }
        "on" | "off" | "toggle" | "left" | "right" | "swap-left" | "swap-right" | "sync" | "width" | "float" => {
            if read_pid(&p).is_none() {
                start_daemon(&p);
            }
            send(&p, &argv[1..].join(" "));
            0
        }
        _ => {
            eprint!("{}", USAGE);
            1
        }
    };
    std::process::exit(code);
}
