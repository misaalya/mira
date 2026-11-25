//! misa-clip: a clipboard history daemon for the bspwm rice.
//!
//! The daemon owns a hidden window and asks XFixes to tell it whenever the
//! CLIPBOARD selection changes hands, so it sleeps in poll(2) until something
//! is copied: no timers, no polling. Each new text is fetched once
//! (UTF8_STRING, STRING as a fallback, INCR transfers included) and put on
//! top of the history. When the clipboard offers no text but an image/png
//! (a screenshot, an image copied from a browser), the picture is stored as
//! a PNG file in $XDG_DATA_HOME/misa-clip/images and its entry is that
//! file's path; selecting it serves image/png again. Entries that a password
//! manager marks with `x-kde-passwordManagerHint` are never stored.
//!
//! When the program that owned the clipboard exits, the clipboard would be
//! empty; with `keep_clipboard` the daemon takes it over and serves the last
//! text itself. Picking an entry from the history works the same way.
//!
//! Clients talk to the daemon over a Unix socket in $XDG_RUNTIME_DIR, one
//! per X display:
//!
//!   misa-clip daemon [--config FILE]  run the daemon (exits if one runs)
//!   misa-clip list                    one preview line per entry, newest first
//!                                     (an image entry: its PNG file's path)
//!   misa-clip get N                   print entry N in full (image: the path)
//!   misa-clip select N                put entry N on the clipboard
//!   misa-clip delete N                remove entry N
//!   misa-clip clear                   remove every entry
//!   misa-clip ping | quit

use std::collections::VecDeque;
use std::env;
use std::ffi::{c_char, c_int, c_long, c_uchar, c_uint, c_ulong, c_void, CString};
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::ptr;
use std::time::{Duration, Instant};

const INCR_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------- Xlib FFI

type Display = c_void;
type Window = c_ulong;
type Atom = c_ulong;
type Time = c_ulong;
type Bool = c_int;

const NONE: c_ulong = 0;
const CURRENT_TIME: Time = 0;
const XA_ATOM: Atom = 4;
const PROP_MODE_REPLACE: c_int = 0;
const PROPERTY_CHANGE_MASK: c_long = 1 << 22;
const PROPERTY_NOTIFY: c_int = 28;
const SELECTION_CLEAR: c_int = 29;
const SELECTION_REQUEST: c_int = 30;
const SELECTION_NOTIFY: c_int = 31;
const PROPERTY_NEW_VALUE: c_int = 0;
const XFIXES_SELECTION_NOTIFY: c_int = 0;
const XFIXES_SET_SELECTION_OWNER_NOTIFY_MASK: c_ulong = 1;

#[repr(C)]
#[derive(Clone, Copy)]
union XEvent {
    type_: c_int,
    pad: [c_long; 24],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XPropertyEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
    atom: Atom,
    time: Time,
    state: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XSelectionRequestEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    owner: Window,
    requestor: Window,
    selection: Atom,
    target: Atom,
    property: Atom,
    time: Time,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XSelectionEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    requestor: Window,
    selection: Atom,
    target: Atom,
    property: Atom,
    time: Time,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XSelectionClearEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
    selection: Atom,
    time: Time,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XFixesSelectionNotifyEvent {
    type_: c_int,
    serial: c_ulong,
    send_event: Bool,
    display: *mut Display,
    window: Window,
    subtype: c_int,
    owner: Window,
    selection: Atom,
    timestamp: Time,
    selection_timestamp: Time,
}

#[link(name = "X11")]
extern "C" {
    fn XOpenDisplay(name: *const c_char) -> *mut Display;
    fn XDefaultRootWindow(d: *mut Display) -> Window;
    fn XConnectionNumber(d: *mut Display) -> c_int;
    fn XCreateSimpleWindow(
        d: *mut Display,
        parent: Window,
        x: c_int,
        y: c_int,
        w: c_uint,
        h: c_uint,
        bw: c_uint,
        border: c_ulong,
        bg: c_ulong,
    ) -> Window;
    fn XSelectInput(d: *mut Display, w: Window, mask: c_long) -> c_int;
    fn XInternAtom(d: *mut Display, name: *const c_char, only_if_exists: Bool) -> Atom;
    fn XPending(d: *mut Display) -> c_int;
    fn XNextEvent(d: *mut Display, ev: *mut XEvent) -> c_int;
    fn XFlush(d: *mut Display) -> c_int;
    fn XConvertSelection(
        d: *mut Display,
        selection: Atom,
        target: Atom,
        property: Atom,
        requestor: Window,
        time: Time,
    ) -> c_int;
    fn XGetWindowProperty(
        d: *mut Display,
        w: Window,
        property: Atom,
        long_offset: c_long,
        long_length: c_long,
        delete: Bool,
        req_type: Atom,
        actual_type: *mut Atom,
        actual_format: *mut c_int,
        nitems: *mut c_ulong,
        bytes_after: *mut c_ulong,
        prop: *mut *mut c_uchar,
    ) -> c_int;
    fn XDeleteProperty(d: *mut Display, w: Window, property: Atom) -> c_int;
    fn XChangeProperty(
        d: *mut Display,
        w: Window,
        property: Atom,
        type_: Atom,
        format: c_int,
        mode: c_int,
        data: *const c_uchar,
        nelements: c_int,
    ) -> c_int;
    fn XSetSelectionOwner(d: *mut Display, selection: Atom, owner: Window, time: Time) -> c_int;
    fn XGetSelectionOwner(d: *mut Display, selection: Atom) -> Window;
    fn XSendEvent(d: *mut Display, w: Window, propagate: Bool, mask: c_long, ev: *mut XEvent)
        -> c_int;
    fn XFree(data: *mut c_void) -> c_int;
    fn XMaxRequestSize(d: *mut Display) -> c_long;
    fn XExtendedMaxRequestSize(d: *mut Display) -> c_long;
}

#[link(name = "Xfixes")]
extern "C" {
    fn XFixesQueryExtension(d: *mut Display, event_base: *mut c_int, error_base: *mut c_int)
        -> Bool;
    fn XFixesSelectSelectionInput(d: *mut Display, w: Window, selection: Atom, mask: c_ulong);
}

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: i16,
    revents: i16,
}
const POLLIN: i16 = 1;

extern "C" {
    fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
}

// ------------------------------------------------------------------ config

struct Config {
    max_items: usize,
    max_bytes: usize,
    max_image_bytes: usize,
    keep_clipboard: bool,
    persist: bool,
    preview_chars: usize,
}

impl Config {
    fn load(path: Option<&Path>) -> Config {
        let mut c = Config {
            max_items: 200,
            max_bytes: 1 << 20,
            max_image_bytes: 32 << 20,
            keep_clipboard: true,
            persist: true,
            preview_chars: 120,
        };
        let Some(text) = path.and_then(|p| fs::read_to_string(p).ok()) else {
            return c;
        };
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let Some((key, value)) = line.split_once('=') else { continue };
            let (key, value) = (key.trim(), value.trim());
            let flag = matches!(value, "true" | "yes" | "on" | "1");
            match key {
                "max_items" => c.max_items = value.parse().unwrap_or(c.max_items).max(1),
                "max_bytes" => c.max_bytes = value.parse().unwrap_or(c.max_bytes).max(1),
                "max_image_bytes" => {
                    c.max_image_bytes = value.parse().unwrap_or(c.max_image_bytes)
                }
                "keep_clipboard" => c.keep_clipboard = flag,
                "persist" => c.persist = flag,
                "preview_chars" => {
                    c.preview_chars = value.parse().unwrap_or(c.preview_chars).max(10)
                }
                _ => eprintln!("misa-clip: unknown config key {key:?}"),
            }
        }
        c
    }
}

// ----------------------------------------------------------------- history

const MAGIC_V1: &[u8] = b"MISACLIP1\n";
const MAGIC: &[u8] = b"MISACLIP2\n";

/// One history entry. An image lives in its own PNG file (named by a hash
/// of its bytes) so the history file stays small and ClipPanel can show it.
#[derive(Clone, PartialEq)]
enum Item {
    Text(Vec<u8>),
    Image(PathBuf),
}

impl Item {
    fn image(&self) -> Option<&Path> {
        match self {
            Item::Image(p) => Some(p),
            Item::Text(_) => None,
        }
    }
}

fn data_dir() -> PathBuf {
    let base = env::var_os("XDG_DATA_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env::var_os("HOME").unwrap_or_default()).join(".local/share"));
    base.join("misa-clip")
}

fn history_path() -> PathBuf {
    data_dir().join("history")
}

fn images_dir() -> PathBuf {
    data_dir().join("images")
}

/// FNV-1a, 64 bit: enough to name and de-duplicate pictures.
fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir)?;
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)?;
        f.write_all(bytes)?;
    }
    fs::rename(&tmp, path)
}

/// `MISACLIP2\n` followed by `<kind><length>\n<bytes>\n` per entry, newest
/// first; kind is `t` (text) or `i` (image: the bytes are its file's path).
/// A `MISACLIP1` file (text only, no kind) is still read.
fn load_history(path: &Path, max_items: usize) -> VecDeque<Item> {
    let mut out = VecDeque::new();
    let Ok(data) = fs::read(path) else { return out };
    let (mut rest, v1) = match (data.strip_prefix(MAGIC), data.strip_prefix(MAGIC_V1)) {
        (Some(r), _) => (r, false),
        (None, Some(r)) => (r, true),
        _ => return out,
    };
    while out.len() < max_items && !rest.is_empty() {
        let mut kind = b't';
        if !v1 {
            kind = rest[0];
            rest = &rest[1..];
        }
        let Some(nl) = rest.iter().position(|&b| b == b'\n') else { break };
        let Some(len) = std::str::from_utf8(&rest[..nl]).ok().and_then(|s| s.parse::<usize>().ok())
        else {
            break;
        };
        rest = &rest[nl + 1..];
        if rest.len() < len + 1 {
            break;
        }
        let bytes = rest[..len].to_vec();
        rest = &rest[len + 1..];
        match kind {
            b'i' => {
                let p = PathBuf::from(String::from_utf8_lossy(&bytes).into_owned());
                if p.is_file() {
                    out.push_back(Item::Image(p));
                }
            }
            _ => out.push_back(Item::Text(bytes)),
        }
    }
    out
}

/// Removes picture files no entry points at (deleted, cut off by
/// max_items, or left by a history that was not persisted).
fn prune_images(items: &VecDeque<Item>) {
    let Ok(dir) = fs::read_dir(images_dir()) else { return };
    for e in dir.flatten() {
        let p = e.path();
        if !items.iter().any(|i| i.image() == Some(p.as_path())) {
            let _ = fs::remove_file(&p);
        }
    }
}

fn save_history(path: &Path, items: &VecDeque<Item>) -> io::Result<()> {
    let mut buf = Vec::new();
    buf.extend_from_slice(MAGIC);
    for item in items {
        let (kind, bytes) = match item {
            Item::Text(t) => (b't', t.as_slice()),
            Item::Image(p) => (b'i', p.as_os_str().as_encoded_bytes()),
        };
        buf.push(kind);
        buf.extend_from_slice(bytes.len().to_string().as_bytes());
        buf.push(b'\n');
        buf.extend_from_slice(bytes);
        buf.push(b'\n');
    }
    write_private(path, &buf)
}

/// One line for the rofi picker: newlines shown as ⏎, whitespace runs
/// collapsed, cut at `max` characters.
fn preview(item: &[u8], max: usize) -> String {
    let text = String::from_utf8_lossy(item);
    let mut out = String::new();
    let mut count = 0;
    let mut space = false;
    for ch in text.trim().chars() {
        let piece = match ch {
            '\n' => {
                space = false;
                Some('⏎')
            }
            c if c.is_whitespace() || c.is_control() => {
                space = true;
                None
            }
            c => Some(c),
        };
        let Some(c) = piece else { continue };
        if space && count > 0 {
            out.push(' ');
            count += 1;
            space = false;
        }
        if count >= max {
            out.push('…');
            break;
        }
        out.push(c);
        count += 1;
    }
    if out.is_empty() {
        out.push_str("(blank)");
    }
    out
}

// ------------------------------------------------------------------ socket

fn socket_path() -> PathBuf {
    let display = env::var("DISPLAY").unwrap_or_default();
    let tag: String = display
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    match env::var_os("XDG_RUNTIME_DIR").filter(|v| !v.is_empty()) {
        Some(dir) => PathBuf::from(dir).join(format!("misa-clip{tag}.sock")),
        None => {
            let uid = fs::metadata("/proc/self").map(|m| m.uid()).unwrap_or(0);
            PathBuf::from(format!("/tmp/misa-clip-{uid}{tag}.sock"))
        }
    }
}

fn client(cmd: &str) -> ExitCode {
    let path = socket_path();
    let mut stream = match UnixStream::connect(&path) {
        Ok(s) => s,
        Err(_) => {
            if cmd != "ping" {
                eprintln!("misa-clip: no daemon on {}", path.display());
            }
            return ExitCode::from(2);
        }
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    if stream.write_all(format!("{cmd}\n").as_bytes()).is_err() {
        return ExitCode::from(2);
    }
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let mut reply = Vec::new();
    if stream.read_to_end(&mut reply).is_err() {
        return ExitCode::from(2);
    }
    // The first byte is the status: '+' ok, '-' error (message follows).
    match reply.split_first() {
        Some((b'+', body)) => {
            let _ = io::stdout().write_all(body);
            ExitCode::SUCCESS
        }
        Some((_, body)) => {
            let _ = io::stderr().write_all(body);
            ExitCode::from(1)
        }
        None => ExitCode::from(1),
    }
}

// ------------------------------------------------------------------ daemon

struct Atoms {
    clipboard: Atom,
    targets: Atom,
    utf8: Atom,
    string: Atom,
    text: Atom,
    plain_utf8: Atom,
    plain: Atom,
    incr: Atom,
    png: Atom,
    password_hint: Atom,
    prop_targets: Atom,
    prop_data: Atom,
}

enum Fetch {
    Idle,
    Targets,
    Data { target: Atom },
    Incr { target: Atom, buf: Vec<u8>, overflow: bool, started: Instant },
}

struct Daemon {
    dpy: *mut Display,
    win: Window,
    a: Atoms,
    cfg: Config,
    items: VecDeque<Item>,
    path: PathBuf,
    fetch: Fetch,
    /// What we serve while we own CLIPBOARD.
    owned: Option<Item>,
    /// Largest property we can write in one request, in bytes.
    max_request: usize,
    /// The text the clipboard holds now is history[0] (so taking the
    /// clipboard over when its owner exits restores the right thing).
    head_is_current: bool,
    running: bool,
    xfixes_event: c_int,
}

fn intern(dpy: *mut Display, name: &str) -> Atom {
    let c = CString::new(name).unwrap();
    unsafe { XInternAtom(dpy, c.as_ptr(), 0) }
}

/// Reads and deletes a property. None if it is missing or longer than `max`.
fn take_property(
    dpy: *mut Display,
    win: Window,
    prop: Atom,
    max: usize,
) -> Option<(Atom, c_int, Vec<u8>)> {
    let mut actual_type: Atom = 0;
    let mut format: c_int = 0;
    let mut nitems: c_ulong = 0;
    let mut after: c_ulong = 0;
    let mut data: *mut c_uchar = ptr::null_mut();
    let long_len = (max / 4 + 1) as c_long;
    let rc = unsafe {
        XGetWindowProperty(
            dpy, win, prop, 0, long_len, 1, 0, &mut actual_type, &mut format, &mut nitems,
            &mut after, &mut data,
        )
    };
    if rc != 0 {
        return None;
    }
    let bytes = if data.is_null() {
        Vec::new()
    } else {
        // Format 32 items are C longs in client memory.
        let unit = match format {
            8 => 1,
            16 => 2,
            32 => std::mem::size_of::<c_long>(),
            _ => 0,
        };
        let v = unsafe { std::slice::from_raw_parts(data, nitems as usize * unit) }.to_vec();
        unsafe { XFree(data.cast()) };
        v
    };
    if after > 0 {
        unsafe { XDeleteProperty(dpy, win, prop) };
        return None;
    }
    if actual_type == NONE {
        return None;
    }
    Some((actual_type, format, bytes))
}

fn longs(bytes: &[u8]) -> Vec<c_ulong> {
    let n = std::mem::size_of::<c_ulong>();
    bytes
        .chunks_exact(n)
        .map(|c| c_ulong::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn latin1_to_utf8(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().map(|&b| b as char).collect::<String>().into_bytes()
}

fn utf8_to_latin1(bytes: &[u8]) -> Vec<u8> {
    String::from_utf8_lossy(bytes)
        .chars()
        .map(|c| if (c as u32) < 256 { c as u8 } else { b'?' })
        .collect()
}

impl Daemon {
    fn new(cfg: Config) -> Result<Daemon, String> {
        let dpy = unsafe { XOpenDisplay(ptr::null()) };
        if dpy.is_null() {
            return Err("cannot open the X display".into());
        }
        let (mut ev_base, mut err_base) = (0, 0);
        if unsafe { XFixesQueryExtension(dpy, &mut ev_base, &mut err_base) } == 0 {
            return Err("the X server has no XFixes".into());
        }
        let root = unsafe { XDefaultRootWindow(dpy) };
        let win = unsafe { XCreateSimpleWindow(dpy, root, -10, -10, 1, 1, 0, 0, 0) };
        unsafe { XSelectInput(dpy, win, PROPERTY_CHANGE_MASK) };
        let a = Atoms {
            clipboard: intern(dpy, "CLIPBOARD"),
            targets: intern(dpy, "TARGETS"),
            utf8: intern(dpy, "UTF8_STRING"),
            string: 31, // XA_STRING
            text: intern(dpy, "TEXT"),
            plain_utf8: intern(dpy, "text/plain;charset=utf-8"),
            plain: intern(dpy, "text/plain"),
            incr: intern(dpy, "INCR"),
            png: intern(dpy, "image/png"),
            password_hint: intern(dpy, "x-kde-passwordManagerHint"),
            prop_targets: intern(dpy, "MISA_CLIP_TARGETS"),
            prop_data: intern(dpy, "MISA_CLIP_DATA"),
        };
        unsafe {
            XFixesSelectSelectionInput(dpy, win, a.clipboard, XFIXES_SET_SELECTION_OWNER_NOTIFY_MASK)
        };
        let path = history_path();
        let items = if cfg.persist { load_history(&path, cfg.max_items) } else { VecDeque::new() };
        prune_images(&items);
        let words = match unsafe { XExtendedMaxRequestSize(dpy) } {
            0 => unsafe { XMaxRequestSize(dpy) },
            n => n,
        };
        let max_request = (words.max(0) as usize * 4).saturating_sub(64);
        Ok(Daemon {
            dpy,
            win,
            a,
            cfg,
            items,
            path,
            fetch: Fetch::Idle,
            owned: None,
            max_request,
            head_is_current: false,
            running: true,
            xfixes_event: ev_base + XFIXES_SELECTION_NOTIFY,
        })
    }

    fn save(&self) {
        if self.cfg.persist {
            if let Err(e) = save_history(&self.path, &self.items) {
                eprintln!("misa-clip: saving {}: {e}", self.path.display());
            }
        }
    }

    fn limit(&self, target: Atom) -> usize {
        if target == self.a.png { self.cfg.max_image_bytes } else { self.cfg.max_bytes }
    }

    fn convert(&mut self, target: Atom, prop: Atom, time: Time) {
        unsafe { XConvertSelection(self.dpy, self.a.clipboard, target, prop, self.win, time) };
    }

    // A new owner took CLIPBOARD (or the old one went away).
    fn on_owner_change(&mut self, ev: &XFixesSelectionNotifyEvent) {
        if ev.selection != self.a.clipboard || ev.owner == self.win {
            return;
        }
        if ev.owner == NONE {
            self.fetch = Fetch::Idle;
            if self.cfg.keep_clipboard && self.head_is_current {
                if let Some(head) = self.items.front().cloned() {
                    self.own(head);
                }
            }
            return;
        }
        self.head_is_current = false;
        self.fetch = Fetch::Targets;
        self.convert(self.a.targets, self.a.prop_targets, ev.selection_timestamp);
    }

    fn on_selection_notify(&mut self, ev: &XSelectionEvent) {
        if ev.selection != self.a.clipboard || ev.requestor != self.win {
            return;
        }
        let time = ev.time;
        match std::mem::replace(&mut self.fetch, Fetch::Idle) {
            Fetch::Targets => {
                let target = if ev.property == NONE {
                    // No TARGETS support: just try text.
                    Some(self.a.utf8)
                } else {
                    let Some((_, _, raw)) =
                        take_property(self.dpy, self.win, self.a.prop_targets, 64 * 1024)
                    else {
                        return;
                    };
                    let offered = longs(&raw);
                    if offered.contains(&self.a.password_hint) {
                        return;
                    }
                    // Text first; a picture only when there is no text
                    // (an office app offers both for a cell range).
                    [self.a.utf8, self.a.plain_utf8, self.a.string]
                        .into_iter()
                        .find(|t| offered.contains(t))
                        .or_else(|| {
                            let png = self.a.png;
                            (self.cfg.max_image_bytes > 0 && offered.contains(&png)).then_some(png)
                        })
                };
                if let Some(target) = target {
                    self.fetch = Fetch::Data { target };
                    self.convert(target, self.a.prop_data, time);
                }
            }
            Fetch::Data { target } => {
                if ev.property == NONE {
                    if target != self.a.string && target != self.a.png {
                        self.fetch = Fetch::Data { target: self.a.string };
                        self.convert(self.a.string, self.a.prop_data, time);
                    }
                    return;
                }
                let Some((kind, _, bytes)) =
                    take_property(self.dpy, self.win, self.a.prop_data, self.limit(target))
                else {
                    return;
                };
                if kind == self.a.incr {
                    // Deleting the property (done above) starts the transfer.
                    self.fetch =
                        Fetch::Incr { target, buf: Vec::new(), overflow: false, started: Instant::now() };
                } else {
                    self.store(target, bytes);
                }
            }
            other => self.fetch = other,
        }
    }

    fn on_property(&mut self, ev: &XPropertyEvent) {
        if ev.window != self.win || ev.atom != self.a.prop_data || ev.state != PROPERTY_NEW_VALUE {
            return;
        }
        let Fetch::Incr { target, buf, overflow, .. } = &mut self.fetch else { return };
        let target = *target;
        let max = if target == self.a.png { self.cfg.max_image_bytes } else { self.cfg.max_bytes };
        match take_property(self.dpy, self.win, self.a.prop_data, max) {
            Some((_, _, chunk)) if chunk.is_empty() => {
                let done = std::mem::take(buf);
                let too_big = *overflow;
                self.fetch = Fetch::Idle;
                if !too_big {
                    self.store(target, done);
                }
            }
            Some((_, _, chunk)) => {
                if buf.len() + chunk.len() > max {
                    *overflow = true;
                    buf.clear();
                } else if !*overflow {
                    buf.extend_from_slice(&chunk);
                }
            }
            None => *overflow = true,
        }
    }

    fn store(&mut self, target: Atom, bytes: Vec<u8>) {
        let item = if target == self.a.png {
            if bytes.len() > self.cfg.max_image_bytes || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
                return;
            }
            let path = images_dir().join(format!("{:016x}.png", fnv64(&bytes)));
            if !path.is_file() {
                if let Err(e) = write_private(&path, &bytes) {
                    eprintln!("misa-clip: saving {}: {e}", path.display());
                    return;
                }
            }
            Item::Image(path)
        } else {
            let bytes = if target == self.a.string { latin1_to_utf8(&bytes) } else { bytes };
            if bytes.len() > self.cfg.max_bytes || bytes.iter().all(|b| b.is_ascii_whitespace()) {
                return;
            }
            Item::Text(bytes)
        };
        if let Some(i) = self.items.iter().position(|x| *x == item) {
            self.items.remove(i);
        }
        self.items.push_front(item);
        self.trim();
        self.head_is_current = true;
        self.save();
    }

    /// Cuts the history to max_items and deletes the pictures that fell off.
    fn trim(&mut self) {
        while self.items.len() > self.cfg.max_items {
            if let Some(Item::Image(p)) = self.items.pop_back() {
                let _ = fs::remove_file(p);
            }
        }
    }

    fn forget(&mut self, item: Item) {
        if let Item::Image(p) = item {
            if self.owned.as_ref().and_then(Item::image) != Some(p.as_path()) {
                let _ = fs::remove_file(p);
            }
        }
    }

    fn own(&mut self, data: Item) {
        self.owned = Some(data);
        unsafe { XSetSelectionOwner(self.dpy, self.a.clipboard, self.win, CURRENT_TIME) };
        if unsafe { XGetSelectionOwner(self.dpy, self.a.clipboard) } == self.win {
            self.head_is_current = true;
        } else {
            self.owned = None;
        }
    }

    fn on_selection_request(&mut self, req: &XSelectionRequestEvent) {
        let a = &self.a;
        let mut prop = if req.property == NONE { req.target } else { req.property };
        let served = match (&self.owned, req.selection == a.clipboard) {
            (Some(Item::Image(path)), true) => {
                let t = req.target;
                if t == a.targets {
                    let list: [c_ulong; 2] = [a.targets, a.png];
                    unsafe {
                        XChangeProperty(
                            self.dpy, req.requestor, prop, XA_ATOM, 32, PROP_MODE_REPLACE,
                            list.as_ptr().cast(), list.len() as c_int,
                        )
                    };
                    true
                } else if t == a.png {
                    // Served in one piece: a screenshot PNG is far below
                    // the request limit (16 MiB on a usual server).
                    match fs::read(path) {
                        Ok(png) if png.len() <= self.max_request => {
                            unsafe {
                                XChangeProperty(
                                    self.dpy, req.requestor, prop, a.png, 8, PROP_MODE_REPLACE,
                                    png.as_ptr(), png.len() as c_int,
                                )
                            };
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            }
            (Some(Item::Text(data)), true) => {
                let t = req.target;
                if t == a.targets {
                    let list: [c_ulong; 6] =
                        [a.targets, a.utf8, a.plain_utf8, a.text, a.string, a.plain];
                    unsafe {
                        XChangeProperty(
                            self.dpy, req.requestor, prop, XA_ATOM, 32, PROP_MODE_REPLACE,
                            list.as_ptr().cast(), list.len() as c_int,
                        )
                    };
                    true
                } else if t == a.utf8 || t == a.text || t == a.plain_utf8 {
                    let kind = if t == a.plain_utf8 { a.plain_utf8 } else { a.utf8 };
                    unsafe {
                        XChangeProperty(
                            self.dpy, req.requestor, prop, kind, 8, PROP_MODE_REPLACE,
                            data.as_ptr(), data.len() as c_int,
                        )
                    };
                    true
                } else if t == a.string || t == a.plain {
                    let latin = utf8_to_latin1(data);
                    unsafe {
                        XChangeProperty(
                            self.dpy, req.requestor, prop, a.string, 8, PROP_MODE_REPLACE,
                            latin.as_ptr(), latin.len() as c_int,
                        )
                    };
                    true
                } else {
                    false
                }
            }
            _ => false,
        };
        if !served {
            prop = NONE;
        }
        let reply = XSelectionEvent {
            type_: SELECTION_NOTIFY,
            serial: 0,
            send_event: 1,
            display: self.dpy,
            requestor: req.requestor,
            selection: req.selection,
            target: req.target,
            property: prop,
            time: req.time,
        };
        let mut ev = XEvent { pad: [0; 24] };
        unsafe {
            ptr::copy_nonoverlapping(
                (&reply as *const XSelectionEvent).cast::<u8>(),
                (&mut ev as *mut XEvent).cast::<u8>(),
                std::mem::size_of::<XSelectionEvent>(),
            );
            XSendEvent(self.dpy, req.requestor, 0, 0, &mut ev);
        }
    }

    fn dispatch(&mut self, ev: &XEvent) {
        let kind = unsafe { ev.type_ };
        let p = (ev as *const XEvent).cast::<u8>();
        unsafe {
            match kind {
                SELECTION_NOTIFY => self.on_selection_notify(&*p.cast()),
                SELECTION_REQUEST => self.on_selection_request(&*p.cast()),
                PROPERTY_NOTIFY => self.on_property(&*p.cast()),
                SELECTION_CLEAR => {
                    let ev: &XSelectionClearEvent = &*p.cast();
                    if ev.selection == self.a.clipboard {
                        self.owned = None;
                    }
                }
                k if k == self.xfixes_event => self.on_owner_change(&*p.cast()),
                _ => {}
            }
        }
    }

    fn index(&self, arg: Option<&str>) -> Result<usize, String> {
        let n: usize = arg.and_then(|s| s.trim().parse().ok()).ok_or("expected an index")?;
        if n < self.items.len() {
            Ok(n)
        } else {
            Err(format!("no entry {n}"))
        }
    }

    fn command(&mut self, line: &str) -> Result<Vec<u8>, String> {
        let mut parts = line.trim().splitn(2, ' ');
        let verb = parts.next().unwrap_or("");
        let arg = parts.next();
        match verb {
            "ping" => Ok(Vec::new()),
            "list" => {
                let mut out = String::new();
                for item in &self.items {
                    match item {
                        Item::Text(t) => out.push_str(&preview(t, self.cfg.preview_chars)),
                        Item::Image(p) => out.push_str(&p.to_string_lossy()),
                    }
                    out.push('\n');
                }
                Ok(out.into_bytes())
            }
            "get" => Ok(match &self.items[self.index(arg)?] {
                Item::Text(t) => t.clone(),
                Item::Image(p) => p.as_os_str().as_encoded_bytes().to_vec(),
            }),
            "select" => {
                let n = self.index(arg)?;
                let item = self.items.remove(n).unwrap();
                self.items.push_front(item.clone());
                self.own(item);
                self.save();
                if self.owned.is_some() {
                    Ok(Vec::new())
                } else {
                    Err("could not take the clipboard".into())
                }
            }
            "delete" => {
                let n = self.index(arg)?;
                let item = self.items.remove(n).unwrap();
                self.forget(item);
                if n == 0 {
                    self.head_is_current = false;
                }
                self.save();
                Ok(Vec::new())
            }
            "clear" => {
                for item in std::mem::take(&mut self.items) {
                    self.forget(item);
                }
                self.head_is_current = false;
                self.save();
                Ok(Vec::new())
            }
            "quit" => {
                self.running = false;
                Ok(Vec::new())
            }
            _ => Err(format!("unknown command {verb:?}")),
        }
    }

    fn serve(&mut self, stream: UnixStream) {
        let _ = stream.set_nonblocking(false);
        let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
        let mut line = String::new();
        let mut reader = BufReader::new(&stream);
        if reader.read_line(&mut line).is_err() {
            return;
        }
        let reply = match self.command(&line) {
            Ok(mut body) => {
                body.insert(0, b'+');
                body
            }
            Err(msg) => format!("-misa-clip: {msg}\n").into_bytes(),
        };
        let _ = (&stream).write_all(&reply);
        unsafe { XFlush(self.dpy) };
    }

    fn run(&mut self, listener: UnixListener) {
        let xfd = unsafe { XConnectionNumber(self.dpy) };
        let mut ev = XEvent { pad: [0; 24] };
        while self.running {
            while unsafe { XPending(self.dpy) } > 0 {
                unsafe { XNextEvent(self.dpy, &mut ev) };
                self.dispatch(&ev);
            }
            unsafe { XFlush(self.dpy) };
            if let Fetch::Incr { started, .. } = &self.fetch {
                if started.elapsed() > INCR_TIMEOUT {
                    self.fetch = Fetch::Idle;
                }
            }
            let timeout = match self.fetch {
                Fetch::Incr { .. } => 1000,
                _ => -1,
            };
            let mut fds = [
                PollFd { fd: xfd, events: POLLIN, revents: 0 },
                PollFd { fd: listener.as_raw_fd(), events: POLLIN, revents: 0 },
            ];
            if unsafe { poll(fds.as_mut_ptr(), 2, timeout) } < 0 {
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                break;
            }
            if fds[1].revents != 0 {
                while let Ok((stream, _)) = listener.accept() {
                    self.serve(stream);
                }
            }
        }
    }
}

fn daemon(config: Option<PathBuf>) -> ExitCode {
    let sock = socket_path();
    if UnixStream::connect(&sock).is_ok() {
        eprintln!("misa-clip: already running ({})", sock.display());
        return ExitCode::SUCCESS;
    }
    let _ = fs::remove_file(&sock);
    let cfg = Config::load(config.as_deref());
    let mut d = match Daemon::new(cfg) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("misa-clip: {e}");
            return ExitCode::FAILURE;
        }
    };
    let listener = match UnixListener::bind(&sock) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("misa-clip: {}: {e}", sock.display());
            return ExitCode::FAILURE;
        }
    };
    let _ = fs::set_permissions(&sock, fs::Permissions::from_mode(0o600));
    let _ = listener.set_nonblocking(true);
    d.run(listener);
    let _ = fs::remove_file(&sock);
    ExitCode::SUCCESS
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: misa-clip daemon [--config FILE]\n       \
         misa-clip list | get N | select N | delete N | clear | ping | quit"
    );
    ExitCode::from(64)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("daemon") => {
            let config = match args.get(1).map(String::as_str) {
                Some("--config") => match args.get(2) {
                    Some(p) => Some(PathBuf::from(p)),
                    None => return usage(),
                },
                Some(_) => return usage(),
                None => None,
            };
            daemon(config)
        }
        Some(cmd @ ("list" | "clear" | "ping" | "quit")) if args.len() == 1 => client(cmd),
        Some(cmd @ ("get" | "select" | "delete")) if args.len() == 2 => {
            client(&format!("{cmd} {}", args[1]))
        }
        _ => usage(),
    }
}
