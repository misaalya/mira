//! misa-backdrop: the dark, blurred backdrop behind the rice's popups.
//!
//!   misa-backdrop X Y W H [--pid PID] [--click CMD]
//!
//! Maps a plain black override-redirect window (class "Backdrop") over the
//! rectangle, waits until the X server has it, then returns: the caller
//! opens its popup right after, so the popup lands above it. picom turns
//! the window into a translucent dim with the desktop blurred behind it
//! (rule "Backdrop" in picom-rules.conf), with a fade.
//!
//! The backdrop itself lives on in the background (its PID is printed) and
//! goes away when:
//!   - PID (the popup's process, or a script) is gone,  checked every 100 ms;
//!   - it is clicked, or Escape is pressed (when `--click CMD` is set): CMD
//!     runs first (`sh -c`), e.g. to close an eww panel that has no pointer or
//!     keyboard grab of its own;
//!   - it gets SIGTERM.
//! No ARGB visual and no drawing: one black window, so it costs nothing.

use std::ffi::{c_char, c_int, c_long, c_uint, c_ulong, c_void, CString};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::sleep;
use std::time::Duration;

type Display = c_void;
type Window = c_ulong;

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

#[repr(C)]
struct XClassHint {
    res_name: *const c_char,
    res_class: *const c_char,
}

#[repr(C)]
struct XEvent {
    kind: c_int,
    pad: [c_long; 24],
}

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: i16,
    revents: i16,
}

const CW_BACK_PIXEL: c_ulong = 1 << 1;
const CW_OVERRIDE_REDIRECT: c_ulong = 1 << 9;
const CW_EVENT_MASK: c_ulong = 1 << 11;
const BUTTON_PRESS_MASK: c_long = 1 << 2;
const KEY_PRESS: c_int = 2;
const BUTTON_PRESS: c_int = 4;
const COPY_FROM_PARENT: c_int = 0;
const INPUT_OUTPUT: c_uint = 1;
const GRAB_MODE_ASYNC: c_int = 1;
const XK_ESCAPE: c_ulong = 0xFF1B;
const LOCK_MASK: c_uint = 1 << 1;
const MOD2_MASK: c_uint = 1 << 4;

#[link(name = "X11")]
extern "C" {
    fn XOpenDisplay(name: *const c_char) -> *mut Display;
    fn XDefaultRootWindow(d: *mut Display) -> Window;
    fn XDefaultScreen(d: *mut Display) -> c_int;
    fn XBlackPixel(d: *mut Display, screen: c_int) -> c_ulong;
    fn XCreateWindow(
        d: *mut Display, parent: Window, x: c_int, y: c_int, w: c_uint, h: c_uint,
        border: c_uint, depth: c_int, class: c_uint, visual: *mut c_void,
        mask: c_ulong, attrs: *mut XSetWindowAttributes,
    ) -> Window;
    fn XSetClassHint(d: *mut Display, w: Window, hint: *mut XClassHint) -> c_int;
    fn XStoreName(d: *mut Display, w: Window, name: *const c_char) -> c_int;
    fn XMapRaised(d: *mut Display, w: Window) -> c_int;
    fn XKeysymToKeycode(d: *mut Display, keysym: c_ulong) -> u8;
    fn XGrabKey(
        d: *mut Display, keycode: c_int, modifiers: c_uint, grab_window: Window,
        owner_events: c_int, pointer_mode: c_int, keyboard_mode: c_int,
    ) -> c_int;
    fn XSetErrorHandler(
        handler: Option<extern "C" fn(*mut Display, *mut c_void) -> c_int>,
    ) -> *mut c_void;
    fn XSync(d: *mut Display, discard: c_int) -> c_int;
    fn XConnectionNumber(d: *mut Display) -> c_int;
    fn XPending(d: *mut Display) -> c_int;
    fn XNextEvent(d: *mut Display, ev: *mut XEvent) -> c_int;
}

extern "C" {
    fn fork() -> c_int;
    fn setsid() -> c_int;
    fn kill(pid: c_int, sig: c_int) -> c_int;
    fn poll(fds: *mut PollFd, n: c_ulong, timeout: c_int) -> c_int;
    fn _exit(code: c_int) -> !;
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn dup2(old: c_int, new: c_int) -> c_int;
}

static X_ERR: AtomicBool = AtomicBool::new(false);

extern "C" fn ignore_x_err(_d: *mut Display, _e: *mut c_void) -> c_int {
    X_ERR.store(true, Ordering::Relaxed);
    0
}

fn usage() -> ! {
    eprintln!("usage: misa-backdrop X Y W H [--pid PID] [--click CMD]");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 4 {
        usage();
    }
    let num = |s: &String| s.parse::<i64>().unwrap_or_else(|_| usage());
    let (x, y, w, h) = (num(&args[0]), num(&args[1]), num(&args[2]), num(&args[3]));
    if w <= 0 || h <= 0 {
        usage();
    }
    let mut pid: c_int = 0;
    let mut click: Option<String> = None;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--pid" if i + 1 < args.len() => {
                pid = args[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            "--click" if i + 1 < args.len() => {
                click = Some(args[i + 1].clone());
                i += 1;
            }
            _ => usage(),
        }
        i += 1;
    }

    unsafe {
        let dpy = XOpenDisplay(std::ptr::null());
        if dpy.is_null() {
            eprintln!("misa-backdrop: cannot open display");
            std::process::exit(1);
        }
        let screen = XDefaultScreen(dpy);
        let mut attrs: XSetWindowAttributes = std::mem::zeroed();
        attrs.background_pixel = XBlackPixel(dpy, screen);
        attrs.override_redirect = 1;
        attrs.event_mask = BUTTON_PRESS_MASK;
        let win = XCreateWindow(
            dpy, XDefaultRootWindow(dpy), x as c_int, y as c_int, w as c_uint, h as c_uint,
            0, COPY_FROM_PARENT, INPUT_OUTPUT, std::ptr::null_mut(),
            CW_BACK_PIXEL | CW_OVERRIDE_REDIRECT | CW_EVENT_MASK, &mut attrs,
        );
        let name = CString::new("misa-backdrop").unwrap();
        let class = CString::new("Backdrop").unwrap();
        let mut hint = XClassHint { res_name: name.as_ptr(), res_class: class.as_ptr() };
        XSetClassHint(dpy, win, &mut hint);
        XStoreName(dpy, win, name.as_ptr());
        XMapRaised(dpy, win);
        // When --click is set (eww panels such as volume/music that have no
        // keyboard grab of their own), also catch Escape (ignoring CapsLock /
        // NumLock, without interfering with Super+Escape) to close the panel.
        // Non-fatal X error handler + brief retry avoids BadAccess if an
        // outgoing backdrop is still closing its X connection.
        if click.is_some() {
            XSetErrorHandler(Some(ignore_x_err));
            let root = XDefaultRootWindow(dpy);
            let esc = XKeysymToKeycode(dpy, XK_ESCAPE) as c_int;
            if esc != 0 {
                for attempt in 0..10 {
                    X_ERR.store(false, Ordering::Relaxed);
                    for mods in [0, LOCK_MASK, MOD2_MASK, LOCK_MASK | MOD2_MASK] {
                        XGrabKey(dpy, esc, mods, root, 0, GRAB_MODE_ASYNC, GRAB_MODE_ASYNC);
                    }
                    XSync(dpy, 0);
                    if !X_ERR.load(Ordering::Relaxed) || attempt == 9 {
                        break;
                    }
                    sleep(Duration::from_millis(10));
                }
            }
        } else {
            XSync(dpy, 0);
        }

        // Mapped: hand control back to the caller, keep the window alive in
        // a child (it owns the X connection from here on).
        match fork() {
            -1 => {}
            0 => {
                setsid();
                let null = CString::new("/dev/null").unwrap();
                let fd = open(null.as_ptr(), 2);
                if fd >= 0 {
                    dup2(fd, 0);
                    // stdout and stderr: a caller reading them (`$(...)`,
                    // subprocess) sees EOF now, not when the backdrop ends
                    dup2(fd, 1);
                    dup2(fd, 2);
                }
            }
            child => {
                println!("{child}");
                _exit(0);
            }
        }

        let mut fds = PollFd { fd: XConnectionNumber(dpy), events: 1, revents: 0 };
        let mut ev: XEvent = std::mem::zeroed();
        loop {
            poll(&mut fds, 1, 100);
            while XPending(dpy) > 0 {
                XNextEvent(dpy, &mut ev);
                if ev.kind == BUTTON_PRESS || ev.kind == KEY_PRESS {
                    if let Some(cmd) = &click {
                        let _ = Command::new("sh").arg("-c").arg(cmd).spawn();
                    }
                    _exit(0);
                }
            }
            if fds.revents & (8 | 16) != 0 {
                _exit(0);                    // X connection gone
            }
            if pid > 0 && kill(pid, 0) != 0 {
                _exit(0);
            }
        }
    }
}
