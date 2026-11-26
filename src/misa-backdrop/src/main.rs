//! misa-backdrop: the dark, blurred backdrop behind the rice's popups.
//!
//!   misa-backdrop X Y W H [--pid PID] [--click CMD]
//!
//! Maps a plain black override-redirect window (class "Backdrop") over the
//! full monitor rectangle, raises any viewable Polybar / dock windows on that
//! monitor directly above it so floating/island/minimal topbars stay sharp
//! while their transparent gaps get full blur & dim seamlessly, then returns:
//! the caller opens its popup right after, so the popup lands above both.
//! picom turns the window into a translucent dim with the desktop blurred
//! behind it (rule "Backdrop" in picom-rules.conf), with a fade.
//!
//! The backdrop itself lives on in the background (its PID is printed) and
//! goes away when:
//!   - PID (the popup's process, or a script) is gone, checked every 100 ms;
//!   - it is clicked, or Escape is pressed (when `--click CMD` is set): CMD
//!     runs first (`sh -c`);
//!   - it gets SIGTERM / SIGINT (gracefully unmaps, waits for picom's fade-out,
//!     and restores raised bar stacking if no newer overlay is active).
//! No ARGB visual and no drawing: one black window, so it costs nothing.

use std::ffi::{c_char, c_int, c_long, c_uchar, c_uint, c_ulong, c_void, CStr, CString};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::sleep;
use std::time::Duration;

type Display = c_void;
type Window = c_ulong;
type Atom = c_ulong;

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
    visual: *mut c_void,
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
struct XWindowChanges {
    x: c_int,
    y: c_int,
    width: c_int,
    height: c_int,
    border_width: c_int,
    sibling: Window,
    stack_mode: c_int,
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

#[repr(C)]
struct SigAction {
    sa_handler: extern "C" fn(c_int),
    sa_mask: [u64; 16],
    sa_flags: c_int,
    _pad: c_int,
    sa_restorer: *const c_void,
}

const CW_BACK_PIXEL: c_ulong = 1 << 1;
const CW_OVERRIDE_REDIRECT: c_ulong = 1 << 9;
const CW_EVENT_MASK: c_ulong = 1 << 11;
const CW_SIBLING: c_uint = 0x20;
const CW_STACK_MODE: c_uint = 0x40;
const STACK_ABOVE: c_int = 0;
const STACK_BELOW: c_int = 1;
const IS_VIEWABLE: c_int = 2;
const INPUT_OUTPUT: c_uint = 1;
const COPY_FROM_PARENT: c_int = 0;
const XA_ATOM: Atom = 4;
const BUTTON_PRESS_MASK: c_long = 1 << 2;
const KEY_PRESS: c_int = 2;
const BUTTON_PRESS: c_int = 4;
const GRAB_MODE_ASYNC: c_int = 1;
const XK_ESCAPE: c_ulong = 0xFF1B;
const LOCK_MASK: c_uint = 1 << 1;
const MOD2_MASK: c_uint = 1 << 4;
const ANY_MODIFIER: c_uint = 0x8000;
const SIGINT: c_int = 2;
const SIGTERM: c_int = 15;

#[link(name = "X11")]
extern "C" {
    fn XOpenDisplay(name: *const c_char) -> *mut Display;
    fn XCloseDisplay(d: *mut Display) -> c_int;
    fn XDefaultRootWindow(d: *mut Display) -> Window;
    fn XDefaultScreen(d: *mut Display) -> c_int;
    fn XBlackPixel(d: *mut Display, screen: c_int) -> c_ulong;
    fn XCreateWindow(
        d: *mut Display, parent: Window, x: c_int, y: c_int, w: c_uint, h: c_uint,
        border: c_uint, depth: c_int, class: c_uint, visual: *mut c_void,
        mask: c_ulong, attrs: *mut XSetWindowAttributes,
    ) -> Window;
    fn XSetClassHint(d: *mut Display, w: Window, hint: *mut XClassHint) -> c_int;
    fn XGetClassHint(d: *mut Display, w: Window, hint: *mut XClassHint) -> c_int;
    fn XStoreName(d: *mut Display, w: Window, name: *const c_char) -> c_int;
    fn XMapRaised(d: *mut Display, w: Window) -> c_int;
    fn XUnmapWindow(d: *mut Display, w: Window) -> c_int;
    fn XDestroyWindow(d: *mut Display, w: Window) -> c_int;
    fn XConfigureWindow(d: *mut Display, w: Window, mask: c_uint, changes: *mut XWindowChanges) -> c_int;
    fn XQueryTree(
        d: *mut Display, w: Window, root_ret: *mut Window, parent_ret: *mut Window,
        children_ret: *mut *mut Window, nchildren_ret: *mut c_uint,
    ) -> c_int;
    fn XGetWindowAttributes(d: *mut Display, w: Window, attr: *mut XWindowAttributes) -> c_int;
    fn XInternAtom(d: *mut Display, name: *const c_char, only_if_exists: c_int) -> Atom;
    fn XGetWindowProperty(
        d: *mut Display, w: Window, prop: Atom, offset: c_long, len: c_long,
        delete: c_int, req_type: Atom, actual_type: *mut Atom, actual_fmt: *mut c_int,
        nitems: *mut c_ulong, bytes_after: *mut c_ulong, prop_ret: *mut *mut c_uchar,
    ) -> c_int;
    fn XFree(data: *mut c_void) -> c_int;
    fn XKeysymToKeycode(d: *mut Display, keysym: c_ulong) -> u8;
    fn XGrabKey(
        d: *mut Display, keycode: c_int, modifiers: c_uint, grab_window: Window,
        owner_events: c_int, pointer_mode: c_int, keyboard_mode: c_int,
    ) -> c_int;
    fn XUngrabKey(d: *mut Display, keycode: c_int, modifiers: c_uint, grab_window: Window) -> c_int;
    fn XSetErrorHandler(
        handler: Option<extern "C" fn(*mut Display, *mut c_void) -> c_int>,
    ) -> *mut c_void;
    fn XSync(d: *mut Display, discard: c_int) -> c_int;
    fn XFlush(d: *mut Display) -> c_int;
    fn XConnectionNumber(d: *mut Display) -> c_int;
    fn XPending(d: *mut Display) -> c_int;
    fn XNextEvent(d: *mut Display, ev: *mut XEvent) -> c_int;
}

extern "C" {
    fn fork() -> c_int;
    fn setsid() -> c_int;
    fn kill(pid: c_int, sig: c_int) -> c_int;
    fn poll(fds: *mut PollFd, n: c_ulong, timeout: c_int) -> c_int;
    fn sigaction(sig: c_int, act: *const SigAction, old: *mut SigAction) -> c_int;
    fn _exit(code: c_int) -> !;
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn dup2(old: c_int, new: c_int) -> c_int;
}

static X_ERR: AtomicBool = AtomicBool::new(false);
static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn ignore_x_err(_d: *mut Display, _e: *mut c_void) -> c_int {
    X_ERR.store(true, Ordering::Relaxed);
    0
}

extern "C" fn handle_stop(_sig: c_int) {
    STOP.store(true, Ordering::Relaxed);
}

#[derive(Clone, Copy)]
struct Rect {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

struct RaisedBar {
    bar: Window,
    prev_above: Window,
    prev_below: Window,
}

unsafe fn window_class_strings(dpy: *mut Display, w: Window) -> (Option<String>, Option<String>) {
    let mut ch = XClassHint { res_name: std::ptr::null_mut(), res_class: std::ptr::null_mut() };
    if XGetClassHint(dpy, w, &mut ch) == 0 {
        return (None, None);
    }
    let name = if !ch.res_name.is_null() {
        Some(CStr::from_ptr(ch.res_name).to_string_lossy().into_owned())
    } else {
        None
    };
    let cls = if !ch.res_class.is_null() {
        Some(CStr::from_ptr(ch.res_class).to_string_lossy().into_owned())
    } else {
        None
    };
    if !ch.res_name.is_null() {
        XFree(ch.res_name as *mut c_void);
    }
    if !ch.res_class.is_null() {
        XFree(ch.res_class as *mut c_void);
    }
    (name, cls)
}

unsafe fn window_intersects_mon(dpy: *mut Display, w: Window, mon: Rect) -> bool {
    let mut attr: XWindowAttributes = std::mem::zeroed();
    if XGetWindowAttributes(dpy, w, &mut attr) == 0 {
        return false;
    }
    if attr.map_state != IS_VIEWABLE || attr.class as c_uint != INPUT_OUTPUT {
        return false;
    }
    !(attr.x >= mon.x + mon.w
        || attr.x + attr.width <= mon.x
        || attr.y >= mon.y + mon.h
        || attr.y + attr.height <= mon.y)
}

unsafe fn is_transient_overlay(dpy: *mut Display, w: Window) -> bool {
    let (_, cls) = window_class_strings(dpy, w);
    matches!(
        cls.as_deref(),
        Some(
            "Backdrop"
                | "CardPicker"
                | "SoundPanel"
                | "MusicPanel"
                | "WifiPanel"
                | "BluetoothPanel"
                | "ClipPanel"
                | "NotifCenter"
                | "NotifToast"
                | "WindowSwitch"
                | "MediaPill"
                | "Rofi"
        )
    )
}

unsafe fn is_monitor_bar_window(
    dpy: *mut Display,
    w: Window,
    self_win: Window,
    a_wm_type: Atom,
    a_dock: Atom,
    mon: Rect,
) -> bool {
    if w == 0 || w == self_win || !window_intersects_mon(dpy, w, mon) {
        return false;
    }
    let (name, cls) = window_class_strings(dpy, w);
    if cls.as_deref().is_some_and(|c| c.eq_ignore_ascii_case("Polybar"))
        || name.as_deref().is_some_and(|n| n.eq_ignore_ascii_case("polybar"))
    {
        return true;
    }
    if a_wm_type != 0 && a_dock != 0 {
        let mut actual_type: Atom = 0;
        let mut actual_fmt: c_int = 0;
        let mut nitems: c_ulong = 0;
        let mut bytes_after: c_ulong = 0;
        let mut prop: *mut c_uchar = std::ptr::null_mut();
        if XGetWindowProperty(
            dpy,
            w,
            a_wm_type,
            0,
            16,
            0,
            XA_ATOM,
            &mut actual_type,
            &mut actual_fmt,
            &mut nitems,
            &mut bytes_after,
            &mut prop,
        ) == 0
            && !prop.is_null()
        {
            let mut matched = false;
            if actual_type == XA_ATOM && actual_fmt == 32 {
                let atoms = std::slice::from_raw_parts(prop as *const Atom, nitems as usize);
                matched = atoms.contains(&a_dock);
            }
            XFree(prop as *mut c_void);
            if matched {
                return true;
            }
        }
    }
    false
}

unsafe fn has_other_active_overlay(dpy: *mut Display, self_win: Window, mon: Rect) -> bool {
    let mut root_ret: Window = 0;
    let mut parent_ret: Window = 0;
    let mut kids_ptr: *mut Window = std::ptr::null_mut();
    let mut nk: c_uint = 0;
    if XQueryTree(dpy, XDefaultRootWindow(dpy), &mut root_ret, &mut parent_ret, &mut kids_ptr, &mut nk) == 0
        || kids_ptr.is_null()
    {
        return false;
    }
    let kids = std::slice::from_raw_parts(kids_ptr, nk as usize);
    let mut found = false;
    for &w in kids {
        if w == 0 || w == self_win {
            continue;
        }
        if !window_intersects_mon(dpy, w, mon) {
            continue;
        }
        let (_, cls) = window_class_strings(dpy, w);
        if matches!(cls.as_deref(), Some("Backdrop" | "CardPicker")) {
            found = true;
            break;
        }
    }
    XFree(kids_ptr as *mut c_void);
    found
}

unsafe fn raise_monitor_bars(dpy: *mut Display, self_win: Window, mon: Rect) -> Vec<RaisedBar> {
    let mut out = Vec::new();
    let s_wm_type = CString::new("_NET_WM_WINDOW_TYPE").unwrap();
    let s_dock = CString::new("_NET_WM_WINDOW_TYPE_DOCK").unwrap();
    let a_wm_type = XInternAtom(dpy, s_wm_type.as_ptr(), 1);
    let a_dock = XInternAtom(dpy, s_dock.as_ptr(), 1);

    let mut root_ret: Window = 0;
    let mut parent_ret: Window = 0;
    let mut kids_ptr: *mut Window = std::ptr::null_mut();
    let mut nk: c_uint = 0;
    if XQueryTree(dpy, XDefaultRootWindow(dpy), &mut root_ret, &mut parent_ret, &mut kids_ptr, &mut nk) == 0
        || kids_ptr.is_null()
        || nk == 0
    {
        return out;
    }
    let kids = std::slice::from_raw_parts(kids_ptr, nk as usize);
    let mut is_bar = vec![false; kids.len()];
    let mut is_stable = vec![false; kids.len()];
    for (i, &w) in kids.iter().enumerate() {
        if is_monitor_bar_window(dpy, w, self_win, a_wm_type, a_dock, mon) {
            is_bar[i] = true;
        } else if w != self_win {
            let mut attr: XWindowAttributes = std::mem::zeroed();
            if XGetWindowAttributes(dpy, w, &mut attr) != 0
                && attr.map_state == IS_VIEWABLE
                && attr.class as c_uint == INPUT_OUTPUT
                && !is_transient_overlay(dpy, w)
            {
                is_stable[i] = true;
            }
        }
    }

    let mut anchor = self_win;
    for (i, &w) in kids.iter().enumerate() {
        if !is_bar[i] {
            continue;
        }
        let prev_above = (i + 1..kids.len())
            .find(|&k| is_stable[k])
            .map(|k| kids[k])
            .unwrap_or(0);
        let prev_below = (0..i)
            .rev()
            .find(|&k| is_stable[k])
            .map(|k| kids[k])
            .unwrap_or(0);
        out.push(RaisedBar { bar: w, prev_above, prev_below });
        let mut wc: XWindowChanges = std::mem::zeroed();
        wc.sibling = anchor;
        wc.stack_mode = STACK_ABOVE;
        XConfigureWindow(dpy, w, CW_SIBLING | CW_STACK_MODE, &mut wc);
        anchor = w;
    }
    XFree(kids_ptr as *mut c_void);
    if !out.is_empty() {
        XFlush(dpy);
    }
    out
}

unsafe fn restore_monitor_bars(dpy: *mut Display, self_win: Window, raised: &[RaisedBar], mon: Rect) {
    if raised.is_empty() || has_other_active_overlay(dpy, self_win, mon) {
        return;
    }
    let mut root_ret: Window = 0;
    let mut parent_ret: Window = 0;
    let mut kids_ptr: *mut Window = std::ptr::null_mut();
    let mut nk: c_uint = 0;
    if XQueryTree(dpy, XDefaultRootWindow(dpy), &mut root_ret, &mut parent_ret, &mut kids_ptr, &mut nk) == 0
        || kids_ptr.is_null()
    {
        return;
    }
    let kids = std::slice::from_raw_parts(kids_ptr, nk as usize);
    for rb in raised {
        if !kids.contains(&rb.bar) {
            continue;
        }
        let mut wc: XWindowChanges = std::mem::zeroed();
        if rb.prev_below != 0 && kids.contains(&rb.prev_below) {
            wc.sibling = rb.prev_below;
            wc.stack_mode = STACK_ABOVE;
            XConfigureWindow(dpy, rb.bar, CW_SIBLING | CW_STACK_MODE, &mut wc);
        } else if rb.prev_above != 0 && kids.contains(&rb.prev_above) {
            wc.sibling = rb.prev_above;
            wc.stack_mode = STACK_BELOW;
            XConfigureWindow(dpy, rb.bar, CW_SIBLING | CW_STACK_MODE, &mut wc);
        }
    }
    XFree(kids_ptr as *mut c_void);
    XFlush(dpy);
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
    let mon = Rect { x: x as i32, y: y as i32, w: w as i32, h: h as i32 };

    unsafe {
        let sa = SigAction {
            sa_handler: handle_stop,
            sa_mask: [0; 16],
            sa_flags: 0,
            _pad: 0,
            sa_restorer: std::ptr::null(),
        };
        sigaction(SIGTERM, &sa, std::ptr::null_mut());
        sigaction(SIGINT, &sa, std::ptr::null_mut());

        let dpy = XOpenDisplay(std::ptr::null());
        if dpy.is_null() {
            eprintln!("misa-backdrop: cannot open display");
            std::process::exit(1);
        }
        XSetErrorHandler(Some(ignore_x_err));

        let screen = XDefaultScreen(dpy);
        let root = XDefaultRootWindow(dpy);
        let mut attrs: XSetWindowAttributes = std::mem::zeroed();
        attrs.background_pixel = XBlackPixel(dpy, screen);
        attrs.override_redirect = 1;
        attrs.event_mask = BUTTON_PRESS_MASK;
        let win = XCreateWindow(
            dpy, root, x as c_int, y as c_int, w as c_uint, h as c_uint,
            0, COPY_FROM_PARENT, INPUT_OUTPUT, std::ptr::null_mut(),
            CW_BACK_PIXEL | CW_OVERRIDE_REDIRECT | CW_EVENT_MASK, &mut attrs,
        );
        let name = CString::new("misa-backdrop").unwrap();
        let class = CString::new("Backdrop").unwrap();
        let mut hint = XClassHint {
            res_name: name.as_ptr() as *mut c_char,
            res_class: class.as_ptr() as *mut c_char,
        };
        XSetClassHint(dpy, win, &mut hint);
        XStoreName(dpy, win, name.as_ptr());
        XMapRaised(dpy, win);

        // Keep topbar / dock windows crisp above the full-monitor backdrop so
        // transparent floating/island gaps get seamless blur & dim.
        let raised = raise_monitor_bars(dpy, win, mon);

        let mut esc_code: c_int = 0;
        if click.is_some() {
            esc_code = XKeysymToKeycode(dpy, XK_ESCAPE) as c_int;
            if esc_code != 0 {
                for attempt in 0..10 {
                    X_ERR.store(false, Ordering::Relaxed);
                    for mods in [0, LOCK_MASK, MOD2_MASK, LOCK_MASK | MOD2_MASK] {
                        XGrabKey(dpy, esc_code, mods, root, 0, GRAB_MODE_ASYNC, GRAB_MODE_ASYNC);
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
        let mut x_gone = false;
        'main_loop: loop {
            if STOP.load(Ordering::Relaxed) {
                break;
            }
            poll(&mut fds, 1, 100);
            if STOP.load(Ordering::Relaxed) {
                break;
            }
            while XPending(dpy) > 0 {
                XNextEvent(dpy, &mut ev);
                if ev.kind == BUTTON_PRESS || ev.kind == KEY_PRESS {
                    if let Some(cmd) = &click {
                        let _ = Command::new("sh").arg("-c").arg(cmd).spawn();
                    }
                    break 'main_loop;
                }
            }
            if fds.revents & (8 | 16) != 0 {
                x_gone = true;
                break;
            }
            if pid > 0 && kill(pid, 0) != 0 {
                break;
            }
        }

        if !x_gone {
            if esc_code != 0 {
                XUngrabKey(dpy, esc_code, ANY_MODIFIER, root);
            }
            XUnmapWindow(dpy, win);
            XFlush(dpy);
            if !raised.is_empty() {
                // Let picom finish Backdrop's 140 ms close fade before restoring bar stacking.
                sleep(Duration::from_millis(160));
                restore_monitor_bars(dpy, win, &raised, mon);
            }
            XDestroyWindow(dpy, win);
            XCloseDisplay(dpy);
        }
        _exit(0);
    }
}
