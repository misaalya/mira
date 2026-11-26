//! One-shot offscreen OpenGL shader baker for `misa-wall` v2.
//!
//! `libGL.so.1` is loaded dynamically via `dlopen` ONLY inside this module
//! during transient `--bake-frame` (CardPicker Theme Showcase) or `--bake-loop`
//! (one-time `.mwa` cache generation) calls, which exit immediately afterward so
//! the persistent desktop wallpaper daemon uses 0 MiB of Mesa/GLX RAM forever!

use std::ffi::{c_char, c_int, c_long, c_uchar, c_uint, c_ulong, c_void, CStr, CString};
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::ptr::{null, null_mut};

use crate::codec::{
    lz4_compress, pack_planar_delta, write_mwa_atomic, EncodedDelta, MWA_FLAG_LINEAR, TILE_ROWS,
};
use crate::{
    decode_rgb_image_scaled, getpid, ignore_x_err, Display, Visual, Window, XDefaultRootWindow,
    XDefaultScreen, XFree, XOpenDisplay, XSetErrorHandler, XSync,
};

const RTLD_LAZY: c_int = 1;
const RTLD_LOCAL: c_int = 0;

const INPUT_OUTPUT: c_uint = 1;
const ALLOC_NONE: c_int = 0;
const CW_BACK_PIXMAP: c_ulong = 1 << 0;
const CW_BORDER_PIXEL: c_ulong = 1 << 3;
const CW_OVERRIDE_REDIRECT: c_ulong = 1 << 9;
const CW_COLORMAP: c_ulong = 1 << 13;

const GLX_RGBA: c_int = 4;
const GLX_DOUBLEBUFFER: c_int = 5;
const GLX_RED_SIZE: c_int = 8;
const GLX_GREEN_SIZE: c_int = 9;
const GLX_BLUE_SIZE: c_int = 10;

const GL_TRIANGLES: u32 = 0x0004;
const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_FLOAT: u32 = 0x1406;
const GL_RGB: u32 = 0x1907;
const GL_RGBA: u32 = 0x1908;
const GL_NEAREST: i32 = 0x2600;
const GL_LINEAR: i32 = 0x2601;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_WRAP_S: u32 = 0x2802;
const GL_TEXTURE_WRAP_T: u32 = 0x2803;
const GL_CLAMP_TO_EDGE: i32 = 0x812F;
const GL_UNPACK_ALIGNMENT: u32 = 0x0CF5;
const GL_PACK_ALIGNMENT: u32 = 0x0D05;
const GL_TEXTURE0: u32 = 0x84C0;
const GL_ARRAY_BUFFER: u32 = 0x8892;
const GL_STATIC_DRAW: u32 = 0x88E4;
const GL_FRAGMENT_SHADER: u32 = 0x8B30;
const GL_VERTEX_SHADER: u32 = 0x8B31;
const GL_COMPILE_STATUS: u32 = 0x8B81;
const GL_LINK_STATUS: u32 = 0x8B82;
const GL_FRAMEBUFFER: u32 = 0x8D40;
const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;

type Colormap = c_ulong;
type GLXContext = *mut c_void;

#[repr(C)]
struct XVisualInfo {
    visual: *mut Visual,
    visualid: c_ulong,
    screen: c_int,
    depth: c_int,
    class: c_int,
    red_mask: c_ulong,
    green_mask: c_ulong,
    blue_mask: c_ulong,
    colormap_size: c_int,
    bits_per_rgb: c_int,
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
    colormap: Colormap,
    cursor: c_ulong,
}

#[link(name = "X11")]
extern "C" {
    fn XCreateColormap(d: *mut Display, w: Window, v: *mut Visual, alloc: c_int) -> Colormap;
    fn XFreeColormap(d: *mut Display, c: Colormap) -> c_int;
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
    fn XCloseDisplay(d: *mut Display) -> c_int;
}

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

type FnGlXChooseVisual = unsafe extern "C" fn(*mut Display, c_int, *mut c_int) -> *mut XVisualInfo;
type FnGlXCreateContext =
    unsafe extern "C" fn(*mut Display, *mut XVisualInfo, GLXContext, c_int) -> GLXContext;
type FnGlXDestroyContext = unsafe extern "C" fn(*mut Display, GLXContext);
type FnGlXMakeCurrent = unsafe extern "C" fn(*mut Display, Window, GLXContext) -> c_int;
type FnGlXGetProcAddress = unsafe extern "C" fn(*const c_uchar) -> *mut c_void;

type FnGlViewport = unsafe extern "C" fn(i32, i32, i32, i32);
type FnGlCreateShader = unsafe extern "C" fn(u32) -> u32;
type FnGlShaderSource = unsafe extern "C" fn(u32, i32, *const *const c_char, *const i32);
type FnGlCompileShader = unsafe extern "C" fn(u32);
type FnGlGetShaderiv = unsafe extern "C" fn(u32, u32, *mut i32);
type FnGlGetShaderInfoLog = unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char);
type FnGlDeleteShader = unsafe extern "C" fn(u32);
type FnGlCreateProgram = unsafe extern "C" fn() -> u32;
type FnGlAttachShader = unsafe extern "C" fn(u32, u32);
type FnGlBindAttribLocation = unsafe extern "C" fn(u32, u32, *const c_char);
type FnGlLinkProgram = unsafe extern "C" fn(u32);
type FnGlGetProgramiv = unsafe extern "C" fn(u32, u32, *mut i32);
type FnGlDeleteProgram = unsafe extern "C" fn(u32);
type FnGlUseProgram = unsafe extern "C" fn(u32);
type FnGlGetUniformLocation = unsafe extern "C" fn(u32, *const c_char) -> i32;
type FnGlUniform1i = unsafe extern "C" fn(i32, i32);
type FnGlUniform1f = unsafe extern "C" fn(i32, f32);
type FnGlUniform2f = unsafe extern "C" fn(i32, f32, f32);
type FnGlUniform4f = unsafe extern "C" fn(i32, f32, f32, f32, f32);
type FnGlGenBuffers = unsafe extern "C" fn(i32, *mut u32);
type FnGlDeleteBuffers = unsafe extern "C" fn(i32, *const u32);
type FnGlBindBuffer = unsafe extern "C" fn(u32, u32);
type FnGlBufferData = unsafe extern "C" fn(u32, isize, *const c_void, u32);
type FnGlEnableVertexAttribArray = unsafe extern "C" fn(u32);
type FnGlVertexAttribPointer = unsafe extern "C" fn(u32, i32, u32, u8, i32, *const c_void);
type FnGlGenTextures = unsafe extern "C" fn(i32, *mut u32);
type FnGlDeleteTextures = unsafe extern "C" fn(i32, *const u32);
type FnGlActiveTexture = unsafe extern "C" fn(u32);
type FnGlBindTexture = unsafe extern "C" fn(u32, u32);
type FnGlTexParameteri = unsafe extern "C" fn(u32, u32, i32);
type FnGlPixelStorei = unsafe extern "C" fn(u32, i32);
type FnGlTexImage2D = unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void);
type FnGlDrawArrays = unsafe extern "C" fn(u32, i32, i32);
type FnGlReadPixels = unsafe extern "C" fn(i32, i32, i32, i32, u32, u32, *mut c_void);
type FnGlGenFramebuffers = unsafe extern "C" fn(i32, *mut u32);
type FnGlDeleteFramebuffers = unsafe extern "C" fn(i32, *const u32);
type FnGlBindFramebuffer = unsafe extern "C" fn(u32, u32);
type FnGlFramebufferTexture2D = unsafe extern "C" fn(u32, u32, u32, u32, i32);
type FnGlCheckFramebufferStatus = unsafe extern "C" fn(u32) -> u32;

struct BakerGl {
    choose_visual: FnGlXChooseVisual,
    create_context: FnGlXCreateContext,
    destroy_context: FnGlXDestroyContext,
    make_current: FnGlXMakeCurrent,
    viewport: FnGlViewport,
    create_shader: FnGlCreateShader,
    shader_source: FnGlShaderSource,
    compile_shader: FnGlCompileShader,
    get_shaderiv: FnGlGetShaderiv,
    get_shader_info_log: FnGlGetShaderInfoLog,
    delete_shader: Option<FnGlDeleteShader>,
    create_program: FnGlCreateProgram,
    attach_shader: FnGlAttachShader,
    bind_attrib_location: FnGlBindAttribLocation,
    link_program: FnGlLinkProgram,
    get_programiv: FnGlGetProgramiv,
    delete_program: Option<FnGlDeleteProgram>,
    use_program: FnGlUseProgram,
    get_uniform_location: FnGlGetUniformLocation,
    uniform1i: FnGlUniform1i,
    uniform1f: FnGlUniform1f,
    uniform2f: FnGlUniform2f,
    uniform4f: FnGlUniform4f,
    gen_buffers: FnGlGenBuffers,
    delete_buffers: Option<FnGlDeleteBuffers>,
    bind_buffer: FnGlBindBuffer,
    buffer_data: FnGlBufferData,
    enable_vertex_attrib_array: FnGlEnableVertexAttribArray,
    vertex_attrib_pointer: FnGlVertexAttribPointer,
    gen_textures: FnGlGenTextures,
    delete_textures: FnGlDeleteTextures,
    active_texture: FnGlActiveTexture,
    bind_texture: FnGlBindTexture,
    tex_parameteri: FnGlTexParameteri,
    pixel_storei: FnGlPixelStorei,
    tex_image_2d: FnGlTexImage2D,
    draw_arrays: FnGlDrawArrays,
    read_pixels: FnGlReadPixels,
    gen_framebuffers: FnGlGenFramebuffers,
    delete_framebuffers: FnGlDeleteFramebuffers,
    bind_framebuffer: FnGlBindFramebuffer,
    framebuffer_texture_2d: FnGlFramebufferTexture2D,
    check_framebuffer_status: FnGlCheckFramebufferStatus,
}

const LIVE_VS_SRC: &str = "#version 120
attribute vec2 aPos;
varying vec2 vUv;
void main() {
    vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
    gl_Position = vec4(aPos, 0.0, 1.0);
}
";

unsafe fn load_sym(lib: *mut c_void, gpa: Option<FnGlXGetProcAddress>, name: &str) -> *mut c_void {
    let cs = CString::new(name).unwrap();
    if let Some(f) = gpa {
        let p = f(cs.as_ptr() as *const c_uchar);
        if !p.is_null() {
            return p;
        }
    }
    dlsym(lib, cs.as_ptr())
}

unsafe fn load_gl() -> Option<BakerGl> {
    let mut lib = null_mut();
    for soname in ["libGL.so.1", "libGL.so"] {
        let cs = CString::new(soname).unwrap();
        lib = dlopen(cs.as_ptr(), RTLD_LAZY | RTLD_LOCAL);
        if !lib.is_null() {
            break;
        }
    }
    if lib.is_null() {
        return None;
    }

    let gpa_ptr = {
        let a = dlsym(lib, CString::new("glXGetProcAddressARB").unwrap().as_ptr());
        if !a.is_null() {
            a
        } else {
            dlsym(lib, CString::new("glXGetProcAddress").unwrap().as_ptr())
        }
    };
    let gpa: Option<FnGlXGetProcAddress> = if gpa_ptr.is_null() {
        None
    } else {
        Some(std::mem::transmute(gpa_ptr))
    };

    macro_rules! req {
        ($n:expr, $t:ty) => {{
            let p = load_sym(lib, gpa, $n);
            if p.is_null() {
                return None;
            }
            std::mem::transmute::<*mut c_void, $t>(p)
        }};
    }
    macro_rules! opt {
        ($n:expr, $t:ty) => {{
            let p = load_sym(lib, gpa, $n);
            if p.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut c_void, $t>(p))
            }
        }};
    }

    Some(BakerGl {
        choose_visual: req!("glXChooseVisual", FnGlXChooseVisual),
        create_context: req!("glXCreateContext", FnGlXCreateContext),
        destroy_context: req!("glXDestroyContext", FnGlXDestroyContext),
        make_current: req!("glXMakeCurrent", FnGlXMakeCurrent),
        viewport: req!("glViewport", FnGlViewport),
        create_shader: req!("glCreateShader", FnGlCreateShader),
        shader_source: req!("glShaderSource", FnGlShaderSource),
        compile_shader: req!("glCompileShader", FnGlCompileShader),
        get_shaderiv: req!("glGetShaderiv", FnGlGetShaderiv),
        get_shader_info_log: req!("glGetShaderInfoLog", FnGlGetShaderInfoLog),
        delete_shader: opt!("glDeleteShader", FnGlDeleteShader),
        create_program: req!("glCreateProgram", FnGlCreateProgram),
        attach_shader: req!("glAttachShader", FnGlAttachShader),
        bind_attrib_location: req!("glBindAttribLocation", FnGlBindAttribLocation),
        link_program: req!("glLinkProgram", FnGlLinkProgram),
        get_programiv: req!("glGetProgramiv", FnGlGetProgramiv),
        delete_program: opt!("glDeleteProgram", FnGlDeleteProgram),
        use_program: req!("glUseProgram", FnGlUseProgram),
        get_uniform_location: req!("glGetUniformLocation", FnGlGetUniformLocation),
        uniform1i: req!("glUniform1i", FnGlUniform1i),
        uniform1f: req!("glUniform1f", FnGlUniform1f),
        uniform2f: req!("glUniform2f", FnGlUniform2f),
        uniform4f: req!("glUniform4f", FnGlUniform4f),
        gen_buffers: req!("glGenBuffers", FnGlGenBuffers),
        delete_buffers: opt!("glDeleteBuffers", FnGlDeleteBuffers),
        bind_buffer: req!("glBindBuffer", FnGlBindBuffer),
        buffer_data: req!("glBufferData", FnGlBufferData),
        enable_vertex_attrib_array: req!("glEnableVertexAttribArray", FnGlEnableVertexAttribArray),
        vertex_attrib_pointer: req!("glVertexAttribPointer", FnGlVertexAttribPointer),
        gen_textures: req!("glGenTextures", FnGlGenTextures),
        delete_textures: req!("glDeleteTextures", FnGlDeleteTextures),
        active_texture: req!("glActiveTexture", FnGlActiveTexture),
        bind_texture: req!("glBindTexture", FnGlBindTexture),
        tex_parameteri: req!("glTexParameteri", FnGlTexParameteri),
        pixel_storei: req!("glPixelStorei", FnGlPixelStorei),
        tex_image_2d: req!("glTexImage2D", FnGlTexImage2D),
        draw_arrays: req!("glDrawArrays", FnGlDrawArrays),
        read_pixels: req!("glReadPixels", FnGlReadPixels),
        gen_framebuffers: req!("glGenFramebuffers", FnGlGenFramebuffers),
        delete_framebuffers: req!("glDeleteFramebuffers", FnGlDeleteFramebuffers),
        bind_framebuffer: req!("glBindFramebuffer", FnGlBindFramebuffer),
        framebuffer_texture_2d: req!("glFramebufferTexture2D", FnGlFramebufferTexture2D),
        check_framebuffer_status: req!("glCheckFramebufferStatus", FnGlCheckFramebufferStatus),
    })
}

unsafe fn compile_shader(gl: &BakerGl, kind: u32, src: &str) -> Option<u32> {
    let sh = (gl.create_shader)(kind);
    if sh == 0 {
        return None;
    }
    let cs = CString::new(src).ok()?;
    let ptr = cs.as_ptr();
    (gl.shader_source)(sh, 1, &ptr, null());
    (gl.compile_shader)(sh);
    let mut ok: i32 = 0;
    (gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
    if ok == 0 {
        let mut buf = [0i8; 1024];
        let mut len: i32 = 0;
        (gl.get_shader_info_log)(sh, buf.len() as i32, &mut len, buf.as_mut_ptr());
        let msg = CStr::from_ptr(buf.as_ptr()).to_string_lossy();
        eprintln!("misa-wall baker: shader compile error:\n{msg}");
        return None;
    }
    Some(sh)
}

unsafe fn link_prog(gl: &BakerGl, fs_src: &str) -> Option<u32> {
    let vs = compile_shader(gl, GL_VERTEX_SHADER, LIVE_VS_SRC)?;
    let Some(fs_sh) = compile_shader(gl, GL_FRAGMENT_SHADER, fs_src) else {
        if let Some(del) = gl.delete_shader {
            del(vs);
        }
        return None;
    };
    let prog = (gl.create_program)();
    (gl.attach_shader)(prog, vs);
    (gl.attach_shader)(prog, fs_sh);
    let c_apos = CString::new("aPos").unwrap();
    (gl.bind_attrib_location)(prog, 0, c_apos.as_ptr());
    (gl.link_program)(prog);
    if let Some(del) = gl.delete_shader {
        del(vs);
        del(fs_sh);
    }
    let mut linked: i32 = 0;
    (gl.get_programiv)(prog, GL_LINK_STATUS, &mut linked);
    if linked == 0 {
        if let Some(delp) = gl.delete_program {
            delp(prog);
        }
        return None;
    }
    Some(prog)
}

struct OffscreenRenderer {
    dpy: *mut Display,
    win: Window,
    cmap: Colormap,
    ctx: GLXContext,
    gl: BakerGl,
    prog: u32,
    fbo: u32,
    fbo_tex: u32,
    img_tex: u32,
    vbo: u32,
    loc_time: i32,
    fw: i32,
    fh: i32,
}

impl OffscreenRenderer {
    unsafe fn new(img_path: &str, frag_path: &str, fw: i32, fh: i32) -> Option<Self> {
        let frag_src = fs::read_to_string(frag_path).ok()?;
        let gl = load_gl()?;
        let dpy = XOpenDisplay(null());
        if dpy.is_null() {
            return None;
        }
        XSetErrorHandler(Some(ignore_x_err));
        let scr = XDefaultScreen(dpy);
        let root = XDefaultRootWindow(dpy);

        let (tex_w, tex_h, tex_bytes) = decode_rgb_image_scaled(img_path, fw * 2, fh * 2)?;
        let im_aspect = (tex_w as f32) / (tex_h.max(1) as f32);
        let vp_a = (fw as f32) / (fh.max(1) as f32);
        let cover = if vp_a > im_aspect {
            let sy = im_aspect / vp_a;
            [1.0f32, sy, 0.0, (1.0 - sy) * 0.5]
        } else {
            let sx = vp_a / im_aspect;
            [sx, 1.0f32, (1.0 - sx) * 0.5, 0.0]
        };

        let mut vis_attribs = [
            GLX_RGBA,
            GLX_DOUBLEBUFFER,
            GLX_RED_SIZE,
            8,
            GLX_GREEN_SIZE,
            8,
            GLX_BLUE_SIZE,
            8,
            0,
        ];
        let vi = (gl.choose_visual)(dpy, scr, vis_attribs.as_mut_ptr());
        if vi.is_null() {
            XCloseDisplay(dpy);
            return None;
        }
        let cmap = XCreateColormap(dpy, root, (*vi).visual, ALLOC_NONE);
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
            event_mask: 0,
            do_not_propagate_mask: 0,
            override_redirect: 1,
            colormap: cmap,
            cursor: 0,
        };
        // Unmapped 1x1 hidden window: never appears on screen!
        let win = XCreateWindow(
            dpy,
            root,
            -16,
            -16,
            1,
            1,
            0,
            (*vi).depth,
            INPUT_OUTPUT,
            (*vi).visual,
            CW_BACK_PIXMAP | CW_BORDER_PIXEL | CW_OVERRIDE_REDIRECT | CW_COLORMAP,
            &mut swa,
        );
        let ctx = (gl.create_context)(dpy, vi, null_mut(), 1);
        XFree(vi as *mut c_void);
        if ctx.is_null() || (gl.make_current)(dpy, win, ctx) == 0 {
            if !ctx.is_null() {
                (gl.destroy_context)(dpy, ctx);
            }
            XDestroyWindow(dpy, win);
            XFreeColormap(dpy, cmap);
            XCloseDisplay(dpy);
            return None;
        }

        let Some(prog) = link_prog(&gl, &frag_src) else {
            (gl.make_current)(dpy, 0, null_mut());
            (gl.destroy_context)(dpy, ctx);
            XDestroyWindow(dpy, win);
            XFreeColormap(dpy, cmap);
            XCloseDisplay(dpy);
            return None;
        };

        let mut fbo: u32 = 0;
        let mut fbo_tex: u32 = 0;
        (gl.gen_framebuffers)(1, &mut fbo);
        (gl.gen_textures)(1, &mut fbo_tex);
        (gl.bind_texture)(GL_TEXTURE_2D, fbo_tex);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        (gl.tex_image_2d)(GL_TEXTURE_2D, 0, GL_RGBA as i32, fw, fh, 0, GL_RGBA, GL_UNSIGNED_BYTE, null());
        (gl.bind_framebuffer)(GL_FRAMEBUFFER, fbo);
        (gl.framebuffer_texture_2d)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_tex, 0);
        if (gl.check_framebuffer_status)(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE {
            (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
            (gl.delete_framebuffers)(1, &fbo);
            (gl.delete_textures)(1, &fbo_tex);
            if let Some(delp) = gl.delete_program {
                delp(prog);
            }
            (gl.make_current)(dpy, 0, null_mut());
            (gl.destroy_context)(dpy, ctx);
            XDestroyWindow(dpy, win);
            XFreeColormap(dpy, cmap);
            XCloseDisplay(dpy);
            return None;
        }

        let verts: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let mut vbo: u32 = 0;
        (gl.gen_buffers)(1, &mut vbo);
        (gl.bind_buffer)(GL_ARRAY_BUFFER, vbo);
        (gl.buffer_data)(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(&verts) as isize,
            verts.as_ptr() as *const c_void,
            GL_STATIC_DRAW,
        );
        (gl.enable_vertex_attrib_array)(0);
        (gl.vertex_attrib_pointer)(0, 2, GL_FLOAT, 0, 0, null());

        let mut img_tex: u32 = 0;
        (gl.active_texture)(GL_TEXTURE0);
        (gl.gen_textures)(1, &mut img_tex);
        (gl.bind_texture)(GL_TEXTURE_2D, img_tex);
        (gl.pixel_storei)(GL_UNPACK_ALIGNMENT, 1);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        (gl.tex_image_2d)(
            GL_TEXTURE_2D,
            0,
            GL_RGB as i32,
            tex_w,
            tex_h,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            tex_bytes.as_ptr() as *const c_void,
        );

        let uloc = |name: &str| {
            let cs = CString::new(name).unwrap();
            (gl.get_uniform_location)(prog, cs.as_ptr())
        };
        (gl.use_program)(prog);
        let lt = uloc("uTex");
        if lt >= 0 {
            (gl.uniform1i)(lt, 0);
        }
        let lr = uloc("uRes");
        if lr >= 0 {
            (gl.uniform2f)(lr, fw as f32, fh as f32);
        }
        let lc = uloc("uCover");
        if lc >= 0 {
            (gl.uniform4f)(lc, cover[0], cover[1], cover[2], cover[3]);
        }
        let lm = uloc("uMouse");
        if lm >= 0 {
            (gl.uniform2f)(lm, 0.0, 0.0);
        }
        let lce = uloc("uCell");
        if lce >= 0 {
            (gl.uniform1f)(lce, 1.0);
        }
        let loc_time = uloc("uTime");

        (gl.viewport)(0, 0, fw, fh);
        (gl.pixel_storei)(GL_PACK_ALIGNMENT, 1);

        Some(Self {
            dpy,
            win,
            cmap,
            ctx,
            gl,
            prog,
            fbo,
            fbo_tex,
            img_tex,
            vbo,
            loc_time,
            fw,
            fh,
        })
    }

    /// Render one frame at `t_sec` into top-down `dst_rgb` (`fw * fh * 3` bytes).
    unsafe fn render_rgb_topdown(&self, t_sec: f32, gl_flip_scratch: &mut [u8], dst_rgb: &mut [u8]) {
        if self.loc_time >= 0 {
            (self.gl.uniform1f)(self.loc_time, t_sec);
        }
        (self.gl.draw_arrays)(GL_TRIANGLES, 0, 3);
        (self.gl.read_pixels)(
            0,
            0,
            self.fw,
            self.fh,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            gl_flip_scratch.as_mut_ptr() as *mut c_void,
        );
        let row = (self.fw as usize) * 3;
        let h = self.fh as usize;
        for y in 0..h {
            let src_y = h - 1 - y;
            dst_rgb[y * row..(y + 1) * row]
                .copy_from_slice(&gl_flip_scratch[src_y * row..(src_y + 1) * row]);
        }
    }
}

impl Drop for OffscreenRenderer {
    fn drop(&mut self) {
        unsafe {
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
            (self.gl.delete_framebuffers)(1, &self.fbo);
            (self.gl.delete_textures)(1, &self.fbo_tex);
            (self.gl.delete_textures)(1, &self.img_tex);
            if let Some(delb) = self.gl.delete_buffers {
                delb(1, &self.vbo);
            }
            if let Some(delp) = self.gl.delete_program {
                delp(self.prog);
            }
            (self.gl.make_current)(self.dpy, 0, null_mut());
            (self.gl.destroy_context)(self.dpy, self.ctx);
            XDestroyWindow(self.dpy, self.win);
            XFreeColormap(self.dpy, self.cmap);
            XSync(self.dpy, 0);
            XCloseDisplay(self.dpy);
        }
    }
}

/// Offscreen single-frame baker for static Theme Picker snapshots (`--bake-frame`).
pub fn bake_shader_frame(img_path: &str, frag_path: &str, out_ppm: &str, time_sec: f32) -> i32 {
    let fw = 640i32;
    let fh = 360i32;
    unsafe {
        let Some(r) = OffscreenRenderer::new(img_path, frag_path, fw, fh) else {
            return 1;
        };
        let len = (fw as usize) * (fh as usize) * 3;
        let mut scratch = vec![0u8; len];
        let mut rgb = vec![0u8; len];
        r.render_rgb_topdown(time_sec, &mut scratch, &mut rgb);

        let mut ppm = Vec::with_capacity(32 + len);
        let _ = write!(ppm, "P6\n{fw} {fh}\n255\n");
        ppm.extend_from_slice(&rgb);
        let tmp = format!("{out_ppm}.{}.tmp", getpid());
        if fs::write(&tmp, &ppm).is_ok() && fs::rename(&tmp, out_ppm).is_ok() {
            0
        } else {
            let _ = fs::remove_file(&tmp);
            1
        }
    }
}

const BAYER4_CELL: [[u8; 4]; 4] = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
];

/// Crisp per-cell Bayer-ordered temporal cross-dissolve between `tail` (fading out)
/// and `head` (fading in) at weight `w` in `(0, 1)`. Preserves 100% pure quantized
/// 3px Bayer colors with zero blurry ghosting and zero extra palette colors!
fn bayer_temporal_dissolve(head: &[u8], tail: &[u8], w: usize, h: usize, frac: f32, out: &mut [u8]) {
    let thresh = (frac * 16.0).clamp(0.0, 16.0);
    for y in 0..h {
        let brow = BAYER4_CELL[y & 3];
        let row_off = y * w * 3;
        for x in 0..w {
            let off = row_off + x * 3;
            // Hash cell slightly with its 4x4 block position so overlap dissolves organically.
            let b = brow[x & 3] as f32 + 0.5;
            let src = if b < thresh { head } else { tail };
            out[off] = src[off];
            out[off + 1] = src[off + 1];
            out[off + 2] = src[off + 2];
        }
    }
}

/// One-shot seamless animation loop baker (`--bake-loop`).
/// Renders `frames + overlap` frames at `fw x fh` on a hidden 1x1 GLX FBO,
/// dissolves the tail cleanly back into frame 0 via crisp ordered-Bayer cell
/// selection (zero loop seam/pop!), encodes every frame transition (including
/// `frames-1 -> 0`) with `awww`-style BitPack + LZ4 + exact dirty bounding box,
/// and writes the `.mwa` cache file atomically before exiting.
pub fn bake_shader_loop(
    img_path: &str,
    frag_path: &str,
    out_mwa: &Path,
    src_hash: u64,
    fw: u32,
    fh: u32,
    fps: u32,
    frames: u32,
) -> i32 {
    let n = frames.clamp(12, 120) as usize;
    let overlap = (n / 4).clamp(6, 16);
    let dt = 1.0f32 / (fps.max(1) as f32);
    let t0 = 2.0f32; // start with all particles/rain/steam already in flight
    let w = fw as usize;
    let h = fh as usize;
    let frame_bytes = w * h * 3;

    unsafe {
        let Some(r) = OffscreenRenderer::new(img_path, frag_path, fw as i32, fh as i32) else {
            return 1;
        };

        let mut scratch = vec![0u8; frame_bytes];
        // Render head overlap frames (0..overlap) first so we can dissolve tail into them.
        let mut head_raw: Vec<Vec<u8>> = Vec::with_capacity(overlap);
        for f in 0..overlap {
            let mut buf = vec![0u8; frame_bytes];
            r.render_rgb_topdown(t0 + (f as f32) * dt, &mut scratch, &mut buf);
            head_raw.push(buf);
        }

        // Render middle steady frames (overlap..n) and tail overlap frames (n..n+overlap).
        // We stream encoding directly so peak RAM stays tiny (~15 MiB max during bake!).
        let mut tail_raw: Vec<Vec<u8>> = Vec::with_capacity(overlap);
        for f in 0..overlap {
            let mut buf = vec![0u8; frame_bytes];
            r.render_rgb_topdown(t0 + ((n + f) as f32) * dt, &mut scratch, &mut buf);
            tail_raw.push(buf);
        }

        // Build seamlessly dissolved frames 0..overlap:
        // At f=0, frac = 1/(overlap+1) -> mostly tail[0] (= frame n, seamless from frame n-1!).
        // At f=overlap-1, frac = overlap/(overlap+1) -> mostly head[overlap-1] (seamless into frame overlap!).
        let mut blended_head: Vec<Vec<u8>> = Vec::with_capacity(overlap);
        for f in 0..overlap {
            let mut out = vec![0u8; frame_bytes];
            let frac = ((f + 1) as f32) / ((overlap + 1) as f32);
            bayer_temporal_dissolve(&head_raw[f], &tail_raw[f], w, h, frac, &mut out);
            blended_head.push(out);
        }
        drop(head_raw);
        drop(tail_raw);

        let frame_delay = ((1000 + fps / 2) / fps.max(1)).clamp(16, 1000) as u16;
        let base_rgb = blended_head[0].clone();
        let mut deltas: Vec<EncodedDelta> = Vec::with_capacity(n);
        // Placeholder for frame 0 wrap delta (frame n-1 -> frame 0), filled at the end.
        deltas.push(EncodedDelta {
            blob: Vec::new(),
            raw_len: 0,
            x0: 0,
            y0: 0,
            x1: 0,
            y1: 0,
            tile_mask: [0u16; TILE_ROWS],
            delay_ms: frame_delay,
        });

        let mut raw_delta = Vec::with_capacity(frame_bytes / 2);
        let mut prev = blended_head[0].clone();

        // Encode frames 1..overlap from blended_head.
        for f in 1..overlap {
            let cur = &blended_head[f];
            if let Some(m) = pack_planar_delta(&prev, cur, w, h, &mut raw_delta) {
                let comp = lz4_compress(&raw_delta);
                deltas.push(EncodedDelta {
                    blob: comp,
                    raw_len: raw_delta.len() as u32,
                    x0: m.x0,
                    y0: m.y0,
                    x1: m.x1,
                    y1: m.y1,
                    tile_mask: m.tile_mask,
                    delay_ms: frame_delay,
                });
            } else {
                deltas.push(EncodedDelta {
                    blob: Vec::new(),
                    raw_len: 0,
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                    tile_mask: [0u16; TILE_ROWS],
                    delay_ms: frame_delay,
                });
            }
            prev.copy_from_slice(cur);
        }
        drop(blended_head);

        // Stream-render and encode frames overlap..n on the fly (only 2 RGB frames in RAM!).
        let mut cur = vec![0u8; frame_bytes];
        for f in overlap..n {
            r.render_rgb_topdown(t0 + (f as f32) * dt, &mut scratch, &mut cur);
            if let Some(m) = pack_planar_delta(&prev, &cur, w, h, &mut raw_delta) {
                let comp = lz4_compress(&raw_delta);
                deltas.push(EncodedDelta {
                    blob: comp,
                    raw_len: raw_delta.len() as u32,
                    x0: m.x0,
                    y0: m.y0,
                    x1: m.x1,
                    y1: m.y1,
                    tile_mask: m.tile_mask,
                    delay_ms: frame_delay,
                });
            } else {
                deltas.push(EncodedDelta {
                    blob: Vec::new(),
                    raw_len: 0,
                    x0: 0,
                    y0: 0,
                    x1: 0,
                    y1: 0,
                    tile_mask: [0u16; TILE_ROWS],
                    delay_ms: frame_delay,
                });
            }
            std::mem::swap(&mut prev, &mut cur);
        }

        // Encode seamless wrap delta: frame (n-1) -> frame 0 (`base_rgb`)!
        if let Some(m) = pack_planar_delta(&prev, &base_rgb, w, h, &mut raw_delta) {
            let comp = lz4_compress(&raw_delta);
            deltas[0] = EncodedDelta {
                blob: comp,
                raw_len: raw_delta.len() as u32,
                x0: m.x0,
                y0: m.y0,
                x1: m.x1,
                y1: m.y1,
                tile_mask: m.tile_mask,
                delay_ms: frame_delay,
            };
        }

        match write_mwa_atomic(out_mwa, fw, fh, fps, 0, src_hash, &base_rgb, &deltas) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("misa-wall baker: {e}");
                1
            }
        }
    }
}

// ============================================================================
// User-supplied live wallpaper detection & baking (GIF, WebP, APNG, Video)
// Inspired by awww: per-frame variable delays, duplicate-frame coalescing,
// wraparound deltas, and zero decoder libraries loaded in the persistent daemon!
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimKind {
    AnimatedGif,
    AnimatedWebp,
    AnimatedPng,
    Video,
}

pub fn is_video_ext(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "mp4" | "webm" | "mkv" | "mov" | "m4v"
    )
}

#[inline]
fn read_byte<R: Read>(r: &mut R) -> Option<u8> {
    let mut b = [0u8; 1];
    if r.read_exact(&mut b).is_ok() { Some(b[0]) } else { None }
}

/// Streaming pure-Rust GIF block walker using an 8 KB `BufReader` (0 full-file RAM!).
/// Stops early once `stop_at` frames are seen (e.g. `Some(2)` for <0.05 ms detection).
pub fn gif_frame_count_limit(path: &Path, stop_at: Option<usize>) -> Option<usize> {
    let file = File::open(path).ok()?;
    let mut r = BufReader::with_capacity(8192, file);
    let mut hdr = [0u8; 13];
    r.read_exact(&mut hdr).ok()?;
    if &hdr[0..6] != b"GIF87a" && &hdr[0..6] != b"GIF89a" {
        return None;
    }
    let packed = hdr[10];
    if (packed & 0x80) != 0 {
        let gct = 3i64 << ((packed & 0x07) + 1);
        r.seek(SeekFrom::Current(gct)).ok()?;
    }
    let mut frames = 0usize;
    while let Some(tag) = read_byte(&mut r) {
        match tag {
            0x3B => break, // trailer
            0x00 => {}
            0x21 => {
                // Extension block: skip 1-byte label + sub-blocks until 0x00
                let _ = read_byte(&mut r)?;
                while let Some(sz) = read_byte(&mut r) {
                    if sz == 0 {
                        break;
                    }
                    if r.seek(SeekFrom::Current(sz as i64)).is_err() {
                        return Some(frames);
                    }
                }
            }
            0x2C => {
                // Image Descriptor: 9 bytes following 0x2C
                frames += 1;
                if let Some(lim) = stop_at {
                    if frames >= lim {
                        return Some(frames);
                    }
                }
                let mut desc = [0u8; 9];
                if r.read_exact(&mut desc).is_err() {
                    break;
                }
                let ipacked = desc[8];
                if (ipacked & 0x80) != 0 {
                    let lct = 3i64 << ((ipacked & 0x07) + 1);
                    if r.seek(SeekFrom::Current(lct)).is_err() {
                        break;
                    }
                }
                if read_byte(&mut r).is_none() {
                    break; // LZW minimum code size
                }
                while let Some(sz) = read_byte(&mut r) {
                    if sz == 0 {
                        break;
                    }
                    if r.seek(SeekFrom::Current(sz as i64)).is_err() {
                        return Some(frames);
                    }
                }
            }
            _ => break,
        }
    }
    Some(frames)
}

/// Pure-Rust WebP RIFF header check for `VP8X` animation bit or `ANIM` chunk.
pub fn webp_is_animated(path: &Path) -> bool {
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    let mut buf = vec![0u8; 65536];
    let n = f.read(&mut buf).unwrap_or(0);
    if n < 20 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WEBP" {
        return false;
    }
    let mut p = 12usize;
    while p + 8 <= n {
        let tag = &buf[p..p + 4];
        let clen = u32::from_le_bytes(buf[p + 4..p + 8].try_into().unwrap()) as usize;
        if tag == b"VP8X" && p + 9 <= n {
            if (buf[p + 8] & 0x02) != 0 {
                return true;
            }
        } else if tag == b"ANIM" {
            return true;
        }
        let step = 8usize.saturating_add(clen).saturating_add(clen & 1);
        if step == 0 || p.saturating_add(step) > n {
            break;
        }
        p += step;
    }
    false
}

/// Pure-Rust PNG chunk scanner for `acTL` (APNG) with `num_frames > 1`.
pub fn png_is_animated(path: &Path) -> bool {
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    let mut buf = vec![0u8; 65536];
    let n = f.read(&mut buf).unwrap_or(0);
    if n < 24 || &buf[0..8] != b"\x89PNG\r\n\x1a\n" {
        return false;
    }
    let mut p = 8usize;
    while p + 12 <= n {
        let clen = u32::from_be_bytes(buf[p..p + 4].try_into().unwrap()) as usize;
        let ctype = &buf[p + 4..p + 8];
        if ctype == b"acTL" && clen >= 4 && p + 12 <= n {
            let num_frames = u32::from_be_bytes(buf[p + 8..p + 12].try_into().unwrap());
            return num_frames > 1;
        }
        if ctype == b"IDAT" || ctype == b"IEND" {
            break;
        }
        let step = 12usize.saturating_add(clen);
        if step == 0 || p.saturating_add(step) > n {
            break;
        }
        p += step;
    }
    false
}

/// Detect whether `path` is a user-supplied animated wallpaper (GIF, animated WebP,
/// APNG, or video loop).
pub fn detect_media_anim(path: &Path) -> Option<AnimKind> {
    if std::env::var("MISA_WALL_LIVE").ok().as_deref() == Some("0") {
        return None;
    }
    if is_video_ext(path) {
        return Some(AnimKind::Video);
    }
    let mut magic = [0u8; 16];
    let mut f = File::open(path).ok()?;
    let n = f.read(&mut magic).ok()?;
    if n >= 6 && (&magic[0..6] == b"GIF87a" || &magic[0..6] == b"GIF89a") {
        if gif_frame_count_limit(path, Some(2)).unwrap_or(0) > 1 {
            return Some(AnimKind::AnimatedGif);
        }
        return None;
    }
    if n >= 12 && &magic[0..4] == b"RIFF" && &magic[8..12] == b"WEBP" {
        if webp_is_animated(path) {
            return Some(AnimKind::AnimatedWebp);
        }
        return None;
    }
    if n >= 8 && &magic[0..8] == b"\x89PNG\r\n\x1a\n" {
        if png_is_animated(path) {
            return Some(AnimKind::AnimatedPng);
        }
        return None;
    }
    None
}

fn choose_media_dims(
    sw: u32,
    sh: u32,
    max_w: u32,
    max_h: u32,
    force_nearest: bool,
) -> (u32, u32, u32) {
    let sw = sw.max(2);
    let sh = sh.max(2);
    let scale = (max_w as f64 / sw as f64)
        .min(max_h as f64 / sh as f64)
        .min(1.0);
    let dw = (((sw as f64 * scale).round() as u32).max(2) & !1).max(2);
    let dh = (((sh as f64 * scale).round() as u32).max(2) & !1).max(2);
    let flags = if force_nearest || (sw <= 320 && sh <= 240) {
        0
    } else {
        MWA_FLAG_LINEAR
    };
    (dw, dh, flags)
}

/// Reusable bilinear coordinate LUT built once per animation bake (zero per-frame trig/float!).
struct ScaleLut {
    sw: usize,
    sh: usize,
    ch: usize,
    dw: usize,
    dh: usize,
    x0: Vec<usize>,
    x1: Vec<usize>,
    wx: Vec<u32>,
    y0: Vec<usize>,
    y1: Vec<usize>,
    wy: Vec<u32>,
}

impl ScaleLut {
    fn new(sw: usize, sh: usize, ch: usize, dw: usize, dh: usize) -> Self {
        let mut x0 = Vec::with_capacity(dw);
        let mut x1 = Vec::with_capacity(dw);
        let mut wx = Vec::with_capacity(dw);
        let xs = (sw as f64) / (dw as f64);
        for x in 0..dw {
            let fx = ((x as f64 + 0.5) * xs - 0.5).clamp(0.0, (sw - 1) as f64);
            let i0 = fx.floor() as usize;
            let i1 = (i0 + 1).min(sw - 1);
            x0.push(i0 * ch);
            x1.push(i1 * ch);
            wx.push(((fx - i0 as f64) * 256.0).round() as u32);
        }
        let mut y0 = Vec::with_capacity(dh);
        let mut y1 = Vec::with_capacity(dh);
        let mut wy = Vec::with_capacity(dh);
        let ys = (sh as f64) / (dh as f64);
        for y in 0..dh {
            let fy = ((y as f64 + 0.5) * ys - 0.5).clamp(0.0, (sh - 1) as f64);
            let j0 = fy.floor() as usize;
            let j1 = (j0 + 1).min(sh - 1);
            y0.push(j0);
            y1.push(j1);
            wy.push(((fy - j0 as f64) * 256.0).round() as u32);
        }
        Self { sw, sh, ch, dw, dh, x0, x1, wx, y0, y1, wy }
    }

    /// Specialized RGB / RGBA-with-opaque-fast-path bilinear scaler (~3-4x faster).
    fn scale_into(&self, src: &[u8], rowstride: usize, out: &mut [u8]) {
        let dw = self.dw;
        let dh = self.dh;
        if self.sw == dw && self.sh == dh {
            if self.ch == 3 {
                for y in 0..dh {
                    out[y * dw * 3..(y + 1) * dw * 3]
                        .copy_from_slice(&src[y * rowstride..y * rowstride + dw * 3]);
                }
            } else {
                for y in 0..dh {
                    let s_row = &src[y * rowstride..];
                    let d_row = &mut out[y * dw * 3..(y + 1) * dw * 3];
                    for x in 0..dw {
                        let so = x * 4;
                        let a = s_row[so + 3] as u32;
                        if a == 255 {
                            d_row[x * 3] = s_row[so];
                            d_row[x * 3 + 1] = s_row[so + 1];
                            d_row[x * 3 + 2] = s_row[so + 2];
                        } else if a == 0 {
                            d_row[x * 3] = 0;
                            d_row[x * 3 + 1] = 0;
                            d_row[x * 3 + 2] = 0;
                        } else {
                            // Fast exact /255: (v * a + 128 + ((v * a + 128) >> 8)) >> 8
                            for c in 0..3 {
                                let t = s_row[so + c] as u32 * a + 128;
                                d_row[x * 3 + c] = ((t + (t >> 8)) >> 8) as u8;
                            }
                        }
                    }
                }
            }
            return;
        }

        if self.ch == 3 {
            for y in 0..dh {
                let wy = self.wy[y];
                let iy = 256 - wy;
                let r0 = &src[self.y0[y] * rowstride..];
                let r1 = &src[self.y1[y] * rowstride..];
                let d_row = &mut out[y * dw * 3..(y + 1) * dw * 3];
                for x in 0..dw {
                    let o0 = self.x0[x];
                    let o1 = self.x1[x];
                    let wx = self.wx[x];
                    let ix = 256 - wx;
                    let w00 = ix * iy;
                    let w10 = wx * iy;
                    let w01 = ix * wy;
                    let w11 = wx * wy;
                    d_row[x * 3] = ((r0[o0] as u32 * w00 + r0[o1] as u32 * w10 + r1[o0] as u32 * w01 + r1[o1] as u32 * w11 + 32768) >> 16) as u8;
                    d_row[x * 3 + 1] = ((r0[o0 + 1] as u32 * w00 + r0[o1 + 1] as u32 * w10 + r1[o0 + 1] as u32 * w01 + r1[o1 + 1] as u32 * w11 + 32768) >> 16) as u8;
                    d_row[x * 3 + 2] = ((r0[o0 + 2] as u32 * w00 + r0[o1 + 2] as u32 * w10 + r1[o0 + 2] as u32 * w01 + r1[o1 + 2] as u32 * w11 + 32768) >> 16) as u8;
                }
            }
        } else {
            for y in 0..dh {
                let wy = self.wy[y];
                let iy = 256 - wy;
                let r0 = &src[self.y0[y] * rowstride..];
                let r1 = &src[self.y1[y] * rowstride..];
                let d_row = &mut out[y * dw * 3..(y + 1) * dw * 3];
                for x in 0..dw {
                    let o0 = self.x0[x];
                    let o1 = self.x1[x];
                    let wx = self.wx[x];
                    let ix = 256 - wx;
                    let w00 = ix * iy;
                    let w10 = wx * iy;
                    let w01 = ix * wy;
                    let w11 = wx * wy;
                    let a00 = r0[o0 + 3];
                    let a10 = r0[o1 + 3];
                    let a01 = r1[o0 + 3];
                    let a11 = r1[o1 + 3];
                    if (a00 & a10 & a01 & a11) == 255 {
                        d_row[x * 3] = ((r0[o0] as u32 * w00 + r0[o1] as u32 * w10 + r1[o0] as u32 * w01 + r1[o1] as u32 * w11 + 32768) >> 16) as u8;
                        d_row[x * 3 + 1] = ((r0[o0 + 1] as u32 * w00 + r0[o1 + 1] as u32 * w10 + r1[o0 + 1] as u32 * w01 + r1[o1 + 1] as u32 * w11 + 32768) >> 16) as u8;
                        d_row[x * 3 + 2] = ((r0[o0 + 2] as u32 * w00 + r0[o1 + 2] as u32 * w10 + r1[o0 + 2] as u32 * w01 + r1[o1 + 2] as u32 * w11 + 32768) >> 16) as u8;
                    } else {
                        let premul = |v: u8, a: u8| -> u32 {
                            let t = v as u32 * a as u32 + 128;
                            (t + (t >> 8)) >> 8
                        };
                        for c in 0..3 {
                            let p00 = premul(r0[o0 + c], a00);
                            let p10 = premul(r0[o1 + c], a10);
                            let p01 = premul(r1[o0 + c], a01);
                            let p11 = premul(r1[o1 + c], a11);
                            d_row[x * 3 + c] = ((p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11 + 32768) >> 16) as u8;
                        }
                    }
                }
            }
        }
    }
}

fn scale_to_rgb(
    src: &[u8],
    sw: usize,
    sh: usize,
    rowstride: usize,
    ch: usize,
    dw: usize,
    dh: usize,
    out: &mut [u8],
) {
    ScaleLut::new(sw, sh, ch, dw, dh).scale_into(src, rowstride, out);
}

/// Streaming planar-delta MWA3 encoder with `awww`-style duplicate-frame coalescing.
struct MediaFrameEncoder {
    w: usize,
    h: usize,
    base_rgb: Vec<u8>,
    prev_rgb: Vec<u8>,
    deltas: Vec<EncodedDelta>,
    raw_delta: Vec<u8>,
    has_base: bool,
}

impl MediaFrameEncoder {
    fn new(w: u32, h: u32) -> Self {
        let n = (w as usize) * (h as usize) * 3;
        Self {
            w: w as usize,
            h: h as usize,
            base_rgb: Vec::new(),
            prev_rgb: Vec::new(),
            deltas: Vec::new(),
            raw_delta: Vec::with_capacity(n / 2),
            has_base: false,
        }
    }

    fn push_frame(&mut self, rgb: &[u8], delay_ms: u16) {
        let d = delay_ms.max(16);
        if !self.has_base {
            self.base_rgb = rgb.to_vec();
            self.prev_rgb = rgb.to_vec();
            // Slot 0 stores the wraparound delta (last -> frame 0) and frame 0's delay.
            self.deltas.push(EncodedDelta {
                blob: Vec::new(),
                raw_len: 0,
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
                tile_mask: [0u16; TILE_ROWS],
                delay_ms: d,
            });
            self.has_base = true;
            return;
        }
        if let Some(m) = pack_planar_delta(&self.prev_rgb, rgb, self.w, self.h, &mut self.raw_delta) {
            let comp = lz4_compress(&self.raw_delta);
            self.deltas.push(EncodedDelta {
                blob: comp,
                raw_len: self.raw_delta.len() as u32,
                x0: m.x0,
                y0: m.y0,
                x1: m.x1,
                y1: m.y1,
                tile_mask: m.tile_mask,
                delay_ms: d,
            });
            self.prev_rgb.copy_from_slice(rgb);
        } else if let Some(last) = self.deltas.last_mut() {
            // Coalesce identical consecutive frame (awww optimization): sum duration!
            last.delay_ms = last.delay_ms.saturating_add(d);
        }
    }

    fn finish(mut self, out_mwa: &Path, flags: u32, src_hash: u64) -> Result<(), String> {
        if self.deltas.len() <= 1 {
            return Err("animation has only 1 unique frame".into());
        }
        // Encode seamless wraparound delta: last unique frame -> frame 0 (`base_rgb`).
        if let Some(m) = pack_planar_delta(&self.prev_rgb, &self.base_rgb, self.w, self.h, &mut self.raw_delta) {
            let comp = lz4_compress(&self.raw_delta);
            let f0_delay = self.deltas[0].delay_ms;
            self.deltas[0] = EncodedDelta {
                blob: comp,
                raw_len: self.raw_delta.len() as u32,
                x0: m.x0,
                y0: m.y0,
                x1: m.x1,
                y1: m.y1,
                tile_mask: m.tile_mask,
                delay_ms: f0_delay,
            };
        } else if self.deltas.len() > 2 {
            // Last frame is identical to frame 0: merge its duration into frame 0 and drop duplicate tail!
            let tail = self.deltas.pop().unwrap();
            self.deltas[0].delay_ms = self.deltas[0].delay_ms.saturating_add(tail.delay_ms);
        }

        let total_ms: u64 = self.deltas.iter().map(|d| d.delay_ms.max(16) as u64).sum();
        let avg_ms = (total_ms / self.deltas.len() as u64).max(16);
        let nominal_fps = ((1000 + avg_ms / 2) / avg_ms).clamp(1, 60) as u32;

        write_mwa_atomic(
            out_mwa,
            self.w as u32,
            self.h as u32,
            nominal_fps,
            flags,
            src_hash,
            &self.base_rgb,
            &self.deltas,
        )
    }
}

// --- Native Animated WebP decoder via lazy dlopen(libwebpdemux.so.2) ---

const WEBP_DEMUX_ABI_VERSION: c_int = 0x0107;

#[repr(C)]
struct WebPData {
    bytes: *const u8,
    size: usize,
}

#[repr(C)]
struct WebPAnimInfo {
    canvas_width: u32,
    canvas_height: u32,
    loop_count: u32,
    bgcolor: u32,
    frame_count: u32,
    pad: [u32; 4],
}

type FnWebPAnimDecoderNewInternal =
    unsafe extern "C" fn(*const WebPData, *const c_void, c_int) -> *mut c_void;
type FnWebPAnimDecoderGetInfo = unsafe extern "C" fn(*const c_void, *mut WebPAnimInfo) -> c_int;
type FnWebPAnimDecoderHasMoreFrames = unsafe extern "C" fn(*const c_void) -> c_int;
type FnWebPAnimDecoderGetNext =
    unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_int) -> c_int;
type FnWebPAnimDecoderDelete = unsafe extern "C" fn(*mut c_void);

unsafe fn bake_webp_native(
    path: &Path,
    out_mwa: &Path,
    src_hash: u64,
    max_w: u32,
    max_h: u32,
    max_frames: u32,
    force_nearest: bool,
) -> Result<(), String> {
    let mut lib = null_mut();
    for soname in ["libwebpdemux.so.2", "libwebpdemux.so"] {
        let cs = CString::new(soname).unwrap();
        lib = dlopen(cs.as_ptr(), RTLD_LAZY | RTLD_LOCAL);
        if !lib.is_null() {
            break;
        }
    }
    if lib.is_null() {
        return Err("libwebpdemux unavailable".into());
    }
    macro_rules! sym {
        ($n:expr, $t:ty) => {{
            let cs = CString::new($n).unwrap();
            let p = dlsym(lib, cs.as_ptr());
            if p.is_null() {
                return Err(concat!("missing ", $n).into());
            }
            std::mem::transmute::<*mut c_void, $t>(p)
        }};
    }
    let dec_new: FnWebPAnimDecoderNewInternal =
        sym!("WebPAnimDecoderNewInternal", FnWebPAnimDecoderNewInternal);
    let dec_info: FnWebPAnimDecoderGetInfo =
        sym!("WebPAnimDecoderGetInfo", FnWebPAnimDecoderGetInfo);
    let dec_more: FnWebPAnimDecoderHasMoreFrames =
        sym!("WebPAnimDecoderHasMoreFrames", FnWebPAnimDecoderHasMoreFrames);
    let dec_next: FnWebPAnimDecoderGetNext =
        sym!("WebPAnimDecoderGetNext", FnWebPAnimDecoderGetNext);
    let dec_del: FnWebPAnimDecoderDelete =
        sym!("WebPAnimDecoderDelete", FnWebPAnimDecoderDelete);

    let raw = fs::read(path).map_err(|e| format!("read webp: {e}"))?;
    let wdata = WebPData {
        bytes: raw.as_ptr(),
        size: raw.len(),
    };
    // Passing NULL options selects MODE_RGBA cleanly regardless of enum values.
    let dec = dec_new(&wdata, null(), WEBP_DEMUX_ABI_VERSION);
    if dec.is_null() {
        return Err("WebPAnimDecoderNew failed".into());
    }
    let mut info = WebPAnimInfo {
        canvas_width: 0,
        canvas_height: 0,
        loop_count: 0,
        bgcolor: 0,
        frame_count: 0,
        pad: [0; 4],
    };
    if dec_info(dec, &mut info) == 0 || info.canvas_width == 0 || info.canvas_height == 0 || info.frame_count <= 1 {
        dec_del(dec);
        return Err("not an animated webp".into());
    }

    let sw = info.canvas_width as usize;
    let sh = info.canvas_height as usize;
    let (dw, dh, flags) = choose_media_dims(info.canvas_width, info.canvas_height, max_w, max_h, force_nearest);
    let lut = ScaleLut::new(sw, sh, 4, dw as usize, dh as usize);
    let mut enc = MediaFrameEncoder::new(dw, dh);
    let mut rgb = vec![0u8; (dw as usize) * (dh as usize) * 3];
    let mut prev_ts: c_int = 0;
    let mut count = 0u32;

    while dec_more(dec) != 0 && count < max_frames {
        let mut buf_ptr: *mut u8 = null_mut();
        let mut ts: c_int = 0;
        if dec_next(dec, &mut buf_ptr, &mut ts) == 0 || buf_ptr.is_null() {
            break;
        }
        let delay = (ts - prev_ts).clamp(20, 10_000) as u16;
        prev_ts = ts;
        let rgba = std::slice::from_raw_parts(buf_ptr, sw * sh * 4);
        lut.scale_into(rgba, sw * 4, &mut rgb);
        enc.push_frame(&rgb, delay);
        count += 1;
    }
    dec_del(dec);
    enc.finish(out_mwa, flags, src_hash)
}

// --- Native Animated GIF decoder via lazy dlopen(libgdk_pixbuf-2.0.so.0) ---

#[repr(C)]
struct GTimeVal {
    tv_sec: c_long,
    tv_usec: c_long,
}

type FnAnimNewFromFile = unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> *mut c_void;
type FnAnimIsStatic = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnAnimGetInt = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnAnimGetIter = unsafe extern "C" fn(*mut c_void, *const GTimeVal) -> *mut c_void;
type FnIterGetDelay = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnIterGetPixbuf = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type FnIterAdvance = unsafe extern "C" fn(*mut c_void, *const GTimeVal) -> c_int;
type FnPbGetInt = unsafe extern "C" fn(*const c_void) -> c_int;
type FnPbGetPixels = unsafe extern "C" fn(*const c_void) -> *const c_uchar;
type FnObjUnref = unsafe extern "C" fn(*mut c_void);

unsafe fn bake_gif_native(
    path: &Path,
    out_mwa: &Path,
    src_hash: u64,
    max_w: u32,
    max_h: u32,
    max_frames: u32,
    force_nearest: bool,
) -> Result<(), String> {
    let total_frames = gif_frame_count_limit(path, Some(max_frames as usize))
        .filter(|&n| n > 1)
        .ok_or_else(|| "GIF has <= 1 frame".to_string())?
        .min(max_frames as usize);

    let mut lib_pb = null_mut();
    for soname in ["libgdk_pixbuf-2.0.so.0", "libgdk_pixbuf-2.0.so"] {
        let cs = CString::new(soname).unwrap();
        lib_pb = dlopen(cs.as_ptr(), RTLD_LAZY | RTLD_LOCAL);
        if !lib_pb.is_null() {
            break;
        }
    }
    if lib_pb.is_null() {
        return Err("libgdk_pixbuf unavailable".into());
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
        ($lib:expr, $n:expr, $t:ty) => {{
            let cs = CString::new($n).unwrap();
            let p = dlsym($lib, cs.as_ptr());
            if p.is_null() {
                return Err(concat!("missing ", $n).into());
            }
            std::mem::transmute::<*mut c_void, $t>(p)
        }};
    }

    let anim_new: FnAnimNewFromFile = sym!(lib_pb, "gdk_pixbuf_animation_new_from_file", FnAnimNewFromFile);
    let anim_static: FnAnimIsStatic = sym!(lib_pb, "gdk_pixbuf_animation_is_static_image", FnAnimIsStatic);
    let anim_w: FnAnimGetInt = sym!(lib_pb, "gdk_pixbuf_animation_get_width", FnAnimGetInt);
    let anim_h: FnAnimGetInt = sym!(lib_pb, "gdk_pixbuf_animation_get_height", FnAnimGetInt);
    let anim_iter: FnAnimGetIter = sym!(lib_pb, "gdk_pixbuf_animation_get_iter", FnAnimGetIter);
    let iter_delay: FnIterGetDelay = sym!(lib_pb, "gdk_pixbuf_animation_iter_get_delay_time", FnIterGetDelay);
    let iter_pb: FnIterGetPixbuf = sym!(lib_pb, "gdk_pixbuf_animation_iter_get_pixbuf", FnIterGetPixbuf);
    let iter_adv: FnIterAdvance = sym!(lib_pb, "gdk_pixbuf_animation_iter_advance", FnIterAdvance);
    let pb_w: FnPbGetInt = sym!(lib_pb, "gdk_pixbuf_get_width", FnPbGetInt);
    let pb_h: FnPbGetInt = sym!(lib_pb, "gdk_pixbuf_get_height", FnPbGetInt);
    let pb_rs: FnPbGetInt = sym!(lib_pb, "gdk_pixbuf_get_rowstride", FnPbGetInt);
    let pb_ch: FnPbGetInt = sym!(lib_pb, "gdk_pixbuf_get_n_channels", FnPbGetInt);
    let pb_px: FnPbGetPixels = sym!(lib_pb, "gdk_pixbuf_get_pixels", FnPbGetPixels);
    let unref: FnObjUnref = sym!(lib_go, "g_object_unref", FnObjUnref);

    let c_path = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| "bad path")?;
    let anim = anim_new(c_path.as_ptr(), null_mut());
    if anim.is_null() {
        return Err("gdk_pixbuf_animation_new_from_file failed".into());
    }
    if anim_static(anim) != 0 {
        unref(anim);
        return Err("GIF reported static".into());
    }
    let aw = anim_w(anim);
    let ah = anim_h(anim);
    if aw <= 0 || ah <= 0 {
        unref(anim);
        return Err("invalid GIF dimensions".into());
    }
    let (dw, dh, flags) = choose_media_dims(aw as u32, ah as u32, max_w, max_h, force_nearest);
    let mut enc = MediaFrameEncoder::new(dw, dh);
    let mut rgb = vec![0u8; (dw as usize) * (dh as usize) * 3];

    let mut tv = GTimeVal { tv_sec: 1, tv_usec: 0 };
    let iter = anim_iter(anim, &tv);
    if iter.is_null() {
        unref(anim);
        return Err("gdk_pixbuf_animation_get_iter failed".into());
    }

    let mut cached_lut: Option<ScaleLut> = None;
    for f in 0..total_frames {
        let raw_delay = iter_delay(iter);
        let delay_ms = if raw_delay < 0 {
            100u16
        } else if raw_delay == 0 {
            40u16
        } else {
            (raw_delay as u32).clamp(20, 10_000) as u16
        };
        let pb = iter_pb(iter);
        if !pb.is_null() {
            let sw = pb_w(pb) as usize;
            let sh = pb_h(pb) as usize;
            let rs = pb_rs(pb) as usize;
            let ch = pb_ch(pb) as usize;
            let px = pb_px(pb);
            if sw > 0 && sh > 0 && (ch == 3 || ch == 4) && !px.is_null() {
                let slice = std::slice::from_raw_parts(px, rs * sh);
                let need_new = cached_lut
                    .as_ref()
                    .map(|l| l.sw != sw || l.sh != sh || l.ch != ch)
                    .unwrap_or(true);
                if need_new {
                    cached_lut = Some(ScaleLut::new(sw, sh, ch, dw as usize, dh as usize));
                }
                cached_lut.as_ref().unwrap().scale_into(slice, rs, &mut rgb);
                enc.push_frame(&rgb, delay_ms);
            }
        }
        if raw_delay < 0 {
            break;
        }
        if f + 1 < total_frames {
            let step_ms = if raw_delay > 0 { raw_delay as c_long } else { 20 };
            tv.tv_usec += step_ms * 1000;
            tv.tv_sec += tv.tv_usec / 1_000_000;
            tv.tv_usec %= 1_000_000;
            let _ = iter_adv(iter, &tv);
        }
    }

    unref(iter);
    unref(anim);
    enc.finish(out_mwa, flags, src_hash)
}

// --- FFmpeg streaming decoder for APNG, video loops (MP4/WebM/MKV/MOV), & fallback ---

fn probe_media_size(path: &Path) -> Option<(u32, u32)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?.trim();
    let (ws, hs) = line.split_once('x')?;
    let w: u32 = ws.trim().parse().ok()?;
    let h: u32 = hs.trim().parse().ok()?;
    if w >= 2 && h >= 2 {
        Some((w, h))
    } else {
        None
    }
}

fn bake_ffmpeg_stream(
    path: &Path,
    out_mwa: &Path,
    src_hash: u64,
    max_w: u32,
    max_h: u32,
    max_fps: u32,
    max_frames: u32,
    force_nearest: bool,
) -> Result<(), String> {
    let (sw, sh) = probe_media_size(path).unwrap_or((max_w, max_h));
    let (dw, dh, flags) = choose_media_dims(sw, sh, max_w, max_h, force_nearest);
    let fps = max_fps.clamp(6, 20);
    let delay_ms = ((1000 + fps / 2) / fps).clamp(16, 1000) as u16;
    let scale_flags = if dw < sw || dh < sh { "area" } else { "bilinear" };
    let vf = format!("fps={fps},scale={dw}:{dh}:flags={scale_flags}");

    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args([
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &vf,
            "-frames:v",
            &max_frames.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("spawn ffmpeg: {e}"))?;

    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| "ffmpeg stdout unavailable".to_string())?;
    let frame_bytes = (dw as usize) * (dh as usize) * 3;
    let mut rgb = vec![0u8; frame_bytes];
    let mut enc = MediaFrameEncoder::new(dw, dh);
    let mut count = 0u32;

    while count < max_frames {
        match stdout.read_exact(&mut rgb) {
            Ok(()) => {
                enc.push_frame(&rgb, delay_ms);
                count += 1;
            }
            Err(_) => break,
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    enc.finish(out_mwa, flags, src_hash)
}

/// One-shot user animated media baker (`--bake-media`).
/// Supports Animated GIF, Animated WebP, APNG, and video loops (MP4/WebM/MKV/MOV).
pub fn bake_media_loop(
    media_path: &str,
    out_mwa: &Path,
    src_hash: u64,
    max_w: u32,
    max_h: u32,
    max_fps: u32,
    max_frames: u32,
    force_nearest: bool,
) -> i32 {
    let path = Path::new(media_path);
    let Some(kind) = detect_media_anim(path) else {
        return 1;
    };
    let res = match kind {
        AnimKind::AnimatedWebp => unsafe {
            bake_webp_native(path, out_mwa, src_hash, max_w, max_h, max_frames, force_nearest)
                .or_else(|_| {
                    bake_ffmpeg_stream(path, out_mwa, src_hash, max_w, max_h, max_fps, max_frames, force_nearest)
                })
        },
        AnimKind::AnimatedGif => unsafe {
            bake_gif_native(path, out_mwa, src_hash, max_w, max_h, max_frames, force_nearest)
                .or_else(|_| {
                    bake_ffmpeg_stream(path, out_mwa, src_hash, max_w, max_h, max_fps, max_frames, force_nearest)
                })
        },
        AnimKind::AnimatedPng | AnimKind::Video => {
            bake_ffmpeg_stream(path, out_mwa, src_hash, max_w, max_h, max_fps, max_frames, force_nearest)
        }
    };
    match res {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("misa-wall media baker: {e}");
            1
        }
    }
}

/// Extract a cover-cropped `tw x th` first-frame thumbnail as atomic P6 PPM (`--thumb`).
/// Used by `CardPicker` and `SddmBackground` for video loops and fallback media.
pub fn bake_media_thumb(media_path: &str, out_ppm: &str, tw: u32, th: u32) -> i32 {
    let tw = tw.clamp(16, 1920);
    let th = th.clamp(16, 1080);
    let vf = format!("scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th}");
    let frame_bytes = (tw as usize) * (th as usize) * 3;

    let mut rgb = vec![0u8; frame_bytes];
    let mut ok = false;

    if let Ok(mut child) = Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i", media_path])
        .args([
            "-an", "-sn", "-dn", "-frames:v", "1", "-vf", &vf, "-f", "rawvideo", "-pix_fmt",
            "rgb24", "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        if let Some(mut stdout) = child.stdout.take() {
            ok = stdout.read_exact(&mut rgb).is_ok();
        }
        let _ = child.kill();
        let _ = child.wait();
    }

    if !ok {
        unsafe {
            if let Some((sw, sh, src)) = decode_rgb_image_scaled(media_path, tw as i32 * 2, th as i32 * 2) {
                if sw > 0 && sh > 0 {
                    scale_to_rgb(&src, sw as usize, sh as usize, (sw as usize) * 3, 3, tw as usize, th as usize, &mut rgb);
                    ok = true;
                }
            }
        }
    }

    if !ok {
        return 1;
    }

    let mut ppm = Vec::with_capacity(32 + frame_bytes);
    let _ = write!(ppm, "P6\n{tw} {th}\n255\n");
    ppm.extend_from_slice(&rgb);
    let tmp = format!("{out_ppm}.{}.tmp", unsafe { getpid() });
    if fs::write(&tmp, &ppm).is_ok() && fs::rename(&tmp, out_ppm).is_ok() {
        0
    } else {
        let _ = fs::remove_file(&tmp);
        1
    }
}

/// Extract a single cover-cropped BGRX frame from a video for `--static` fallback.
pub fn decode_video_static_bgrx(path: &str, sw: i32, sh: i32, monitors: &[crate::MonRect]) -> Option<Vec<u8>> {
    let mut target = vec![0u8; (sw as usize) * (sh as usize) * 4];
    let mut any = false;
    for m in monitors {
        let mw = m.w.max(2) as usize;
        let mh = m.h.max(2) as usize;
        let vf = format!("scale={mw}:{mh}:force_original_aspect_ratio=increase,crop={mw}:{mh}");
        let Ok(mut child) = Command::new("ffmpeg")
            .args(["-v", "error", "-nostdin", "-i", path])
            .args([
                "-an", "-sn", "-dn", "-frames:v", "1", "-vf", &vf, "-f", "rawvideo", "-pix_fmt",
                "rgb24", "-",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        else {
            continue;
        };
        let mut rgb = vec![0u8; mw * mh * 3];
        let got = child
            .stdout
            .take()
            .map(|mut s| s.read_exact(&mut rgb).is_ok())
            .unwrap_or(false);
        let _ = child.kill();
        let _ = child.wait();
        if !got {
            continue;
        }
        any = true;
        for y in 0..m.h {
            let dy = m.y + y;
            if dy < 0 || dy >= sh {
                continue;
            }
            let max_cols = (m.w.min(sw - m.x.max(0))).max(0) as usize;
            let s_row = &rgb[(y as usize) * mw * 3..];
            let dst_off = ((dy as usize) * (sw as usize) + (m.x.max(0) as usize)) * 4;
            for x in 0..max_cols {
                target[dst_off + x * 4] = s_row[x * 3 + 2];
                target[dst_off + x * 4 + 1] = s_row[x * 3 + 1];
                target[dst_off + x * 4 + 2] = s_row[x * 3];
                target[dst_off + x * 4 + 3] = 0;
            }
        }
    }
    if any { Some(target) } else { None }
}
